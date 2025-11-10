from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from .. import losses
from ..helper import load_dataset

from ..evaluation import evaluate_incremental, IncrementalMetrics
from ...trainer.train_cgcd_incremental import Stage2Result

logger = logging.getLogger(__name__)


@dataclass
class Stage3Result:
    best_acc_all: float
    best_acc_old: float
    best_acc_new: float
    best_epoch_all: int
    best_epoch_old: int
    best_epoch_new: int
    last_checkpoint: Path
    best_checkpoint_all: Optional[Path]
    best_checkpoint_old: Optional[Path]
    best_checkpoint_new: Optional[Path]


def _freeze_batch_norm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def _param_difference(main_params, subset_params):
    return list(set(main_params).difference(set(subset_params)))


def train_incremental_stage(
    args,
    base_model: torch.nn.Module,
    base_criterion,
    stage2: Stage2Result,
    experiment_dir: Path,
    initial_acc: float,
    dataloader_eval: DataLoader,
) -> Stage3Result:
    base_model.eval()

    nb_classes_now = stage2.nb_classes_prev + stage2.nb_classes_new
    criterion_now = losses.Proxy_Anchor(
        nb_classes=nb_classes_now,
        sz_embed=args.sz_embedding,
        mrg=args.mrg,
        alpha=args.alpha,
    ).cuda()
    criterion_now.proxies.data[: stage2.nb_classes_prev] = base_criterion.proxies.data
    criterion_now.proxies.data[stage2.nb_classes_prev :] = torch.from_numpy(stage2.cluster_centers).cuda()

    model_now = copy.deepcopy(base_model).cuda()

    if args.gpu_id != -1:
        embedding_params = list(model_now.model.embedding.parameters())
        backbone_params = _param_difference(model_now.parameters(), embedding_params)
    else:
        embedding_params = list(model_now.module.model.embedding.parameters())
        backbone_params = _param_difference(model_now.module.parameters(), embedding_params)

    param_groups = [
        {"params": backbone_params},
        {"params": embedding_params, "lr": float(args.lr)},
        {"params": criterion_now.parameters(), "lr": float(args.lr) * 100},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(args.lr),
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    best_acc_all = 0.0
    best_acc_old = 0.0
    best_acc_new = 0.0
    best_epoch_all = 0
    best_epoch_old = 0
    best_epoch_new = 0
    best_projection_all: Optional[np.ndarray] = None
    best_projection_old: Optional[np.ndarray] = None
    best_projection_new: Optional[np.ndarray] = None

    best_all_path = experiment_dir / f"{args.dataset}_{args.model}_model_last_step_best_all.pth"
    best_old_path = experiment_dir / f"{args.dataset}_{args.model}_model_last_step_best_old.pth"
    best_new_path = experiment_dir / f"{args.dataset}_{args.model}_model_last_step_best_new.pth"
    last_path = experiment_dir / f"{args.dataset}_{args.model}_model_last_step_1.pth"
    result_file = experiment_dir / "result.txt"

    for epoch in range(args.nb_epochs):
        model_now.train()

        if args.bn_freeze:
            _freeze_batch_norm(model_now.model if args.gpu_id != -1 else model_now.module.model)

        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_params = list(model_now.model.embedding.parameters()) + list(criterion_now.parameters())
                param_pool = model_now.parameters()
            else:
                unfreeze_params = list(model_now.module.model.embedding.parameters()) + list(
                    criterion_now.parameters()
                )
                param_pool = model_now.module.parameters()

            if epoch == 0:
                for param in _param_difference(param_pool, unfreeze_params):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in _param_difference(param_pool, unfreeze_params):
                    param.requires_grad = True

        metrics: IncrementalMetrics = evaluate_incremental(
            model_now,
            dataloader_eval,
            criterion_now,
            stage2.nb_classes_prev,
            stage2.nb_classes_prev,
        )

        logger.info(
            "Epoch %d — Acc(all/old/new): %.4f / %.4f / %.4f",
            epoch,
            metrics.acc_all,
            metrics.acc_old,
            metrics.acc_new,
        )
        for cls_stats in metrics.per_class:
            logger.debug(
                "Class %-20s Precision %.3f Recall %.3f F1 %.3f Acc %.3f",
                cls_stats["class"],
                cls_stats["precision"],
                cls_stats["recall"],
                cls_stats["f1"],
                cls_stats["accuracy"],
            )

        if metrics.acc_all > best_acc_all:
            best_acc_all = metrics.acc_all
            best_epoch_all = epoch
            best_projection_all = metrics.cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_all,
                    "label2cls": metrics.label2cls,
                    "nb_classes_old": stage2.nb_classes_prev,
                    "nb_classes_new": stage2.nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                best_all_path,
            )

        if metrics.acc_old > best_acc_old:
            best_acc_old = metrics.acc_old
            best_epoch_old = epoch
            best_projection_old = metrics.cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_old,
                    "label2cls": metrics.label2cls,
                    "nb_classes_old": stage2.nb_classes_prev,
                    "nb_classes_new": stage2.nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                best_old_path,
            )

        if metrics.acc_new > best_acc_new:
            best_acc_new = metrics.acc_new
            best_epoch_new = epoch
            best_projection_new = metrics.cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_new,
                    "label2cls": metrics.label2cls,
                    "nb_classes_old": stage2.nb_classes_prev,
                    "nb_classes_new": stage2.nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                best_new_path,
            )

        pbar = tqdm(enumerate(stage2.merged_loader), total=len(stage2.merged_loader))
        for batch_idx, (image, target, _) in pbar:
            feats = model_now(image.squeeze().cuda())

            target_new_flag = torch.where(target > stage2.nb_classes_prev, 1, 0)
            target_old_count = target.size(0) - target_new_flag.sum()

            if target_old_count > 0:
                target_sampling = torch.randint(stage2.nb_classes_prev, (target_old_count,))
                feats_sampling = torch.normal(
                    base_criterion.proxies[target_sampling],
                    stage2.exampler_sampling,
                ).cuda()
                target = torch.cat((target, target_sampling), dim=0)
                feats = torch.cat((feats, feats_sampling), dim=0)

            loss_pa = criterion_now(feats, target.squeeze().cuda())

            target_old_mask = torch.nonzero(target_new_flag)
            if target_old_mask.size(0) > 1:
                target_old_mask = target_old_mask.squeeze()
                imag_old = torch.unsqueeze(image[target_old_mask[0]], dim=0)
                feats_new = torch.unsqueeze(feats[target_old_mask[0]], dim=0)

                for kd_idx in range(1, target_old_mask.size(0)):
                    imag_old = torch.cat(
                        (imag_old, torch.unsqueeze(image[target_old_mask[kd_idx]], dim=0)),
                        dim=0,
                    )
                    feats_new = torch.cat(
                        (feats_new, torch.unsqueeze(feats[target_old_mask[kd_idx]], dim=0)),
                        dim=0,
                    )

                with torch.no_grad():
                    feats_old = base_model(imag_old.squeeze().cuda())
                feats_new = feats_new.cuda()
                loss_kd = torch.dist(
                    F.normalize(feats_old.view(feats_old.size(0) * feats_old.size(1), 1), dim=0).detach(),
                    F.normalize(feats_new.view(feats_old.size(0) * feats_old.size(1), 1), dim=0),
                )
            else:
                loss_kd = torch.tensor(0.0).cuda()

            loss = loss_pa * 1.0 + loss_kd * 10.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                f"Train Epoch: {epoch} [{batch_idx + 1}/{len(stage2.merged_loader)} "
                f"({100.0 * batch_idx / len(stage2.merged_loader):.0f}%)] "
                f"Loss total [{loss.item():4.3f} pa: {loss_pa.item():4.3f} kd: {loss_kd.item():4.3f}]"
            )

        scheduler.step()

        summary_line = (
            f"Valid Epoch: {epoch} Acc_0: {initial_acc:4.3f} "
            f"Acc: {metrics.acc_all:4.3f}/{metrics.acc_old:4.3f}/{metrics.acc_new:4.3f} "
            f"Best result: {best_epoch_all}/{best_epoch_old}/{best_epoch_new} "
            f"{best_acc_all:4.3f}/{best_acc_old:4.3f}/{best_acc_new:4.3f}"
        )
        logger.info(summary_line)
        with open(result_file, "a+", encoding="utf-8") as handle:
            handle.write(summary_line + "\n")

        torch.save(
            {
                "model_pa_state_dict": model_now.state_dict(),
                "proxies_param": criterion_now.proxies,
                "best_projection": best_projection_all,
                "nb_classes_old": stage2.nb_classes_prev,
                "nb_classes_new": stage2.nb_classes_new,
                "nb_classes_total": nb_classes_now,
                "best_acc_all": best_acc_all,
                "best_epoch": best_epoch_all,
            },
            last_path,
        )

    return Stage3Result(
        best_acc_all=best_acc_all,
        best_acc_old=best_acc_old,
        best_acc_new=best_acc_new,
        best_epoch_all=best_epoch_all,
        best_epoch_old=best_epoch_old,
        best_epoch_new=best_epoch_new,
        last_checkpoint=last_path,
        best_checkpoint_all=best_all_path if best_acc_all > 0 else None,
        best_checkpoint_old=best_old_path if best_acc_old > 0 else None,
        best_checkpoint_new=best_new_path if best_acc_new > 0 else None,
    )
