from __future__ import annotations

import json
import logging
import copy
import tqdm

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
from numpy.typing import NDArray

import cv2
import numpy as np
from PIL import Image
import os


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from trainer.cgcd_stage_utils import build_cgcd_args

import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support, classification_report

from CGCD import dataset
from CGCD import losses
from CGCD import utils
from CGCD.net.splitNet import SplitModlue


logger = logging.getLogger(__name__)


# @dataclass
# class Stage2Result:
#     cgcd_model: nn.Module
#     criterion: nn.Module
#     checkpoint_info: Dict
#     merged_loader: DataLoader
#     old_loader: DataLoader
#     new_loader: DataLoader
#     best_projection: NDArray
#     label2cls: Dict
#     nb_classes_prev: int
#     nb_classes_new: int
#     exampler_sampling: torch.Tensor
#     pseudo_labels_old: np.ndarray
#     pseudo_labels_new: np.ndarray
#     preference: float
#     cluster_centers: np.ndarray
#     split_data_path: Path
#     index_old: np.ndarray
#     index_new: np.ndarray
#     affinity_model: AffinityPropagation
#     gaussian_model: Optional[GaussianMixture]


@dataclass
class CGCDStageArtifacts:
    model: torch.nn.Module
    criterion: torch.nn.Module
    checkpoint_path: Path
    checkpoint_info: np.ndarray
    label2cls: np.ndarray
    experiment_dir: Path
    nb_classes_prev: int


class create_ccd_dataset(Dataset):
    """
    Input: dataset class and splitted data index list
    Return: a new dataset class that consists only the splitted data considering CCD stage
            where stage 0 is labelled data and stage > 0 is unlabelled data
    """

    def __init__(self, dataset, transform, stage) -> None:
        super(create_ccd_dataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.batch_labeled_or_not = 1 if stage == 0 else 0

    def __getitem__(self, index):
        batch_data = cv2.imread(self.dataset["paths"][index])
        batch_data = cv2.cvtColor(batch_data, cv2.COLOR_BGR2RGB)
        batch_data = Image.fromarray(batch_data)

        batch_label = self.dataset["labels"][index]
        batch_unique_index = self.dataset["uq_idx"][index]

        batch_data = self.transform(batch_data)
        return batch_data, batch_label, batch_unique_index, np.array([self.batch_labeled_or_not])

    def __len__(self):
        return self.dataset["len"]


def generate_my_dataset(ccd_dataset, indices):
    if isinstance(indices, torch.Tensor):
        indices = indices.detach().cpu().numpy()
    subset_idx = np.asarray(indices, dtype=np.int64)

    src = ccd_dataset.dataset  # {"paths", "labels", "uq_idx", "len"}
    subset = {
        "paths": [src["paths"][i] for i in subset_idx],
        "labels": [src["labels"][i] for i in subset_idx],
        "uq_idx": np.asarray(src["uq_idx"])[subset_idx],
        "len": len(subset_idx),
    }

    stage_flag = 0 if getattr(ccd_dataset, "batch_labeled_or_not", 0) == 1 else 1
    return create_ccd_dataset(subset, ccd_dataset.transform, stage_flag)


def merge_my_dataset(dataset_old, dataset_new):
    """
    Merge two CCD datasets built from `create_ccd_dataset` preserving transforms and masks.
    """

    def _extract(ccd_dataset):
        if isinstance(ccd_dataset, create_ccd_dataset):
            return ccd_dataset.dataset, ccd_dataset.transform, getattr(ccd_dataset, "batch_labeled_or_not", 0)
        if isinstance(ccd_dataset, dict):
            return ccd_dataset, None, 0
        raise ValueError("Unsupported dataset type for merge_my_dataset")

    src_old, transform_old, flag_old = _extract(dataset_old)
    src_new, transform_new, flag_new = _extract(dataset_new)

    transform = transform_old or transform_new
    if transform_old and transform_new and transform_old is not transform_new:
        logger.warning("merge_my_dataset received datasets with different transforms; using the first one.")

    paths = list(src_old["paths"]) + list(src_new["paths"])
    labels = list(src_old["labels"]) + list(src_new["labels"])
    uq_old = np.asarray(src_old["uq_idx"])
    uq_new = np.asarray(src_new["uq_idx"])
    if uq_old.size == 0:
        uq_idx = uq_new.copy()
    elif uq_new.size == 0:
        uq_idx = uq_old.copy()
    else:
        uq_idx = np.concatenate([uq_old, uq_new])

    subset = {
        "paths": paths,
        "labels": labels,
        "uq_idx": uq_idx,
        "len": len(paths),
    }

    stage_flag = 0 if (flag_old == 1 and flag_new == 1) else 1
    return create_ccd_dataset(subset, transform, stage_flag)


def _infer_nb_classes(dataset) -> int:
    if hasattr(dataset, "nb_classes"):
        return dataset.nb_classes()
    labels = None
    if hasattr(dataset, "dataset"):
        inner = dataset.dataset
        if isinstance(inner, dict) and "labels" in inner:
            labels = inner["labels"]
    if labels is None and hasattr(dataset, "labels"):
        labels = dataset.labels
    if labels is None:
        raise ValueError("Unable to infer number of classes for CGCD training.")
    return len(set(int(x) for x in labels))


class _CGCDLoaderAdapter:
    """
    Wrap a dataloader that yields (image, label, uq_idx, mask) tuples so CGCD's
    training utilities, which expect (image, label, index), can reuse them
    without duplicating datasets.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        for batch in self.dataloader:
            if len(batch) == 4:
                image, label, index, _ = batch
            else:
                image, label, index = batch
            yield image, label, index


def run_discover_novel_categories(
    args,
    model: torch.nn.Module,
    criterion,
    experiment_dir: Path,
    nb_classes_prev: int,
    previous_loader: DataLoader,
    dataset_train_now: dataset,
    dataloader_train_now: DataLoader,
    dataloader_eval_now: DataLoader,
) -> CGCDStageArtifacts:
    model.eval()

    previous_loader = _CGCDLoaderAdapter(previous_loader)
    dataloader_train_now = _CGCDLoaderAdapter(dataloader_train_now)

    cgcd_args = build_cgcd_args(args, exp_suffix="stage1")

    print("==> Init. Evaluation..")
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, dataloader_eval_now)  # valid_0
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds = preds_lb.detach().cpu().numpy()
        acc_0, _ = utils._hungarian_match_(np.array(dataloader_eval_now.dataset.getlabels()), preds)

    print(f"Valid Epoch: -1 Acc_0: {acc_0:.4f}")
    print("===> Discovering novel category step Calc. proxy mean and sigma for exemplar..")
    with torch.no_grad():
        feats_prev, _ = utils.evaluate_cos_(model, previous_loader)
        feats_prev = losses.l2_norm(feats_prev)
        exampler_sampling = feats_prev.std(dim=0).cuda()

    print("==> Init. Split old and new..")
    with torch.no_grad():
        feats_now, labels_now = utils.evaluate_cos_(model, dataloader_train_now)
        cos_sim = F.linear(losses.l2_norm(feats_now), losses.l2_norm(criterion.proxies))
        preds_cs, _ = torch.max(cos_sim, dim=1)
        utils.show_OnN(
            feats_now,
            labels_now,
            preds_cs,
            nb_classes_prev,
            str(experiment_dir),
            cgcd_args.thres,
            True,
        )

    print("==> Fine. Split old and new..")
    ev_dataset = copy.deepcopy(dataset_train_now)
    ev_dataset_train = copy.deepcopy(dataset_train_now)

    split_module = SplitModlue(
        save_path=str(experiment_dir),
        sz_feature=cgcd_args.sz_embedding,
        sz_embed=128,
    )
    index_new, index_old = split_module.split_old_and_new(
        main_model=model,
        proxy=criterion,
        old_new_dataset_eval=ev_dataset,
        old_new_dataset_train=ev_dataset_train,
        last_old_num=nb_classes_prev,
        thres_cos=cgcd_args.thres,
        confidence_thres=cgcd_args.confidence_thres,
    )

    split_data_path = experiment_dir / "split_data.json"
    with open(split_data_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"index_old": index_old.tolist(), "index_new": index_new.tolist()},
            handle,
        )
    logger.info("Saved split indices to %s", split_data_path)

    dataset_train_old = generate_my_dataset(dataset_train_now, index_old)
    dataset_train_new = generate_my_dataset(dataset_train_now, index_new)

    dataloader_train_old = DataLoader(
        dataset_train_old,
        batch_size=cgcd_args.sz_batch,
        shuffle=False,
        num_workers=cgcd_args.nb_workers,
    )
    dataloader_train_new = DataLoader(
        dataset_train_new,
        batch_size=cgcd_args.sz_batch,
        shuffle=False,
        num_workers=cgcd_args.nb_workers,
    )

    with torch.no_grad():
        feats_old, _ = utils.evaluate_cos_(model, dataloader_train_old)
        cos_sim_old = F.linear(losses.l2_norm(feats_old), losses.l2_norm(criterion.proxies))
        _, pseudo_labels_old = torch.max(cos_sim_old, dim=1)
        pseudo_labels_old = pseudo_labels_old.detach().cpu().numpy()

    with torch.no_grad():
        feats_new, _ = utils.evaluate_cos_(model, dataloader_train_new)

    if cgcd_args.preference is None:
        preference, _, _ = utils.find_optimal_preference(
            feats_new.cpu().numpy(),
            preference_range=range(-10, 0),
        )
    else:
        preference = cgcd_args.preference
    logger.info("AffinityPropagation preference: %s", preference)

    affinity_model = AffinityPropagation(preference=preference).fit(feats_new.cpu().numpy())
    unique_labels, _ = np.unique(affinity_model.labels_, return_counts=True)
    nb_classes_new = len(unique_labels)
    pseudo_labels_new = affinity_model.labels_

    gaussian_model: Optional[GaussianMixture] = None
    if cgcd_args.use_GM_clustering:
        gaussian_model = GaussianMixture(
            n_components=nb_classes_new,
            max_iter=1000,
            tol=1e-4,
            init_params="kmeans",
        ).fit(feats_new.cpu().numpy())
        pseudo_labels_new = gaussian_model.predict(feats_new.cpu().numpy())
        logger.info("GaussianMixture refined %d novel clusters", nb_classes_new)

    dataset_train_old.dataset["labels"] = pseudo_labels_old.tolist()
    dataset_train_old.dataset["len"] = len(dataset_train_old.dataset["labels"])

    new_labels = (pseudo_labels_new + nb_classes_prev).tolist()
    dataset_train_new.dataset["labels"] = new_labels
    dataset_train_new.dataset["len"] = len(dataset_train_new.dataset["labels"])

    dataset_train_merged = merge_my_dataset(dataset_train_old, dataset_train_new)

    # dataloader_train_merged = DataLoader(
    #     dataset_train_merged,
    #     batch_size=cgcd_args.sz_batch,
    #     shuffle=True,
    #     num_workers=cgcd_args.nb_workers,
    # )

    dataloader_train_merged = torch.utils.data.DataLoader(
        dataset_train_merged,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        sampler=None,
    )

    print("==> Training splitted new.., Initializing new criterion and model.")
    nb_classes_now = nb_classes_prev + nb_classes_new
    criterion_now = losses.Proxy_Anchor(
        nb_classes=nb_classes_now, sz_embed=cgcd_args.sz_embedding, mrg=cgcd_args.mrg, alpha=cgcd_args.alpha
    ).cuda()
    criterion_now.proxies.data[:nb_classes_prev] = criterion.proxies.data  # Old proxies
    criterion_now.proxies.data[nb_classes_prev:] = torch.from_numpy(
        affinity_model.cluster_centers_
    ).cuda()  # New proxies

    best_acc_all, best_acc_old, best_acc_new = 0.0, 0.0, 0.0
    best_epoch_all, best_epoch_old, best_epoch_new = 0.0, 0.0, 0.0
    best_projection_all = None  # 최고 정확도일 때의 projection mapping 저장

    model_now = copy.deepcopy(model)
    model_now = model_now.cuda()

    param_groups = [
        {
            "params": (
                list(set(model_now.parameters()).difference(set(model_now.model.embedding.parameters())))
                if cgcd_args.gpu_id != -1
                else list(
                    set(model_now.module.parameters()).difference(set(model_now.module.model.embedding.parameters()))
                )
            )
        },
        {
            "params": (
                model_now.model.embedding.parameters()
                if cgcd_args.gpu_id != -1
                else model_now.module.model.embedding.parameters()
            ),
            "lr": float(cgcd_args.lr) * 1,
        },
    ]
    param_groups.append({"params": criterion_now.parameters(), "lr": float(cgcd_args.lr) * 100})
    opt = torch.optim.AdamW(
        param_groups, lr=float(cgcd_args.lr), weight_decay=cgcd_args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cgcd_args.lr_decay_step, gamma=cgcd_args.lr_decay_gamma)

    print("==> Stage 3. Training  start..")

    result_root = Path("./result") / args.dataset
    experiment_dir = result_root / cgcd_args.model / cgcd_args.exp
    os.makedirs(experiment_dir, exist_ok=True)

    nb_epochs = cgcd_args.nb_epochs
    for epoch in range(0, nb_epochs):
        model_now.train()

        ####
        bn_freeze = cgcd_args.bn_freeze
        if bn_freeze:
            modules = model_now.model.modules() if cgcd_args.gpu_id != -1 else model_now.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        if cgcd_args.warm > 0:
            if cgcd_args.gpu_id != -1:
                unfreeze_model_param = list(model_now.model.embedding.parameters()) + list(criterion_now.parameters())
            else:
                unfreeze_model_param = list(model_now.module.model.embedding.parameters()) + list(
                    criterion_now.parameters()
                )

            if epoch == 0:
                for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == cgcd_args.warm:
                for param in list(set(model_now.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        print("\n==> Category incremental step Evaluation")
        experiment_dir = experiment_dir
        model_now.eval()
        with torch.no_grad():
            feats, _ = utils.evaluate_cos_(model_now, dataloader_eval_now)  # valid_n_1
            cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_now.proxies))
            _, prediction_label = torch.max(cos_sim, dim=1)
            prediction_label = prediction_label.detach().cpu().numpy()

            target_label = np.array(dataloader_eval_now.dataset.getlabels())  # valid_n_1

            cluster2label = utils.cluster_pred_2_gt(prediction_label.astype(int), target_label.astype(int))
            acc_all = utils.pred_2_gt_proj_acc(cluster2label, target_label.astype(int), prediction_label.astype(int))

            old_mask = target_label < nb_classes_prev
            acc_old = utils.pred_2_gt_proj_acc(
                cluster2label, target_label[old_mask].astype(int), prediction_label[old_mask].astype(int)
            )

            new_mask = target_label >= nb_classes_prev
            acc_new = utils.pred_2_gt_proj_acc(
                cluster2label, target_label[new_mask].astype(int), prediction_label[new_mask].astype(int)
            )

            # 헝가리안 매칭 적용된 prediction 사용
            prediction_label_mapped = cluster2label[prediction_label]

            # 클래스별 precision, recall, f1-score, accuracy 계산
            precision, recall, f1, support = precision_recall_fscore_support(
                target_label, prediction_label_mapped, average=None, zero_division=0
            )

            # 전체 평균
            precision_macro = precision_recall_fscore_support(
                target_label, prediction_label_mapped, average="macro", zero_division=0
            )[0]
            recall_macro = precision_recall_fscore_support(
                target_label, prediction_label_mapped, average="macro", zero_division=0
            )[1]
            f1_macro = precision_recall_fscore_support(
                target_label, prediction_label_mapped, average="macro", zero_division=0
            )[2]

            # 클래스별 상세 결과 (accuracy 포함)
            unique_classes = np.unique(target_label)
            label2cls = dataloader_eval_now.dataset.label2cls
            class_accuracies = []
            for i, cls in enumerate(unique_classes):
                # 클래스별 accuracy 계산
                cls_mask = target_label == cls
                cls_acc = (prediction_label_mapped[cls_mask] == target_label[cls_mask]).sum() / cls_mask.sum()
                class_accuracies.append(cls_acc)
                print(
                    f"Class {label2cls[cls]:<20}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Accuracy={cls_acc:.3f}"
                )
            accuracy_macro = np.mean(class_accuracies)
            print(
                f"Macro avg: Precision={precision_macro:.3f}, Recall={recall_macro:.3f}, F1={f1_macro:.3f}, Accuracy={accuracy_macro:.3f}"
            )

        if acc_all > best_acc_all:
            best_acc_all = acc_all
            best_epoch_all = epoch
            best_projection_all = cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_all,  # 최적 projection mapping 저장
                    "label2cls": label2cls,  # 클래스 인덱스-이름 매핑 저장
                    "nb_classes_old": nb_classes_prev,
                    "nb_classes_new": nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                "{}/{}_{}_model_last_step_best_all.pth".format(experiment_dir, cgcd_args.dataset, cgcd_args.model),
            )

        if acc_old > best_acc_old:
            best_acc_old = acc_old
            best_epoch_old = epoch
            best_projection_old = cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_old,  # 최적 projection mapping 저장
                    "label2cls": label2cls,  # 클래스 인덱스-이름 매핑 저장
                    "nb_classes_old": nb_classes_prev,
                    "nb_classes_new": nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                "{}/{}_{}_model_last_step_best_old.pth".format(experiment_dir, cgcd_args.dataset, cgcd_args.model),
            )

        if acc_new > best_acc_new:
            best_acc_new = acc_new
            best_epoch_new = epoch
            best_projection_new = cluster2label.copy()
            torch.save(
                {
                    "model_pa_state_dict": model_now.state_dict(),
                    "proxies_param": criterion_now.proxies,
                    "best_projection": best_projection_new,  # 최적 projection mapping 저장
                    "label2cls": label2cls,  # 클래스 인덱스-이름 매핑 저장
                    "nb_classes_old": nb_classes_prev,
                    "nb_classes_new": nb_classes_new,
                    "nb_classes_total": nb_classes_now,
                },
                "{}/{}_{}_model_last_step_best_new.pth".format(experiment_dir, cgcd_args.dataset, cgcd_args.model),
            )

        ####
        print("==> Stage 3. Category incremental step Train")
        pbar = tqdm.tqdm(enumerate(dataloader_train_merged))
        for batch_idx, (image, target, index, _) in pbar:
            image = image.cuda()
            target = target.cuda()
            feats = model_now(image)

            #### Exampler
            target_new = torch.where(target > nb_classes_prev, 1, 0)
            target_old = target.size(0) - target_new.sum()
            # target_old = y.size(0)

            if target_old > 0:
                target_sampling = torch.randint(nb_classes_prev, (target_old,), device="cuda")
                feats_sampling = torch.normal(criterion.proxies[target_sampling], exampler_sampling)
                target = torch.cat((target, target_sampling), dim=0)
                feats = torch.cat((feats, feats_sampling), dim=0)

            loss_pa = criterion_now(feats, target)

            #### KD
            target_old_mask = torch.nonzero(target_new)
            if target_old_mask.size(0) > 1:
                target_old_mask = torch.nonzero(target_new).squeeze()
                imag_old = torch.unsqueeze(image[target_old_mask[0]], dim=0)
                feats_new = torch.unsqueeze(feats[target_old_mask[0]], dim=0)

                for kd_idx in range(1, target_old_mask.size(0)):
                    imag_old_ = torch.unsqueeze(image[target_old_mask[kd_idx]], dim=0)
                    imag_old = torch.cat((imag_old, imag_old_), dim=0)
                    feats_new_ = torch.unsqueeze(feats[target_old_mask[kd_idx]], dim=0)
                    feats_new = torch.cat((feats_new, feats_new_), dim=0)

                with torch.no_grad():
                    feats_old = model(imag_old.squeeze())

                # FRoST
                loss_kd = torch.dist(
                    F.normalize(feats_old.view(feats_old.size(0) * feats_old.size(1), 1), dim=0).detach(),
                    F.normalize(feats_new.view(feats_old.size(0) * feats_old.size(1), 1), dim=0),
                )
            else:
                loss_kd = torch.tensor(0.0, device="cuda")

            loss = loss_pa * 1.0 + loss_kd * 10.0

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_description(
                f"Train Epoch: {epoch} [{batch_idx + 1}/{len(dataloader_train_merged)} ({100.0 * batch_idx / len(dataloader_train_merged):.0f}%)] "
                f"Loss total [ {loss.item():4.3f} pa: {loss_pa.item():4.3f} kd: {loss_kd.item():4.3f} ]"
            )

        scheduler.step()

        print(
            f"Valid Epoch: {epoch} Acc_0: {acc_0:4.3f} Acc: {acc_all:4.3f}/{acc_old:4.3f}/{acc_new:4.3f} "
            f"Best result: {best_epoch_all}/{best_epoch_old}/{best_epoch_new} {best_acc_all:4.3f}/{best_acc_old:4.3f}/{best_acc_new:4.3f}"
        )

        experiment_dir_log = experiment_dir / "result.txt"
        with open(experiment_dir_log, "a+") as fval:
            fval.write(
                f"Valid Epoch: {epoch} Acc_0: {acc_0:4.3f} Acc: {acc_all:4.3f}/{acc_old:4.3f}/{acc_new:4.3f} "
                f"Best result: {best_epoch_all}/{best_epoch_old}/{best_epoch_new} {best_acc_all:4.3f}/{best_acc_old:4.3f}/{best_acc_new:4.3f}\n"
            )

        step = 1
        torch.save(
            {
                "model_pa_state_dict": model_now.state_dict(),
                "proxies_param": criterion_now.proxies,
                "best_projection": best_projection_all,  # 최적 projection mapping 저장
                "nb_classes_old": nb_classes_prev,
                "nb_classes_new": nb_classes_new,
                "nb_classes_total": nb_classes_now,
                "best_acc_all": best_acc_all,
                "best_epoch": best_epoch_all,
            },
            "{}/{}_{}_model_last_step_{}.pth".format(experiment_dir, cgcd_args.dataset, cgcd_args.model, step),
        )

        checkpoint_info = {
            "best_projection": best_projection_all,
            "nb_classes_old": nb_classes_prev,
            "nb_classes_total": nb_classes_new,
        }

    # return Stage2Result(
    #     cgcd_model=model_now,
    #     criterion=criterion_now,
    #     checkpoint_info=checkpoint_info,
    #     merged_loader=dataloader_train_merged,
    #     old_loader=dataloader_train_old,
    #     new_loader=dataloader_train_new,
    #     best_projection=best_projection_all,
    #     label2cls=label2cls,
    #     nb_classes_prev=nb_classes_prev,
    #     nb_classes_new=nb_classes_new,
    #     exampler_sampling=exampler_sampling,
    #     pseudo_labels_old=pseudo_labels_old,
    #     pseudo_labels_new=pseudo_labels_new,
    #     preference=preference,
    #     cluster_centers=affinity_model.cluster_centers_,
    #     split_data_path=split_data_path,
    #     index_old=np.asarray(index_old),
    #     index_new=np.asarray(index_new),
    #     affinity_model=affinity_model,
    #     gaussian_model=gaussian_model,
    # )

    checkpoint_path = f"{experiment_dir}/{cgcd_args.dataset}_{cgcd_args.model}_model_last_step_best_all.pth"
    return CGCDStageArtifacts(
        model=model_now,
        criterion=criterion_now,
        checkpoint_path=checkpoint_path,
        checkpoint_info=checkpoint_info,
        label2cls=label2cls,
        experiment_dir=experiment_dir,
        nb_classes_prev=(pseudo_labels_new + nb_classes_prev),
    )
