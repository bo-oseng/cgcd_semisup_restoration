from __future__ import annotations

import math, os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from CGCD import losses as cgcd_losses, utils as cgcd_utils
from CGCD.environment import prepare_environment
from CGCD.helper import load_dataset, setup_model_and_training, train_initial_step
from OneRestore.model.OneRestore import OneRestore
from OneRestore.model.loss import SSIM
from OneRestore.utils.utils import tensor_metric
from trainer.cgcd_stage_utils import build_cgcd_args


@dataclass
class CGCDStageArtifacts:
    model: torch.nn.Module
    criterion: torch.nn.Module
    checkpoint_path: Path
    cluster2label: np.ndarray
    label_names: Sequence[str]
    nb_classes_prev: int
    nb_total: int
    experiment_dir: Path


@dataclass
class OneRestoreArtifacts:
    checkpoint_path: Path


@dataclass
class StageOutputs:
    cgcd: CGCDStageArtifacts
    onerestore: OneRestoreArtifacts


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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_for_cgcd(inputs: torch.Tensor, crop_size: int = 224) -> torch.Tensor:
    if inputs.shape[-1] != crop_size or inputs.shape[-2] != crop_size:
        inputs = TF.center_crop(inputs, [crop_size, crop_size])
    mean = torch.tensor((0.485, 0.456, 0.406), device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=inputs.device).view(1, 3, 1, 1)
    return (inputs - mean) / std


def _predict_proxy_embeddings(
    inputs: torch.Tensor,
    cgcd_model: torch.nn.Module,
    criterion,
) -> torch.Tensor:
    cgcd_model.eval()
    with torch.no_grad():
        embed_inputs = _prepare_for_cgcd(inputs)
        feats = cgcd_model(embed_inputs)
        cos_sim = F.linear(cgcd_losses.l2_norm(feats), cgcd_losses.l2_norm(criterion.proxies))
        _, prediction_cluster = torch.max(cos_sim, dim=1)
        proxies = criterion.proxies[prediction_cluster]
    return proxies


def train_cgcd_stage0(
    args, train_dataloader: DataLoader, val_dataloader: DataLoader, label_names: Sequence[str]
) -> CGCDStageArtifacts:
    cgcd_args = build_cgcd_args(args, exp_suffix="stage0")
    env = prepare_environment(cgcd_args)

    train_loader = _CGCDLoaderAdapter(train_dataloader)
    eval_loader = _CGCDLoaderAdapter(val_dataloader)

    nb_classes = _infer_nb_classes(train_loader.dataset)
    model, criterion, optimizer, scheduler = setup_model_and_training(cgcd_args, nb_classes)

    checkpoint_path = env.experiment_dir / f"{cgcd_args.dataset}_{cgcd_args.model}_best_step_0.pth"
    if not os.path.exists(checkpoint_path):
        train_initial_step(
            cgcd_args,
            model,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            eval_loader,
            str(env.experiment_dir),
        )

    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["model_pa_state_dict"])
    criterion.proxies = checkpoint["proxies_param"]

    with torch.no_grad():
        feats, targets = cgcd_utils.evaluate_cos_(model, eval_loader)
        cos_sim = F.linear(cgcd_losses.l2_norm(feats), cgcd_losses.l2_norm(criterion.proxies))
        _, prediction_cluster = torch.max(cos_sim, dim=1)
    cluster2label = cgcd_utils.cluster_pred_2_gt(
        prediction_cluster.detach().cpu().numpy().astype(int),
        targets.detach().cpu().numpy().astype(int),
    )

    nb_total = criterion.proxies.size(0)

    return CGCDStageArtifacts(
        model=model,
        criterion=criterion,
        checkpoint_path=checkpoint_path,
        cluster2label=cluster2label,
        label_names=list(label_names),
        nb_classes_prev=nb_classes,
        nb_total=nb_total,
        experiment_dir=env.experiment_dir,
    )


def train_onerestore_stage0(args, dataloader: DataLoader, cgcd_artifacts: CGCDStageArtifacts) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restorer = OneRestore(q_dim=getattr(args, "cgcd_embedding_dim", 324)).to(device)
    optimizer = torch.optim.Adam(
        restorer.parameters(),
        lr=getattr(args, "restore_lr", 1e-4),
        weight_decay=getattr(args, "restore_weight_decay", 0.0),
    )
    l1_loss = torch.nn.L1Loss()
    ssim_loss = SSIM(window_size=11).to(device)
    checkpoint_dir = _ensure_dir(Path(getattr(args, "restore_checkpoint_dir", "exp/stage0/onerestore")))
    checkpoint_path = checkpoint_dir / "onerestore_stage0_best.pth"

    if os.path.exists(checkpoint_path):
        return OneRestoreArtifacts(checkpoint_path=checkpoint_path)

    best_psnr = -math.inf
    restore_epochs = getattr(args, "restore_epochs", 100)
    l1_weight = getattr(args, "restore_l1_weight", 1.0)
    ssim_weight = getattr(args, "restore_ssim_weight", 0.1)

    cgcd_model = cgcd_artifacts.model.to(device)
    cgcd_criterion = cgcd_artifacts.criterion.to(device)

    for epoch in range(restore_epochs):
        restorer.train()
        epoch_psnr = []
        print("Initial Restoration model train start")
        for degraded, gt in tqdm.tqdm(dataloader):
            degraded = degraded.to(device)
            gt = gt.to(device)

            embeddings = _predict_proxy_embeddings(degraded, cgcd_model, cgcd_criterion)
            output = restorer(degraded, embeddings)

            loss = l1_weight * l1_loss(output, gt) + ssim_weight * (1 - ssim_loss(output, gt))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = tensor_metric(gt, output, "PSNR", data_range=1)
            epoch_psnr.append(psnr)

        avg_psnr = float(np.mean(epoch_psnr)) if epoch_psnr else -math.inf
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": restorer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_psnr": best_psnr,
                },
                checkpoint_path,
            )

    return OneRestoreArtifacts(checkpoint_path=checkpoint_path)


def run_cgcd_onerestore_old(
    args,
    cgcd_train_loader: DataLoader,
    cgcd_val_loader: DataLoader,
    restoration_loader: DataLoader,
    label_names: Sequence[str],
) -> StageOutputs:
    cgcd_artifacts = train_cgcd_stage0(args, cgcd_train_loader, cgcd_val_loader, label_names)
    onerestore_artifacts = train_onerestore_stage0(args, restoration_loader, cgcd_artifacts)
    return cgcd_artifacts, onerestore_artifacts
