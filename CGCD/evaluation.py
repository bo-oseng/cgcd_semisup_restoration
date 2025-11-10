from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from . import losses
from . import utils


logger = logging.getLogger(__name__)


@dataclass
class Stage0Metrics:
    acc_0: float


@dataclass
class IncrementalMetrics:
    acc_all: float
    acc_old: float
    acc_new: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: List[Dict[str, float]]
    cluster2label: np.ndarray
    prediction_label: np.ndarray
    label2cls: Dict[int, str]


def evaluate_initial(model: torch.nn.Module, loader, criterion) -> Stage0Metrics:
    model.eval()
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, loader)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion.proxies))
        _, preds_lb = torch.max(cos_sim, dim=1)
        preds = preds_lb.detach().cpu().numpy()
        acc_0, _ = utils._hungarian_match_(np.array(loader.dataset.labels), preds)
    return Stage0Metrics(acc_0=acc_0)


def evaluate_incremental(
    model: torch.nn.Module,
    loader,
    criterion,
    nb_classes_prev: int,
    nb_classes_eval_threshold: int,
) -> IncrementalMetrics:
    model.eval()
    with torch.no_grad():
        feats, _ = utils.evaluate_cos_(model, loader)
        cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion.proxies))
        _, prediction_label = torch.max(cos_sim, dim=1)
        prediction_label = prediction_label.detach().cpu().numpy()

    target_label = np.array(loader.dataset.labels)
    cluster2label = utils.cluster_pred_2_gt(
        prediction_label.astype(int),
        target_label.astype(int),
    )
    acc_all = utils.pred_2_gt_proj_acc(
        cluster2label,
        target_label.astype(int),
        prediction_label.astype(int),
    )

    old_mask = target_label < nb_classes_prev
    acc_old = utils.pred_2_gt_proj_acc(
        cluster2label,
        target_label[old_mask].astype(int),
        prediction_label[old_mask].astype(int),
    )

    new_mask = target_label >= nb_classes_eval_threshold
    acc_new = utils.pred_2_gt_proj_acc(
        cluster2label,
        target_label[new_mask].astype(int),
        prediction_label[new_mask].astype(int),
    )

    prediction_label_mapped = cluster2label[prediction_label]

    precision, recall, f1, _ = precision_recall_fscore_support(
        target_label,
        prediction_label_mapped,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        target_label,
        prediction_label_mapped,
        average="macro",
        zero_division=0,
    )

    label2cls = loader.dataset.idx2cls
    per_class = []
    unique_classes = np.unique(target_label)
    for idx, cls in enumerate(unique_classes):
        cls_mask = target_label == cls
        cls_acc = 0.0
        if cls_mask.any():
            cls_acc = float((prediction_label_mapped[cls_mask] == target_label[cls_mask]).sum() / cls_mask.sum())
        per_class.append(
            {
                "class": label2cls[cls],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "accuracy": cls_acc,
            }
        )

    return IncrementalMetrics(
        acc_all=float(acc_all),
        acc_old=float(acc_old),
        acc_new=float(acc_new),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        per_class=per_class,
        cluster2label=cluster2label,
        prediction_label=prediction_label,
        label2cls=label2cls,
    )
