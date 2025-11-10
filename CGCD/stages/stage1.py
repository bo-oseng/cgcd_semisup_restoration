from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

from ..helper import train_initial_step

from ..evaluation import evaluate_initial, Stage0Metrics

logger = logging.getLogger(__name__)


@dataclass
class Stage1Result:
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    losses: list
    best_recall: list
    best_epoch: int
    acc_0: float
    checkpoint_path: Path


def run_initial_training(
    args,
    model: torch.nn.Module,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    eval_loader,
    experiment_dir: Path,
) -> Stage1Result:
    checkpoint_path = experiment_dir / f"{args.dataset}_{args.model}_best_step_0.pth"
    losses_list = []
    best_recall = [-sys.maxsize]
    best_epoch = 0

    if not checkpoint_path.exists():
        losses_list, best_recall, best_epoch = train_initial_step(
            args,
            model,
            criterion,
            optimizer,
            scheduler,
            train_loader,
            eval_loader,
            str(experiment_dir),
        )
    else:
        logger.info("Resuming Stage 1 from checkpoint %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_pa_state_dict"])
        criterion.proxies = checkpoint["proxies_param"]

    model = model.cuda()
    metrics: Stage0Metrics = evaluate_initial(model, eval_loader, criterion)
    logger.info("Initial validation Acc_0: %.4f", metrics.acc_0)

    return Stage1Result(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses_list,
        best_recall=best_recall,
        best_epoch=best_epoch,
        acc_0=metrics.acc_0,
        checkpoint_path=checkpoint_path,
    )
