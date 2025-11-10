from __future__ import annotations

import copy
import logging

from .helper import load_dataset, setup_model_and_training

from .cli import parse_args
from .environment import EnvironmentContext, prepare_environment

from .stages.stage1 import run_initial_training
from ..trainer.train_cgcd_incremental import discover_novel_categories
from .stages.stage3 import train_incremental_stage, Stage3Result

logger = logging.getLogger(__name__)


def run_pipeline(args) -> Stage3Result:
    env: EnvironmentContext = prepare_environment(args)
    dset_tr_0, dlod_tr_0 = load_dataset(args, str(env.target_dataset_path), "train_0", is_train=True)
    _, dlod_ev_0 = load_dataset(args, str(env.target_dataset_path), "eval_0", is_train=False)
    nb_classes = dset_tr_0.nb_classes()

    model, criterion, optimizer, scheduler = setup_model_and_training(args, nb_classes)

    
    stage1 = run_initial_training(
        args,
        model,
        criterion,
        optimizer,
        scheduler,
        dlod_tr_0,
        dlod_ev_0,
        env.experiment_dir,
    )
    
    if args.oldonly:
        return stage1

    stage2 = discover_novel_categories(
        args,
        stage1.model,
        stage1.criterion,
        str(env.target_dataset_path),
        env.experiment_dir,
        nb_classes,
        dlod_tr_0,
    )

    stage3_args = copy.deepcopy(args)
    stage3_args.nb_epochs = 60
    stage3_args.warm = 10
    stage3_args.steps = 1

    stage3 = train_incremental_stage(
        stage3_args,
        stage1.model,
        stage1.criterion,
        stage2,
        str(env.target_dataset_path),
        env.experiment_dir,
        eval_mode="eval_1",
        initial_acc=stage1.acc_0,
    )

    logger.info(
        "Finished training — best overall/old/new accuracy: %.4f / %.4f / %.4f",
        stage3.best_acc_all,
        stage3.best_acc_old,
        stage3.best_acc_new,
    )
    return stage3


def main(argv=None) -> Stage3Result:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":
    main()
