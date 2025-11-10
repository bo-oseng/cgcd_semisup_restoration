from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .helper import set_seed

logger = logging.getLogger(__name__)

_DATASET_ROOT = Path("/mnt/sdd/kbs/proxyrestore/CGCD/data")
_DATASET_MAP = {
    "cub": "cub_cgcd_split_static",
    "deg": "degradations18",
    "mit": "MIT67",
    "dog": "DOG120",
    "air": "AIR100",
}


@dataclass
class EnvironmentContext:
    result_root: Path
    experiment_dir: Path
    dataset_root: Path
    target_dataset: str
    target_dataset_path: Path


def prepare_environment(args) -> EnvironmentContext:
    set_seed(args.seed)
    logger.info("Random seed set to %s", args.seed)

    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)
        logger.info("Using GPU %s", args.gpu_id)

    result_root = Path("./result") / args.dataset
    experiment_dir = result_root / args.model / args.exp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    target_dataset_name = args.target_dataset or _DATASET_MAP.get(args.dataset)
    if target_dataset_name is None:
        raise ValueError(f"Unsupported dataset '{args.dataset}' and no target_dataset override provided.")
    target_dataset_path = _DATASET_ROOT / target_dataset_name

    return EnvironmentContext(
        result_root=result_root,
        experiment_dir=experiment_dir,
        dataset_root=_DATASET_ROOT,
        target_dataset=target_dataset_name,
        target_dataset_path=target_dataset_path,
    )
