from dataset_utils.util import data_util
from dataset_utils.data import (
    create_dataloader,
    create_dataloader_wo_contra,
    build_CCD_dataset,
    build_my_CCD_train_dataset,
    build_my_CCD_val_dataset,
    add_dataset_attribute,
    create_ccd_dataset,
)
import os
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import argparse

from dataset_utils.util import config, util
from dataset_semisup.dataset import TrainLabeled, TrainUnlabeled, ValLabeled

from trainer import train_initialize_old
from trainer import train_cgcd_incremental
from trainer import train_semisup_restoration
from dataclasses import dataclass


@lru_cache(maxsize=8)
def _cached_class_id_to_name(mapping_path: str):
    if not os.path.isfile(mapping_path):
        raise FileNotFoundError(f"Class-id map not found: {mapping_path}")
    with open(mapping_path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _load_class_id_to_name(args):
    mapping_path = getattr(args, "class_map_path", "data/CDD_11/Train/class_id_to_name.json")
    resolved_path = os.path.abspath(os.path.expanduser(mapping_path))
    return _cached_class_id_to_name(resolved_path)


def _build_dir_lookup(search_root: Path):
    search_root = Path(search_root)
    lookup = {}
    if search_root.is_dir():
        for entry in search_root.iterdir():
            if entry.is_dir():
                lookup.setdefault(entry.name, entry.name)
                suffix = entry.name.split(".", 1)[-1]
                lookup.setdefault(suffix, entry.name)
    return lookup


def _match_degradation_dir(class_name, class_id, lookup, search_root: Path):
    search_root = Path(search_root)
    candidates = [
        class_name,
        class_name.replace("_", "-"),
        f"{class_id:03d}.{class_name}",
        f"{class_id:03d}.{class_name.replace('_', '-')}",
    ]
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
        candidate_path = search_root / candidate
        if candidate_path.is_dir():
            lookup.setdefault(candidate, candidate)
            suffix = candidate.split(".", 1)[-1]
            lookup.setdefault(suffix, candidate)
            return candidate
    return None


def _resolve_degradation_subset(class_ids, class_id_to_name, search_root: Path):
    if not class_ids:
        return []
    lookup = _build_dir_lookup(search_root)
    subset = []
    missing = []
    for class_id in class_ids:
        class_name = class_id_to_name.get(class_id)
        if class_name is None:
            missing.append((class_id, None))
            continue
        folder_name = _match_degradation_dir(class_name, class_id, lookup, search_root)
        if folder_name is None:
            missing.append((class_id, class_name))
            folder_name = f"{class_name}"
            # folder_name = f"{class_id:03d}.{class_name}"
        subset.append(folder_name)
    if missing:
        print(f"[Restoration] Missing degradation dirs for classes {missing} under {search_root}")
    return subset


def _stage_key_from_index(stage_i: int) -> str:
    return "labelled" if stage_i == 0 else f"stage_{stage_i}"


def get_parser():
    """
    Input: takes arguments from yaml config file located in config/$DATASET$/*.yaml
    Return: a dict with key for argument query
    """
    parser = argparse.ArgumentParser(description="PyTorch implementation of PromptCCD framework")
    parser.add_argument("--exp_name", type=str, default="ViT_ssk", help="To differentiate each experiments")
    parser.add_argument(
        "--config", type=str, default="/mnt/sdd/kbs/proxyrestore/SemiAIORCL/tmp.yaml", help="config file"
    )
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_folder", type=str, default="data/CDD_11", help="Root folder for restoration datasets")
    parser.add_argument("--cgcd_model", type=str, default="dinov3")
    parser.add_argument("--cgcd_epochs", type=int, default=1)
    parser.add_argument("--cgcd_lr", type=float, default=3.5e-4)
    parser.add_argument("--cgcd_weight_decay", type=float, default=1e-4)
    parser.add_argument("--cgcd_lr_decay_step", type=int, default=30)
    parser.add_argument("--cgcd_lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--cgcd_warmup", type=int, default=0)
    parser.add_argument("--cgcd_alpha", type=float, default=32.0)
    parser.add_argument("--cgcd_margin", type=float, default=0.1)
    parser.add_argument("--cgcd_embedding_dim", type=int, default=324)
    parser.add_argument("--cgcd_exp_dir", type=str, default="exp/stage0/cgcd")
    parser.add_argument("--restore_epochs", type=int, default=1)
    parser.add_argument("--restore_lr", type=float, default=1e-4)
    parser.add_argument("--restore_weight_decay", type=float, default=0.0)
    parser.add_argument("--restore_l1_weight", type=float, default=1.0)
    parser.add_argument("--restore_ssim_weight", type=float, default=0.1)
    parser.add_argument("--restore_checkpoint_dir", type=str, default="exp/stage0/onerestore")

    # ! 임시
    parser.add_argument("--semi_epochs", type=int, default=1)
    parser.add_argument("--resume_path", default=None, type=str, help="if resume")
    parser.add_argument("--initialize", default=False, action="store_true", help="initail teacher inference")
    parser.add_argument("--save_path", default="./model/ckpt/", type=str)
    parser.add_argument("--log_dir", default="./model/log", type=str)

    args = parser.parse_args()

    # Check if config file is specified
    try:
        assert args.config is not None
    except:
        raise ValueError("Please specify config file")

    # Check if train or test mode is specified
    try:
        assert args.train != args.test
    except:
        raise ValueError("Please specify either train or test mode")

    # Load arguments from config file
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_cfg(cfg, vars(args))

    return cfg, args.config


def setup_restoration_dataloaders(args, stage_class_info, stage_i):
    stage_key = _stage_key_from_index(stage_i)
    stage_info = stage_class_info.get(stage_key, {"old": [], "new": []})
    class_id_to_name = _load_class_id_to_name(args)

    train_folder = Path(args.train_folder)

    labelled_subset = _resolve_degradation_subset(stage_info.get("old", []), class_id_to_name, train_folder / "Train")
    unlabeled_subset = _resolve_degradation_subset(stage_info.get("new", []), class_id_to_name, train_folder / "Train")

    crop_size = getattr(args, "crop_size", 256)
    paired_dataset = TrainLabeled(
        dataroot=str(train_folder),
        phase="train",
        finesize=crop_size,
        degradation_subset=labelled_subset,
    )
    paired_loader = DataLoader(
        paired_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    if stage_i == 0:
        return paired_loader, None, None

    unpaired_dataset = TrainUnlabeled(
        args=args,
        dataroot=str(train_folder),
        phase="train",
        finesize=crop_size,
        degradation_subset=unlabeled_subset,
    )

    val_subset = _resolve_degradation_subset(stage_info.get("new", []), class_id_to_name, train_folder / "Test")
    val_dataset = ValLabeled(
        dataroot=str(train_folder), phase="valid", finesize=crop_size, degradation_subset=val_subset
    )

    unpaired_loader = None
    if unpaired_dataset is not None:
        unpaired_loader = DataLoader(
            unpaired_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    return paired_loader, unpaired_loader, val_loader


if __name__ == "__main__":
    args, config_path = get_parser()

    dataset_train, stage_class_info = build_my_CCD_train_dataset(args, split="train")
    dataset_test, stage_class_info = build_my_CCD_train_dataset(args, split="test")
    dataset_val = build_my_CCD_val_dataset(args, split="val")
    class_id_to_name = _load_class_id_to_name(args)

    stage_outputs = {
        0: {"cgcd": None, "restoratin": None},
        1: {"cgcd": None, "restoratin": None},
        2: {"cgcd": None, "restoratin": None},
    }

    for stage_i in range(args.n_stage + 1):
        cgcd_train_dataset, transform, _ = add_dataset_attribute(dataset_train[stage_i])
        cgcd_train_dataset = create_ccd_dataset(cgcd_train_dataset, transform, stage=stage_i)
        cgcd_train_dataloader = torch.utils.data.DataLoader(
            cgcd_train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            sampler=None,
        )

        cgcd_val_dataloader = create_dataloader(args, dataset_val, -1)
        cgcd_test_dataloader = create_dataloader(args, dataset_test[: stage_i + 1], -2)

        restoration_paired_loader, restoration_unpaired_loader, restoration_val_loader = setup_restoration_dataloaders(
            args, stage_class_info, stage_i
        )

        stage_key = _stage_key_from_index(stage_i)
        stage_info = stage_class_info.get(stage_key, {"old": [], "new": []})
        label_names = [class_id_to_name.get(str(class_id), str(class_id)) for class_id in stage_info.get("old", [])]

        if stage_i == 0:
            cgcd_artifacts, onerestore_artifacts = train_initialize_old.run_cgcd_onerestore_old(
                args=args,
                cgcd_train_loader=cgcd_train_dataloader,
                cgcd_val_loader=cgcd_val_dataloader,
                restoration_loader=restoration_paired_loader,
                label_names=label_names,
            )

            stage_outputs[stage_i]["cgcd"] = cgcd_artifacts
            stage_outputs[stage_i]["restoration"] = onerestore_artifacts

        else:
            previous_loader = create_dataloader_wo_contra(args, dataset_train[stage_i - 1], stage_i - 1)["default"]

            prev_cgcd_setup = stage_outputs[stage_i - 1]["cgcd"]
            prev_restoration_ckpt = stage_outputs[stage_i - 1]["restoration"].checkpoint_path

            incremental_result = train_cgcd_incremental.run_discover_novel_categories(
                args=args,
                model=prev_cgcd_setup.model,
                criterion=prev_cgcd_setup.criterion,
                experiment_dir=prev_cgcd_setup.experiment_dir,
                nb_classes_prev=prev_cgcd_setup.nb_classes_prev,
                previous_loader=previous_loader,
                dataset_train_now=cgcd_train_dataset,
                dataloader_train_now=cgcd_train_dataloader,
                dataloader_eval_now=cgcd_train_dataloader,
            )

            semisup_result = train_semisup_restoration.run_semisup(
                args,
                restoration_paired_loader,
                restoration_unpaired_loader,
                restoration_val_loader,
                incremental_result.model,
                incremental_result.criterion,
                incremental_result.checkpoint_info,
                incremental_result.label2cls,
                prev_restoration_ckpt,
            )

            stage_outputs[stage_i]["cgcd"] = incremental_result
            stage_outputs[stage_i]["restoratin"] = semisup_result

        cgcd_ckpt = stage_outputs[stage_i]["cgcd"].checkpoint_path
        restoration_ckpt = stage_outputs[stage_i]["restoration"].checkpoint_path
        print(f"[Stage {stage_i}] CGCD ckpt: {cgcd_ckpt}, " f"OneRestore ckpt: {restoration_ckpt}")
