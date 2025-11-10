"""
Utilities for building restoration-aware CCD datasets.

This module mirrors the dataset construction pipeline defined in
`dataset_utils.data`, but augments each sample with a paired clean target so
that CGCD and OneRestore can be trained in tandem.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from dataset_utils.data import get_dataset_funcs


def _resolve_interpolation(value: Union[int, str, InterpolationMode]) -> InterpolationMode:
    """
    Normalize interpolation inputs so users can keep using the integer codes
    (e.g., 3 -> bicubic) that torchvision accepts elsewhere in the codebase.
    """
    if isinstance(value, InterpolationMode):
        return value

    if isinstance(value, str):
        value = value.upper()
        if not value.startswith("INTERPOLATIONMODE."):
            value = f"INTERPOLATIONMODE.{value}"
        return InterpolationMode[value.split("INTERPOLATIONMODE.")[-1]]

    mapping = {
        0: InterpolationMode.NEAREST,
        1: InterpolationMode.LANCZOS,
        2: InterpolationMode.BILINEAR,
        3: InterpolationMode.BICUBIC,
        4: InterpolationMode.BOX,
    }
    return mapping.get(value, InterpolationMode.BICUBIC)


class RestorationPairTransform:
    """
    Apply paired spatial augmentations to degraded/clean images.

    The implementation follows OneRestore's `data_augmentation` routine:
    1. Optional resize so the shorter side is at least `crop_size`.
    2. Random (or center) crop.
    3. Shared rotation/flip sampled from the 8 augmentation modes.
    """

    def __init__(
        self,
        crop_size: int,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
        random_crop: bool = True,
        apply_aug: bool = True,
    ) -> None:
        self.crop_size = crop_size
        self.interpolation = interpolation
        self.random_crop = random_crop
        self.apply_aug = apply_aug

    def __call__(self, degraded: Image.Image, clean: Image.Image):
        degraded, clean = self._resize_if_needed(degraded, clean)
        degraded, clean = self._crop_pair(degraded, clean)

        if self.apply_aug:
            mode = random.randint(0, 7)
            degraded = self._apply_aug(degraded, mode)
            clean = self._apply_aug(clean, mode)

        return TF.to_tensor(degraded), TF.to_tensor(clean)

    def _resize_if_needed(
        self, degraded: Image.Image, clean: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = degraded.size
        min_side = min(w, h)
        if min_side >= self.crop_size:
            return degraded, clean

        scale = self.crop_size / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        degraded = degraded.resize((new_w, new_h), self.interpolation)
        clean = clean.resize((new_w, new_h), self.interpolation)
        return degraded, clean

    def _crop_pair(self, degraded: Image.Image, clean: Image.Image):
        w, h = degraded.size
        crop = min(self.crop_size, w, h)

        if self.random_crop:
            left = random.randint(0, w - crop) if w > crop else 0
            top = random.randint(0, h - crop) if h > crop else 0
        else:
            left = max((w - crop) // 2, 0)
            top = max((h - crop) // 2, 0)

        box = (left, top, left + crop, top + crop)
        return degraded.crop(box), clean.crop(box)

    @staticmethod
    def _apply_aug(img: Image.Image, mode: int) -> Image.Image:
        if mode == 0:
            return img
        if mode == 1:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 2:
            return img.rotate(90, expand=False)
        if mode == 3:
            return img.rotate(90, expand=False).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 4:
            return img.rotate(180, expand=False)
        if mode == 5:
            return img.rotate(180, expand=False).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 6:
            return img.rotate(270, expand=False)
        if mode == 7:
            return img.rotate(270, expand=False).transpose(Image.FLIP_TOP_BOTTOM)
        raise ValueError(f"Unsupported augmentation mode: {mode}")


def build_restoration_transform(mode: str, args) -> RestorationPairTransform:
    """
    Factory for restoration transforms.

    Args:
        mode: "train" enables random crop + augmentation; any other value falls
              back to deterministic center crops without flips.
        args: experiment configuration. The following optional attributes are
              consumed:
                * restoration_crop_size (default: args.input_size or 256)
                * restoration_interpolation (default: args.interpolation or 3)
    """

    crop_size = getattr(args, "restoration_crop_size", getattr(args, "input_size", 256))
    interpolation_value = getattr(args, "restoration_interpolation", getattr(args, "interpolation", 3))
    interpolation = _resolve_interpolation(interpolation_value)
    random_crop = mode == "train"
    apply_aug = mode == "train"

    return RestorationPairTransform(
        crop_size=crop_size,
        interpolation=interpolation,
        random_crop=random_crop,
        apply_aug=apply_aug,
    )


def _swap_to_clean_path(
    degraded_path: Union[str, Path],
    clean_dir: str,
    fallback_dir: str,
) -> Path:
    """
    Replace the degradation directory (e.g., '006.rain') with the clean one
    (e.g., '000.clear'). If the preferred directory does not exist, fall back to
    `fallback_dir`.
    """

    degraded_path = Path(degraded_path)
    if len(degraded_path.parts) < 2:
        raise ValueError(f"Expected at least two path levels, got {degraded_path}")

    parts = list(degraded_path.parts)
    parts[-2] = clean_dir
    clean_path = Path(*parts)

    if not clean_path.exists() and fallback_dir:
        parts[-2] = fallback_dir
        clean_path = Path(*parts)

    return clean_path


class create_restoration_ccd_dataset(Dataset):
    """
    CCD dataset wrapper that also returns the clean GT pair for restoration.
    """

    def __init__(
        self,
        dataset,
        transform: Callable[[Image.Image, Image.Image], Tuple[object, object]],
        stage: int,
        clean_dir: str = "000.clear",
        clean_fallback_dir: str = "clear",
    ) -> None:
        super(create_restoration_ccd_dataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.batch_labeled_or_not = 1 if stage == 0 else 0
        self.clean_dir = clean_dir
        self.clean_fallback_dir = clean_fallback_dir

    def __getitem__(self, index):
        degraded_path = self.dataset["paths"][index]
        clean_path = _swap_to_clean_path(degraded_path, self.clean_dir, self.clean_fallback_dir)

        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        batch_label = self.dataset["labels"][index]
        batch_unique_index = self.dataset["uq_idx"][index]

        if self.transform is not None:
            degraded, clean = self.transform(degraded, clean)

        return (
            degraded,
            batch_label,
            batch_unique_index,
            np.array([self.batch_labeled_or_not]),
            clean,
        )

    def __len__(self):
        return self.dataset["len"]


def build_restoration_dataset(args, split: str):
    """
    Mirror `build_CCD_dataset` while swapping in the restoration transform.
    """

    dataset_name = args.dataset
    number_of_stage = args.n_stage
    ccd_dataset: List[np.ndarray] = [None] * number_of_stage
    datasets: List[Tuple[Tuple[Sequence[str], Sequence[int]], np.ndarray, object]] = []
    dataset_l = None

    dataset_func = get_dataset_funcs.get(dataset_name)
    if dataset_func is None:
        raise ValueError(f"Dataset {dataset_name} not found.")

    dataset_train, dataset_val = dataset_func(split)

    if split == "train":
        transform = build_restoration_transform("train", args)
        dataset = dataset_train
    elif split == "val":
        transform = build_restoration_transform("test", args)
        indices = np.arange(len(dataset_val[1]))
        return (dataset_val, indices, transform)
    elif split == "test":
        transform = build_restoration_transform("test", args)
        dataset = dataset_val
    else:
        raise ValueError(f"Split {split} is not supported for CCD training.")

    dataset_target = np.array(dataset[1])
    for class_idx in range(args.classes):
        class_indices = np.where(dataset_target == class_idx)[0]
        np.random.shuffle(class_indices)

        if class_idx < args.labelled_data:
            labelled_ratios = args.ccd_split_ratio[0]
            class_i_l, class_i_u = np.split(class_indices, [int(len(class_indices) * labelled_ratios[0])])
            dataset_l = class_i_l if dataset_l is None else np.concatenate((dataset_l, class_i_l), axis=0)

            tmp_indices = class_i_u
            for stage_i in range(number_of_stage):
                split_point = int(len(tmp_indices) * labelled_ratios[stage_i + 1])
                tmp_indices, tmp_indices_ = np.split(tmp_indices, [split_point])
                if ccd_dataset[stage_i] is None:
                    ccd_dataset[stage_i] = tmp_indices
                else:
                    ccd_dataset[stage_i] = np.concatenate((ccd_dataset[stage_i], tmp_indices), axis=0)
                tmp_indices = tmp_indices_
        else:
            stage_selector = (class_idx % args.labelled_data) // (
                (args.classes - args.labelled_data) // args.n_stage
            )
            tmp_indices = class_indices
            for stage_i in range(number_of_stage - stage_selector):
                ratio = args.ccd_split_ratio[stage_selector + 1][stage_i]
                split_point = int(len(tmp_indices) * ratio)
                tmp_indices, tmp_indices_ = np.split(tmp_indices, [split_point])
                target_stage = stage_i + stage_selector
                if ccd_dataset[target_stage] is None:
                    ccd_dataset[target_stage] = tmp_indices
                else:
                    ccd_dataset[target_stage] = np.concatenate((ccd_dataset[target_stage], tmp_indices), axis=0)
                tmp_indices = tmp_indices_

    dataset_l = (dataset, dataset_l, transform)
    datasets.append(dataset_l)

    for dataset_u_i in ccd_dataset:
        dataset_u_i = (dataset, dataset_u_i, transform)
        datasets.append(dataset_u_i)

    return datasets
