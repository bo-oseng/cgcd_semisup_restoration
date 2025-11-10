"""Shared helpers for CGCD stage orchestration."""

from __future__ import annotations

from pathlib import Path

from CGCD.cli import build_parser as build_cgcd_parser


def build_cgcd_args(main_args, *, exp_suffix: str) -> object:
    """Create a CGCD args namespace aligned with the main pipeline settings."""

    parser = build_cgcd_parser()
    cgcd_args = parser.parse_args([])

    cgcd_args.dataset = getattr(main_args, "cgcd_dataset_name", cgcd_args.dataset)
    cgcd_args.target_dataset = getattr(main_args, "cgcd_target_dataset", cgcd_args.target_dataset)
    cgcd_args.exp = str(Path(getattr(main_args, "cgcd_exp_root", "exp/cgcd")) / exp_suffix)

    cgcd_args.sz_embedding = getattr(main_args, "cgcd_embedding_dim", cgcd_args.sz_embedding)
    cgcd_args.sz_batch = getattr(main_args, "cgcd_batch_size", cgcd_args.sz_batch)
    cgcd_args.nb_workers = getattr(main_args, "cgcd_num_workers", cgcd_args.nb_workers)

    cgcd_args.nb_epochs = getattr(main_args, "cgcd_epochs", cgcd_args.nb_epochs)
    cgcd_args.warm = getattr(main_args, "cgcd_warmup", cgcd_args.warm)
    cgcd_args.lr = getattr(main_args, "cgcd_lr", cgcd_args.lr)
    cgcd_args.weight_decay = getattr(main_args, "cgcd_weight_decay", cgcd_args.weight_decay)
    cgcd_args.lr_decay_step = getattr(main_args, "cgcd_lr_decay_step", cgcd_args.lr_decay_step)
    cgcd_args.lr_decay_gamma = getattr(main_args, "cgcd_lr_decay_gamma", cgcd_args.lr_decay_gamma)

    cgcd_args.alpha = getattr(main_args, "cgcd_alpha", cgcd_args.alpha)
    cgcd_args.mrg = getattr(main_args, "cgcd_margin", cgcd_args.mrg)
    cgcd_args.thres = getattr(main_args, "cgcd_thres", getattr(cgcd_args, "thres", 0.0))
    cgcd_args.confidence_thres = getattr(
        main_args, "cgcd_confidence_thres", getattr(cgcd_args, "confidence_thres", 0.0)
    )
    cgcd_args.preference = getattr(main_args, "cgcd_preference", getattr(cgcd_args, "preference", None))
    cgcd_args.use_GM_clustering = getattr(main_args, "cgcd_use_gm", getattr(cgcd_args, "use_GM_clustering", False))
    cgcd_args.bn_freeze = getattr(main_args, "cgcd_bn_freeze", getattr(cgcd_args, "bn_freeze", True))
    cgcd_args.steps = getattr(main_args, "cgcd_steps", getattr(cgcd_args, "steps", 1))

    return cgcd_args
