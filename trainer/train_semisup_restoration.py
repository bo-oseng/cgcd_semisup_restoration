import os
import socket
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from SemiUIR.trainer import DDPTrainer
from OneRestore.model.OneRestore import OneRestore

from SemiUIR.utils import create_emamodel, count_parameters, setup_seed


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _init_or_reuse_process_group(args):
    """
    Ensure a process group exists so the DDP trainer can run even when the caller
    did not launch the script via `torchrun`. Returns True when initialization
    happened inside this helper.
    """

    if dist.is_available() and dist.is_initialized():
        args.local_rank = dist.get_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        return False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size == 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        rank = 0
        local_rank = 0

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    args.local_rank = local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True


def _cleanup_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_semisup(
    args,
    paired_loader,
    unpaired_loader,
    val_loader,
    cgcd_model,
    criterion,
    checkpoint_info,
    label2cls,
    restoration_path,
):
    """
    Launch the semi-supervised OneRestore trainer inside the multi-stage pipeline.
    """

    initialized_here = _init_or_reuse_process_group(args)
    setup_seed(getattr(args, "manual_seed", 2022))

    if args.local_rank == 0 and paired_loader is not None:
        print(f"[Semisup] labeled batches: {len(paired_loader)}")
    if args.local_rank == 0 and val_loader is not None:
        print(f"[Semisup] val batches: {len(val_loader)}")

    checkpoint = torch.load(restoration_path, map_location="cuda")
    model_info = checkpoint.get("state_dict", checkpoint)
    weights_dict = {k.replace("module.", ""): v for k, v in model_info.items()}

    student_model = OneRestore(q_dim=getattr(args, "sz_embedding", 324))
    teacher_model = OneRestore(q_dim=getattr(args, "sz_embedding", 324))
    teacher_model.load_state_dict(weights_dict, strict=False)
    teacher_model = create_emamodel(teacher_model)

    if args.local_rank == 0:
        print("student model params: %d" % count_parameters(student_model))

    writer = SummaryWriter(log_dir=args.log_dir) if args.local_rank == 0 else None

    trainer = DDPTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=args,
        supervised_loader=paired_loader,
        unsupervised_loader=unpaired_loader,
        val_loader=val_loader,
        iter_per_epoch=len(unpaired_loader) if unpaired_loader is not None else len(paired_loader),
        writer=writer,
        cgcd_model=cgcd_model,
        criterion=criterion,
        checkpoint_info=checkpoint_info,
        label2cls=label2cls,
    )
    state = trainer.train()

    if writer is not None:
        writer.close()
    if initialized_here:
        _cleanup_process_group()

    return state


if __name__ == "__main__":
    raise SystemExit(
        "This module is intended to be used from `train_all_stages.py`. "
        "Please refer to that entry point for dataset construction."
    )
