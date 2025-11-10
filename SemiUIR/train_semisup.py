import os
import warnings

warnings.filterwarnings("ignore")

import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from SemiUIR.trainer import DDPTrainer

from ..CGCD.helper import load_model_with_projection
from ..OneRestore.model.OneRestore import OneRestore


from .dataset_my import TrainLabeled, TrainUnlabeled, ValLabeled
from .utils import *


def main(args):
    # Initialize process group
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")

    # random seed
    setup_seed(2022)

    # load data
    train_folder = os.path.join(args.data_dir, "train")
    paired_dataset = TrainLabeled(dataroot=train_folder, phase="labeled", finesize=args.crop_size)
    unpaired_dataset = TrainUnlabeled(
        dataroot=train_folder,
        phase="unlabeled",
        unlabel_dir=args.unlabel_dir,
        candidate_dir=args.candidate,
        finesize=args.crop_size,
    )
    val_dataset = ValLabeled(dataroot=args.data_dir, phase="val", finesize=args.crop_size)

    # Create DistributedSampler
    paired_sampler = DistributedSampler(paired_dataset, rank=args.local_rank, shuffle=True)
    unpaired_sampler = DistributedSampler(unpaired_dataset, rank=args.local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=args.local_rank, shuffle=False)

    paired_loader = DataLoader(
        paired_dataset,
        batch_size=args.train_batchsize,
        sampler=paired_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    unpaired_loader = DataLoader(
        unpaired_dataset,
        batch_size=args.train_batchsize,
        sampler=unpaired_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batchsize, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )

    if args.local_rank == 0:
        print("there are total %s batches for train" % (len(paired_loader)))
        print("there are total %s batches for val" % (len(val_loader)))

    # cgcd for proxy
    cgcd_proxies_info = load_model_with_projection(args)

    # checkpoint
    checkpoint = torch.load(args.pretrained_path)
    model_info = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    weights_dict = {}
    for k, v in model_info.items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v

    # create model
    student_model = OneRestore(q_dim=324)

    teacher_model = OneRestore(q_dim=324)
    teacher_model.load_state_dict(weights_dict)
    teacher_model = create_emamodel(teacher_model)

    if args.local_rank == 0:
        print("student model params: %d" % count_parameters(student_model))

    # tensorboard (only on rank 0)
    writer = None
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    trainer = DDPTrainer(
        model=student_model,
        teacher_model=teacher_model,
        cgcd_proxies_info=cgcd_proxies_info,
        args=args,
        supervised_loader=paired_loader,
        unsupervised_loader=unpaired_loader,
        val_loader=val_loader,
        iter_per_epoch=len(unpaired_loader),
        writer=writer,
    )

    trainer.train()

    if args.local_rank == 0 and writer is not None:
        writer.close()

    # Cleanup
    dist.destroy_process_group()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# /mnt/sdd/kbs/proxyrestore/CGCD/result/deg/dinov3/012_1_CDD_11_train_w_clear_dinov3_thr_03_324_optimal_conf_02/deg_dinov3_model_last_step_best_all.pth
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument("-g", "--gpus", default=1, type=int, metavar="N", help="number of GPUs to use")
    parser.add_argument("--num_epochs", default=300, type=int)
    parser.add_argument("--train_batchsize", default=20, type=int, help="train batchsize per GPU")
    parser.add_argument("--val_batchsize", default=4, type=int, help="val batchsize per GPU")
    parser.add_argument("--crop_size", default=256, type=int, help="crop size")
    parser.add_argument("--resume", default="False", type=str, help="if resume")
    parser.add_argument("--resume_path", default=None, type=str, help="if resume")
    parser.add_argument("--use_pretain", default="False", type=str, help="use pretained model")
    parser.add_argument(
        "--pretrained_path",
        default="/mnt/sdd/kbs/proxyrestore/OneRestore/ckpts/004_direct_proxy_restore_cddbalance/OneRestore_model_292_26.7448_0.8543_26.7448_0.8543.tar",
        type=str,
        help="if pretrained",
    )
    parser.add_argument("--data_dir", default="./data", type=str, help="data root path")
    parser.add_argument("--save_path", default="./model/ckpt/", type=str)
    parser.add_argument("--log_dir", default="./model/log", type=str)
    parser.add_argument("--local_rank", default=-1, type=int, help="node rank for distributed training")

    parser.add_argument("--alpha", default=32, type=float, help="Scaling Parameter setting")
    parser.add_argument("--mrg", default=0.1, type=float, help="Margin parameter setting")
    parser.add_argument("--num_workers", default=64, type=float, help="num_workers")
    parser.add_argument("--initialize", default=False, action="store_true", help="initail teacher inference")

    parser.add_argument(
        "--embedding_size",
        default=324,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument("--model", default="dinov3", help="Model for training")
    parser.add_argument(
        "--cgcd_ckpt",
        required=True,
        help="cgcd checkpoint",
    )
    parser.add_argument("--unlabel_dir", default=None, required=True, type=str, help="candidate directory name")
    parser.add_argument("--candidate", default=None, required=True, type=str, help="candidate directory name")

    args = parser.parse_args()

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # torchrun will handle distributed training
    main(args)
