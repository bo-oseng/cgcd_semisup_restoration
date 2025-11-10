import copy, sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


from tqdm import *

from . import utils

# try:
from . import dataset, losses
from .net.resnet import *
from .net.vision_transformer import *

# except ImportError:
#     import dataset, utils, losses
#     from net.resnet import *
#     from net.vision_transformer import *

import argparse
import logging

logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seed for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_dataset(dataset, index, index_target=None, target=None):
    dataset_ = copy.deepcopy(dataset)

    if target is not None:
        for i, v in enumerate(index_target):
            dataset_.labels[v] = target[i]

    for removed_count, original_index in enumerate(index):
        adjusted_index = original_index - removed_count
        dataset_.I.pop(adjusted_index)
        dataset_.labels.pop(adjusted_index)
        dataset_.im_paths.pop(adjusted_index)
    return dataset_


def merge_dataset(dataset_o, dataset_n):
    dataset_ = copy.deepcopy(dataset_o)
    # if len(dataset_n.classes) > len(dataset_.classes):
    #     dataset_.classes = dataset_n.classes
    dataset_.I.extend(dataset_n.I)
    dataset_.im_paths.extend(dataset_n.im_paths)
    dataset_.labels.extend(dataset_n.labels)

    return dataset_


def load_dataset(args, pth_dataset, mode, is_train=True):
    dset = dataset.load(
        name=args.dataset, root=pth_dataset, mode=mode, transform=dataset.utils.make_transform(is_train=is_train)
    )
    dlod = torch.utils.data.DataLoader(dset, batch_size=args.sz_batch, shuffle=is_train, num_workers=args.nb_workers)
    return dset, dlod


def setup_model_and_training(args, nb_classes):
    if args.model.find("resnet18") > -1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=False, is_norm=True, bn_freeze=True)
    elif args.model.find("resnet50") > -1:
        model = Resnet50(
            embedding_size=args.sz_embedding,
            pretrained=False,
            is_norm=True,
            bn_freeze=True,
            num_classes=None,
        )
    elif args.model.find("VIT") > -1:
        base_vit = vit_base()
        base_vit.load_state_dict(torch.load("./pre/dino_vitbase16_pretrain.pth"), strict=True)
        model = VisionTransformerWithLinear(
            base_vit=base_vit,
            embedding_size=args.sz_embedding,
            is_norm=True,
            bn_freeze=True,
        )

    elif args.model.find("daclip") > -1:
        import open_clip
        from net.vision_transformer import DaclipWithLinear

        checkpoint = "pre/daclip_ViT-B-32.pt"
        model, _ = open_clip.create_model_from_pretrained("daclip_ViT-B-32", pretrained=checkpoint)
        model = DaclipWithLinear(
            base_vit=model.visual,
            embedding_size=args.sz_embedding,
            is_norm=True,
            bn_freeze=True,
        )

    elif args.model.find("dinov3") > -1:
        REPO_DIR = "/mnt/sdd/kbs/proxyrestore/CGCD/dinov3"
        base_vit = torch.hub.load(
            REPO_DIR,
            "dinov3_vitb16",
            source="local",
            weights="pre/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        )
        model = VisionTransformerWithLinear(
            base_vit=base_vit,
            embedding_size=args.sz_embedding,
            is_norm=True,
            bn_freeze=True,
        )

    model = model.cuda()

    criterion_pa = losses.Proxy_Anchor(
        nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha
    ).cuda()

    param_groups = [
        {
            "params": (
                list(set(model.parameters()).difference(set(model.model.embedding.parameters())))
                if args.gpu_id != -1
                else list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))
            )
        },
        {
            "params": (
                model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters()
            ),
            "lr": float(args.lr) * 1,
        },
    ]
    param_groups.append({"params": criterion_pa.parameters(), "lr": float(args.lr) * 100})

    opt_pa = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    scheduler_pa = torch.optim.lr_scheduler.StepLR(opt_pa, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    return model, criterion_pa, opt_pa, scheduler_pa


def load_model_with_projection(args, device="cuda"):
    cgcd_ckpt_path = args.cgcd_ckpt
    cgcd_model_name = args.model
    checkpoint = torch.load(cgcd_ckpt_path, map_location="cuda", weights_only=False)

    # 모델 클래스 결정 및 생성
    if cgcd_model_name == "resnet18":
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=False, is_norm=True, bn_freeze=True)
    elif cgcd_model_name == "resnet50":
        model = Resnet50(
            embedding_size=args.sz_embedding,
            pretrained=False,
            is_norm=True,
            bn_freeze=True,
            num_classes=None,
        )
    elif cgcd_model_name == "dinov3":
        REPO_DIR = "/mnt/sdd/kbs/proxyrestore/CGCD/dinov3"
        base_vit = torch.hub.load(
            REPO_DIR,
            "dinov3_vitb16",
            source="local",
            weights="pre/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        )
        model = VisionTransformerWithLinear(
            base_vit=base_vit,
            embedding_size=args.sz_embedding,
            is_norm=True,
            bn_freeze=True,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    model.load_state_dict(checkpoint["model_pa_state_dict"])

    # Criterion 로드
    nb_classes_total = checkpoint["nb_classes_total"]
    criterion = losses.Proxy_Anchor(
        nb_classes=nb_classes_total,
        sz_embed=args.sz_embedding,
        mrg=args.mrg,
        alpha=args.alpha,
    ).to(device)
    criterion.proxies = checkpoint["proxies_param"]

    checkpoint_info = {
        "best_projection": checkpoint["best_projection"],
        "nb_classes_old": checkpoint["nb_classes_old"],
        "nb_classes_total": checkpoint["nb_classes_total"],
        # "label2cls": checkpoint["label2cls"],
    }

    if "label2cls" in checkpoint:
        checkpoint_info["label2cls"] = checkpoint["label2cls"]

    return model, criterion, checkpoint_info


def train_initial_step(args, model, criterion_pa, opt_pa, scheduler_pa, dlod_tr_0, dlod_ev, pth_rst_exp):
    losses_list = []
    best_recall = [-sys.maxsize]
    best_accuracy = -sys.maxsize
    best_epoch = 0

    for epoch in range(0, args.nb_epochs):
        model.train()

        bn_freeze = True
        if bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion_pa.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(
                    criterion_pa.parameters()
                )

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0))
        for batch_idx, (image, target, index) in pbar:
            feats = model(image.squeeze().cuda())
            loss_pa = criterion_pa(feats, target.squeeze().cuda())
            opt_pa.zero_grad()
            loss_pa.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == "Proxy_Anchor":
                torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

            losses_per_epoch.append(loss_pa.data.cpu().numpy())
            opt_pa.step()

            with torch.no_grad():
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
                _, preds = torch.max(cos_sim, dim=1)
                correct = (preds == target.squeeze().cuda()).float().sum()
                accuracy = correct / target.size(0)

            pbar.set_description(
                f"Train Epoch: {epoch} [{batch_idx + 1}/{len(dlod_tr_0)} "
                f"({100.0 * batch_idx / len(dlod_tr_0):.0f}%)] "
                f"Loss: {loss_pa.item():.4f}/{0:.4f} Acc: {accuracy.item():.4f}"
            )

        losses_list.append(np.mean(losses_per_epoch))
        scheduler_pa.step()

        if epoch >= 0:
            with torch.no_grad():
                print("Evaluating..")
                Recalls = utils.evaluate_cos(model, dlod_ev, epoch)

            if best_recall[0] < Recalls[0]:
                best_recall = Recalls
                best_epoch = epoch
                torch.save(
                    {"model_pa_state_dict": model.state_dict(), "proxies_param": criterion_pa.proxies},
                    "{}/{}_{}_best_step_0.pth".format(pth_rst_exp, args.dataset, args.model),
                )
                with open("{}/{}_{}_best_results.txt".format(pth_rst_exp, args.dataset, args.model), "w") as f:
                    f.write("Best Epoch: {}\tBest Recall@{}: {:.4f}\n".format(best_epoch, 1, best_recall[0] * 100))
                    f.write(f" Best ACC: {accuracy.item():.4f}")

            if best_accuracy < accuracy.item():
                best_accuracy = accuracy.item()
                torch.save(
                    {"model_pa_state_dict": model.state_dict(), "proxies_param": criterion_pa.proxies},
                    f"{pth_rst_exp}/{args.dataset}_{args.model}_best_accuray.pth",
                )

    return losses_list, best_recall, best_epoch


def train_initial_freq_step(args, model, criterion_pa, opt_pa, scheduler_pa, dlod_tr_0, dlod_ev, pth_rst_exp):
    losses_list = []
    best_recall = [-sys.maxsize]
    best_accuracy = -sys.maxsize
    best_epoch = 0

    for epoch in range(0, args.nb_epochs):
        model.train()

        bn_freeze = True
        if bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion_pa.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(
                    criterion_pa.parameters()
                )

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True

        total, correct = 0, 0
        pbar = tqdm(enumerate(dlod_tr_0))
        for batch_idx, (image, target, index) in pbar:
            feats = model(image.squeeze().cuda())
            loss_pa = criterion_pa(feats, target.squeeze().cuda())
            opt_pa.zero_grad()
            loss_pa.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == "Proxy_Anchor":
                torch.nn.utils.clip_grad_value_(criterion_pa.parameters(), 10)

            losses_per_epoch.append(loss_pa.data.cpu().numpy())
            opt_pa.step()

            with torch.no_grad():
                cos_sim = F.linear(losses.l2_norm(feats), losses.l2_norm(criterion_pa.proxies))
                _, preds = torch.max(cos_sim, dim=1)
                correct = (preds == target.squeeze().cuda()).float().sum()
                accuracy = correct / target.size(0)

            pbar.set_description(
                f"Train Epoch: {epoch} [{batch_idx + 1}/{len(dlod_tr_0)} "
                f"({100.0 * batch_idx / len(dlod_tr_0):.0f}%)] "
                f"Loss: {loss_pa.item():.4f}/{0:.4f} Acc: {accuracy.item():.4f}"
            )

        losses_list.append(np.mean(losses_per_epoch))
        scheduler_pa.step()

        if epoch >= 0:
            with torch.no_grad():
                print("Evaluating..")
                Recalls = utils.evaluate_cos(model, dlod_ev, epoch)

            if best_recall[0] < Recalls[0]:
                best_recall = Recalls
                best_epoch = epoch
                torch.save(
                    {"model_pa_state_dict": model.state_dict(), "proxies_param": criterion_pa.proxies},
                    "{}/{}_{}_best_step_0.pth".format(pth_rst_exp, args.dataset, args.model),
                )
                with open("{}/{}_{}_best_results.txt".format(pth_rst_exp, args.dataset, args.model), "w") as f:
                    f.write("Best Epoch: {}\tBest Recall@{}: {:.4f}\n".format(best_epoch, 1, best_recall[0] * 100))
                    f.write(f" Best ACC: {accuracy.item():.4f}")

            if best_accuracy < accuracy.item():
                best_accuracy = accuracy.item()
                torch.save(
                    {"model_pa_state_dict": model.state_dict(), "proxies_param": criterion_pa.proxies},
                    f"{pth_rst_exp}/{args.dataset}_{args.model}_best_accuray.pth",
                )

    return losses_list, best_recall, best_epoch


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Official implementation of `Proxy Anchor Loss for Deep Metric Learning`"
        + "Our code is modified from `https://github.com/dichotomies/proxy-nca`"
    )
    # export directory, training and val datasets, test datasets
    parser.add_argument("--LOG_DIR", default="./logs", help="Path to log folder")
    parser.add_argument(
        "--dataset", default="deg", help="Training dataset, e.g. cub, cars, SOP, Inshop"
    )  # cub # mit # dog # air
    parser.add_argument(
        "--embedding_size",
        default=1024,
        type=int,
        dest="sz_embedding",
        help="Size of embedding that is appended to backbone model.",
    )
    parser.add_argument(
        "--batch_size", default=120, type=int, dest="sz_batch", help="Number of samples per batch."
    )  # 150
    parser.add_argument("--epochs", default=60, type=int, dest="nb_epochs", help="Number of training epochs.")

    parser.add_argument("--gpu_id", default=0, type=int, help="ID of GPU that is used for training.")

    parser.add_argument("--workers", default=8, type=int, dest="nb_workers", help="Number of workers for dataloader.")
    parser.add_argument("--model", default="dinov3", help="Model for training")  # resnet50 #resnet18  VIT
    parser.add_argument("--loss", default="Proxy_Anchor", help="Criterion for training")  # Proxy_Anchor #Contrastive
    parser.add_argument("--optimizer", default="adamw", help="Optimizer setting")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate setting")  # 1e-4
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay setting")
    parser.add_argument("--lr_decay_step", default=5, type=int, help="Learning decay step setting")  #
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float, help="Learning decay gamma setting")
    parser.add_argument("--alpha", default=32, type=float, help="Scaling Parameter setting")
    parser.add_argument("--mrg", default=0.1, type=float, help="Margin parameter setting")
    parser.add_argument("--warm", default=5, type=int, help="Warmup training epochs")  # 1
    parser.add_argument("--bn_freeze", default=True, type=bool, help="Batch normalization parameter freeze")
    parser.add_argument("--l2_norm", default=True, type=bool, help="L2 normlization")
    parser.add_argument("--remark", default="", help="Any reamrk")

    parser.add_argument("--use_split_modlue", type=bool, default=True)
    parser.add_argument("--use_GM_clustering", type=bool, default=True)  # False

    parser.add_argument("--exp", type=str, default="debug")
    parser.add_argument("--target_dataset", type=str, default="degradations_with_real", help="target_dataset")
    parser.add_argument("--thres", type=float, default=0.0, help="old new split threshold")
    parser.add_argument("--preference", type=float, default=None, help="AffinityPropagation preference")
    parser.add_argument(
        "--cgcd_ckpt",
        default=None,
        help="cgcd checkpoint",
    )  # resnet50 #resnet18  VIT

    parser.add_argument("--confidence_thres", type=float, default=0.03, help="init split confidence threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results")

    return parser.parse_args()
