import os
from itertools import cycle

import numpy as np
import PIL.Image as Image
import pyiqa
from adamp import AdamP

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.models import vgg16
from torchvision.utils import save_image as imwrite
from tqdm import tqdm


from .utils import *

from .loss.contrast import ContrastLoss
from .loss.losses import *

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class OneRestoreArtifacts:
    epoch: int
    state_dict: Dict[str, Any]
    teacher_state_dict: Dict[str, Any]
    optimizer_dict: Dict[str, Any]
    curiter: int
    checkpoint_path: str


train_transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

cgcd_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class DDPTrainer:
    def __init__(
        self,
        model,
        teacher_model,
        args,
        supervised_loader,
        unsupervised_loader,
        val_loader,
        iter_per_epoch,
        writer,
        *,
        cgcd_proxies_info=None,
        cgcd_model=None,
        criterion=None,
        checkpoint_info=None,
        label2cls=None,
    ):

        if cgcd_proxies_info is not None:
            if not isinstance(cgcd_proxies_info, (list, tuple)) or len(cgcd_proxies_info) != 3:
                raise ValueError("`cgcd_proxies_info` must be a (model, criterion, checkpoint_info) tuple.")
            cgcd_model = cgcd_model or cgcd_proxies_info[0]
            criterion = criterion or cgcd_proxies_info[1]
            checkpoint_info = checkpoint_info or cgcd_proxies_info[2]

        if cgcd_model is None or criterion is None or checkpoint_info is None:
            raise ValueError("CGCD model, criterion and checkpoint info must be provided.")

        self.args = args

        self.supervised_loader = supervised_loader
        self.unsupervised_loader = unsupervised_loader
        self.val_loader = val_loader

        self.curiter = 0
        self.iter_per_epoch = iter_per_epoch
        self.writer = writer
        self.gamma = 0.5
        self.start_epoch = 1
        self.epochs = args.semi_epochs
        self.save_period = 20
        self.consistency = 0.2
        self.consistency_rampup = 100.0

        self.loss_unsup = nn.L1Loss()
        self.loss_str = MyLoss().cuda()
        self.loss_cr = ContrastLoss().cuda()

        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.cuda()
        self.loss_per = PerpetualLoss(vgg_model).cuda()
        self.iqa_metric = pyiqa.create_metric("musiq", as_loss=True).cuda()

        self.cgcd = cgcd_model
        self.criterion = criterion
        self.checkpoint_info = checkpoint_info

        self.cgcd.eval()

        self.cluster2label = self.checkpoint_info["best_projection"]
        max_label = max(self.cluster2label)
        self.label2cluster = np.full(max_label + 1, -1)
        for cluster_idx, label in enumerate(self.cluster2label):
            self.label2cluster[label] = cluster_idx

        self.label2cls = {i: f"unknown{i + 1}" for i in range(20)}
        for label, degra in label2cls.items():
            self.label2cls[int(label)] = degra
        self.cls2label = {clss: label for label, clss in self.label2cls.items()}

        self.clusetr_nums = len(self.criterion.proxies)
        clear_idx = int(self.label2cluster[self.cls2label["clear"]])
        proxies = self.criterion.proxies

        all_idx = torch.arange(proxies.size(0), device=proxies.device)
        neg_idx = all_idx[all_idx != clear_idx]
        neg = torch.index_select(proxies, 0, neg_idx)

        self.positive_cluster = proxies[clear_idx : clear_idx + 1]
        self.negative_cluster = neg

        self.model = model
        self.teacher_model = teacher_model

        # Move models to GPU
        self.model.cuda()
        self.teacher_model.cuda()

        # Wrap model with DDP
        self.model = DDP(
            self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

        # set optimizer and learning rate
        self.optimizer_s = AdamP(self.model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)
        self.lr_scheduler_s = lr_scheduler.MultiStepLR(self.optimizer_s, milestones=[100, 150], gamma=0.1)

        self.pseudo_label_update_cnt = 0
        self.cossim_mean = None
        self.cossim_std = None
        self.musiq_mean = None
        self.musiq_std = None
        self.score_stats_epoch = None

    @torch.no_grad()
    def _update_teachers(self, teacher, itera, keep_rate=0.996):
        # exponential moving average(EMA)
        alpha = min(1 - 1 / (itera + 1), keep_rate)
        for ema_param, param in zip(teacher.parameters(), self.model.module.parameters()):
            ema_param.data = (alpha * ema_param.data) + (1 - alpha) * param.data

    def _predict_with_out_grad(self, image, embedding):
        with torch.no_grad():
            teacher_predict_target_ul = self.teacher_model(image, embedding)

        return teacher_predict_target_ul

    def _freeze_teachers_parameters(self):
        for p in self.teacher_model.parameters():
            p.requires_grad = False

    def _get_reliable(
        self,
        teacher_predict,
        student_predict,
        positive_list,
        p_name,
        score_reference,
    ):
        score_t = self._combine_scores(teacher_predict)
        score_s = self._combine_scores(student_predict)

        positive_sample = positive_list.clone()
        mask = (score_t > score_s) & (score_t > score_reference)
        valid_indices = torch.where(mask)[0]

        positive_sample[mask] = teacher_predict[mask]

        if self.args.local_rank == 0:
            for idx in valid_indices:
                self.pseudo_label_update_cnt += 1
                imwrite(teacher_predict[idx], p_name[idx])
        return positive_sample

    def _sigmoid_rampup(self, current, rampup_length):
        # Exponential rampup
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def _get_current_consistency_weight(self, epoch):
        return self.consistency * self._sigmoid_rampup(epoch, self.consistency_rampup)

    def _get_proxy_predict(self, input_img):
        with torch.no_grad():
            lq_em = train_transform(input_img)
            feats = self.cgcd(lq_em)
            cos_sim = F.linear(l2_norm(feats), l2_norm(self.criterion.proxies))
            prediction_value, prediction_cluster = torch.max(cos_sim, dim=1)
        return prediction_value, prediction_cluster

    def _get_constratstive_scores(self, input_img, tau: float = 0.1):
        with torch.no_grad():
            lq_em = train_transform(input_img)
            feats = self.cgcd(lq_em)  # (B, D)
            feats_norm = l2_norm(feats)  # (B, D)

            pos_proxy = l2_norm(self.positive_cluster)  # (1, D)
            neg_proxies = l2_norm(self.negative_cluster)  # (K-1, D)

            pos_sim = F.linear(feats_norm, pos_proxy)  # (B, 1)
            neg_sim = F.linear(feats_norm, neg_proxies)  # (B, K-1)

            pos_logit = torch.logsumexp(pos_sim / tau, dim=-1)
            neg_logit = torch.logsumexp(neg_sim / tau, dim=-1)
            contrastive_score = torch.sigmoid(pos_logit - neg_logit)

        return contrastive_score

    def _get_musiq_score(self, input_img):
        with torch.no_grad():
            score = self.iqa_metric(input_img)
        return score.squeeze()

    def _evaluate_scores(self, image_list):
        num_candidates = image_list.shape[0]
        cossim_scores = torch.zeros(num_candidates, device=image_list.device)
        musiq_scores = torch.zeros(num_candidates, device=image_list.device)
        for idx in range(num_candidates):
            cossim_scores[idx] = self._get_constratstive_scores(image_list[idx : idx + 1])
            musiq_scores[idx] = self._get_musiq_score(image_list[idx : idx + 1])
        return cossim_scores, musiq_scores

    def _combine_scores(self, image_list):
        cossim_scores, musiq_scores = self._evaluate_scores(image_list)
        if self.cossim_mean is None or self.cossim_std is None or self.musiq_mean is None or self.musiq_std is None:
            raise RuntimeError("Score statistics have not been computed yet. Call `_compute_score_statistics` first.")

        eps = 1e-6
        cossim_mean = torch.as_tensor(self.cossim_mean, device=cossim_scores.device, dtype=cossim_scores.dtype)
        cossim_std = torch.clamp(
            torch.as_tensor(self.cossim_std, device=cossim_scores.device, dtype=cossim_scores.dtype), min=eps
        )
        musiq_mean = torch.as_tensor(self.musiq_mean, device=musiq_scores.device, dtype=musiq_scores.dtype)
        musiq_std = torch.clamp(
            torch.as_tensor(self.musiq_std, device=musiq_scores.device, dtype=musiq_scores.dtype), min=eps
        )

        z_cossim = (cossim_scores - cossim_mean) / cossim_std
        z_musiq = (musiq_scores - musiq_mean) / musiq_std
        score_reference = z_cossim + z_musiq
        return score_reference

    @torch.no_grad()
    def _compute_score_statistics(self, epoch):
        if self.score_stats_epoch == epoch:
            return

        device = torch.device(f"cuda:{self.args.local_rank}")
        cossim_sum = torch.zeros(1, device=device, dtype=torch.float64)
        cossim_sumsq = torch.zeros(1, device=device, dtype=torch.float64)
        musiq_sum = torch.zeros(1, device=device, dtype=torch.float64)
        musiq_sumsq = torch.zeros(1, device=device, dtype=torch.float64)
        sample_count = torch.zeros(1, device=device, dtype=torch.float64)

        data_iter = self.unsupervised_loader
        if self.args.local_rank == 0:
            data_iter = tqdm(data_iter, ncols=130, desc=f"Computing score stats (epoch {epoch})")

        for _, pseudo_list, _ in data_iter:
            pseudo_list = pseudo_list.cuda(non_blocking=True)
            cossim_scores, musiq_scores = self._evaluate_scores(pseudo_list)
            cossim_sum += cossim_scores.double().sum()
            cossim_sumsq += (cossim_scores.double() ** 2).sum()
            musiq_sum += musiq_scores.double().sum()
            musiq_sumsq += (musiq_scores.double() ** 2).sum()
            sample_count += pseudo_list.size(0)

        if dist.is_initialized():
            dist.all_reduce(cossim_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cossim_sumsq, op=dist.ReduceOp.SUM)
            dist.all_reduce(musiq_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(musiq_sumsq, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)

        total_samples = sample_count.item()
        if total_samples == 0:
            raise RuntimeError("Failed to compute score statistics: unsupervised dataset is empty.")

        self.cossim_mean = (cossim_sum / total_samples).item()
        self.musiq_mean = (musiq_sum / total_samples).item()

        cossim_var = cossim_sumsq / total_samples - self.cossim_mean**2
        musiq_var = musiq_sumsq / total_samples - self.musiq_mean**2

        self.cossim_std = max(cossim_var.clamp(min=0.0).sqrt().item(), 1e-6)
        self.musiq_std = max(musiq_var.clamp(min=0.0).sqrt().item(), 1e-6)
        self.score_stats_epoch = epoch

        if self.args.local_rank == 0:
            print(
                f"[Epoch {epoch}] Score stats -> "
                f"cossim mean/std: ({self.cossim_mean:.4f}, {self.cossim_std:.4f}), "
                f"musiq mean/std: ({self.musiq_mean:.4f}, {self.musiq_std:.4f})"
            )

    def _initialize_pseudo_labels(self):
        """Initialize pseudo labels using teacher model predictions before training starts"""
        if self.args.local_rank == 0:
            print("\n=== Initializing pseudo labels with teacher predictions ===")

        self.model.eval()
        self.teacher_model.eval()

        # Only process on rank 0 to avoid duplicate writes
        if self.args.local_rank == 0:
            tbar = tqdm(self.unsupervised_loader, ncols=130, desc="Initializing pseudo labels")
        else:
            tbar = self.unsupervised_loader

        initialized_count = 0

        with torch.no_grad():
            for unpaired_data, pseudo_list, pseudo_name in tbar:
                unpaired_data = unpaired_data.cuda(non_blocking=True)

                # Get teacher predictions
                unpaired_proxy_cossim, unpaired_proxy_cluster = self._get_proxy_predict(unpaired_data)
                unpaired_proxy = self.criterion.proxies[unpaired_proxy_cluster]
                teacher_predict_target_ul = self._predict_with_out_grad(unpaired_data, unpaired_proxy)

                # Save teacher predictions as initial pseudo labels (only on rank 0)
                if self.args.local_rank == 0:
                    for idx in range(teacher_predict_target_ul.shape[0]):
                        imwrite(teacher_predict_target_ul[idx], pseudo_name[idx])
                        initialized_count += 1

        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()

        if self.args.local_rank == 0:
            print(f"Initialized {initialized_count} pseudo labels with teacher predictions\n")

        # Set back to train mode
        self.model.train()

    def _train_epoch(self, epoch):
        sup_loss, unsup_loss = AverageMeter(), AverageMeter()
        loss_total_ave = 0.0
        total_score_reference = AverageMeter()
        psnr_train = []
        self.model.train()
        self._freeze_teachers_parameters()

        train_loader = iter(zip(cycle(self.supervised_loader), self.unsupervised_loader))

        # Only show progress bar on rank 0
        if self.args.local_rank == 0:
            tbar = tqdm(range(len(self.unsupervised_loader)), ncols=130, leave=True)
        else:
            tbar = range(len(self.unsupervised_loader))

        for i in tbar:
            (pair_data, label), (unpaired_data, pseudo_list, pseudo_name) = next(train_loader)

            pair_data = pair_data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            unpaired_data = unpaired_data.cuda(non_blocking=True)
            pseudo_list = pseudo_list.cuda(non_blocking=True)

            # cgcd proxy prediction
            unpaired_proxy_cossim, unpaired_proxy_cluster = self._get_proxy_predict(unpaired_data)
            unpaired_proxy = self.criterion.proxies[unpaired_proxy_cluster]

            pair_proxy_cossim, pair_proxy_cluster = self._get_proxy_predict(pair_data)
            pair_proxy = self.criterion.proxies[pair_proxy_cluster]

            # restoration outputs
            outputs_labeled = self.model(pair_data, pair_proxy)  # student output - label
            student_outputs_ul = self.model(unpaired_data, unpaired_proxy)  # student output - unlabel

            teacher_predict_target_ul = self._predict_with_out_grad(  # teacher output - unlabel
                unpaired_data, unpaired_proxy
            )

            # label supervise
            structure_loss = self.loss_str(outputs_labeled, label)
            perpetual_loss = self.loss_per(outputs_labeled, label)

            loss_sup = structure_loss + 0.3 * perpetual_loss
            sup_loss.update(loss_sup.mean().item())

            # Calculate combined IQA score for each sample in pseudo_list
            with torch.no_grad():
                score_reference = self._combine_scores(pseudo_list)
                batch_mean = score_reference.mean().item()
                batch_size = score_reference.numel()
                total_score_reference.update(batch_mean, n=batch_size)

                # unlabled semi-supervise
                pseudo_label_sample = self._get_reliable(
                    teacher_predict_target_ul,
                    student_outputs_ul,
                    pseudo_list,
                    pseudo_name,
                    score_reference,
                )

            loss_unsu = self.loss_unsup(student_outputs_ul, pseudo_label_sample) + self.loss_cr(
                student_outputs_ul, pseudo_label_sample, unpaired_data
            )

            unsup_loss.update(loss_unsu.mean().item())

            consistency_weight = self._get_current_consistency_weight(epoch)

            total_loss = consistency_weight * loss_unsu + loss_sup
            total_loss = total_loss.mean()

            psnr_train.extend(to_psnr(outputs_labeled, label))

            self.optimizer_s.zero_grad()
            total_loss.backward()
            self.optimizer_s.step()

            if self.args.local_rank == 0:
                tbar.set_description(
                    "Train-Student Epoch {} | Ls {:.4f} Lu {:.4f} | quality {:.10f}|".format(
                        epoch, sup_loss.avg, unsup_loss.avg, total_score_reference.avg
                    )
                )

            with torch.no_grad():
                self._update_teachers(teacher=self.teacher_model, itera=self.curiter)
                self.curiter = self.curiter + 1

        loss_total_ave = loss_total_ave + total_loss

        # Only log on rank 0
        if self.args.local_rank == 0 and self.writer is not None:
            self.writer.add_scalar("Train_loss", total_loss, global_step=epoch)
            self.writer.add_scalar("sup_loss", sup_loss.avg, global_step=epoch)
            self.writer.add_scalar("unsup_loss", unsup_loss.avg, global_step=epoch)
            self.writer.add_scalar("Candidate quality score", total_score_reference.avg, global_step=epoch)

        self.lr_scheduler_s.step(epoch=epoch - 1)
        return loss_total_ave, psnr_train

    def _log_pseudo_labels(self, epoch):
        """각 degradation class별 첫 번째 수도라벨을 TensorBoard에 로깅"""
        # Only log on rank 0
        if self.args.local_rank != 0 or self.writer is None:
            return

        candidate_dir = os.path.join(self.args.train_folder, "Pseudo")
        degradation_classes = ["low_haze_snow", "haze_snow", "haze_rain"]

        print(f"[Pseudo_label_update_cnt epoch: {epoch}]: {self.pseudo_label_update_cnt}")
        self.pseudo_label_update_cnt = 0
        print("!log_pseudo_labels\n")

        for deg_class in degradation_classes:
            dir_candidate = os.path.join(candidate_dir, deg_class)

            if os.path.isdir(dir_candidate):
                images = sorted(
                    [
                        f
                        for f in os.listdir(dir_candidate)
                        if f.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"))
                    ]
                )

                if len(images) > 0:
                    first_image_path = os.path.join(dir_candidate, images[0])
                    last_image_path = os.path.join(dir_candidate, images[-1])

                    first_pseudo_img = Image.open(first_image_path).convert("RGB")
                    first_pseudo_tensor = torchvision.transforms.ToTensor()(first_pseudo_img)

                    last_pseudo_img = Image.open(last_image_path).convert("RGB")
                    last_pseudo_tensor = torchvision.transforms.ToTensor()(last_pseudo_img)

                    # TensorBoard에 저장
                    self.writer.add_image(
                        f"PseudoLabel/{deg_class}/{images[0][:-4]}", first_pseudo_tensor, global_step=epoch
                    )
                    self.writer.add_image(
                        f"PseudoLabel/{deg_class}/{images[-1][:-4]}", last_pseudo_tensor, global_step=epoch
                    )
                else:
                    print(f"Warning: No images found in {dir_candidate}")
            else:
                print(f"Warning: Directory not found: {dir_candidate}")

    def _valid_epoch(self, epoch):
        psnr_val = []
        self.model.eval()
        self.teacher_model.eval()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        val_iqa = AverageMeter()  # 추가

        # Only show progress bar on rank 0
        if self.args.local_rank == 0:
            tbar = tqdm(self.val_loader, ncols=130)
        else:
            tbar = self.val_loader

        with torch.no_grad():
            for i, (val_data, val_label) in enumerate(tbar):
                val_data = val_data.cuda()
                val_label = val_label.cuda()

                # ? student forward
                val_proxy_cossim, val_proxy_cluster = self._get_proxy_predict(val_data)
                val_embedding = self.criterion.proxies[val_proxy_cluster]
                val_output = self.model(val_data, val_embedding)
                temp_psnr, temp_ssim, N = compute_psnr_ssim(val_output, val_label)

                iqa_score = self.iqa_metric(val_output).mean().item()

                val_psnr.update(temp_psnr, N)
                val_ssim.update(temp_ssim, N)
                val_iqa.update(iqa_score, N)  # 추가
                psnr_val.extend(to_psnr(val_output, val_label))

                if self.args.local_rank == 0:
                    tbar.set_description(
                        "{} Epoch {} | PSNR: {:.4f}, SSIM: {:.4f}, IQA: {:.4f}|".format(
                            "Eval-Student", epoch, val_psnr.avg, val_ssim.avg, val_iqa.avg
                        )
                    )

            # Only log on rank 0
            if self.args.local_rank == 0 and self.writer is not None:
                self.writer.add_scalar("Val_psnr", val_psnr.avg, global_step=epoch)
                self.writer.add_scalar("Val_ssim", val_ssim.avg, global_step=epoch)
                self.writer.add_scalar("Val_iqa", val_iqa.avg, global_step=epoch)  # 추가

            return psnr_val

    def train(self):
        self._freeze_teachers_parameters()
        if self.args.resume_path is None:
            initialize_weights(self.model.module)
        else:
            if self.args.local_rank == 0:
                print("Resuming from checkpoint ...")
            checkpoint = torch.load(self.args.resume_path, map_location=f"cuda:{self.args.local_rank}")
            self.model.module.load_state_dict(checkpoint["state_dict"])

            # Restore teacher model state and EMA counter
            if "teacher_state_dict" in checkpoint:
                self.teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
                if self.args.local_rank == 0:
                    print("Teacher model state restored")

            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1
                if self.args.local_rank == 0:
                    print("Set resume epoch")

            if "curiter" in checkpoint:
                self.curiter = checkpoint["curiter"]
                if self.args.local_rank == 0:
                    print(f"EMA iteration counter restored: {self.curiter}")

            if "optimizer_dict" in checkpoint:
                self.optimizer_s.load_state_dict(checkpoint["optimizer_dict"])
                if self.args.local_rank == 0:
                    print("Optimizer state restored")

        # Initialize pseudo labels with teacher predictions before training
        if self.args.initialize:
            self._initialize_pseudo_labels()

        # Compute dataset-wide score statistics for this epoch
        self._compute_score_statistics(0)

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Set epoch for DistributedSampler
            if hasattr(self.supervised_loader.sampler, "set_epoch"):
                self.supervised_loader.sampler.set_epoch(epoch)
            if hasattr(self.unsupervised_loader.sampler, "set_epoch"):
                self.unsupervised_loader.sampler.set_epoch(epoch)

            # ? train start
            loss_ave, psnr_train = self._train_epoch(epoch)
            loss_val = loss_ave.item() / 256 * self.args.batch_size
            train_psnr = sum(psnr_train) / len(psnr_train)

            # ? log pseudo labels
            self._log_pseudo_labels(epoch)

            # ? validation start
            psnr_val = self._valid_epoch(max(0, epoch))
            val_psnr = sum(psnr_val) / len(psnr_val)

            # Only print and save on rank 0
            if self.args.local_rank == 0:
                print(
                    "[%d] main_loss: %.6f, train psnr: %.6f, val psnr: %.6f, lr: %.8f"
                    % (epoch, loss_val, train_psnr, val_psnr, self.lr_scheduler_s.get_last_lr()[0])
                )

                for name, param in self.model.named_parameters():
                    if self.writer is not None:
                        self.writer.add_histogram(f"{name}", param, 0)

            # Synchronize before checkpoint saving
            if dist.is_initialized():
                dist.barrier()

            # Save checkpoint on rank 0
            checkpoint_path = str(self.args.save_path) + "model_e{}.pth".format(str(epoch))
            if self.args.local_rank == 0:
                if epoch % self.save_period == 0:
                    state = {
                        "arch": type(self.model.module).__name__,
                        "epoch": epoch,
                        "state_dict": self.model.module.state_dict(),
                        "teacher_state_dict": self.teacher_model.state_dict(),
                        "optimizer_dict": self.optimizer_s.state_dict(),
                        "curiter": self.curiter,
                    }
                    print("Saving a checkpoint: {} ...".format(str(checkpoint_path)))
                    torch.save(state, checkpoint_path)

            # Synchronize after checkpoint saving
            if dist.is_initialized():
                dist.barrier()

        return OneRestoreArtifacts(
            epoch=epoch,
            state_dict=self.model.module.state_dict(),
            teacher_state_dict=self.teacher_model.state_dict(),
            optimizer_dict=self.optimizer_s.state_dict(),
            curiter=self.curiter,
            checkpoint_path=checkpoint_path,
        )
