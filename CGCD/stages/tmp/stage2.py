from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture

from ..helper import generate_dataset, merge_dataset, load_dataset

from .. import dataset
from .. import losses
from .. import utils
from ..net.splitNet import SplitModlue

logger = logging.getLogger(__name__)


@dataclass
class Stage2Result:
    merged_loader: torch.utils.data.DataLoader
    old_loader: torch.utils.data.DataLoader
    new_loader: torch.utils.data.DataLoader
    nb_classes_prev: int
    nb_classes_new: int
    exampler_sampling: torch.Tensor
    pseudo_labels_old: np.ndarray
    pseudo_labels_new: np.ndarray
    preference: float
    cluster_centers: np.ndarray
    split_data_path: Path
    index_old: np.ndarray
    index_new: np.ndarray
    affinity_model: AffinityPropagation
    gaussian_model: Optional[GaussianMixture]


def discover_novel_categories(
    args,
    model: torch.nn.Module,
    criterion,
    dataset_path: str,
    experiment_dir: Path,
    nb_classes_prev: int,
    previous_loader,
    train_mode: str = "train_1",
    eval_mode: str = "eval_1",
) -> Stage2Result:
    model.eval()

    print("===> Discovering novel category step Calc. proxy mean and sigma for exemplar..")
    with torch.no_grad():
        feats_prev, _ = utils.evaluate_cos_(model, previous_loader)
        feats_prev = losses.l2_norm(feats_prev)
        exampler_sampling = feats_prev.std(dim=0).cuda()

    dataset_train_now, dataloader_train_now = load_dataset(args, dataset_path, train_mode, is_train=False)

    print("==> Init. Split old and new..")
    with torch.no_grad():
        feats_now, labels_now = utils.evaluate_cos_(model, dataloader_train_now)
        cos_sim = F.linear(losses.l2_norm(feats_now), losses.l2_norm(criterion.proxies))
        preds_cs, _ = torch.max(cos_sim, dim=1)
        utils.show_OnN(
            feats_now,
            labels_now,
            preds_cs,
            nb_classes_prev,
            str(experiment_dir),
            args.thres,
            True,
        )

    print("==> Fine. Split old and new..")
    ev_dataset = dataset.load(
        name=args.dataset,
        root=dataset_path,
        mode=train_mode,
        transform=dataset.utils.make_transform(is_train=False),
    )

    ev_dataset_train = dataset.load(
        name=args.dataset,
        root=dataset_path,
        mode=train_mode,
        transform=dataset.utils.make_transform(is_train=True),
    )

    split_module = SplitModlue(
        save_path=str(experiment_dir),
        sz_feature=args.sz_embedding,
        sz_embed=128,
    )
    index_new, index_old = split_module.split_old_and_new(
        main_model=model,
        proxy=criterion,
        old_new_dataset_eval=ev_dataset,
        old_new_dataset_train=ev_dataset_train,
        last_old_num=nb_classes_prev,
        thres_cos=args.thres,
        confidence_thres=args.confidence_thres,
    )

    split_data_path = experiment_dir / "split_data.json"
    with open(split_data_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"index_old": index_old.tolist(), "index_new": index_new.tolist()},
            handle,
        )
    logger.info("Saved split indices to %s", split_data_path)

    #!
    dataset_train_old = generate_dataset(dataset_train_now, index_old)
    dataset_train_new = generate_dataset(dataset_train_now, index_new)

    dataloader_train_old = torch.utils.data.DataLoader(
        dataset_train_old,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
    )
    dataloader_train_new = torch.utils.data.DataLoader(
        dataset_train_new,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
    )

    with torch.no_grad():
        feats_old, _ = utils.evaluate_cos_(model, dataloader_train_old)
        cos_sim_old = F.linear(losses.l2_norm(feats_old), losses.l2_norm(criterion.proxies))
        _, pseudo_labels_old = torch.max(cos_sim_old, dim=1)
        pseudo_labels_old = pseudo_labels_old.detach().cpu().numpy()

    with torch.no_grad():
        feats_new, _ = utils.evaluate_cos_(model, dataloader_train_new)

    if args.preference is None:
        preference, _, _ = utils.find_optimal_preference(
            feats_new.cpu().numpy(),
            preference_range=range(-20, 0),
        )
    else:
        preference = args.preference
    logger.info("AffinityPropagation preference: %s", preference)

    affinity_model = AffinityPropagation(preference=preference).fit(feats_new.cpu().numpy())
    unique_labels, _ = np.unique(affinity_model.labels_, return_counts=True)
    nb_classes_new = len(unique_labels)
    pseudo_labels_new = affinity_model.labels_

    gaussian_model: Optional[GaussianMixture] = None
    if args.use_GM_clustering:
        gaussian_model = GaussianMixture(
            n_components=nb_classes_new,
            max_iter=1000,
            tol=1e-4,
            init_params="kmeans",
        ).fit(feats_new.cpu().numpy())
        pseudo_labels_new = gaussian_model.predict(feats_new.cpu().numpy())
        logger.info("GaussianMixture refined %d novel clusters", nb_classes_new)

    dataset_train_old.labels = pseudo_labels_old.tolist()
    dataset_train_new.labels = (pseudo_labels_new + nb_classes_prev).tolist()

    dset_tr_merged = merge_dataset(dataset_train_old, dataset_train_new)
    dlod_tr_merged = torch.utils.data.DataLoader(
        dset_tr_merged,
        batch_size=args.sz_batch,
        shuffle=True,
        num_workers=args.nb_workers,
    )

    return Stage2Result(
        merged_loader=dlod_tr_merged,
        old_loader=dataloader_train_old,
        new_loader=dataloader_train_new,
        nb_classes_prev=nb_classes_prev,
        nb_classes_new=nb_classes_new,
        exampler_sampling=exampler_sampling,
        pseudo_labels_old=pseudo_labels_old,
        pseudo_labels_new=pseudo_labels_new,
        preference=preference,
        cluster_centers=affinity_model.cluster_centers_,
        split_data_path=split_data_path,
        index_old=np.asarray(index_old),
        index_new=np.asarray(index_new),
        affinity_model=affinity_model,
        gaussian_model=gaussian_model,
    )
