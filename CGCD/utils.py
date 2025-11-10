import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import logging

# import losses
# import json
from tqdm import tqdm

# import math
# import os
# import sys

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN, Birch, AffinityPropagation, MeanShift, OPTICS, AgglomerativeClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# import hdbscan


# def cluster_pred_2_gt(y_pred, y_true):
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#     _, col_idx = linear_sum_assignment(w.max() - w)
#     return col_idx


def cluster_pred_2_gt(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    num_classes = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for sample_idx in range(y_pred.size):
        confusion_matrix[y_pred[sample_idx], y_true[sample_idx]] += 1
    _, optimal_mapping = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    return optimal_mapping


def calculate_cluster_accuracy(y_pred, y_true):
    """클러스터 예측의 정확도를 계산"""
    optimal_mapping = cluster_pred_2_gt(y_pred, y_true)
    return pred_2_gt_proj_acc(optimal_mapping, y_true, y_pred)


def pred_2_gt_proj_acc(proj, y_true, y_pred):
    proj_pred = proj[y_pred]
    acc_score = accuracy_score(y_true, proj_pred)
    return acc_score


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert preds_k == targets_k  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return


def _hungarian_match_(y_pred, y_true):
    # y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # ind = linear_sum_assignment(w.max() - w)
    # acc = 0.
    # for i in range(D):
    #     acc += w[ind[0][i], ind[1][i]]
    # acc = acc * 1. / y_pred.size
    # return acc

    ind_arr, jnd_arr = linear_sum_assignment(w.max() - w)
    ind = np.array(list(zip(ind_arr, jnd_arr)))

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def predict_batchwise(model, dataloader):
    """
    Run `model` over the entire dataloader and collect feature/label/index tensors.

    The dataloader batches are expected to be tuples where the first three items are
    (image_batch, label_batch, index_batch); any additional elements (e.g., masks)
    are ignored.
    """

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    feats, labels, indices = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                raise ValueError("Each batch must provide at least (images, labels[, indices]).")

            images = batch[0].to(device, non_blocking=True)
            label_batch = batch[1]
            index_batch = batch[2] if len(batch) > 2 else torch.arange(
                label_batch.size(0),
                device=label_batch.device if isinstance(label_batch, torch.Tensor) else "cpu",
            )

            feats.append(model(images))
            labels.append(label_batch if isinstance(label_batch, torch.Tensor) else torch.tensor(label_batch))
            indices.append(index_batch if isinstance(index_batch, torch.Tensor) else torch.tensor(index_batch))

    model.train(was_training)

    return [torch.cat(tensors).to(device) for tensors in (feats, labels, indices)]


def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, _ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T == class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean


def evaluate_cos_ev(model, dataloader, proxies_new):
    nb_classes = dataloader.dataset.nb_classes()

    # acc, _ = _hungarian_match_(clustering.labels_, np.array(dlod_tr_n.dataset.labels)) #pred, true

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()
    T = T.float().cpu()

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall


def evaluate_cos_(model, dataloader):
    # nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    # X = torch.softmax(X, dim=1)

    # cos_sim = F.linear(X, X)  # 2158x2158
    # v, i = cos_sim.topk(1 + 5)
    # T1 = T[i[:, 1]]
    # V = v[:, 1].float().cpu()

    # return X[i[:, 1]], T, T1
    # return X, T, T1
    return X, T

    # clustering = AffinityPropagation(damping=0.5).fit(X.cpu().numpy())  ###
    # u, c = np.unique(clustering.labels_, return_counts=True)
    # print(u, c)

    # get predictions by assigning nearest 8 neighbors with cosine

    xs = []

    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall


def saveImage(strPath, input):
    normalize_un = transforms.Compose(
        [
            transforms.Normalize(
                mean=[-0.4914 / 0.2471, -0.4824 / 0.2435, -0.4413 / 0.2616], std=[1 / 0.2471, 1 / 0.2435, 1 / 0.2616]
            )
        ]
    )

    sqinput = input.squeeze()
    unnorminput = normalize_un(sqinput)
    npinput = unnorminput.cpu().numpy()
    npinput = np.transpose(npinput, (1, 2, 0))
    npinput = np.clip(npinput, 0.0, 1.0)

    plt.imsave(strPath, npinput)


def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1.0 * len(T))


def evaluate_cos(model, dataloader, epoch):
    # nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T, _ = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []

    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:, 1:]]
    Y = Y.float().cpu()
    T = T.float().cpu()

    recall = []
    r_at_k = calc_recall_at_k(T, Y, 1)
    recall.append(r_at_k)
    print("R@{} : {:.3f}".format(1, 100 * r_at_k))
    return recall


def show_OnN(feats, labels, predict, nb_classes, pth_result, thres=0.0, is_hist=False, iter=0):
    old_correctly_identified, old_misidentified_as_new, new_misidentified_as_old, new_correctly_identified = 0, 0, 0, 0
    old, new = [], []

    for j in range(feats.size(0)):
        if labels[j] < nb_classes:
            old.append(predict[j].cpu().numpy())
            if predict[j] >= thres:
                old_correctly_identified += 1
            else:
                old_misidentified_as_new += 1
        else:
            new.append(predict[j].cpu().numpy())
            if predict[j] >= thres:
                new_misidentified_as_old += 1
            else:
                new_correctly_identified += 1

    if is_hist is True:
        plt.hist((old, new), histtype="bar", bins=200, label=["old", "new"])
        plt.xticks(np.arange(0, 0.5, 0.05))
        plt.legend()
        plt.savefig(pth_result + "/" + "Init_Split_" + str(iter) + ".png")
        plt.close()
        # plt.clf()

    print(
        "Init. Split result(0.)\t old_correctly_identified: {}\t old_misidentified_as_new: {}\t new_misidentified_as_old: {}\t new_correctly_identified: {}".format(
            old_correctly_identified, old_misidentified_as_new, new_misidentified_as_old, new_correctly_identified
        )
    )

    # 혼동 행렬의 구성 요소 정의
    tp = old_correctly_identified
    fn = old_misidentified_as_new
    fp = new_misidentified_as_old
    tn = new_correctly_identified

    epsilon = 1e-7
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    print("\n--- Performance Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("---------------------------\n")


def find_optimal_preference(X, preference_range=None, verbose=True):
    """
    Find optimal preference for AffinityPropagation using silhouette score

    Args:
        X: feature matrix (n_samples, n_features)
        preference_range: list/array of preference values to test
        verbose: print progress

    Returns:
        best_preference: optimal preference value
        best_score: corresponding silhouette score
        results: dict with all tested preferences and their scores
    """
    if preference_range is None:
        # Default range based on similarity matrix
        similarity_matrix = -euclidean_distances(X, squared=True)
        min_sim = np.min(similarity_matrix)
        median_sim = np.median(similarity_matrix)
        preference_range = np.linspace(min_sim, median_sim, 20)

    best_score = -1
    best_preference = None
    results = {"preferences": [], "silhouette_scores": [], "n_clusters": []}

    for pref in preference_range:
        try:
            af = AffinityPropagation(preference=pref, random_state=42)
            labels = af.fit_predict(X)
            n_clusters = len(np.unique(labels))

            if n_clusters > 1 and n_clusters < len(X):  # Valid clustering
                score = silhouette_score(X, labels)
                results["preferences"].append(pref)
                results["silhouette_scores"].append(score)
                results["n_clusters"].append(n_clusters)

                if score > best_score:
                    best_score = score
                    best_preference = pref

                if verbose:
                    print(f"Preference: {pref:.2f}, Clusters: {n_clusters}, Silhouette: {score:.3f}")
            else:
                if verbose:
                    print(f"Preference: {pref:.2f}, Clusters: {n_clusters} (invalid)")

        except Exception as e:
            if verbose:
                print(f"Preference: {pref:.2f}, Error: {str(e)}")

    return best_preference, best_score, results


def plot_preference_analysis(results):
    """
    Plot preference analysis results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Silhouette score vs preference
    ax1.plot(results["preferences"], results["silhouette_scores"], "bo-")
    ax1.set_xlabel("Preference")
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Score vs Preference")
    ax1.grid(True)

    # Number of clusters vs preference
    ax2.plot(results["preferences"], results["n_clusters"], "ro-")
    ax2.set_xlabel("Preference")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title("Number of Clusters vs Preference")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def preference_for_target_clusters(X, target_n_clusters, max_iter=50):
    """
    Find preference value to achieve target number of clusters

    Args:
        X: feature matrix
        target_n_clusters: desired number of clusters
        max_iter: maximum binary search iterations

    Returns:
        preference: preference value that gives closest to target clusters
    """
    similarity_matrix = -euclidean_distances(X, squared=True)

    low = np.min(similarity_matrix)
    high = np.median(similarity_matrix)

    best_preference = None
    best_diff = float("inf")

    for _ in range(max_iter):
        mid = (low + high) / 2

        try:
            af = AffinityPropagation(preference=mid, random_state=42)
            labels = af.fit_predict(X)
            n_clusters = len(np.unique(labels))

            diff = abs(n_clusters - target_n_clusters)
            if diff < best_diff:
                best_diff = diff
                best_preference = mid

            if n_clusters == target_n_clusters:
                return mid
            elif n_clusters > target_n_clusters:
                low = mid
            else:
                high = mid

        except Exception:
            high = mid

    return best_preference
