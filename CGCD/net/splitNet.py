import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm

import copy, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import logging

logger = logging.getLogger(__name__)

from .. import losses


class create_ccd_dataset(Dataset):
    """
    Input: dataset class and splitted data index list
    Return: a new dataset class that consists only the splitted data considering CCD stage
            where stage 0 is labelled data and stage > 0 is unlabelled data
    """

    def __init__(self, dataset, transform, stage) -> None:
        super(create_ccd_dataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.batch_labeled_or_not = 1 if stage == 0 else 0

    def __getitem__(self, index):
        batch_data = cv2.imread(self.dataset["paths"][index])
        batch_data = cv2.cvtColor(batch_data, cv2.COLOR_BGR2RGB)
        batch_data = Image.fromarray(batch_data)

        batch_label = self.dataset["labels"][index]
        batch_unique_index = self.dataset["uq_idx"][index]

        batch_data = self.transform(batch_data)
        return batch_data, batch_label, batch_unique_index, np.array([self.batch_labeled_or_not])

    def __len__(self):
        return self.dataset["len"]


def Utils_SaveTxt(epoch, result_path, value1, value2=-1, value3=-1, value4=-1, name="Loss"):
    path = result_path + "/" + name + ".txt"
    val = open(path, "a+")
    if value2 == -1 and value3 == -1 and value4 == -1:
        val.write("%d, %f \n" % (epoch, value1))
    elif value3 == -1 and value4 == -1:
        val.write("%d, %f, %f \n" % (epoch, value1, value2))
    elif value4 == -1:
        val.write("%d, %f, %f, %f \n" % (epoch, value1, value2, value3))
    else:
        val.write("%d, %f, %f, %f, %f \n" % (epoch, value1, value2, value3, value4))
    val.close()


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


class Soft_cross_entropy(object):
    def __call__(self, outputs_x, targets_x):
        return -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))


class SplitNet(torch.nn.Module):
    def __init__(self, sz_feature=512, sz_embed=128):
        torch.nn.Module.__init__(self)
        self.fc2 = nn.Linear(sz_feature, sz_embed // 2)
        self.batch2 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu2 = nn.Sigmoid()
        self.fc3 = nn.Linear(sz_embed // 2, sz_embed // 2)
        self.batch3 = torch.nn.BatchNorm1d(sz_embed // 2)
        self.relu3 = nn.Sigmoid()
        self.fc6 = nn.Linear(sz_embed // 2, 2)

    def forward(self, X):
        out_f = self.fc2(X)
        out_f = self.batch2(out_f)
        out_f = self.relu2(out_f)
        out_f = self.fc3(out_f)
        out_f = self.batch3(out_f)
        out_f = self.relu3(out_f)
        out_f = self.fc6(out_f)

        return out_f


class SplitModlue:
    def __init__(self, save_path, sz_feature=512, sz_embed=64):
        self.siplitnet = SplitNet(sz_feature=sz_feature, sz_embed=sz_embed)
        self.siplitnet = self.siplitnet.cuda()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean").cuda()
        self.cross_entropy_none = nn.CrossEntropyLoss(reduction="none").cuda()
        self.soft_cross_entropy = Soft_cross_entropy()
        self.save_path = save_path

    def show_OnN(
        self,
        labels,
        predict,
        nb_classes,
        pth_result,
        pth_name,
        thres=0.0,
        is_hist=False,
        iter=0,
        loss=None,
    ):
        old_correctly_identified, old_misidentified_as_new = 0, 0
        new_misidentified_as_old, new_correctly_identified = 0, 0
        old, new = [], []

        for j in range(len(labels)):
            if labels[j] < nb_classes:
                if loss is not None:
                    old.append(loss[j])
                else:
                    old.append(predict[j])

                if predict[j] >= thres:
                    old_correctly_identified += 1
                else:
                    old_misidentified_as_new += 1
            else:
                if loss is not None:
                    new.append(loss[j])
                else:
                    new.append(predict[j])

                if predict[j] >= thres:
                    new_misidentified_as_old += 1
                else:
                    new_correctly_identified += 1

        if is_hist is True:
            plt.hist((old, new), histtype="bar", bins=100, label=["old", "new"])
            plt.legend()
            plt.savefig(pth_result + "/" + pth_name + str(iter) + ".png")
            plt.close()

        return old_correctly_identified, old_misidentified_as_new, new_misidentified_as_old, new_correctly_identified

    # def predict_batchwise(self, model, dataloader):
    #     model_is_training = model.training
    #     model.eval()

    #     ds = dataloader.dataset
    #     A = [[] for i in range(len(ds[0]))]
    #     with torch.no_grad():
    #         for batch in tqdm(dataloader):
    #             for i, J in enumerate(batch):
    #                 if i == 0:
    #                     J = model(J.cuda())
    #                 for j in J:
    #                     A[i].append(j)
    #     model.train()
    #     model.train(model_is_training)

    #     return [torch.stack(A[i]) for i in range(len(A))]

    def predict_batchwise(self, model, dataloader):
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
                if not isinstance(label_batch, torch.Tensor):
                    label_batch = torch.as_tensor(label_batch)

                if len(batch) > 2:
                    index_batch = batch[2]
                    if not isinstance(index_batch, torch.Tensor):
                        index_batch = torch.as_tensor(index_batch)
                else:
                    index_batch = torch.arange(label_batch.size(0))

                feats.append(model(images))
                labels.append(label_batch)
                indices.append(index_batch)

        model.train(was_training)

        return [torch.cat(tensors).to(device) for tensors in (feats, labels, indices)]

    def predict_batchwise_split(self, model, splitnet, dataloader):
        """
        Run `model` and `splitnet` over the entire dataloader and collect tensors:
        [split_logits, labels, indices]. Any additional batch entries (e.g., masks)
        are ignored to stay compatible with the new CCD dataset structure.
        """

        device = next(model.parameters()).device
        split_device = next(splitnet.parameters()).device
        was_training_model = model.training
        was_training_split = splitnet.training
        model.eval()
        splitnet.eval()

        logits, labels, indices = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                    raise ValueError("Each batch must provide at least (images, labels[, indices]).")

                images = batch[0].to(device, non_blocking=True)
                label_batch = batch[1]
                if not isinstance(label_batch, torch.Tensor):
                    label_batch = torch.as_tensor(label_batch)

                if len(batch) > 2:
                    index_batch = batch[2]
                    if not isinstance(index_batch, torch.Tensor):
                        index_batch = torch.as_tensor(index_batch)
                else:
                    index_batch = torch.arange(label_batch.size(0))

                feats = model(images)
                logits.append(splitnet(feats.to(split_device)))
                labels.append(label_batch)
                indices.append(index_batch)

        model.train(was_training_model)
        splitnet.train(was_training_split)

        stacked_logits = torch.cat(logits).to(split_device)
        stacked_labels = torch.cat(labels).to(device)
        stacked_indices = torch.cat(indices).to(device)

        return [stacked_logits, stacked_labels, stacked_indices]

    def evaluate_cos_(self, model, dataloader):
        X, T, _ = self.predict_batchwise(model, dataloader)
        X = l2_norm(X)

        return X, T

    def generate_dataset(self, dataset, remove_index, index_target=None, target=None, index_target_new=None):
        """
        Build a CCD dataset subset that keeps indices not present in remove_index and
        remaps labels so SplitNet can run binary classification on old/new instances.
        Works with both legacy datasets (with .im_paths) and the new CCD dict-based datasets.
        """

        # Backwards compatibility for legacy datasets that expose im_paths/labels/I lists.
        if hasattr(dataset, "im_paths"):
            dataset_ = copy.deepcopy(dataset)

            if index_target is not None:
                for v in index_target:
                    dataset_.labels[v] = 0

            if index_target_new is not None:
                for v in index_target_new:
                    dataset_.labels[v] = 1

            for removed_count, original_index in enumerate(remove_index):
                adjusted_index = original_index - removed_count
                dataset_.I.pop(adjusted_index)
                dataset_.labels.pop(adjusted_index)
                dataset_.im_paths.pop(adjusted_index)

            return dataset_

        if not hasattr(dataset, "dataset"):
            raise ValueError("Unsupported dataset type for SplitNet.generate_dataset")

        src = dataset.dataset  # {"paths", "labels", "uq_idx", "len"}
        total_len = src.get("len", len(src.get("paths", [])))

        def _normalize_indices(indices):
            if indices is None:
                return []
            if isinstance(indices, torch.Tensor):
                indices = indices.detach().cpu().numpy()
            arr = np.asarray(indices, dtype=np.int64).reshape(-1)
            return [int(idx) for idx in arr if 0 <= int(idx) < total_len]

        remove_index = set(_normalize_indices(remove_index))
        idx_old = _normalize_indices(index_target)
        idx_new = _normalize_indices(index_target_new)

        paths = list(src["paths"])
        labels = list(src["labels"])
        uq_idx = np.asarray(src["uq_idx"])

        for idx in idx_old:
            labels[idx] = 0
        for idx in idx_new:
            labels[idx] = 1

        keep_indices = [i for i in range(total_len) if i not in remove_index]

        subset = {
            "paths": [paths[i] for i in keep_indices],
            "labels": [labels[i] for i in keep_indices],
            "uq_idx": uq_idx[keep_indices] if len(keep_indices) > 0 else np.empty((0,), dtype=uq_idx.dtype),
            "len": len(keep_indices),
        }

        stage_flag = 0 if getattr(dataset, "batch_labeled_or_not", 0) == 1 else 1
        return create_ccd_dataset(subset, dataset.transform, stage_flag)

    def calc_old_new(self, preds_cs, thres_cos, y, is_hist=True, last_old_num=160, step=0, confidence_thres=0.03):

        if step > 0:
            print("Normalizing...")
            preds_cs = ((preds_cs - preds_cs.min()) / (preds_cs.max() - preds_cs.min())) * 2.0 - 1.0

        thres_cos_min = thres_cos - confidence_thres
        thres_cos_max = thres_cos + confidence_thres

        idx_o_t = preds_cs >= thres_cos_max
        idx_o = np.nonzero(idx_o_t)[0]
        idx_n_t = preds_cs < thres_cos_min
        idx_n = np.nonzero(idx_n_t)[0]

        old_new = idx_o_t + idx_n_t
        old_new = ~old_new
        idx_rm = old_new.nonzero()[0]

        oo_i, on_i, no_i, nn_i = 0, 0, 0, 0
        for j in range(len(idx_o)):
            if y[idx_o[j]] < last_old_num:
                oo_i += 1
            else:
                no_i += 1
        for j in range(len(idx_n)):
            if y[idx_n[j]] < last_old_num:
                on_i += 1
            else:
                nn_i += 1

        # print("Init. Split result 1st. witout conf\t oo: {}\t on: {}\t no: {}\t nn: {}".format(oo_i, on_i, no_i, nn_i))
        print(
            "Init. Split result 1st. ({}~{})\t oo: {}\t on: {}\t no: {}\t nn: {}".format(
                str(thres_cos_min), str(thres_cos_max), oo_i, on_i, no_i, nn_i
            )
        )
        print("idx_rm: ", len(idx_rm))

        oo_i, on_i, no_i, nn_i = self.show_OnN(
            y, preds_cs, last_old_num, self.save_path, "Cos_similarity_", thres=thres_cos, is_hist=is_hist, iter=0
        )

        return idx_o, idx_n, idx_rm

    def calc_GMM_cross(
        self,
        total_loss,
        y,
        is_hist=True,
        thres=0.5,
        thres_min=0.05,
        thres_max=0.95,
        last_old_num=160,
        epoch=0,
        training=True,
    ):

        total_loss_org = (total_loss - total_loss.min()) / (total_loss.max() - total_loss.min())
        total_loss = np.reshape(total_loss_org, (-1, 1))
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(total_loss)
        prob = gmm.predict_proba(total_loss)
        prob = prob[:, gmm.means_.argmin()]

        prob_mean = np.mean(prob)
        prob_var = np.var(prob)
        gmm_pro_max = thres_max
        gmm_pro_min = thres_min

        pred_zero = prob >= gmm_pro_max
        label_zero_index = pred_zero.nonzero()[0]  # old
        pred_one = prob < gmm_pro_min
        label_one_index = pred_one.nonzero()[0]  # new

        old_new = pred_zero + pred_one
        old_new = ~old_new
        idx_rm = old_new.nonzero()[0]

        old_correctly_identified, old_misidentified_as_new, new_misidentified_as_old, new_correctly_identified = (
            self.show_OnN(
                y,
                prob,
                last_old_num,
                self.save_path,
                "Fine_Split_",
                thres=thres,
                is_hist=is_hist,
                iter=epoch,
                loss=total_loss_org,
            )
        )
        print(
            "Fine. Split result(0.5)\t oo: {}\t on: {}\t no: {}\t nn: {}".format(
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

        if training:
            return label_zero_index, label_one_index, idx_rm
        else:
            label_zero_index = np.concatenate([idx_rm, label_zero_index], axis=0)
            label_one_index = np.concatenate([idx_rm, label_one_index], axis=0)
            return label_zero_index, label_one_index, idx_rm

    def split_old_and_new(
        self,
        main_model,
        proxy,
        old_new_dataset_eval,
        old_new_dataset_train,
        main_epoch=3,
        sub_epoch=5,
        lr=5e-5,
        weight_decay=5e-3,
        batch_size=64,
        num_workers=4,
        last_old_num=160,
        thres_min=0.05,
        thres_max=0.95,
        thres_cos=0.0,
        confidence_thres=0.03,
        step=0,
    ):

        main_model_ = copy.deepcopy(main_model)
        main_model_ = main_model_.cuda()
        param_groups = [
            {"params": main_model_.parameters()},
            {"params": self.siplitnet.parameters()},
        ]
        opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
        data_loader_ev = torch.utils.data.DataLoader(
            old_new_dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        m_org, y_true = self.evaluate_cos_(main_model, data_loader_ev)
        cos_sim = F.linear(losses.l2_norm(m_org), losses.l2_norm(proxy.proxies))
        v, y_p = torch.max(cos_sim, dim=1)

        y_p_noise = y_p.cpu().detach().numpy()
        v_arr = v.cpu().detach().numpy()
        y_true_arr = y_true.cpu().detach().numpy()

        idx_o, idx_n, idx_rm = self.calc_old_new(
            preds_cs=v_arr,
            thres_cos=thres_cos,
            y=y_true_arr,
            last_old_num=last_old_num,
            step=step,
        )

        ev_dataset_o = self.generate_dataset(
            dataset=old_new_dataset_train,
            remove_index=idx_rm,
            index_target=idx_o,
            target=y_p_noise,
            index_target_new=idx_n,
        )
        data_loader_tr_o = torch.utils.data.DataLoader(
            ev_dataset_o,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
        )

        for epoch_main in range(main_epoch):
            for epoch_sub in range(sub_epoch):
                pbar = enumerate(data_loader_tr_o)
                total_loss_arr = []
                for batch_idx, (x, y, z, _) in pbar:
                    self.siplitnet.train()
                    main_model_.train()
                    y = F.one_hot(y, num_classes=2).cuda()
                    x = x.cuda()
                    m = main_model_(x.cuda())

                    y_bin_pridict = self.siplitnet(m)
                    total_loss = self.soft_cross_entropy(y_bin_pridict, y)

                    opt.zero_grad()
                    total_loss.backward()
                    opt.step()

                    total_loss_arr.append(total_loss.cpu().detach().numpy())
                total_loss_arr = np.array(total_loss_arr)
                print(
                    "Fine. Split result Epoch: {}/{} Loss: {}".format(epoch_main, epoch_sub, np.mean(total_loss_arr))
                )

            m_, y_true, _ = self.predict_batchwise_split(main_model_, self.siplitnet, data_loader_ev)

            y_true_arr = y_true.cpu().detach().numpy()
            m_ = torch.softmax(m_, dim=1)
            y_0_1 = []
            for i in range(len(y_p_noise)):
                y_0_1.append(0)

            y_hot = np.array(y_0_1)
            y_hot = torch.tensor(y_hot)
            y_hot = y_hot.cuda()
            y_hot = y_hot.long()

            loss_total = self.cross_entropy_none(m_, y_hot)

            idx_o, idx_n, idx_rm = self.calc_GMM_cross(
                loss_total.cpu().detach().numpy(),
                y_true_arr,
                epoch=epoch_main,
                last_old_num=last_old_num,
                thres_min=thres_min,
                thres_max=thres_max,
            )
            ev_dataset_o = self.generate_dataset(
                dataset=old_new_dataset_train,
                remove_index=idx_rm,
                index_target=idx_o,
                target=y_p_noise,
                index_target_new=idx_n,
            )
            data_loader_tr_o = torch.utils.data.DataLoader(
                ev_dataset_o,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=True,
            )

        idx_o, idx_n, _ = self.calc_GMM_cross(
            loss_total.cpu().detach().numpy(),
            y_true_arr,
            epoch=epoch_main,
            last_old_num=last_old_num,
            thres_min=thres_min,
            thres_max=thres_max,
            training=False,
        )
        del opt

        return idx_o, idx_n
