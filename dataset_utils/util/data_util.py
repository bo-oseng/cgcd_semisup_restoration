# -----------------------------------------------------------------------------
# Functions for data utility
# -----------------------------------------------------------------------------
import os
import json
import random
from random import shuffle

import torch
import numpy as np
import pandas as pd
from glob import glob
from torchvision import transforms
from torchvision.transforms import functional as F
from scipy import io as mat_io
import logging
import torchvision.transforms.v2 as T


def info(text):
    """
    Output text info in terminal console
    """
    print("-" * 80)
    logging.info(f" {text}")
    print("-" * 80)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class StrongWeakView(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, strong_transform, weak_transform):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __call__(self, x):
        return [self.weak_transform(x), self.strong_transform(x)]


def build_transform(mode, args):
    """
    Return transformed image
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if mode == "default":
        crop_pct = args.crop_pct
        transform = transforms.Compose(
            [
                transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
                transforms.RandomCrop(args.input_size),
                transforms.RandomHorizontalFlip(p=0.5 if args.dataset != "mnist" else 0),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif mode == "weak":
        crop_pct = args.crop_pct
        transform = transforms.Compose(
            [
                transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    elif mode == "test":
        crop_pct = args.crop_pct
        transform = transforms.Compose(
            [
                transforms.Resize(int(args.input_size / crop_pct), interpolation=args.interpolation),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

    else:
        raise ValueError("Transform mode: {} not supported for GCD continual training.".format(mode))
    print(f"Transform for {mode}")
    print(transform)
    return transform


# class _PairedAugment:
#     def __init__(self, patch_size, transforms_list):
#         self.patch_size = patch_size
#         self.transforms_list = transforms_list
#         mean = (0.485, 0.456, 0.406)
#         std = (0.229, 0.224, 0.225)
#         self.norm = T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
#         self.to_tensor = T.ToTensor()

#     def _apply_single(self, image, crop_params, mode):
#         i, j, h, w = crop_params
#         image = F.crop(image, i, j, h, w)
#         image = self.transforms_list[mode](image)
#         return self.norm(self.to_tensor(image))

#     def __call__(self, image, paired_image=None):
#         crop_params = transforms.RandomCrop.get_params(image, (self.patch_size, self.patch_size))
#         mode = random.randrange(len(self.transforms_list))

#         if paired_image is None:
#             return self._apply_single(image, crop_params, mode)

#         image_t = self._apply_single(image, crop_params, mode)
#         paired_t = self._apply_single(paired_image, crop_params, mode)
#         return image_t, paired_t


# def get_paired_transform(patch_size=256):
#     """Randomly pick one of eight modes, keeping degrad/gt pairs aligned."""
#     transforms_list = [
#         T.Identity(),
#         T.RandomVerticalFlip(p=1.0),
#         T.RandomRotation(degrees=(90, 90)),
#         T.Compose([T.RandomRotation(degrees=(90, 90)), T.RandomVerticalFlip(p=1.0)]),
#         T.RandomRotation(degrees=(180, 180)),
#         T.Compose([T.RandomRotation(degrees=(180, 180)), T.RandomVerticalFlip(p=1.0)]),
#         T.RandomRotation(degrees=(270, 270)),
#         T.Compose([T.RandomRotation(degrees=(270, 270)), T.RandomVerticalFlip(p=1.0)]),
#     ]

#     return _PairedAugment(patch_size, transforms_list)


def denorm(tensor):
    """
    (C, H, W) 또는 (B, C, H, W) 텐서를 역정규화합니다.
    """
    # 디바이스(cuda/cpu)를 텐서와 일치시킴
    mean = torch.tensor((0.485, 0.456, 0.406))
    std = torch.tensor((0.229, 0.224, 0.225))
    denorm_mean = mean.view(-1, 1, 1)
    denorm_std = std.view(-1, 1, 1)

    mean = denorm_mean.to(tensor.device)
    std = denorm_std.to(tensor.device)

    if tensor.ndim == 4:  # (B, C, H, W) 배치인 경우
        mean = mean.unsqueeze(0)  # (1, C, 1, 1)
        std = std.unsqueeze(0)  # (1, C, 1, 1)
    return tensor.mul(std).add(mean)


class PairedAugment:
    def __init__(self, patch_size=256):
        self.patch_size = patch_size

        self.transforms_list = [
            T.Identity(),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(degrees=(90, 90)),
            T.Compose([T.RandomRotation(degrees=(90, 90)), T.RandomVerticalFlip(p=1.0)]),
            T.RandomRotation(degrees=(180, 180)),
            T.Compose([T.RandomRotation(degrees=(180, 180)), T.RandomVerticalFlip(p=1.0)]),
            T.RandomRotation(degrees=(270, 270)),
            T.Compose([T.RandomRotation(degrees=(270, 270)), T.RandomVerticalFlip(p=1.0)]),
        ]

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.norm = T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
        self.to_tensor = T.ToTensor()

    def _apply_single(self, image, crop_params, mode):
        i, j, h, w = crop_params
        image = F.crop(image, i, j, h, w)
        image = self.transforms_list[mode](image)
        return self.norm(self.to_tensor(image))

    def __call__(self, image, paired_image=None):
        crop_params = transforms.RandomCrop.get_params(image, (self.patch_size, self.patch_size))
        mode = random.randrange(len(self.transforms_list))

        if paired_image is None:
            return self._apply_single(image, crop_params, mode)

        image_t = self._apply_single(image, crop_params, mode)
        paired_t = self._apply_single(paired_image, crop_params, mode)
        return image_t, paired_t


def get_strong_transform(args):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = args.interpolation
    strong_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size), interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return strong_transform


def load_cifar100_images(split):
    info(f"Loading cifar100 images {split} dataset...")
    ########################## DATASET PATH ##########################
    dataset_path = "./data/cifar-100-images"
    ########################## DATASET PATH ##########################
    class_dict_path = os.path.join("config/cifar100/class_dict.txt")
    with open(class_dict_path, "r") as f:
        class_dict = f.read()
        class_dict = class_dict.replace("'", '"')

    class_dict = json.loads(class_dict)

    # Load train and val images path + labels
    train_label, val_label = [], []

    train_list = glob(os.path.join(dataset_path, "train/*/*.png"))
    val_list = glob(os.path.join(dataset_path, "test/*/*.png"))

    for image_path in train_list:
        label = int(class_dict[os.path.split(image_path)[0].split("/")[-1]])
        train_label.append(label)

    for image_path in val_list:
        label = int(class_dict[os.path.split(image_path)[0].split("/")[-1]])
        val_label.append(label)

    return ((train_list, train_label), (val_list, val_label))


def load_imagenet100(split):
    info(f"Loading ImageNet100 {split} dataset ...")
    train_path = "config/imagenet100/train_100.txt"
    val_path = "config/imagenet100/val_100.txt"
    ########################## DATASET PATH ##########################
    dataset_path = "data/imagenet"
    ########################## DATASET PATH ##########################

    train = pd.read_csv(train_path, sep=" ", names=["filepath", "target"])
    val = pd.read_csv(val_path, sep=" ", names=["filepath", "target"])

    train["filepath"] = train["filepath"].apply(lambda x: os.path.join(dataset_path, x))
    train = [list(train[train["target"] == x][:500].values) for x in range(100)]
    train = np.concatenate(train, axis=0)
    train = pd.DataFrame(train, columns=["filepath", "target"])
    val["filepath"] = val["filepath"].apply(lambda x: os.path.join(dataset_path, x))

    # Load train and val images path + labels
    train_list = train["filepath"].values
    train_label = train["target"].values

    val_list = val["filepath"].values
    val_label = val["target"].values

    return ((train_list, train_label), (val_list, val_label))


def load_tiny_imagenet_200(split):
    info(f"Loading Tiny ImageNet 200 {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = "./data/tiny-imagenet-200"
    ########################## DATASET PATH ##########################
    class_dict_path = os.path.join("config/tinyimagenet/class_dict.txt")
    with open(class_dict_path, "r") as f:
        class_dict = f.read()
        class_dict = class_dict.replace("'", '"')

    class_dict = json.loads(class_dict)

    # Load train and val images path + labels
    train_label, val_label = [], []

    train_list = glob(os.path.join(dataset_path, "train/*/*/*.JPEG"))
    val_list = glob(os.path.join(dataset_path, "val/*/*/*.JPEG"))

    for image_path in train_list:
        label = int(class_dict[os.path.split(image_path)[0].split("/")[-2]])
        train_label.append(label)

    for image_path in val_list:
        label = int(class_dict[os.path.split(image_path)[0].split("/")[-1]])
        val_label.append(label)

    return ((train_list, train_label), (val_list, val_label))


def load_caltech101(split):
    info(f"Loading Caltech101 {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = "./data/caltech101/101_ObjectCategories"
    ########################## DATASET PATH ##########################

    classes = os.listdir(dataset_path)
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    train_list, val_list = [], []
    train_label, val_label = [], []

    for cls in classes:
        images = os.listdir(os.path.join(dataset_path, cls))
        images = [os.path.join(dataset_path, cls, x) for x in images]
        shuffle(images)
        train_list.extend(images[: int(len(images) * 0.8)])
        val_list.extend(images[int(len(images) * 0.8) :])
        train_label.extend([class_to_idx[cls]] * int(len(images) * 0.8))
        val_label.extend([class_to_idx[cls]] * (len(images) - int(len(images) * 0.8)))

    return ((train_list, train_label), (val_list, val_label))


def load_aircraft(split):
    info(f"Loading FGVC Aircraft {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = "./data/fgvc-aircraft-2013b"
    ########################## DATASET PATH ##########################

    train_classes_file = os.path.join(dataset_path, "data", "images_%s_%s.txt" % ("variant", "train"))
    val_classes_file = os.path.join(dataset_path, "data", "images_%s_%s.txt" % ("variant", "val"))

    (image_ids, train_targets, classes, class_to_idx) = find_aircraft_classes(train_classes_file)
    train_samples = make_aircraft_dataset(dataset_path, image_ids, train_targets)

    (image_ids, val_targets, classes, class_to_idx) = find_aircraft_classes(val_classes_file)
    val_samples = make_aircraft_dataset(dataset_path, image_ids, val_targets)

    train_list = [x[0] for x in train_samples]
    train_label = [x[1] for x in train_samples]

    val_list = [x[0] for x in val_samples]
    val_label = [x[1] for x in val_samples]

    return ((train_list, train_label), (val_list, val_label))


def make_aircraft_dataset(dir, image_ids, targets):
    assert len(image_ids) == len(targets)
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, "data", "images", "%s.jpg" % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_aircraft_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, "r")
    for line in f:
        split_line = line.split(" ")
        image_ids.append(split_line[0])
        targets.append(" ".join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)

    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


def load_stanford_cars(split):
    limit = 0
    info(f"Loading Stanford Cars {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = "data/stanford_car"
    ########################## DATASET PATH ##########################
    train_Dir = os.path.join(dataset_path, "cars_train")
    test_Dir = os.path.join(dataset_path, "cars_test")

    train_list, val_list = [], []
    train_label, val_label = [], []

    train_meta_path = os.path.join(dataset_path, "devkit", "cars_train_annos.mat")
    train_labels_meta = mat_io.loadmat(train_meta_path)

    for idx, img_ in enumerate(train_labels_meta["annotations"][0]):
        if limit:
            if idx > limit:
                break
        train_list.append(os.path.join(train_Dir, img_[5][0]))
        train_label.append(int(img_[4][0][0]))

    val_meta_path = os.path.join(dataset_path, "devkit", "cars_test_annos_withlabels.mat")
    val_labels_meta = mat_io.loadmat(val_meta_path)

    for idx, img_ in enumerate(val_labels_meta["annotations"][0]):
        if limit:
            if idx > limit:
                break
        val_list.append(os.path.join(test_Dir, img_[5][0]))
        val_label.append(int(img_[4][0][0]))

    # convert label to start from 0
    train_label = [x - 1 for x in train_label]
    val_label = [x - 1 for x in val_label]

    return ((train_list, train_label), (val_list, val_label))


def load_CUB_200(split="train"):
    info(f"Loading CUB 200 {split} dataset ...")
    ########################## DATASET PATH ##########################
    dataset_path = "data/CUB/CUB_200_2011"
    ########################## DATASET PATH ##########################

    images = pd.read_csv(os.path.join(dataset_path, "images.txt"), sep=" ", names=["img_id", "filepath"])
    image_class_labels = pd.read_csv(
        os.path.join(dataset_path, "image_class_labels.txt"), sep=" ", names=["img_id", "target"]
    )
    train_test_split = pd.read_csv(
        os.path.join(dataset_path, "train_test_split.txt"), sep=" ", names=["img_id", "is_training_img"]
    )

    data = images.merge(image_class_labels, on="img_id").merge(train_test_split, on="img_id")
    data["filepath"] = data["filepath"].apply(lambda x: os.path.join(dataset_path, "images", x))

    # Split train and val
    train = data[data["is_training_img"] == 1]
    val = data[data["is_training_img"] == 0]

    # Load train and val images path + labels
    train_list = train["filepath"].values
    train_label = train["target"].values - 1  # CUB labels start from 1

    val_list = val["filepath"].values
    val_label = val["target"].values - 1  # CUB labels start from 1

    return ((train_list, train_label), (val_list, val_label))


def load_CDD_11(split="train"):
    info(f"Loading CDD_11 {split} dataset ...")
    ########################## DATASET PATH ##########################
    if split != "test":
        dataset_path = "data/CDD_11/Train"
    else:
        dataset_path = "data/CDD_11/Test"

    ########################## DATASET PATH ##########################
    images = pd.read_csv(os.path.join(dataset_path, "images.txt"), sep=" ", names=["img_id", "filepath"])
    image_class_labels = pd.read_csv(
        os.path.join(dataset_path, "image_class_labels.txt"), sep=" ", names=["img_id", "target"]
    )
    train_test_split = pd.read_csv(
        os.path.join(dataset_path, "train_test_split.txt"), sep=" ", names=["img_id", "is_training_img"]
    )

    data = images.merge(image_class_labels, on="img_id").merge(train_test_split, on="img_id")
    data["filepath"] = data["filepath"].apply(lambda x: os.path.join(dataset_path, x))

    # Split train and val
    train = data[data["is_training_img"] == 1]
    val = data[data["is_training_img"] == 0]

    # Load train and val images path + labels
    train_list = train["filepath"].values
    train_label = train["target"].values

    val_list = val["filepath"].values
    val_label = val["target"].values

    return ((train_list, train_label), (val_list, val_label))
