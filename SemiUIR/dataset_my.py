import os.path
import torch
import torch.utils.data as data
from PIL import Image
import random
from random import randrange
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import tqdm

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def rotate(img, rotate_index):
    """
    :return: 8 version of rotating image
    """
    if rotate_index == 0:
        return img
    if rotate_index == 1:
        return img.rotate(90)
    if rotate_index == 2:
        return img.rotate(180)
    if rotate_index == 3:
        return img.rotate(270)
    if rotate_index == 4:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 5:
        return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 6:
        return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
    if rotate_index == 7:
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)


class TrainLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize=256):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        # degradation classes
        self.degradation_classes = sorted(os.listdir(os.path.join(self.root, self.phase, "input")))

        # image path
        self.input_paths = []
        self.gt_paths = []

        print("setup lable data")
        for deg_class in tqdm.tqdm(self.degradation_classes):
            dir_input = os.path.join(self.root, self.phase, "input", deg_class)
            dir_gt = os.path.join(self.root, self.phase, "GT")

            if os.path.isdir(dir_input):
                input_images = sorted(make_dataset(dir_input))
                self.input_paths.extend(input_images)

                # GT는 공통으로 사용하므로 input 이미지 개수만큼 추가
                gt_images = sorted(make_dataset(dir_gt))
                self.gt_paths.extend(gt_images[: len(input_images)])

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        input_image = Image.open(self.input_paths[index]).convert("RGB")
        label_image = Image.open(self.gt_paths[index]).convert("RGB")

        w, h = input_image.size
        x = randrange(max(0, w - self.fineSize + 1))
        y = randrange(max(0, h - self.fineSize + 1))
        cropped_input = input_image.crop((x, y, x + self.fineSize, y + self.fineSize))
        cropped_label = label_image.crop((x, y, x + self.fineSize, y + self.fineSize))

        # rotate
        rotate_index = randrange(0, 8)
        rotated_input = rotate(cropped_input, rotate_index)
        rotated_label = rotate(cropped_label, rotate_index)

        # transform to (0, 1)
        input = self.transform(rotated_input)
        label = self.transform(rotated_label)

        return input, label

    def __len__(self):
        return len(self.input_paths)


class TrainUnlabeled(data.Dataset):
    def __init__(self, dataroot, phase, unlabel_dir, candidate_dir, finesize=256):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        # degradation classes
        # self.degradation_classes = sorted(os.path.join("./data/unlabeled/input"))
        # "RainReal", "SnowReal", "UnannotatedHazyImages"
        self.degradation_classes_syn = ["009.low_haze_snow", "010.haze_snow", "011.haze_rain"]
        self.degradation_classes_real = ["RainReal", "SnowReal", "UnannotatedHazyImages"]

        self.degradation_classes = self.degradation_classes_syn

        # image path
        self.input_dir = []
        self.candidate_paths = []

        print("setup unlable data")
        for deg_class in tqdm.tqdm(self.degradation_classes):
            dir_input = os.path.join(self.root, self.phase, unlabel_dir, deg_class)
            dir_candidate = os.path.join(self.root, self.phase, candidate_dir, deg_class)

            if os.path.isdir(dir_input):
                input_images = sorted(make_dataset(dir_input))
                self.input_dir.extend(input_images)

                # candidate는 공통으로 사용하므로 input 이미지 개수만큼 추가
                candidate_images = sorted(make_dataset(dir_candidate))
                self.candidate_paths.extend(candidate_images[: len(input_images)])

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        input_img = Image.open(self.input_dir[index]).convert("RGB")
        candidate = Image.open(self.candidate_paths[index]).convert("RGB")

        # weak augmentation
        unpaired_data = self.transform(input_img)

        pseudo_list = self.transform(candidate)
        pseudo_name = self.candidate_paths[index]

        return unpaired_data, pseudo_list, pseudo_name

    def __len__(self):
        return len(self.input_dir)


class ValLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize, mode=None):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        # degradation classes
        degradation_classes_new = ["009.low_haze_snow", "010.haze_snow", "011.haze_rain"]
        degradation_classes_old = [
            "001.haze",
            "002.low",
            "003.snow",
            "004.low_rain",
            "005.low_snow",
            "006.rain",
            "007.low_haze",
            "008.low_haze_rain",
        ]

        if mode == "new":
            self.degradation_classes = degradation_classes_new
        elif mode == "old":
            self.degradation_classes = degradation_classes_old
        else:
            self.degradation_classes = degradation_classes_old + degradation_classes_new
        # image path
        self.input_paths = []
        self.gt_paths = []

        for deg_class in self.degradation_classes:
            dir_input = os.path.join(self.root, self.phase, "input", deg_class)
            dir_gt = os.path.join(self.root, self.phase, "GT")

            if os.path.isdir(dir_input):
                input_images = sorted(make_dataset(dir_input))
                self.input_paths.extend(input_images)

                # GT는 공통으로 사용하므로 input 이미지 개수만큼 추가
                gt_images = sorted(make_dataset(dir_gt))
                self.gt_paths.extend(gt_images[: len(input_images)])

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        input_img = Image.open(self.input_paths[index]).convert("RGB")
        gt_img = Image.open(self.gt_paths[index]).convert("RGB")

        resized_input = input_img.resize((self.fineSize, self.fineSize), Image.LANCZOS)
        resized_gt = gt_img.resize((self.fineSize, self.fineSize), Image.LANCZOS)

        # transform to (0, 1)
        tensor_input = self.transform(resized_input)
        tensor_gt = self.transform(resized_gt)

        return tensor_input, tensor_gt

    def __len__(self):
        return len(self.input_paths)


class TestLabeled(data.Dataset):
    def __init__(self, dataroot, phase, finesize, mode=None):
        super().__init__()
        self.phase = phase
        self.root = dataroot
        self.fineSize = finesize

        # degradation classes
        degradation_classes_new = ["009.low_haze_snow", "010.haze_snow", "011.haze_rain"]
        degradation_classes_old = [
            "001.haze",
            "002.low",
            "003.snow",
            "004.low_rain",
            "005.low_snow",
            "006.rain",
            "007.low_haze",
            "008.low_haze_rain",
        ]

        if mode == "new":
            self.degradation_classes = degradation_classes_new
        elif mode == "old":
            self.degradation_classes = degradation_classes_old
        else:
            self.degradation_classes = degradation_classes_old + degradation_classes_new
        # image path
        self.input_paths = []
        self.gt_paths = []

        for deg_class in self.degradation_classes:
            dir_input = os.path.join(self.root, self.phase, "input", deg_class)
            dir_gt = os.path.join(self.root, self.phase, "GT")

            if os.path.isdir(dir_input):
                input_images = sorted(make_dataset(dir_input))
                self.input_paths.extend(input_images)

                # GT는 공통으로 사용하므로 input 이미지 개수만큼 추가
                gt_images = sorted(make_dataset(dir_gt))
                self.gt_paths.extend(gt_images[: len(input_images)])

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        input_img = Image.open(self.input_paths[index]).convert("RGB")
        gt_img = Image.open(self.gt_paths[index]).convert("RGB")

        # transform to (0, 1)
        tensor_input = self.transform(input_img)
        tensor_gt = self.transform(gt_img)
        input_name = "/".join(self.input_paths[index].split("/")[-2:])
        return tensor_input, tensor_gt, input_name

    def __len__(self):
        return len(self.input_paths)


class TestData(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot

        self.dir_input = os.path.join(self.root + "/input")

        # image path
        self.input_paths = sorted(make_dataset(self.dir_input))

        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        input_img = Image.open(self.input_paths[index]).convert("RGB")

        # transform to (0, 1)
        tensor_input = self.transform(input_img)
        return tensor_input

    def __len__(self):
        return len(self.input_paths)


class InferenceDataset(data.Dataset):
    def __init__(self, dataroot):
        super().__init__()
        self.root = dataroot
        # image path
        self.input_paths = []
        self.gt_paths = []

        dir_input = os.path.join(self.root)
        dir_gt = os.path.join("/mnt/sdd/kbs/proxyrestore/OneRestore/data/CDD-11_balance_test_new/000.clear")

        if os.path.isdir(dir_input):
            input_images = sorted(make_dataset(dir_input))
            self.input_paths.extend(input_images)

            gt_images = sorted(make_dataset(dir_gt))
            self.gt_paths.extend(gt_images[: len(input_images)])
        # transform
        self.transform = ToTensor()  # [0,1]

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        input_img = Image.open(self.input_paths[index]).convert("RGB")
        gt_img = Image.open(self.gt_paths[index]).convert("RGB")

        # transform to (0, 1)
        tensor_input = self.transform(input_img)
        tensor_gt = self.transform(gt_img)
        img_name = self.input_paths[index].split("/")[-1]
        return tensor_input, tensor_gt, img_name

    def __len__(self):
        return len(self.input_paths)


def data_aug(input_image):
    # Random Horizontal Flip
    if random.random() < 0.5:
        input_image = transforms.functional.hflip(input_image)

    # Random Vertical Flip
    if random.random() < 0.5:
        input_image = transforms.functional.vflip(input_image)

    # Random Rotation (90, 180, 270 degrees)
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        input_image = transforms.functional.rotate(input_image, angle)

    return input_image
