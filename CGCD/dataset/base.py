from __future__ import print_function
from __future__ import division

import os
import torch
import torchvision
import numpy as np
import PIL.Image


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.labels, self.im_paths, self.I = [], [], []

    def nb_classes(self):
        assert set(self.labels) == set(self.classes)
        return len(self.classes)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        def img_load(index):
            im = PIL.Image.open(self.im_paths[index])
            # convert gray to rgb
            if len(list(im.split())) == 1:
                im = im.convert("RGB")
            # convert RGBA to RGB
            if im.mode == "RGBA":
                im = im.convert("RGB")
            if self.transform is not None:
                im = self.transform(im)
            return im

        im = img_load(index)
        target = self.labels[index]

        return im, target, index

    def get_label(self, index):
        return self.labels[index]

    def set_subset(self, I):
        self.labels = [self.labels[i] for i in I]
        self.I = [self.I[i] for i in I]
        self.im_paths = [self.im_paths[i] for i in I]
