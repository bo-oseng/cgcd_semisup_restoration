import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import exp
from torchvision import transforms
from torchvision.models import vgg16
import torchvision


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs**weights
    pow2 = mssim**weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()

    def forward(self, xs, ys):
        L2_temp = 0.2 * self.L2(xs, ys)
        L1_temp = 0.8 * self.L1(xs, ys)
        L_total = L1_temp + L2_temp
        return L_total


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, res, gt):
        res = (res + 1.0) * 127.5
        gt = (gt + 1.0) * 127.5
        r_mean = (res[:, 0, :, :] + gt[:, 0, :, :]) / 2.0
        r = res[:, 0, :, :] - gt[:, 0, :, :]
        g = res[:, 1, :, :] - gt[:, 1, :, :]
        b = res[:, 2, :, :] - gt[:, 2, :, :]
        p_loss_temp = (((512 + r_mean) * r * r) / 256) + 4 * g * g + (((767 - r_mean) * b * b) / 256)
        p_loss = torch.mean(torch.sqrt(p_loss_temp + 1e-8)) / 255.0
        return p_loss


class PerpetualLoss(nn.Module):
    def __init__(self, vgg_model):
        super(PerpetualLoss, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {"3": "relu1_2", "8": "relu2_2", "15": "relu3_3"}

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, dehaze, gt):
        loss = []
        dehaze_features = self.output_features(dehaze)
        gt_features = self.output_features(gt)
        for dehaze_feature, gt_feature in zip(dehaze_features, gt_features):
            loss.append(F.mse_loss(dehaze_feature, gt_feature))

        return sum(loss) / len(loss)


# Charbonnier loss
class CharLoss(nn.Module):
    def __init__(self):
        super(CharLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        diff = torch.add(pred, -target)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Total_loss(nn.Module):
    def __init__(self, args):
        super(Total_loss, self).__init__()
        self.weight_sl1, self.weight_msssim = (0.6, 0.4)

    # total_loss = loss(inp, pos, neg, out)
    def forward(self, pos, out):
        smooth_loss_l1 = F.smooth_l1_loss(out, pos)
        msssim_loss = 1 - msssim(out, pos, normalize=True)

        total_loss = self.weight_sl1 * smooth_loss_l1 + self.weight_msssim * msssim_loss
        return total_loss
