# Copyright (c) 2021 MSK
import random
import math
import warnings
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T

IMG_SIZE = 128


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class RandomResizedCenterCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if scale[0] > scale[1]:
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale

    @staticmethod
    def get_params(img, scale):
        width, height = _get_image_size(img)
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*scale) * area

            w = int(round(math.sqrt(target_area)))
            h = int(round(math.sqrt(target_area)))

            if 0 < w <= width and 0 < h <= height:
                i = (height - h) // 2
                j = (width - w) // 2
                return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class BaseNuclearTransform:
    """Create nucleus"""
    def __init__(self, base_transform, mode='base'):
        self.base_transform = base_transform
        self.mode = mode

    def __call__(self, sample):
        img = np.array(sample['image'])
        center = sample['center']
        box = sample['bbox']
        # create the original patch
        x = img[int(center[0] - IMG_SIZE / 2):int(center[0] + IMG_SIZE / 2),
                            int(center[1] - IMG_SIZE / 2):int(center[1] + IMG_SIZE / 2)]
        if self.mode == 'br':
            # create patch with black background
            m = np.zeros_like(x[:, :, 0:1])
            m[box[0]:box[1], box[2]:box[3]] = 1
            sample_bk = np.zeros_like(x)
            x = x * m + (1 - m) * sample_bk
        img = self.base_transform(Image.fromarray(x))
        return img


class WRTwoNuclearViewTransform:
    """Create two views of given nucleus by random sampling background"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, sample):
        img = np.array(sample['image'])
        center = sample['center']
        box = sample['bbox']
        # create the original patch
        x = img[int(center[0] - IMG_SIZE / 2):int(center[0] + IMG_SIZE / 2),
                            int(center[1] - IMG_SIZE / 2):int(center[1] + IMG_SIZE / 2)]
        # create patch with a randomly sampled background
        m = np.zeros_like(x[:, :, 0:1])
        m[box[0]:box[1], box[2]:box[3]] = 1
        fsize = img.shape[0]
        a, b = int(random.uniform(0, fsize-IMG_SIZE-1)), int(random.uniform(0, fsize-IMG_SIZE-1))
        sample_bk = img[a:a+IMG_SIZE, b:b+IMG_SIZE]
        y = x*m + (1-m)*sample_bk
        q = self.base_transform(Image.fromarray(x))
        k = self.base_transform(Image.fromarray(y))
        return [q, k]


class BRTwoNuclearViewTransform:
    """Create two views of given nucleus by removing background"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, sample):
        img = np.array(sample['image'])
        center = sample['center']
        box = sample['bbox']
        # create the original patch
        x = img[int(center[0] - IMG_SIZE / 2):int(center[0] + IMG_SIZE / 2),
                            int(center[1] - IMG_SIZE / 2):int(center[1] + IMG_SIZE / 2)]
        # create patch with black background
        m = np.zeros_like(x[:, :, 0:1])
        m[box[0]:box[1], box[2]:box[3]] = 1
        sample_bk = np.zeros_like(x)
        y = x*m + (1-m)*sample_bk
        q = self.base_transform(Image.fromarray(y))
        k = self.base_transform(Image.fromarray(y))
        return [q, k]


class TwoNuclearViewTransform:
    """Create two views of given nucleus by random sampling background"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, sample):
        img = np.array(sample['image'])
        center = sample['center']
        # create the original patch
        x = img[int(center[0] - IMG_SIZE / 2):int(center[0] + IMG_SIZE / 2),
                            int(center[1] - IMG_SIZE / 2):int(center[1] + IMG_SIZE / 2)]
        q = self.base_transform(Image.fromarray(x))
        k = self.base_transform(Image.fromarray(x))
        return [q, k]


class WSINuclearTransform:
    """Create nucleus from WSI"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, sample):
        reg = sample['slide'].read_region(sample['center']-int(IMG_SIZE/4), 0, (int(IMG_SIZE/2), int(IMG_SIZE/2)))

        img = reg.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        img = self.base_transform(img.convert('RGB'))
        return img
