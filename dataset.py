# Copyright (c) 2021 MSK
import pandas as pd
import os
import numpy as np
from PIL import Image
import openslide

import torch.utils.data as data

IMG_SIZE = 128


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(directory, data, box=False):
    instances = []
    for _, item in data.iterrows():
        img_path = os.path.join(directory, item['img'])
        center = eval(item['center'])
        if box:
            box = eval(item['bbox'])
        else:
            box = None
        item = img_path, center, box
        instances.append(item)
    return instances


class TileNucleiDataset(data.Dataset):
    def __init__(self, data_file, img_dir, transform=None, bbox=True):
        """
        Args:
            data_file (string): Path to the csv file of all nuclei.
            img_dir (string): Path to image folder
            transform (callable): Transform to be applied on a sample.
            bbox (bool): whether to read bounding bbox
        """
        data = pd.read_csv(data_file)
        self.data = make_dataset(img_dir, data, box=bbox)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, center, box = self.data[idx]
        image = pil_loader(img_path)
        sample = {'image': image, 'center': center, 'bbox': box}
        sample = self.transform(sample)

        return sample


class WSINucleiDataset(data.Dataset):
    def __init__(self, nucleus_file, img_dir, transform=None):
        """
        Args:
            nucleus_file (string): Path to the csv file of all nuclei.
            transform (callable): Transform to be applied on a sample.
        """
        samples = pd.read_csv(nucleus_file)[['x', 'y']].values.tolist()
        slide_id = nucleus_file.split("/")[-1].split(".")[0] + '.svs'
        self.slide = openslide.OpenSlide(os.path.join(img_dir, slide_id))
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        center = np.array(self.samples[idx]).astype(np.int)
        sample = {'slide': self.slide, 'center': center}
        sample = self.transform(sample)

        return sample
