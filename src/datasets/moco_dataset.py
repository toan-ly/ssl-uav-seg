import os
from pathlib import Path
from glob import glob

import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

from PIL import Image, ImageFilter
import random

import torchvision.transforms as T

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_uav_img_paths(data_root='../../data/UAVid/'):
    data_root = Path(data_root)

    data_folder = ['uavid_train', 'uavid_val']
    img_paths = []
    for folder in data_folder:
        im_paths = sorted(glob(str(data_root / folder / 'seq*' / 'Images' / '*.png')))
        img_paths.extend(im_paths)
    
    print(f'Total {len(img_paths)} images found for MoCo training')
    return img_paths

def get_moco_transform(patch_size=224, version=2):
    if version == 2:
        return T.Compose([
            T.RandomResizedCrop(patch_size, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif version == 1:
        return T.Compose([
            T.RandomResizedCrop(patch_size, scale=(0.2, 1.0)),
            T.RandomGrayscale(p=0.2),
            T.ColorJitter(0.4, 0.4, 0.4, 0.4),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class UAVMocoDataset(Dataset):
    def __init__(self, img_paths, patch_size=224, moco_version=2):
        self.img_paths = img_paths
        self.patch_size = patch_size
        self.transform = get_moco_transform(patch_size, version=moco_version)
        self.two_crops_transform = TwoCropsTransform(self.transform)

    def __len__(self):
        return len(self.img_paths)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self._load_image(img_path)

        im_q, im_k = self.two_crops_transform(img)
        return im_q, im_k


