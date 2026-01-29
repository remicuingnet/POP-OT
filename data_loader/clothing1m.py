"""
Key differences from CIFAR-10/100:

Image loading: CLOTHING1M uses file paths instead of in-memory arrays
Image size: Uses 224x224 (ImageNet-style) instead of 32x32
Normalization: Uses ImageNet statistics
No synthetic noise: CLOTHING1M has real-world noisy labels (39% noise rate)
File structure: Reads from text files (noisy_train.txt, clean_val.txt, clean_test.txt)
14 classes: Instead of 10 or 100
Clean subset: Has clean training labels available for some samples
Expected directory structure:

data_dir/
├── noisy_train.txt
├── clean_train.txt (optional, for clean subset)
├── clean_val.txt
├── clean_test.txt
└── images/
    ├── 0/
    ├── 1/
    └── ...

"""

from typing import Literal
import os
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def get_clothing1m(
    root,
    train=True,
    transform_train=None,
    transform_train_aug=None,
    transform_val=None,
):
    if train:
        train_dataset = Clothing1M(
            root,
            mode="train",
            transform=transform_train,
            transform_aug=transform_train_aug,
        )
        val_dataset = Clothing1MVal(
            root,
            mode="val",
            transform=transform_val,
        )
        noise_level = train_dataset.noise_level

    else:
        train_dataset = []
        val_dataset = Clothing1MVal(
            root,
            mode="test",
            transform=transform_val,
        )
        noise_level = 0

    return train_dataset, val_dataset, noise_level


class Clothing1M(Dataset):
    def __init__(
        self,
        root,
        mode: Literal["train", "test", "val"] = "train",
        transform=None,
        transform_aug=None,
    ):
        self.root = root
        self.transform = transform
        self.transform_aug = transform_aug
        self.num_classes = 14
        self.train = mode == "train"

        noisy = self.train

        # Load data
        self.data, self.labels, self.noise_level = load_db(
            root=root, mode=mode, noisy=noisy
        )
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.labels[index]

        # Load image
        img = Image.open(os.path.join(self.root, "clothing1m", img_path)).convert("RGB")

        if self.transform_aug is not None:
            img2 = self.transform_aug(img)
        else:
            img2 = self.transform(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, img2, target, index, 0  # target_gt

    def __len__(self):
        return len(self.data)


class Clothing1MVal(Clothing1M):

    def __init__(
        self,
        root,
        mode: Literal["train", "test", "val"] = "train",
        transform=None,
        transform_aug=None,
    ):
        super().__init__(
            root=root, mode=mode, transform=transform, transform_aug=transform_aug
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.data[index]
        target = self.labels[index]

        # Load image
        img = Image.open(os.path.join(self.root, "clothing1m", img_path)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index, 0  # target_gt


def load_db(root, mode: Literal["train", "test", "val"], noisy=False):
    noise = 0.0

    fname = "clothing1m/clean_label_kv.txt"
    file_path = os.path.join(root, fname)
    labels = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            img_path, label = line.strip().split()
            labels[img_path] = int(label)

    if noisy:
        clean_labels = labels
        labels = {}
        fname = "clothing1m/noisy_label_kv.txt"
        file_path = os.path.join(root, fname)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                img_path, label = line.strip().split()
                labels[img_path] = int(label)
        errors = 0
        cnt = 0
        for k, v in clean_labels.items():
            if k in labels:
                cnt += 1
                errors += int(v != labels[k])
        clean_labels.update(labels)
        labels = clean_labels
        print(f"{cnt} not in  noisy and clean")
        noise = float(errors) / float(cnt)

    fname = {
        "train": (
            "clothing1m/noisy_train_key_list.txt"
            if noisy
            else "clothing1m/clean_train_key_list.txt"
        ),
        "test": "clothing1m/clean_test_key_list.txt",
        "val": "clothing1m/clean_val_key_list.txt",
    }
    file_path = os.path.join(root, fname[mode])

    label_list = []
    data_list = []
    missing = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for key in lines:
        if key in labels:  # and os.path.exists(os.path.join(root, "clothing1m", key)):
            label_list.append(labels[key])
            data_list.append(key)
        else:
            print(key)
            # print(os.path.normpath(os.path.join(root, "clothing1m", key)))
            missing.append(key)
    print(f"{len(missing)} missing files")
    # data_list = data_list[:1000]
    # label_list = label_list[:1000]
    return data_list, label_list, noise
