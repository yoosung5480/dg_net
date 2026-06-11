# =========================================================
# Classification DataLoader (ImageNet / STL10 / Custom)
# =========================================================

import os
import random
import numpy as np
from typing import Optional, List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from src.config.DG_config import DataConfig


# =========================================================
# 1. Seed
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Transform (classification은 약하게)
# =========================================================
def build_transform(cfg):

    if cfg.split == "train":
        t = [
            transforms.RandomResizedCrop(cfg.img_size),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        t = [
            transforms.Resize((cfg.img_size, cfg.img_size))
        ]

    t += [
        transforms.ToTensor(),
        transforms.Normalize(cfg.mean, cfg.std)
    ]

    return transforms.Compose(t)


# =========================================================
# 3. Dataset Builders
# =========================================================

# ---------- ImageNet ----------
def build_imagenet_dataset(root, transform, classes=None, max_samples=None):

    dataset = datasets.ImageFolder(root, transform=transform)

    if classes is not None:
        idx = [i for i, (_, label) in enumerate(dataset.samples) if label in classes]
        dataset = Subset(dataset, idx)

    if max_samples is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    return dataset


# ---------- STL10 (중요 차이: label 있음) ----------
def build_stl10_dataset(root, transform, split="train", max_samples=None):

    dataset = datasets.STL10(
        root=root,
        split=split,   # train / test
        download=True,
        transform=transform
    )

    if max_samples is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    return dataset


# ---------- Custom ----------
class CustomClsDataset(Dataset):
    def __init__(self, root, transform):
        self.samples = []
        self.transform = transform

        class_dirs = sorted(os.listdir(root))

        for label, cls in enumerate(class_dirs):
            cls_path = os.path.join(root, cls)

            for f in os.listdir(cls_path):
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_path, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_custom_dataset(root, transform, max_samples=None):

    dataset = CustomClsDataset(root, transform)

    if max_samples is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    return dataset


# =========================================================
# 4. Dataset Interface
# =========================================================
def prepare_dataset(cfg):

    transform = build_transform(cfg)

    if cfg.dataset == "imagenet":
        return build_imagenet_dataset(
            root=cfg.data_path,
            transform=transform,
            classes=cfg.classes,
            max_samples=cfg.max_samples
        )

    elif cfg.dataset == "stl10":
        return build_stl10_dataset(
            root=cfg.data_path,
            transform=transform,
            split=cfg.split,
            max_samples=cfg.max_samples
        )

    elif cfg.dataset == "custom":
        return build_custom_dataset(
            root=cfg.data_path,
            transform=transform,
            max_samples=cfg.max_samples
        )

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")


# =========================================================
# 5. Debug
# =========================================================
def debug_dataset(dataset, name="Dataset"):

    print(f"\n===== {name} (Classification) =====")
    print(f"Total samples: {len(dataset)}")

    base = dataset.dataset if hasattr(dataset, "dataset") else dataset

    print(f"Dataset type: {type(base).__name__}")

    if hasattr(base, "classes"):
        print(f"Num classes: {len(base.classes)}")

    print("===============================\n")


# =========================================================
# 6. DataLoader
# =========================================================
def prepare_dataloader(cfg):

    set_seed(cfg.seed)

    dataset = prepare_dataset(cfg)
    debug_dataset(dataset, cfg.dataset)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.pin_memory and torch.cuda.is_available()),
        drop_last=cfg.drop_last,
    )

    return loader


# =========================================================
# 7. Inspect / Visualization
# =========================================================
def inspect_loader(loader):

    print("===== Inspect Loader =====")

    for batch in loader:
        x, y = batch

        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"label example: {y[:5]}")
        break

    print("==========================\n")


def visualize_batch(loader):

    x, y = next(iter(loader))

    img = x[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.title(f"Label: {y[0].item()}")
    plt.axis("off")
    plt.show()