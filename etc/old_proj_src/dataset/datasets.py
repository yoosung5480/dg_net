# =========================================================
# Universal SSL DataLoader (ImageNet / STL10 / Custom)
# - Extended Config
# - Debugging
# - Transform Control
# - Subset / Slicing
# =========================================================

import os
import random
import numpy as np
from typing import List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


from dataclasses import dataclass
from typing import Optional, List, Tuple

from src.config.DG_config import DataConfig

# @dataclass
# class DataConfig:
#     # -------------------------
#     # dataset
#     # -------------------------
#     dataset: str = "imagenet"   # "imagenet", "stl10", "custom"
#     data_path: str = "/data/imagenet/train"

#     split: str = "train"
#     classes: Optional[List[int]] = None
#     max_samples: Optional[int] = None

#     # -------------------------
#     # transform
#     # -------------------------
#     img_size: int = 224

#     use_augmentation: bool = True
#     crop_scale: Tuple[float, float] = (0.08, 1.0)
#     hflip_prob: float = 0.5

#     color_jitter: Optional[Tuple[float, float, float, float]] = None

#     gaussian_blur: bool = False
#     blur_prob: float = 0.5

#     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
#     std: Tuple[float, float, float]  = (0.229, 0.224, 0.225)

#     # -------------------------
#     # dataloader
#     # -------------------------
#     batch_size: int = 64
#     num_workers: int = 4

#     shuffle: bool = True
#     drop_last: bool = True
#     pin_memory: bool = True

#     # reproducibility
#     seed: int = 42
    
# =========================================================
# 2. Seed
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# 3. Transform
# =========================================================
def build_transform(cfg):
    t = []

    if cfg.use_augmentation:
        t.append(transforms.RandomResizedCrop(
            cfg.img_size,
            scale=cfg.crop_scale
        ))

        if cfg.hflip_prob > 0:
            t.append(transforms.RandomHorizontalFlip(cfg.hflip_prob))

        if cfg.color_jitter is not None:
            t.append(transforms.ColorJitter(*cfg.color_jitter))

        if cfg.gaussian_blur:
            t.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23)
            ], p=cfg.blur_prob))
    else:
        t.append(transforms.Resize((cfg.img_size, cfg.img_size)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(cfg.mean, cfg.std))

    return transforms.Compose(t)


# =========================================================
# 4. Dataset Builders
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


# ---------- STL10 ----------
def build_stl10_dataset(root, transform, split="train", max_samples=None):
    dataset = datasets.STL10(
        root=root,
        split=split,
        download=True,
        transform=transform
    )

    if max_samples is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    return dataset


# ---------- Custom ----------
class CustomImageDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = []
        self.transform = transform

        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.paths.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def build_custom_dataset(root, transform, max_samples=None):
    dataset = CustomImageDataset(root, transform)

    if max_samples is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:max_samples])

    return dataset


# =========================================================
# 5. Dataset Interface
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
# 6. Debug
# =========================================================
def debug_dataset(dataset, name="Dataset"):
    print(f"\n===== {name} Info =====")

    print(f"Total samples: {len(dataset)}")

    if hasattr(dataset, 'dataset'):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    print(f"Dataset type: {type(base_dataset).__name__}")

    if hasattr(base_dataset, 'classes'):
        print(f"Num classes: {len(base_dataset.classes)}")

    if hasattr(base_dataset, 'samples'):
        print(f"Total files: {len(base_dataset.samples)}")

    elif hasattr(base_dataset, 'paths'):
        print(f"Total files: {len(base_dataset.paths)}")

    elif hasattr(base_dataset, 'data'):
        print(f"Total files (from binary): {len(base_dataset.data)}")

    print("========================\n")


# =========================================================
# 7. DataLoader
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
# 8. Inspect / Visualization
# =========================================================
def inspect_loader(loader):
    print("===== Inspect Loader =====")

    for i, batch in enumerate(loader):
        print(f"Batch {i}")

        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        print(f"Shape: {x.shape}")
        print(f"Min: {x.min().item():.4f}, Max: {x.max().item():.4f}")
        break

    print("==========================\n")


def visualize_batch(loader):
    batch = next(iter(loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch

    img = x[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()


# =========================================================
# 9. Example Usage
# =========================================================
if __name__ == "__main__":

    cfg = DataConfig()

    # -------- Example 1: STL10 --------
    cfg.dataset = "stl10"
    cfg.data_path = "/home/jeongyuseong/바탕화면/datasets"
    cfg.split = "train+unlabeled"
    cfg.max_samples = 5000

    # -------- Example 2: ImageNet subset --------
    # cfg.dataset = "imagenet"
    # cfg.data_path = "/data/imagenet/train"
    # cfg.classes = [0,1,2,3,4]
    # cfg.max_samples = 10000

    # -------- Example 3: Custom --------
    # cfg.dataset = "custom"
    # cfg.data_path = "/data/custom"

    loader = prepare_dataloader(cfg)

    inspect_loader(loader)
    visualize_batch(loader)