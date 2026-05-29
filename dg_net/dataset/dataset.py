"""Config-driven dataloaders for SSL pretraining and classification transfer.

Public entry point:

    cfg = DataConfig()
    loader = prepare_dataloader(cfg)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

try:  # supports both package imports and running this module directly
    from .dataset_config import DataConfig
except ImportError:  # pragma: no cover - direct-script compatibility
    from dataset_config import DataConfig


SUPPORTED_DATASETS = ("STL10", "CIFAR100", "Flowers102", "iNaturalist")
SUPPORTED_MODES = ("ssl", "classification")
_DATASET_NAMES = {name.lower(): name for name in SUPPORTED_DATASETS}
_DATASET_NAMES["inat"] = "iNaturalist"


class MultiViewTransform:
    """Apply the same augmentation pipeline independently for SSL views."""

    def __init__(self, transform: Callable[[Any], torch.Tensor], num_views: int) -> None:
        if num_views < 2:
            raise ValueError("SSL training requires ssl_num_views >= 2.")
        self.transform = transform
        self.num_views = num_views

    def __call__(self, image: Any) -> Tuple[torch.Tensor, ...]:
        return tuple(self.transform(image) for _ in range(self.num_views))


def _canonical_dataset(name: str) -> str:
    try:
        return _DATASET_NAMES[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {name!r}. Supported datasets: {', '.join(SUPPORTED_DATASETS)}."
        ) from exc


def _canonical_mode(mode: str) -> str:
    value = mode.strip().lower()
    if value not in SUPPORTED_MODES:
        raise ValueError(f"Unknown mode {mode!r}. Supported modes: {', '.join(SUPPORTED_MODES)}.")
    return value


def validate_config(cfg: DataConfig) -> Tuple[str, str]:
    """Validate mode/dataset constraints and return normalized identifiers."""

    name = _canonical_dataset(cfg.dataset)
    mode = _canonical_mode(cfg.mode)
    if mode == "ssl" and name != "STL10":
        raise ValueError("SSL mode currently supports STL10 only (unlabeled pretraining split).")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must not be negative.")
    if cfg.max_samples is not None and cfg.max_samples <= 0:
        raise ValueError("max_samples must be positive when specified.")
    if cfg.persistent_workers and cfg.num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    return name, mode


def set_seed(seed: int) -> None:
    """Seed dataset selection and PyTorch dataloader randomness."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform(cfg: DataConfig, mode: Optional[str] = None) -> Callable[[Any], Any]:
    """Build a single-view classification or multi-view SSL transform."""

    mode = _canonical_mode(mode or cfg.mode)
    pipeline: List[Callable[[Any], Any]] = []
    if cfg.use_augmentation:
        pipeline.append(transforms.RandomResizedCrop(cfg.img_size, scale=cfg.crop_scale))
        if cfg.hflip_prob > 0:
            pipeline.append(transforms.RandomHorizontalFlip(cfg.hflip_prob))
        if cfg.color_jitter is not None:
            pipeline.append(transforms.ColorJitter(*cfg.color_jitter))
        if cfg.gaussian_blur:
            kernel_size = max(3, (cfg.img_size // 10) * 2 + 1)
            pipeline.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size)], p=cfg.blur_prob)
            )
    else:
        pipeline.append(transforms.Resize((cfg.img_size, cfg.img_size)))
    pipeline.extend((transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)))
    base_transform = transforms.Compose(pipeline)
    if mode == "ssl":
        return MultiViewTransform(base_transform, cfg.ssl_num_views)
    return base_transform


def _is_train_split(split: str) -> bool:
    normalized = split.strip().lower()
    if normalized in {"train", "training"}:
        return True
    if normalized in {"test", "val", "valid", "validation"}:
        return False
    raise ValueError("CIFAR100 split must be 'train' or 'test'.")


def _flowers_split(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {"validation": "val", "valid": "val"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"train", "val", "test"}:
        raise ValueError("Flowers102 split must be 'train', 'val', or 'test'.")
    return normalized


def _inaturalist_version(cfg: DataConfig) -> str:
    if cfg.inaturalist_version:
        return cfg.inaturalist_version
    normalized = cfg.split.strip().lower()
    versions = {
        "train": "2021_train",
        "training": "2021_train",
        "val": "2021_valid",
        "valid": "2021_valid",
        "validation": "2021_valid",
    }
    try:
        return versions[normalized]
    except KeyError as exc:
        raise ValueError(
            "iNaturalist split must be 'train' or 'val', or set inaturalist_version explicitly."
        ) from exc


def _resolve_root(cfg: DataConfig, name: str, mode: str) -> str:
    """Support either a torchvision root or the project's shared dataset directory."""

    root = Path(cfg.data_path).expanduser()
    if name != "STL10":
        return str(root)
    partition = "train" if mode == "ssl" or cfg.split.strip().lower() == "train" else "val"
    candidates = (
        root,
        root / "stl10" / partition,
        root / "STL10" / partition,
        root / "stl10",
        root / "STL10",
    )
    for candidate in candidates:
        if (candidate / "stl10_binary").is_dir():
            return str(candidate)
    return str(root)


def _effective_split(cfg: DataConfig, name: str, mode: str) -> str:
    if mode == "ssl":
        return "unlabeled"
    if name == "CIFAR100":
        return "train" if _is_train_split(cfg.split) else "test"
    if name == "Flowers102":
        return _flowers_split(cfg.split)
    if name == "iNaturalist":
        return _inaturalist_version(cfg)
    return cfg.split


def _dataset_targets(dataset: Dataset[Any]) -> Optional[Sequence[int]]:
    for name in ("targets", "labels", "_labels"):
        targets = getattr(dataset, name, None)
        if targets is not None:
            return targets
    return None


def _apply_subset_options(dataset: Dataset[Any], cfg: DataConfig) -> Dataset[Any]:
    indices = list(range(len(dataset)))
    if cfg.classes is not None:
        allowed = set(cfg.classes)
        targets = _dataset_targets(dataset)
        if targets is None:
            indices = [idx for idx in indices if dataset[idx][1] in allowed]
        else:
            indices = [idx for idx in indices if int(targets[idx]) in allowed]
    if cfg.max_samples is not None and len(indices) > cfg.max_samples:
        random.Random(cfg.seed).shuffle(indices)
        indices = indices[: cfg.max_samples]
    if cfg.classes is not None or cfg.max_samples is not None:
        return Subset(dataset, indices)
    return dataset


def prepare_dataset(cfg: DataConfig) -> Dataset[Any]:
    """Create exactly one dataset selected by ``cfg.dataset`` and ``cfg.mode``."""

    name, mode = validate_config(cfg)
    transform = build_transform(cfg, mode)
    root = _resolve_root(cfg, name, mode)
    if name == "STL10":
        split = "unlabeled" if mode == "ssl" else cfg.split
        dataset = datasets.STL10(root=root, split=split, download=cfg.download, transform=transform)
    elif name == "CIFAR100":
        dataset = datasets.CIFAR100(
            root=root, train=_is_train_split(cfg.split), download=cfg.download, transform=transform
        )
    elif name == "Flowers102":
        dataset = datasets.Flowers102(
            root=root, split=_flowers_split(cfg.split), download=cfg.download, transform=transform
        )
    else:
        dataset = datasets.INaturalist(
            root=root,
            version=_inaturalist_version(cfg),
            target_type="full",
            download=cfg.download,
            transform=transform,
        )
    return _apply_subset_options(dataset, cfg)


def _distributed_sampler(dataset: Dataset[Any], cfg: DataConfig) -> Optional[DistributedSampler]:
    if not cfg.distributed:
        return None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return DistributedSampler(dataset, shuffle=cfg.shuffle, seed=cfg.seed)
    if cfg.rank is None or cfg.world_size is None:
        raise ValueError(
            "distributed=True requires initialized torch.distributed or both rank and world_size."
        )
    if not 0 <= cfg.rank < cfg.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size.")
    return DistributedSampler(
        dataset,
        num_replicas=cfg.world_size,
        rank=cfg.rank,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _tensor_summary(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
    )


def _batch_summary(batch: Any, mode: str) -> str:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    if mode == "ssl" and isinstance(images, (tuple, list)):
        views = "; ".join(f"view_{idx}: {_tensor_summary(view)}" for idx, view in enumerate(images))
        return views
    return _tensor_summary(images)


def write_load_report(loader: DataLoader[Any], cfg: DataConfig, name: str, mode: str) -> Path:
    """Persist successful dataset creation and one representative batch result."""

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_split = _effective_split(cfg, name, mode)
    report_path = output_dir / f"{name.lower()}_{mode}_{effective_split}.txt"
    batch = next(iter(loader), None)
    batch_text = "empty dataset: no batch available" if batch is None else _batch_summary(batch, mode)
    sampler_name = type(loader.sampler).__name__
    report_path.write_text(
        "\n".join(
            (
                f"dataset: {name}",
                f"training_type: {mode}",
                f"split: {effective_split}",
                f"data_path: {Path(cfg.data_path).expanduser()}",
                f"resolved_data_path: {_resolve_root(cfg, name, mode)}",
                f"samples: {len(loader.dataset)}",
                f"batch_size: {cfg.batch_size}",
                f"sampler: {sampler_name}",
                f"batch_sample: {batch_text}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return report_path


def prepare_dataloader(cfg: DataConfig) -> DataLoader[Any]:
    """Return an independent config-created dataloader and optionally log its load result."""

    name, mode = validate_config(cfg)
    set_seed(cfg.seed)
    dataset = prepare_dataset(cfg)
    sampler = _distributed_sampler(dataset, cfg)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        drop_last=cfg.drop_last,
        persistent_workers=cfg.persistent_workers,
        worker_init_fn=_worker_seed if cfg.num_workers else None,
        generator=generator,
    )
    if cfg.write_report:
        write_load_report(loader, cfg, name, mode)
    return loader


def inspect_loader(loader: DataLoader[Any], mode: str = "classification") -> str:
    """Return a compact batch summary useful in notebooks and training smoke tests."""

    batch = next(iter(loader), None)
    return "empty dataset: no batch available" if batch is None else _batch_summary(batch, mode)


__all__ = [
    "DataConfig",
    "MultiViewTransform",
    "SUPPORTED_DATASETS",
    "build_transform",
    "inspect_loader",
    "prepare_dataloader",
    "prepare_dataset",
    "set_seed",
    "validate_config",
    "write_load_report",
]
