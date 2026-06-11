import importlib
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

dataset_module = importlib.import_module("dataset.dataset")
from dataset import DataConfig


class FakeVisionDataset(torch.utils.data.Dataset):
    last_kwargs = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        self.transform = kwargs["transform"]
        self.targets = [0, 1, 2, 1, 0, 2]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.new("RGB", (12, 12), color=(index * 15, 30, 60))
        return self.transform(image), self.targets[index]


@pytest.fixture(autouse=True)
def mocked_torchvision_datasets(monkeypatch):
    for name in ("STL10", "CIFAR100", "Flowers102", "INaturalist"):
        fake_class = type(f"Fake{name}", (FakeVisionDataset,), {"last_kwargs": {}})
        monkeypatch.setattr(dataset_module.datasets, name, fake_class)


def make_config(tmp_path: Path, **overrides) -> DataConfig:
    values = dict(
        data_path=str(tmp_path / "source"),
        output_dir=str(tmp_path / "output"),
        img_size=16,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        use_augmentation=False,
        download=False,
    )
    values.update(overrides)
    return DataConfig(**values)


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (18, 18), color=color).save(path)


def build_imagenet_tree(root: Path) -> None:
    write_image(root / "train" / "class0" / "a.png", (255, 0, 0))
    write_image(root / "train" / "class0" / "b.png", (200, 0, 0))
    write_image(root / "train" / "class1" / "c.png", (0, 255, 0))
    write_image(root / "train" / "class1" / "d.png", (0, 200, 0))
    write_image(root / "val" / "class0" / "e.png", (0, 0, 255))
    write_image(root / "val" / "class1" / "f.png", (255, 255, 0))


def build_flat_imagefolder(root: Path) -> None:
    write_image(root / "class0" / "a.png", (255, 0, 0))
    write_image(root / "class1" / "b.png", (0, 255, 0))


def test_ssl_stl10_uses_unlabeled_multiview_and_writes_report(tmp_path):
    (tmp_path / "source" / "stl10" / "train" / "stl10_binary").mkdir(parents=True)
    cfg = make_config(tmp_path, dataset="STL10", mode="ssl", split="train", ssl_num_views=2)

    loader = dataset_module.prepare_dataloader(cfg)
    views, labels = next(iter(loader))

    assert dataset_module.datasets.STL10.last_kwargs["split"] == "unlabeled"
    assert dataset_module.datasets.STL10.last_kwargs["root"] == str(
        tmp_path / "source" / "stl10" / "train"
    )
    assert len(views) == 2
    assert views[0].shape == (2, 3, 16, 16)
    assert labels.shape == (2,)
    report = (tmp_path / "output" / "stl10_ssl_unlabeled.txt").read_text(encoding="utf-8")
    assert "training_type: ssl" in report
    assert "samples: 6" in report
    assert f"resolved_data_path: {tmp_path / 'source' / 'stl10' / 'train'}" in report
    assert "view_0: shape=(2, 3, 16, 16)" in report


@pytest.mark.parametrize(
    ("dataset_name", "split", "constructor_name", "expected_argument", "report_name"),
    [
        ("STL10", "test", "STL10", ("split", "test"), "stl10_classification_test.txt"),
        ("CIFAR100", "test", "CIFAR100", ("train", False), "cifar100_classification_test.txt"),
        ("Flowers102", "validation", "Flowers102", ("split", "val"), "flowers102_classification_val.txt"),
        (
            "iNaturalist",
            "val",
            "INaturalist",
            ("version", "2021_valid"),
            "inaturalist_classification_2021_valid.txt",
        ),
    ],
)
def test_classification_dataset_dispatch(
    tmp_path, dataset_name, split, constructor_name, expected_argument, report_name
):
    cfg = make_config(tmp_path, dataset=dataset_name, mode="classification", split=split)

    loader = dataset_module.prepare_dataloader(cfg)
    images, labels = next(iter(loader))

    key, value = expected_argument
    assert getattr(dataset_module.datasets, constructor_name).last_kwargs[key] == value
    assert images.shape == (2, 3, 16, 16)
    assert labels.tolist() == [0, 1]
    assert (tmp_path / "output" / report_name).is_file()


@pytest.mark.parametrize(
    ("split", "expected_dir", "expected_labels"),
    [("train", "train", [0, 0]), ("validation", "val", [0, 1])],
)
def test_imagenet_classification_uses_split_root_when_present(
    tmp_path, split, expected_dir, expected_labels
):
    build_imagenet_tree(tmp_path / "source")
    cfg = make_config(tmp_path, dataset="ImageNet", mode="classification", split=split)

    loader = dataset_module.prepare_dataloader(cfg)
    images, labels = next(iter(loader))

    assert Path(loader.dataset.root) == tmp_path / "source" / expected_dir
    assert images.shape == (2, 3, 16, 16)
    assert labels.tolist() == expected_labels
    report = (tmp_path / "output" / f"imagenet_classification_{expected_dir}.txt").read_text(
        encoding="utf-8"
    )
    assert f"resolved_data_path: {tmp_path / 'source' / expected_dir}" in report


def test_imagenet_falls_back_to_data_path_when_split_root_missing(tmp_path):
    build_flat_imagefolder(tmp_path / "source")
    cfg = make_config(tmp_path, dataset="imagenet", mode="classification", split="train", write_report=False)

    dataset = dataset_module.prepare_dataset(cfg)

    assert dataset.root == str(tmp_path / "source")
    assert len(dataset) == 2


def test_imagenet_ssl_returns_multiview_batch_and_supports_imagefolder_alias(tmp_path):
    build_imagenet_tree(tmp_path / "source")
    cfg = make_config(tmp_path, dataset="imagefolder", mode="ssl", ssl_num_views=2)

    loader = dataset_module.prepare_dataloader(cfg)
    views, labels = next(iter(loader))

    assert "ImageNet" in dataset_module.SUPPORTED_DATASETS
    assert Path(loader.dataset.root) == tmp_path / "source" / "train"
    assert len(views) == 2
    assert views[0].shape == (2, 3, 16, 16)
    assert labels.tolist() == [0, 0]
    report = (tmp_path / "output" / "imagenet_ssl_train.txt").read_text(encoding="utf-8")
    assert "dataset: ImageNet" in report
    assert f"resolved_data_path: {tmp_path / 'source' / 'train'}" in report


def test_imagenet_class_filter_and_max_samples_are_config_controlled_and_repeatable(tmp_path):
    build_imagenet_tree(tmp_path / "source")
    first_cfg = make_config(
        tmp_path,
        dataset="ImageNet",
        mode="classification",
        classes=[1],
        max_samples=1,
        seed=123,
        write_report=False,
    )
    second_cfg = make_config(
        tmp_path,
        dataset="ImageNet",
        mode="classification",
        classes=[1],
        max_samples=1,
        seed=123,
        write_report=False,
    )

    first = dataset_module.prepare_dataloader(first_cfg).dataset
    second = dataset_module.prepare_dataloader(second_cfg).dataset

    assert isinstance(first, Subset)
    assert len(first) == 1
    assert first.indices == second.indices
    assert first.dataset.targets[first.indices[0]] == 1


def test_imagenet_distributed_config_uses_ddp_sampler_and_disables_random_sampler(tmp_path):
    build_imagenet_tree(tmp_path / "source")
    cfg = make_config(
        tmp_path,
        dataset="ImageNet",
        mode="classification",
        distributed=True,
        rank=1,
        world_size=2,
        shuffle=True,
        write_report=False,
    )

    loader = dataset_module.prepare_dataloader(cfg)

    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.rank == 1
    assert loader.sampler.num_replicas == 2


def test_ssl_rejects_labeled_transfer_datasets(tmp_path):
    cfg = make_config(tmp_path, dataset="CIFAR100", mode="ssl", write_report=False)

    with pytest.raises(ValueError, match="SSL mode currently supports STL10 and ImageNet only"):
        dataset_module.prepare_dataloader(cfg)
