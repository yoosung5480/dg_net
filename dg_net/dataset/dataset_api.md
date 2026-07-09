# Dataset API

> Canonical dataset package reference. All commands in this document assume the current working directory is `dg_net/`.

## 1. Package role

`dataset/` builds SSL and classification dataloaders from one `DataConfig` object. It owns:

- dataset selection and split normalization
- transform construction
- class filtering / max-sample subsetting
- optional `DistributedSampler`
- load-success reports in `output_dir`

## 2. Public API

```python
from dataset import (
    DataConfig,
    MultiViewTransform,
    SUPPORTED_DATASETS,
    build_transform,
    inspect_loader,
    prepare_dataloader,
    prepare_dataset,
    write_load_report,
)
```

## 3. Supported datasets and return types

| dataset | `mode="ssl"` | `mode="classification"` | effective split behavior |
| --- | --- | --- | --- |
| `STL10` | yes | yes | SSL always uses `unlabeled`; classification uses requested `train`/`test` |
| `ImageNet` / `imagefolder` / `imagenet` | yes | yes | prefers `data_path/train` or `data_path/val`, falls back to `data_path` |
| `Synthetic` | yes | yes | `torchvision.datasets.FakeData` smoke dataset |
| `CIFAR100` | no | yes | `train` or `test` |
| `Flowers102` | no | yes | `train`, `val`/`validation`, `test` |
| `iNaturalist` / `inat` | no | yes | `train -> 2021_train`, `val/validation -> 2021_valid` unless overridden |

Return format:

- SSL: `views, labels`, where `views` is a tuple of `ssl_num_views` tensors
- classification: `images, labels`

## 4. Shortest runnable example

```python
from dataset import DataConfig, prepare_dataloader

cfg = DataConfig(
    dataset="Synthetic",
    mode="ssl",
    img_size=16,
    batch_size=2,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    use_augmentation=False,
    max_samples=4,
    write_report=False,
)
loader = prepare_dataloader(cfg)
views, labels = next(iter(loader))
print(len(views), views[0].shape, labels.shape)
```

## 5. `DataConfig` fields

### Dataset selection

| field | default | notes |
| --- | --- | --- |
| `dataset` | `"STL10"` | canonical names plus aliases `inat`, `imagefolder`, `imagenet` |
| `data_path` | `"/home/jeongyuseong/바탕화면/datasets"` | dataset root or ImageFolder root |
| `mode` | `"ssl"` | `ssl` or `classification` |
| `split` | `"train"` | normalized per dataset |
| `download` | `False` | forwarded to torchvision where supported |
| `classes` | `None` | optional class-id filter |
| `max_samples` | `None` | reproducible subset cap |
| `inaturalist_version` | `None` | overrides split-derived torchvision version |

### Transform fields

| field | default | notes |
| --- | --- | --- |
| `img_size` | `224` | resize / crop target |
| `use_augmentation` | `True` | `False` uses deterministic resize |
| `crop_scale` | `(0.08, 1.0)` | `RandomResizedCrop` scale |
| `hflip_prob` | `0.5` | horizontal flip probability |
| `color_jitter` | `None` | optional torchvision `ColorJitter` tuple |
| `gaussian_blur` | `False` | optional blur branch |
| `blur_prob` | `0.5` | blur apply probability |
| `mean` | `(0.485, 0.456, 0.406)` | normalization mean |
| `std` | `(0.229, 0.224, 0.225)` | normalization std |
| `ssl_num_views` | `2` | must be `>= 2` in SSL mode |

### Loader / runtime fields

| field | default | notes |
| --- | --- | --- |
| `batch_size` | `64` | loader batch size |
| `num_workers` | `4` | worker count |
| `shuffle` | `True` | disabled automatically when sampler is present |
| `drop_last` | `True` | training batches only in engine val config |
| `pin_memory` | `True` | actually enabled only when CUDA is available |
| `persistent_workers` | `False` | requires `num_workers > 0` |
| `seed` | `42` | subset selection and loader RNG |
| `distributed` | `False` | enables distributed sampler |
| `rank`, `world_size` | `None` | required when distributed is requested outside an initialized process group |
| `output_dir` | `dataset/output` | report destination |
| `write_report` | `True` | writes a one-batch load report |

## 6. ImageNet root and split resolution

`ImageNet` uses `torchvision.datasets.ImageFolder`, not a custom parser.

- SSL uses `data_path/train` when it exists, otherwise `data_path`
- classification `split="train"` prefers `data_path/train`
- classification `split="val"`, `"validation"`, or `"test"` prefers `data_path/val`
- if the split directory is missing, the loader falls back to `data_path`

Expected folder layout:

```text
data_path/
├── train/
│   ├── class0/
│   └── class1/
└── val/
    ├── class0/
    └── class1/
```

## 7. SSL vs classification examples

```python
from dataset import DataConfig, inspect_loader, prepare_dataloader

ssl_cfg = DataConfig(
    dataset="Synthetic",
    mode="ssl",
    img_size=16,
    batch_size=2,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    use_augmentation=False,
    max_samples=4,
    write_report=False,
)
ssl_loader = prepare_dataloader(ssl_cfg)
print(inspect_loader(ssl_loader, mode="ssl"))

cls_cfg = DataConfig(
    dataset="Synthetic",
    mode="classification",
    img_size=16,
    batch_size=2,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    use_augmentation=False,
    max_samples=4,
    write_report=False,
)
cls_loader = prepare_dataloader(cls_cfg)
images, labels = next(iter(cls_loader))
print(images.shape, labels.shape)
```

## 8. Distributed sampler and `set_epoch()`

```python
from dataset import DataConfig, prepare_dataloader

cfg = DataConfig(
    dataset="Synthetic",
    mode="classification",
    img_size=16,
    batch_size=2,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    use_augmentation=False,
    max_samples=8,
    distributed=True,
    rank=0,
    world_size=2,
    write_report=False,
)
loader = prepare_dataloader(cfg)
print(type(loader.sampler).__name__)

for epoch in range(2):
    loader.sampler.set_epoch(epoch)
```

Rules:

- `prepare_dataloader()` creates a `DistributedSampler` when `distributed=True`
- if `torch.distributed` is already initialized, rank/world size are read from the process group
- otherwise you must provide both `rank` and `world_size`
- the engine calls `sampler.set_epoch(epoch)` before each distributed epoch to reshuffle consistently

## 9. Mean/std and `CLAMP_DEGRADED`

The model package can clamp `degraded` images to `[0, 1]`, but that only preserves pixel semantics when the dataset normalization is identity:

```text
mean = (0.0, 0.0, 0.0)
std  = (1.0, 1.0, 1.0)
```

`TrainConfig.__post_init__()` enforces this: `clamp_degraded=True` with non-identity normalization raises `ValueError`.

## 10. Smoke test and output artifacts

```python
from pathlib import Path
from dataset import DataConfig, prepare_dataloader

output_dir = Path("dataset/output/api_reports")
loader = prepare_dataloader(DataConfig(
    dataset="Synthetic",
    mode="classification",
    img_size=16,
    batch_size=2,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    use_augmentation=False,
    max_samples=4,
    output_dir=str(output_dir),
    write_report=True,
))
print(next(output_dir.glob("*.txt")))
```

Report contents include:

- resolved dataset name
- training type (`ssl` or `classification`)
- effective split
- original `data_path`
- resolved root path
- sample count
- sampler type
- first-batch tensor summary

## 11. Train script connection

- `engine/train_config.py` builds train/validation loaders through `TrainConfig.build_data_config(...)`
- DG pretraining uses `mode="ssl"`
- `linear_probe`, `finetune`, and `knn_eval` use `mode="classification"`
- the pipeline scripts pass identity normalization (`--mean 0,0,0 --std 1,1,1`) because the model path uses unclamped pixel-space degradation by default

## 12. DDP cautions

- `shuffle` is ignored once a distributed sampler is attached; sampler order owns the shuffle
- rank 0 is the only writer of dataset reports in the engine path (`write_report=(not distributed or rank == 0)`)
- SSL with InfoNCE-style training still depends on `ssl_num_views >= 2`; the dataloader enforces that in `MultiViewTransform`
