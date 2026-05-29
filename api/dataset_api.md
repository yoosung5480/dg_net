# Dataset API 가이드

`dataset/` 패키지는 **`DataConfig` 하나로 데이터셋 선택, transform, subset,
`DataLoader`, DDP sampler, 로드 보고서 저장까지 결정**하는 독립 패키지이다.
SSL 사전학습 입력은 STL10 unlabeled split이며, 분류 전이 입력은 STL10,
CIFAR100, Flowers102, iNaturalist를 지원한다.

이 문서의 코드는 다음 위치에서 그대로 실행하는 것을 기준으로 한다.

```bash
cd /home/jeongyuseong/바탕화면/SSL/DGnet_proj/version1/src
```

## 1. 공개 API와 가장 짧은 실행 예제

```python
from dataset import DataConfig, inspect_loader, prepare_dataloader

cfg = DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    img_size=32,
    batch_size=4,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
    max_samples=16,       # 빠른 smoke test용; 실제 학습에서는 제거
    use_augmentation=False,
)

loader = prepare_dataloader(cfg)
print(inspect_loader(loader, mode=cfg.mode))

views, labels = next(iter(loader))
x_view1, x_view2 = views
print(x_view1.shape, x_view2.shape, labels.shape)
# torch.Size([4, 3, 32, 32]) torch.Size([4, 3, 32, 32]) torch.Size([4])
```

`prepare_dataloader(cfg)`가 반환하는 값은 표준
`torch.utils.data.DataLoader`이다. `write_report=True`인 기본 설정에서는
첫 배치를 한 번 읽어 성공 보고서도 `cfg.output_dir`에 기록한다.

### 공개 심볼

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

| API | 역할 |
| --- | --- |
| `DataConfig` | 데이터 입력과 로더 생성을 결정하는 유일한 config |
| `prepare_dataloader(cfg)` | 학습/평가용 `DataLoader` 생성, 선택적으로 보고서 저장 |
| `prepare_dataset(cfg)` | transform과 subset이 적용된 단일 dataset만 생성 |
| `build_transform(cfg, mode=None)` | classification single-view 또는 SSL multi-view transform 생성 |
| `inspect_loader(loader, mode)` | 첫 batch tensor shape/min/max 문자열 반환 |
| `write_load_report(loader, cfg, name, mode)` | 로드 성공 보고서 파일 저장 |
| `SUPPORTED_DATASETS` | `("STL10", "CIFAR100", "Flowers102", "iNaturalist")` |

## 2. 지원 데이터셋, mode, 실제 반환 형식

| `dataset` | 허용 `mode` | `split` 입력 | torchvision에 전달되는 split/version |
| --- | --- | --- | --- |
| `"STL10"` | `"ssl"` | 값은 무시됨 | 항상 `split="unlabeled"` |
| `"STL10"` | `"classification"` | `"train"` / `"test"` | 입력 split 그대로 |
| `"CIFAR100"` | `"classification"` | `"train"` / `"test"` | `train=True` / `False` |
| `"Flowers102"` | `"classification"` | `"train"` / `"val"` / `"validation"` / `"test"` | `"validation"`은 `"val"`로 정규화 |
| `"iNaturalist"` | `"classification"` | `"train"` / `"val"` / `"validation"` | 기본 `version="2021_train"` / `"2021_valid"` |

`mode="ssl"`은 현재 **STL10에만 허용**된다. `CIFAR100` 등으로 SSL
loader를 만들면 `ValueError`가 발생한다.

### 2.1 SSL 반환 형식

`STL10 + ssl`에서는 원본 하나에 동일 augmentation pipeline을 독립적으로
`ssl_num_views`회 적용한다. DataLoader의 한 batch는 다음 형태이다.

```python
views, labels = next(iter(loader))
assert len(views) == cfg.ssl_num_views
assert views[0].shape == (cfg.batch_size, 3, cfg.img_size, cfg.img_size)
# views[i]: Tensor[B, C, H, W], labels: Tensor[B]
```

STL10 unlabeled의 label 값은 학습 target으로 사용하지 않는다. DGNet 또는
contrastive 학습에는 보통 `images = views[0]` 또는 서로 다른 두 view를
사용한다.

### 2.2 Classification 반환 형식

```python
images, labels = next(iter(loader))
assert images.shape == (cfg.batch_size, 3, cfg.img_size, cfg.img_size)
# images: Tensor[B, C, H, W], labels: Tensor[B]
```

## 3. 저장되어 있어야 하는 데이터 위치와 형태

`data_path`는 개별 이미지 파일 경로가 아니라, **torchvision dataset이 읽을
root 디렉터리**이다. 직접 폴더별 이미지 dataset을 스캔하는 구현이 아니므로,
임의의 `class_name/*.jpg` 구조를 주면 동작하지 않는다. 이미 다운로드한
dataset은 torchvision이 기대하는 공식 파일/메타데이터 형태로 존재해야 한다.
파일이 없고 네트워크 다운로드가 허용되는 환경이라면 `download=True`로
torchvision이 해당 구조를 만들게 할 수 있다.

### 3.1 STL10: 프로젝트 경로 탐색 규칙

STL10만 프로젝트의 공유 dataset 디렉터리 관례를 추가 지원한다.
`data_path=ROOT`라면 패키지는 `stl10_binary/`가 존재하는 첫 경로를 사용한다.

| 경우 | 순서대로 검사하는 경로 |
| --- | --- |
| `mode="ssl"` 또는 `split="train"` | `ROOT`, `ROOT/stl10/train`, `ROOT/STL10/train`, `ROOT/stl10`, `ROOT/STL10` |
| classification의 train 이외 split | `ROOT`, `ROOT/stl10/val`, `ROOT/STL10/val`, `ROOT/stl10`, `ROOT/STL10` |

가장 단순한 권장 형태는 한 torchvision STL10 root에 공식 binary 파일들을
모두 두는 것이다.

```text
/home/jeongyuseong/바탕화면/datasets/
└── stl10/
    └── stl10_binary/
        ├── unlabeled_X.bin
        ├── train_X.bin
        ├── train_y.bin
        ├── test_X.bin
        ├── test_y.bin
        └── ... 공식 STL10 binary/metadata 파일
```

```python
cfg = DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets/stl10",
    download=False,
)
```

기존 프로젝트처럼 train/validation 저장 위치를 나누어 관리할 수도 있다.
단, 최종 선택되는 `stl10_binary/` 폴더에는 요청한 split에 필요한 공식
파일이 들어 있어야 한다.

```text
/home/jeongyuseong/바탕화면/datasets/
└── stl10/
    ├── train/stl10_binary/   # SSL unlabeled 또는 classification train을 읽을 root
    └── val/stl10_binary/     # classification test를 읽을 root로 사용할 때
```

### 3.2 CIFAR100, Flowers102, iNaturalist

이 세 dataset은 `data_path`를 수정 없이 torchvision constructor의 `root`로
넘긴다. 따라서 저장 형태는 설치된 torchvision의 해당 dataset 규약을 따른다.

```python
cfg = DataConfig(
    dataset="Flowers102",
    mode="classification",
    split="validation",       # 내부에서 "val"로 변환
    data_path="/home/jeongyuseong/바탕화면/datasets",
    download=False,           # 이미 torchvision 형식으로 준비되어 있을 때
)
loader = prepare_dataloader(cfg)
```

| dataset | 실제 호출 개념 |
| --- | --- |
| CIFAR100 | `torchvision.datasets.CIFAR100(root=data_path, train=..., download=...)` |
| Flowers102 | `torchvision.datasets.Flowers102(root=data_path, split=..., download=...)` |
| iNaturalist | `torchvision.datasets.INaturalist(root=data_path, version=..., target_type="full", download=...)` |

## 4. `DataConfig` 하이퍼파라미터

### 4.1 Dataset 선택과 subset

| 필드 | 기본값 | 설명 |
| --- | --- | --- |
| `dataset` | `"STL10"` | 지원 dataset 이름. 대소문자는 정규화되며 iNaturalist는 `"inat"` 별칭도 지원한다. |
| `data_path` | `"/home/jeongyuseong/바탕화면/datasets"` | torchvision root 또는 위 STL10 탐색 규칙의 기준 root |
| `mode` | `"ssl"` | `"ssl"` 또는 `"classification"` |
| `split` | `"train"` | classification split 선택. STL10 SSL에서는 무시되고 unlabeled를 사용한다. |
| `download` | `False` | 없는 dataset을 torchvision이 다운로드하도록 할지 여부 |
| `classes` | `None` | 정수 class id 목록만 유지한다. `None`이면 전체 사용한다. |
| `max_samples` | `None` | subset 최대 개수. 설정하면 `seed`로 섞은 고정 subset을 선택한다. |
| `inaturalist_version` | `None` | 지정하면 `split`에서 유도되는 iNaturalist version 대신 직접 사용한다. |

`classes`는 label이 있는 분류 실험에 사용하는 것이 적합하다. STL10
unlabeled 사전학습에서는 보통 설정하지 않는다.

### 4.2 Transform / augmentation

| 필드 | 기본값 | DataLoader 입력에 주는 영향 |
| --- | --- | --- |
| `img_size` | `224` | 모든 반환 이미지의 공간 크기를 `[img_size, img_size]`로 만든다. |
| `use_augmentation` | `True` | `True`: random crop/선택 증강, `False`: deterministic resize |
| `crop_scale` | `(0.08, 1.0)` | `RandomResizedCrop` 면적 범위; augmentation 활성화 시 사용 |
| `hflip_prob` | `0.5` | horizontal flip 확률; `0`이면 flip transform을 추가하지 않는다. |
| `color_jitter` | `None` | `(brightness, contrast, saturation, hue)`; 설정 시 augmentation pipeline에 추가 |
| `gaussian_blur` | `False` | Gaussian blur 사용 여부 |
| `blur_prob` | `0.5` | blur가 선택될 확률 |
| `mean` | `(0.485, 0.456, 0.406)` | `ToTensor()` 뒤 channel normalization 평균 |
| `std` | `(0.229, 0.224, 0.225)` | channel normalization 표준편차 |
| `ssl_num_views` | `2` | SSL에서 한 이미지로부터 반환할 view 수; 반드시 `>= 2` |

실제 pipeline은 다음과 같다.

```text
use_augmentation=True:
  RandomResizedCrop -> (HorizontalFlip) -> (ColorJitter) -> (GaussianBlur)
  -> ToTensor -> Normalize

use_augmentation=False:
  Resize((img_size, img_size)) -> ToTensor -> Normalize
```

> `mean/std` 기본값을 사용하면 모델 입력은 ImageNet-normalized tensor이다.
> `[0, 1]` 픽셀 시각화나 `DGNet(CLAMP_DEGRADED=True)`의 픽셀 범위 의미가
> 중요할 때는 `mean=(0, 0, 0), std=(1, 1, 1)`을 설정한다.

### 4.3 DataLoader / 재현성 / DDP

| 필드 | 기본값 | 설명 |
| --- | ---: | --- |
| `batch_size` | `64` | 반환 batch 크기; 양수여야 한다. |
| `num_workers` | `4` | DataLoader worker 수; notebook/smoke test는 `0`이 간단하다. |
| `shuffle` | `True` | 일반 loader shuffle 또는 DDP sampler shuffle 여부 |
| `drop_last` | `True` | 마지막 작은 batch 제거. SSL 학습에는 보통 `True`, 평가에는 `False`. |
| `pin_memory` | `True` | CUDA 사용 가능할 때만 실제 loader에 활성화된다. |
| `persistent_workers` | `False` | epoch 간 worker 유지. `True`이면 `num_workers > 0`이어야 한다. |
| `seed` | `42` | subset 선택, DataLoader generator, worker random seed의 기준값 |
| `distributed` | `False` | `True`이면 `DistributedSampler`를 사용하고 DataLoader 자체 shuffle은 끈다. |
| `rank` | `None` | process group 미초기화 상태에서 DDP sampler를 만들 때 필요한 rank |
| `world_size` | `None` | process group 미초기화 상태에서 필요한 전체 process 수 |

### 4.4 출력 보고서

| 필드 | 기본값 | 설명 |
| --- | --- | --- |
| `output_dir` | `dataset/output` 절대 경로 | loader 성공 보고서를 저장할 디렉터리 |
| `write_report` | `True` | `prepare_dataloader()` 중 첫 batch를 읽고 `.txt` 보고서를 쓸지 여부 |

보고서 이름은 다음 규칙을 따른다.

```text
{dataset.lower()}_{mode}_{effective_split}.txt
```

예:

```text
dataset/output/stl10_ssl_unlabeled.txt
dataset/output/cifar100_classification_test.txt
dataset/output/flowers102_classification_val.txt
dataset/output/inaturalist_classification_2021_valid.txt
```

내용에는 dataset 이름, mode, effective split, 입력/해석된 root, sample 수,
batch size, sampler 종류와 첫 batch shape/min/max가 기록된다.

## 5. 바로 사용할 수 있는 예제

### 5.1 STL10 SSL 학습 입력

```python
from dataset import DataConfig, prepare_dataloader

cfg = DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    img_size=224,
    ssl_num_views=2,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    use_augmentation=True,
    color_jitter=(0.4, 0.4, 0.4, 0.1),
    gaussian_blur=True,
)
loader = prepare_dataloader(cfg)

for views, _ in loader:
    view_a, view_b = views
    # SSL/DGNet training step(view_a, view_b)
    break
```

### 5.2 분류 전이 평가 loader

```python
from dataset import DataConfig, prepare_dataloader

cfg = DataConfig(
    dataset="CIFAR100",
    mode="classification",
    split="test",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    img_size=224,
    use_augmentation=False,
    batch_size=128,
    num_workers=4,
    shuffle=False,
    drop_last=False,
)
loader = prepare_dataloader(cfg)

images, labels = next(iter(loader))
print(images.shape, labels.shape)
```

### 5.3 빠른 subset smoke test

```python
from dataset import DataConfig, inspect_loader, prepare_dataloader

cfg = DataConfig(
    dataset="Flowers102",
    mode="classification",
    split="train",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    classes=[0, 1, 2],
    max_samples=32,
    seed=123,
    img_size=32,
    use_augmentation=False,
    batch_size=8,
    num_workers=0,
    drop_last=False,
    pin_memory=False,
)
loader = prepare_dataloader(cfg)
print(inspect_loader(loader))
```

### 5.4 DDP sampler 구성

torch distributed process group를 launcher가 초기화한 뒤라면 `rank`와
`world_size`를 생략할 수 있다. 초기화 전 smoke test에서는 둘 다 준다.

```python
from dataset import DataConfig, prepare_dataloader

cfg = DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    distributed=True,
    rank=0,
    world_size=2,
    batch_size=32,
    num_workers=0,
    write_report=False,
)
loader = prepare_dataloader(cfg)

# 실제 epoch loop에서는 DistributedSampler에 epoch를 알려 shuffle을 변경한다.
if hasattr(loader.sampler, "set_epoch"):
    loader.sampler.set_epoch(0)
```

## 6. 실전 연결 체크리스트

1. `img_size`는 사용할 모델의 입력 크기(`DgNetConfig.IMG_SIZE`)와 같게 둔다.
2. `mode="ssl"`이면 `views` sequence에서 학습에 사용할 view를 명시적으로 고른다.
3. 모델이 `[0, 1]` 입력을 전제로 clamp 또는 이미지 저장을 한다면 dataset의
   `mean/std`도 그 계약에 맞춘다.
4. 성능 실험 전에는 `max_samples=None`, `write_report` 사용 여부,
   `drop_last`, DDP sampler 설정을 다시 확인한다.
5. loader 생성 확인은 `inspect_loader()`와 `output_dir/*.txt` 보고서로 먼저
   수행한 뒤 긴 학습을 시작한다.
