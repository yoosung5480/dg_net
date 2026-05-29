# DGNet Model API 가이드

`model/dg_model.py`는 config만으로 DGNet을 조립하고, forward 결과·가중치
저장/로드·구조 보고서·샘플 이미지 저장 기능을 제공한다. 이 문서에서 말하는
공개 모델은 `DGNet`/`DgNetConfig` 기반 구현이며, 별도 외부 `src.config`에
의존하는 초기 prototype인 `model_masking.py`가 아니다.

이 문서의 코드는 다음 위치에서 그대로 실행하는 것을 기준으로 한다.

```bash
cd /home/jeongyuseong/바탕화면/SSL/DGnet_proj/version1/src
```

## 1. 가장 짧은 생성과 forward

```python
import torch

from model.dg_model import DGNet, DgNetConfig

cfg = DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=48,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=32,
    DG_ARCHITECT="HYBRID",
)
model = DGNet(cfg).eval()

images = torch.rand(4, 3, 32, 32)
with torch.no_grad():
    out = model(images)

print(out.degraded.shape)                # torch.Size([4, 3, 32, 32])
print(out.reconstruction.shape)          # torch.Size([4, 3, 32, 32])
print(out.representation.shape)          # torch.Size([4, 32])
print(out.degraded_representation.shape) # torch.Size([4, 32])
```

입력 계약은 `Tensor[B, IN_CHANS, IMG_SIZE, IMG_SIZE]`이다. 크기나 channel이
config와 맞지 않으면 forward 중 `ValueError`가 발생한다.

## 2. DGNet이 config로 조립되는 방식

DGNet은 두 역할을 결합한다.

```text
입력 x
 ├─ M: DegradationGenerator(config로 VIT/CNN/HYBRID 선택)
 │    residual = M(x)
 │    degraded = x - residual
 │
 └─ I: 고정 구조 ViT inference network
      encoder(original/degraded) -> projection representations
      decoder(encoder(degraded))  -> reconstruction
      decoder(encoder(original))  -> original_reconstruction
```

| 구성 요소 | 구현/선택 방식 | 출력 |
| --- | --- | --- |
| `degradation` (`M`) | `DG_ARCHITECT`: `"VIT"`, `"CNN"`, `"HYBRID"` 중 선택 | 이미지 크기의 residual `M(x)` |
| `encoder` (`I`) | 항상 ViT patch encoder | patch token representation |
| `decoder` (`I`) | 항상 ViT patch decoder | 복원 이미지 |
| `projection` | encoder token 평균을 MLP projection | `[B, PROJECTION_DIM]` representation |

`DegradationGenerator.residual_head`는 weight와 bias가 **0으로 초기화**된다.
따라서 새로 생성한 모델은 최초 forward 시 `residual == 0`이다. 기본 설정인
`CLAMP_DEGRADED=False`이거나 입력이 이미 `[0, 1]` 범위이면
`degraded == original`도 성립한다. 학습과 save/load 검증에서 활용할 수 있는
중요한 초기 조건이다.

## 3. `DgNetConfig` 하이퍼파라미터

`DgNetConfig`는 `frozen=True` dataclass이다. 실험 variant는
`dataclasses.replace(base, DG_ARCHITECT="CNN")`처럼 만드는 것이 편리하다.

### 3.1 입력과 patch tokenization

| 필드 | 기본값 | 의미 |
| --- | ---: | --- |
| `IMG_SIZE` | `224` | 입력 이미지 한 변 크기. dataset의 `img_size`와 같아야 한다. |
| `PATCH_SIZE` | `16` | patch 한 변 크기. `IMG_SIZE % PATCH_SIZE == 0`이어야 한다. |
| `IN_CHANS` | `3` | 입력/복원 이미지 channel 수 |

token 개수는 `(IMG_SIZE // PATCH_SIZE) ** 2`이고, decoder가 예측하는 각
token의 pixel 차원은 `PATCH_SIZE * PATCH_SIZE * IN_CHANS`이다.

### 3.2 공통 크기 기본값

| 필드 | 기본값 | 적용 대상 / 의미 |
| --- | ---: | --- |
| `EMBED_DIM` | `512` | component별 override가 없을 때 encoder/decoder/degradation token 차원 |
| `DEPTH` | `6` | override가 없을 때 각 block stack 깊이 |
| `NUM_HEADS` | `8` | override가 없을 때 ViT/Hybrid attention head 수 |
| `MLP_RATIO` | `4.0` | ViT와 Hybrid FFN hidden dimension 배수 |
| `PROJECTION_DIM` | `128` | representation projection 최종 차원 |

attention을 사용하는 encoder, decoder, `DG_ARCHITECT="VIT"` 또는
`"HYBRID"` degradation에서는 해당 component의 embed dimension이 head 수로
나누어져야 한다.

### 3.3 Architecture 선택

| 필드 | 기본값 | 설명 |
| --- | --- | --- |
| `ENCODER_ARCHITECT` | `"VIT"` | 현재 구현에서는 반드시 `"VIT"`이어야 한다. |
| `DECODER_ARCHITECT` | `"VIT"` | 현재 구현에서는 반드시 `"VIT"`이어야 한다. |
| `DG_ARCHITECT` | `"HYBRID"` | degradation `M` block: `"VIT"`, `"CNN"`, `"HYBRID"` |
| `CNN_ARCHITECT` | `"CONVNEXT"` | `DG_ARCHITECT="CNN"`일 때 `"RESNET"` 또는 `"CONVNEXT"` |

각 degradation variant는 같은 image-in / residual-out 계약을 지키므로 config
한 줄만 바꾸어 ablation할 수 있다.

| `DG_ARCHITECT` | 사용 block | 성격 |
| --- | --- | --- |
| `"VIT"` | `ViTBlock` | global self-attention 기반 |
| `"CNN"` | `TokenCNNBlock` | patch grid를 feature map으로 변환해 ResNet/ConvNeXt block 적용 |
| `"HYBRID"` | `HybridBlock` | attention + local CNN branch + FFN |

### 3.4 Component별 크기 override

`None`이면 위 공통 기본값을 사용한다.

| 필드 | 기본값 | 영향 |
| --- | --- | --- |
| `ENCODER_EMBED_DIM` | `None` | encoder token dim |
| `ENCODER_DEPTH` | `None` | encoder ViT block 수 |
| `ENCODER_NUM_HEADS` | `None` | encoder attention head 수 |
| `DECODER_EMBED_DIM` | `None` | decoder token dim; encoder 출력은 linear layer로 이 차원에 투영된다. |
| `DECODER_DEPTH` | `None` | decoder ViT block 수 |
| `DECODER_NUM_HEADS` | `None` | decoder attention head 수 |
| `DG_EMBED_DIM` | `None` | degradation token dim |
| `DG_DEPTH` | `None` | degradation block 수 |
| `DG_NUM_HEADS` | `None` | VIT/HYBRID degradation attention head 수; CNN variant에서는 사용하지 않는다. |

### 3.5 Positional embedding과 출력 제어

| 필드 | 기본값 | 설명 |
| --- | --- | --- |
| `USE_POS_EMBED` | `True` | encoder/decoder/degradation token에 학습 가능한 positional embedding 추가 |
| `POS_EMBED_INIT_ZERO` | `True` | `True`: zero init, `False`: truncated normal init |
| `CLAMP_DEGRADED` | `False` | `degraded = x - M(x)`를 계산한 후 `[0, 1]`로 clamp |

`CLAMP_DEGRADED=True`는 입력 tensor가 `[0, 1]` 픽셀 의미를 가질 때 적절하다.
dataset의 기본 ImageNet normalization을 사용한다면 보통 `False`로 두거나,
시각화 목적의 loader에서 `mean=(0, 0, 0), std=(1, 1, 1)`을 사용한다.

### 3.6 Regularization / block 상세

| 필드 | 기본값 | 적용 범위 |
| --- | ---: | --- |
| `ATTN_DROPOUT` | `0.0` | ViT/HYBRID attention dropout |
| `PROJ_DROPOUT` | `0.0` | ViT attention/MLP projection dropout, HYBRID attention output dropout |
| `DROP_PATH` | `0.0` | ViT 및 CNN degradation block stochastic depth |
| `CNN_KERNEL_SIZE` | `3` | CNN의 ConvNeXt variant 및 HYBRID local CNN kernel; 이 경로에서는 odd 값 사용 (`RESNET`에서는 영향 없음) |
| `CNN_DROPOUT` | `0.0` | HYBRID local CNN branch dropout |
| `FFN_DROPOUT` | `0.0` | HYBRID FFN dropout |

## 4. Validation 제약과 빠른 확인

모델 생성 시 `cfg.validate()`가 실행된다.

| 잘못된 설정 | 결과 |
| --- | --- |
| `IMG_SIZE <= 0`, `PATCH_SIZE <= 0`, 또는 나누어떨어지지 않음 | `ValueError` |
| `IN_CHANS <= 0` 또는 `PROJECTION_DIM <= 0` | `ValueError` |
| encoder/decoder architecture가 `"VIT"`가 아님 | `ValueError` |
| `DG_ARCHITECT`가 VIT/CNN/HYBRID가 아님 | `ValueError` |
| attention component의 dim/depth/head가 유효하지 않거나 dim이 head로 나누어지지 않음 | `ValueError` |

복사해서 실행 가능한 생성/초기 조건 smoke validation:

```python
import torch

from model.dg_model import DGNet, DgNetConfig, parameter_summary

cfg = DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
    PROJECTION_DIM=32,
)
model = DGNet(cfg).eval()
sample = torch.rand(2, 3, 32, 32)

with torch.no_grad():
    out = model(sample)

assert out.residual.shape == sample.shape
assert torch.count_nonzero(out.residual).item() == 0
assert torch.equal(out.degraded, sample)
count, size_mib = parameter_summary(model)
print(f"parameters={count:,}, parameter_size_mib={size_mib:.4f}")
```

## 5. Forward 반환값: `DgNetOutput`

```python
output = model(images)
values = output.as_dict()
```

| 필드 | shape | 의미 |
| --- | --- | --- |
| `original` | `[B, C, H, W]` | 입력 `x` |
| `residual` | `[B, C, H, W]` | degradation generator 출력 `M(x)` |
| `degraded` | `[B, C, H, W]` | `x - M(x)`; 선택적으로 clamp됨 |
| `reconstruction` | `[B, C, H, W]` | degraded 입력의 encoder-decoder 복원 |
| `original_reconstruction` | `[B, C, H, W]` | original 입력의 encoder-decoder 복원 |
| `representation` | `[B, PROJECTION_DIM]` | original encoder token 평균의 projection |
| `degraded_representation` | `[B, PROJECTION_DIM]` | degraded encoder token 평균의 projection |

`loss/loss.py`의 DGNet loss는 이 객체의 `original`, `reconstruction`,
`residual` 필드를 직접 소비할 수 있다.

## 6. 저장 / 로드 API

```python
from pathlib import Path
import torch

from model.dg_model import DGNet, DgNetConfig

output_dir = Path("model/output/experiment_001")
model = DGNet(DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
))

saved_dir = model.save(output_dir)
loaded = DGNet.load(saved_dir, map_location="cpu", verbose=False).eval()

sample = torch.rand(2, 3, 32, 32)
model.eval()
with torch.no_grad():
    before = model(sample).reconstruction
    after = loaded(sample).reconstruction
assert torch.allclose(before, after)
```

저장 디렉터리 구조:

```text
model/output/experiment_001/
├── config.json   # DgNetConfig 전체 값: architecture 재생성 정보
└── model.pt      # state_dict weight
```

`DGNet.load()`는 `config.json`으로 architecture를 먼저 생성한 뒤
`model.pt`를 로드한다. 즉 모델을 배포하거나 checkpoint를 옮길 때는 두 파일을
항상 함께 보관한다.

## 7. 구조 보고서와 샘플 시각화 유틸리티

### 7.1 `architecture_summary()` / `parameter_summary()`

```python
from model.dg_model import DGNet, DgNetConfig, parameter_summary

model = DGNet(DgNetConfig(
    IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2, NUM_HEADS=4
))
print(model.architecture_summary())
parameters, size_mib = parameter_summary(model.degradation)
print(parameters, size_mib)
```

- `model.architecture_summary()`는 encoder, decoder, degradation, projection,
  total parameter 수·메모리와 module 구조 문자열을 반환한다.
- `parameter_summary(module)`는 trainable parameter 수와 MiB 크기를 반환한다.

### 7.2 `save_sample_visualization()`

`DgNetOutput`의 첫 sample에 대해 original / degraded / reconstruction 세 패널
이미지를 저장한다. 내부에서 tensor를 `[0, 1]`로 clamp해 PNG로 변환하므로
raw pixel 범위 입력으로 시각화하는 것이 가장 해석하기 쉽다.

```python
from pathlib import Path
import torch

from model.dg_model import DGNet, DgNetConfig, save_sample_visualization

model = DGNet(DgNetConfig(
    IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2, NUM_HEADS=4
)).eval()
with torch.no_grad():
    output = model(torch.rand(2, 3, 32, 32))

image_path = save_sample_visualization(output, Path("model/output/sample.png"))
print(image_path)
```

### 7.3 `write_model_report()`

한 batch를 추론하여 architecture summary, 주요 tensor shape, 초기 zero residual
확인값을 `.log`에 저장하고 동시에 `dg_model_sample.png`를 만든다.

```python
from pathlib import Path
import torch

from model.dg_model import DGNet, DgNetConfig, write_model_report

model = DGNet(DgNetConfig(
    IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2, NUM_HEADS=4
))
sample = torch.rand(2, 3, 32, 32)
report = write_model_report(model, sample, Path("model/output/validation"))
print(report)
```

결과:

```text
model/output/validation/
├── dg_model.log
└── dg_model_sample.png
```

### 7.4 모듈 자체 검증 실행

아래 명령은 VIT/CNN/HYBRID degradation 모두의 shape 및 zero-init 조건,
보고서 생성, save/load round trip을 검증한다.

```bash
python -m model.dg_model
```

출력 artifact는 `model/output/dg_model.log`,
`model/output/dg_model_sample.png`, `model/output/saved_dgnet/`에 저장된다.

## 8. Architecture 비교 실험 예제

```python
from dataclasses import replace
import torch

from model.dg_model import DGNet, DgNetConfig, parameter_summary

base = DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
    PROJECTION_DIM=32,
)
images = torch.rand(4, 3, 32, 32)

for architecture in ("VIT", "CNN", "HYBRID"):
    cfg = replace(
        base,
        DG_ARCHITECT=architecture,
        CNN_ARCHITECT="CONVNEXT",  # architecture == "CNN"일 때만 영향
    )
    model = DGNet(cfg).eval()
    with torch.no_grad():
        output = model(images)
    parameters, size_mib = parameter_summary(model.degradation)
    print(architecture, output.degraded.shape, parameters, f"{size_mib:.3f} MiB")
```

## 9. Dataset + DGNet + Loss 실제 연결 샘플

아래 예제는 STL10 SSL loader, DGNet, 독립 loss 패키지를 한 번에 연결한다.
시각화와 `CLAMP_DEGRADED`가 같은 픽셀 계약을 사용하도록 normalization을
identity로 설정했다. 실제 학습 정책에서 normalized 입력을 사용하려면
`CLAMP_DEGRADED=False`와 loss scale을 함께 검토한다.

```python
from pathlib import Path

import torch

from dataset import DataConfig, prepare_dataloader
from loss.loss import prepare_dgloss
from loss.loss_config import LossConfig
from model.dg_model import DGNet, DgNetConfig, save_sample_visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = prepare_dataloader(DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    img_size=32,
    ssl_num_views=2,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    batch_size=4,
    num_workers=0,
    drop_last=True,
    pin_memory=False,
    max_samples=64,              # 예제 실행용; 본 학습에서는 None 권장
))

model = DGNet(DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=48,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=32,
    DG_ARCHITECT="HYBRID",
    CLAMP_DEGRADED=True,
)).to(device)
loss_fn = prepare_dgloss(LossConfig())

optimizer_i = torch.optim.Adam(
    list(model.encoder.parameters())
    + list(model.decoder.parameters())
    + list(model.projection.parameters()),
    lr=1e-4,
)
optimizer_m = torch.optim.Adam(model.degradation.parameters(), lr=1e-4)

def trainable(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)

views, _ = next(iter(loader))
images = views[0].to(device)

model.eval()
with torch.no_grad():
    initial = model(images)
save_sample_visualization(initial, Path("model/output/step_000.png"))

model.train()
global_step = 0

# I phase: degradation M은 고정하고 inference encoder/decoder를 업데이트
trainable(model.degradation, False)
trainable(model.encoder, True)
trainable(model.decoder, True)
trainable(model.projection, True)
optimizer_i.zero_grad()
values_i = loss_fn(model(images), step=global_step)
values_i.inference_loss.backward()
optimizer_i.step()

# M phase: inference network는 고정하고 degradation generator만 업데이트
trainable(model.degradation, True)
trainable(model.encoder, False)
trainable(model.decoder, False)
trainable(model.projection, False)
optimizer_m.zero_grad()
values_m = loss_fn(model(images), step=global_step)
values_m.degradation_loss.backward()
optimizer_m.step()

model.eval()
with torch.no_grad():
    after_step = model(images)
save_sample_visualization(after_step, Path("model/output/step_001.png"))
print(values_i.metrics())
print(values_m.metrics())
```

## 10. 실전 사용 체크리스트

1. `DataConfig.img_size == DgNetConfig.IMG_SIZE`를 보장한다.
2. `IMG_SIZE`가 `PATCH_SIZE`로 나누어떨어지고, attention dimension은 head
   수로 나누어떨어지는지 생성 전에 확인한다.
3. 시각화/픽셀 clamp를 쓸 경우 입력 normalization 범위를 명시한다.
4. architecture 비교는 같은 base config에서 `DG_ARCHITECT`만 우선 바꾸어
   비교하고, 이후 dimension/depth ablation을 분리한다.
5. checkpoint는 `config.json`과 `model.pt`를 함께 보관하고, 학습 재개 전에
   `DGNet.load()` round-trip과 샘플 forward를 검증한다.
