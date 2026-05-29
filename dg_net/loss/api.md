# DGNet loss API

`loss.py`는 모델이나 데이터셋 모듈을 import하지 않는 독립 loss 모듈이다.
학습 엔진은 `DGNetOutput`에 포함된 `original`, `reconstruction`, `residual`
tensor만 loss에 전달한다.

## 목적 함수

기본 `reduction="mean"`에서 \(D_{rec}\)는 batch와 픽셀에 대한 평균 squared
reconstruction error이다. 따라서 `tau_deg`도 평균 오차 단위로 지정한다.
`reduction="sum"`을 선택하면 논문의 비정규화 squared L2 합 형태이며
`tau_deg` 역시 크기에 맞춰 조정해야 한다.

\[
\mathcal L_I = \alpha_{inference} D_{rec}, \qquad
D_{rec} = \|\hat{x}-x\|_2^2
\]

\[
\mathcal L_M =
\alpha_{target}(D_{rec}-\tau_{deg})^2
+\lambda_{budget}\left|\operatorname{mean}(|M(x)|)-\beta_{mask}\right|
-\lambda_{reg}D_{KL}(P_{damage}\|U)
\]

네 개 항은 코드에서 각각 `ReconstructionDistance`,
`DegradationTargetDistance`, `MaskBudgetDistance`,
`DamageKLDivergence` 객체로 분리되어 있다.

## 가장 짧은 사용법

```python
from loss_config import LossConfig
from loss import prepare_dgloss

loss_cfg = LossConfig()
loss_fn = prepare_dgloss(loss_cfg)      # optimizer가 아니라 loss nn.Module이다.

output = model(images)                  # DGNetOutput
values = loss_fn(output, step=global_step)
inference_objective = values.inference_loss
degradation_objective = values.degradation_loss
```

`output` 대신 독립적인 tensor로 확인할 수도 있다.

```python
values = loss_fn.from_tensors(
    original=images,
    reconstruction=reconstructed_images,
    residual=mask_residual,
    step=20,
)
print(values.metrics())
```

## Config 필드

| 필드 | 기본값 | 의미 |
| --- | ---: | --- |
| `alpha_inference` | `1.0` | inference encoder/decoder가 최소화하는 `D_rec` 계수 |
| `alpha_target` | `1.0` | degradation model의 목표 난이도 matching 항 계수 |
| `lambda_budget` | `1.0` | residual 평균 크기를 `beta_mask`로 맞추는 항 계수 |
| `lambda_reg` | `0.1` | uniform brightness-damage collapse를 막는 KL 계수; loss에는 음수 부호로 들어간다 |
| `tau_deg` | `0.25` | degradation model이 만들 목표 reconstruction error |
| `beta_mask` | `0.15` | `mean(abs(M(x)))`의 목표 damage budget |
| `eps` | `1e-8` | zero-initialized residual에서도 KL을 finite하게 유지하는 안정화 값 |
| `reduction` | `"mean"` | reconstruction/KL scalar reduction: `"mean"` 또는 `"sum"` |

각 loss 계수에는 다음 schedule 필드가 하나씩 대응한다.

| 계수 | schedule 필드 |
| --- | --- |
| `alpha_inference` | `alpha_inference_schedule` |
| `alpha_target` | `alpha_target_schedule` |
| `lambda_budget` | `lambda_budget_schedule` |
| `lambda_reg` | `lambda_reg_schedule` |

## 항 끄기: 계수 `0.0`

계수가 정확히 `0.0`이면 해당 항은 objective의 autograd 경로에 넣지 않는다.
따라서 하나의 항을 제거한 ablation은 config 변경만으로 수행한다.

```python
from loss_config import LossConfig
from loss import prepare_dgloss

# budget과 anti-collapse 없이 target difficulty만 학습
loss_fn = prepare_dgloss(
    LossConfig(lambda_budget=0.0, lambda_reg=0.0)
)

# degradation objective 전체를 끔: backward()는 호출 가능하지만
# reconstruction/residual에 대한 gradient는 모두 정확히 zero이다.
off_loss_fn = prepare_dgloss(
    LossConfig(alpha_target=0.0, lambda_budget=0.0, lambda_reg=0.0)
)
```

주의: `alpha_inference=0.0`이어도 `alpha_target>0.0`이면
`D_target`은 같은 reconstruction을 사용하므로 target 항의 gradient는 존재한다.
특정 항의 on/off gradient를 비교하려면 나머지 관련 계수도 함께 고정해야 한다.

## Linear / cosine schedule

`ScheduleConfig`는 base coefficient에 곱해지는 factor를 만든다.
`enabled=False`이면 factor는 항상 `1.0`이다. base coefficient를 `0.0`으로
설정하면 schedule을 켜도 그 항은 계속 꺼져 있다.

```python
from loss_config import LossConfig, ScheduleConfig

warmup = ScheduleConfig(
    enabled=True,
    kind="linear",        # 또는 "cosine"
    start_step=0,
    end_step=100,
    start_factor=0.0,
    end_factor=1.0,
)

loss_cfg = LossConfig(
    alpha_target=1.0,
    lambda_reg=0.1,
    alpha_target_schedule=warmup,
    lambda_reg_schedule=ScheduleConfig(
        enabled=True, kind="cosine", start_step=0, end_step=100
    ),
)
loss_fn = prepare_dgloss(loss_cfg)

at_start = loss_fn(output, step=0).effective_weights   # target/reg = 0
at_end = loss_fn(output, step=100).effective_weights    # configured full values
```

## DGNet 교대 최적화 예시

`inference_loss`와 `degradation_loss`는 목적이 다른 두 optimizer용 값이다.
`values.combined_loss`도 제공되지만 log/smoke test용이며, 한 optimizer에서
최소화하는 학습 objective로 사용하지 않는다. 학습 시에는 각 phase에서
업데이트하지 않는 module을 freeze한다.

```python
loss_fn = prepare_dgloss(LossConfig())

# I phase: engine에서 model.degradation parameters를 freeze한다.
optimizer_i.zero_grad()
out_i = model(images)
loss_fn(out_i, step=global_step).inference_loss.backward()
optimizer_i.step()

# M phase: engine에서 model.encoder/model.decoder parameters를 freeze한다.
optimizer_m.zero_grad()
out_m = model(images)
loss_fn(out_m, step=global_step).degradation_loss.backward()
optimizer_m.step()
```

## Dataset / model API를 이용한 smoke validation 설정

프로젝트의 SSL 입력은 `STL10` unlabeled split을 사용한다. 아래 설정은
`batch_size=4`, `mode="ssl"` 요구사항과 DGNet의 작은 확인용 architecture를
그대로 연결한다.

```python
import sys
from pathlib import Path

import torch

ROOT = Path("src").resolve()
sys.path[:0] = [str(ROOT / "dataset"), str(ROOT / "model"), str(ROOT / "loss")]

from dataset import DataConfig, prepare_dataloader
from dg_model import DGNet, DgNetConfig, save_sample_visualization
from loss import prepare_dgloss
from loss_config import LossConfig

data_cfg = DataConfig(
    dataset="STL10", mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    max_samples=400, img_size=32, use_augmentation=False,
    batch_size=4, num_workers=0, drop_last=True, pin_memory=False,
)
loader = prepare_dataloader(data_cfg)
model = DGNet(DgNetConfig(
    IMG_SIZE=32, PATCH_SIZE=8, EMBED_DIM=64, DEPTH=2,
    NUM_HEADS=4, DECODER_EMBED_DIM=48, DECODER_NUM_HEADS=4,
    PROJECTION_DIM=32,
))
loss_fn = prepare_dgloss(LossConfig())

views, _ = next(iter(loader))
images = views[0]
initial = model(images)
save_sample_visualization(initial, Path("src/loss/output/step_000.png"))

# 학습 engine은 위의 I/M phase 분리를 적용하여 정확히 100 step 수행한다.
for global_step in range(100):
    ...

after_100 = model(images)
save_sample_visualization(after_100, Path("src/loss/output/step_100.png"))
```

## Report 저장 helper

loss 모듈 자체의 보고 기능도 모델 의존성 없이 동작한다.

```python
from loss import write_loss_report

write_loss_report(values, "src/loss/output/loss_step_100.log")
```

또한 `python src/loss/loss.py`를 실행하면 tensor-only 독립 smoke test가
`src/loss/output/loss_tensor_smoke.log`에 다음 내용을 검증하여 남긴다.

- 네 term이 finite scalar를 생성하는지
- cosine schedule이 0 step과 100 step에서 예상 coefficient를 생성하는지
- 모든 term coefficient가 `0.0`일 때 backward gradient가 zero인지
