# DGNet 프레임워크 리팩터링 — 결정 완료형 Agent 작업 명세서

## 0. 실행 권한과 최종 목표

이 문서는 후속 agent가 별도 질문이나 승인 없이 DGNet 프레임워크 분석, 리팩터링,
테스트, 문서화를 끝까지 수행하기 위한 결정 완료형 작업 명세서이다.

후속 agent는 아래 순서를 반드시 지킨다.

```text
Phase 0: 구현 없이 분석 및 report.md 작성
Phase 1: 현재 동작 회귀 테스트 고정
Phase 2: loss objective 이중화와 gradient 안전성 구현
Phase 3: 전역 degradation 진단 및 조건부 architecture 보완
Phase 4: ImageNet 지원
Phase 5: visualization/API/script 정리
Phase 6: single GPU/DDP/transfer 검증과 최종 report.md 갱신
```

`report.md` 최초 작성 전에는 production code를 수정하지 않는다. 최초 `report.md`가 작성되고
baseline 회귀 테스트가 추가되면, 별도 승인 없이 나머지 구현을 계속 수행한다.

### 최종 결과

작업 완료 시 저장소는 다음을 만족해야 한다.

1. 현재 실험 실패 원인과 구형/현재 구조 차이가 루트 `report.md`에 근거와 함께 기록된다.
2. cosine과 InfoNCE representation objective를 config/CLI로 선택할 수 있다.
3. I/M 교대 학습의 parameter update와 gradient flow가 테스트로 고정된다.
4. VIT/HYBRID degradation generator의 전역 정보 혼합 여부가 정량 진단된다.
5. patch seam 문제가 확인되면 config 기반 spatial refinement head가 추가된다.
6. ImageNet ImageFolder loader가 SSL/classification/DDP에서 동작한다.
7. checkpoint마다 old framework 수준 이상의 sample visualization이 저장된다.
8. 기존 public API, config 중심 조립, fan-in/fan-out 규칙을 유지한다.
9. package별 API 문서와 실행 script가 실제 코드와 일치한다.

---

## 1. 절대 규칙

### 1.1 수정 금지

아래는 참고 전용이며 수정하지 않는다.

```text
TODO.md
about_project/project.md
old_proj_src/**
version1_api/**
version2_harness/**
```

사용자가 이미 만든 unrelated 변경사항을 되돌리거나 덮어쓰지 않는다.

### 1.2 새 dependency 금지

`dg_net/requirements.txt`에 dependency를 추가하지 않는다. PyTorch, torchvision, numpy,
Pillow, matplotlib, pytest 등 현재 환경에 있는 패키지만 사용한다.

### 1.3 의존 방향

허용:

```text
engine -> model
engine -> loss
engine -> dataset
model/dg_model.py -> model/cnn.py, model/vit.py, model/hybrid.py
```

금지:

```text
loss -> engine/model/dataset
dataset -> engine/model/loss
model -> engine/loss/dataset
```

`loss`는 tensor와 config만 받는 독립 패키지로 유지한다. 분산 통신을 loss 내부에 넣지 않는다.

### 1.4 기존 API 보존

다음을 삭제하거나 이름 변경하지 않는다.

```text
model.dg_model.DGNet
model.dg_model.DgNet
model.dg_model.DgNetConfig
model.dg_model.DGNetConfig
model.dg_model.DgNetOutput
model.dg_model.save_sample_visualization
model.dg_model.write_model_report
loss.loss.DGLoss
loss.loss.DGLossOutput
loss.loss.prepare_dgloss
loss.loss_config.TermWeightConfig
loss.loss_config.LossConfig
dataset.DataConfig
dataset.prepare_dataloader
engine.train_config.TrainConfig
```

기존 config 필드, forward output 필드, save/load checkpoint 형식을 유지한다. 새 기능은 additive
필드와 optional argument로 추가한다.

---

## 2. 현재 구현에 대한 확정된 사실

후속 agent는 아래 사실을 다시 논쟁하지 말고 baseline으로 사용한다.

### Model

- 현재 `dg_net/model/dg_model.py`는 입력을 patch token `[B,N,D]`로 변환한다.
- VIT/HYBRID generator는 self-attention 때문에 patch 간 전역 정보 교환이 가능하다.
- CNN-only generator의 정보 교환 범위는 patch grid CNN receptive field에 제한된다.
- 현재 generator는 token별 residual patch를 출력하여 unpatchify한다.
- 구형 `old_proj_src/models/degradation/model_degradation.py`도 raw input patch와 token별
  delta patch를 조합하여 unpatchify한다.
- 따라서 “구형은 전체 이미지를 하나의 flatten vector로 출력했다”는 기억은 저장소 코드로
  확인되지 않았다.
- 현재 residual head는 zero-init이며 초기 residual은 0이다.

### Loss/Engine

- 현재 representation distance는 cosine만 지원한다.
- InfoNCE는 구현되어 있지 않다.
- I phase는 degradation을 freeze하고 encoder/decoder/projection을 학습한다.
- M phase는 degradation만 학습하고 encoder/decoder/projection을 freeze한다.
- frozen inference network 연산을 통해 M까지 gradient가 전달되어야 한다.
- M phase에 `torch.no_grad()`를 적용하면 안 된다.

### Dataset/Docs/Test

- STL10/Synthetic SSL, STL10/CIFAR100/Flowers102/iNaturalist classification을 지원한다.
- ImageNet은 지원하지 않는다.
- 정식 pytest는 현재 dataset에만 존재한다.
- 현재 sample visualization은 original/degraded/reconstruction만 저장한다.
- canonical API 문서는 `dg_net/{model,loss,dataset,engine}/*_api.md`이다.

---

## 3. 작업 범위와 파일별 책임

### 반드시 생성

```text
report.md
dg_net/model/tests/test_dg_model.py
dg_net/model/tests/test_globality.py
dg_net/loss/tests/test_loss.py
dg_net/engine/tests/test_gradient_flow.py
dg_net/engine/tests/test_smoke_pipeline.py
```

필요하면 각 `tests/` 디렉토리에 빈 `__init__.py`를 추가한다.

### 수정 허용

```text
dg_net/model/dg_model.py
dg_net/model/model_api.md
dg_net/loss/loss.py
dg_net/loss/loss_config.py
dg_net/loss/__init__.py
dg_net/loss/loss_api.md
dg_net/dataset/dataset.py
dg_net/dataset/dataset_config.py
dg_net/dataset/dataset_api.md
dg_net/dataset/tests/test_dataset.py
dg_net/engine/dg_train.py
dg_net/engine/eval.py
dg_net/engine/train_config.py
dg_net/engine/engine_api.md
dg_net/scripts/run_dgnet_pipeline_single_gpu.bash
dg_net/scripts/run_dgnet_pipeline_ddp.bash
```

`cnn.py`, `vit.py`, `hybrid.py`는 진단 결과상 bug가 직접 확인되지 않으면 수정하지 않는다.

### 정리 대상

- `dg_net/model/api.md`는 삭제하지 말고 첫 줄에 canonical 문서가 `model_api.md`임을 알리는
  짧은 안내 문서로 바꾼다.
- `dg_net/loss/dataset_api.md`는 삭제하지 말고 canonical 문서가
  `../dataset/dataset_api.md`임을 알리는 짧은 안내 문서로 바꾼다.
- `dg_net/hybrid_dgnet_train.bash`는 현재 CLI와 맞지 않는다. 삭제하지 말고 최신
  `scripts/run_dgnet_pipeline_ddp.bash`를 호출하는 compatibility wrapper로 바꾼다.

---

## 4. Phase 0 — 구현 없는 분석과 최초 `report.md`

Phase 0에서는 `report.md`와 분석용 임시 artifact 외 production code를 수정하지 않는다.
`old_proj_src`는 `src.*` import 전제를 가지고 있어 현재 위치에서 직접 실행되지 않으므로,
구형 프레임워크 비교는 정적 코드 분석으로 수행한다. 구형 코드를 고치거나 복사하지 않는다.

### 4.1 `report.md` 필수 목차

```markdown
# DGNet 구현 현황 및 실패 원인 분석
## 1. 결론 요약
## 2. 현재/구형 architecture tensor flow 비교
## 3. 현재 loss 비교와 objective 차이
## 4. I/M phase gradient flow 분석
## 5. 패치 독립성/전역성 진단 결과
## 6. 확인된 구현 문제
## 7. 설계 차이
## 8. 미확인 가설과 한계
## 9. 리팩터링 결정
## 10. 실행한 검증 명령과 결과
```

모든 주요 판단에 다음 label 중 하나를 붙인다.

```text
[확인됨] 코드/테스트/실행 결과로 직접 확인
[추론] 확인된 증거로부터 도출했으나 추가 실험 필요
[미확인] 현재 증거로 판단 불가
```

### 4.2 tensor flow 표

구형과 현재 모델에 대해 아래 tensor를 표로 기록한다.

```text
input image
patch embedding
DG blocks output
DG output head
residual 또는 degraded output
encoder tokens
pooled representation
projection output
decoder patch prediction
reconstruction image
```

현재 모델 runtime shape는 다음 고정 config로 측정한다.

```python
DgNetConfig(
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
```

### 4.3 최초 진단 명령

저장소 루트에서 실행한다.

```bash
python -m pytest dg_net/dataset/tests/test_dataset.py -q
python -m compileall -q dg_net
PYTHONPATH=dg_net python -m loss.loss
python dg_net/model/dg_model.py
```

각 명령의 종료 코드와 핵심 결과를 `report.md`에 기록한다. 실패하면 숨기지 말고 원인을 기록한다.

### 4.4 Phase 0 완료 조건

- `report.md` 필수 목차가 모두 존재한다.
- 구형/현재 tensor flow 표가 있다.
- 현재 교대 학습의 의도된 gradient flow가 설명되어 있다.
- 아직 runtime으로 확인하지 못한 항목은 `[미확인]`으로 표시되어 있다.

완료 후 별도 승인 없이 Phase 1로 진행한다.

---

## 5. Phase 1 — Baseline 회귀 테스트

production code 수정 전에 아래 테스트를 추가하고 현재 동작을 고정한다.

### 5.1 `dg_net/model/tests/test_dg_model.py`

필수 테스트:

1. `VIT`, `CNN`, `HYBRID` 모두 forward output shape가 입력 image shape와 일치한다.
2. 초기 residual은 정확히 0이고 degraded는 original과 정확히 같다.
3. `DgNetOutput` 기존 7개 필드가 유지된다.
4. save/load round trip 후 모든 output이 `allclose`이다.
5. 기존 `save_sample_visualization(output, path)` 호출이 PNG를 생성한다.

### 5.2 `dg_net/loss/tests/test_loss.py`

InfoNCE 구현 전 baseline 테스트:

1. cosine representation distance의 동일 vector 결과는 0이다.
2. representation/reconstruction/hybrid preset의 scalar loss가 finite이다.
3. weight 0 term은 effective weight와 weighted term이 0이다.
4. linear/cosine scheduler가 시작/끝 값을 만족한다.
5. 기존 `DGLossOutput` 필드가 유지된다.

### 5.3 `dg_net/engine/tests/test_gradient_flow.py`

작은 CPU model을 사용한다.

```text
IMG_SIZE=16
PATCH_SIZE=8
EMBED_DIM=16
DEPTH=1
NUM_HEADS=4
DECODER_EMBED_DIM=16
DECODER_NUM_HEADS=4
PROJECTION_DIM=8
BATCH=4
```

parameter snapshot 전후를 비교하여 다음을 검사한다.

| Phase | degradation | encoder | decoder | projection |
|---|---|---|---|---|
| I reconstruction | unchanged | changed | changed | unchanged 허용 |
| I cosine representation | unchanged | changed | unchanged 허용 | changed |
| M reconstruction target | changed | unchanged | unchanged | unchanged |
| M cosine representation | changed | unchanged | unchanged | unchanged |

M phase에서는 degradation parameter 중 최소 하나의 finite non-zero gradient와 update가 있어야 한다.
단순히 `.grad is None` 여부만 검사하지 않는다.

### 5.4 Baseline 테스트 명령

```bash
PYTHONPATH=dg_net python -m pytest \
  dg_net/model/tests/test_dg_model.py \
  dg_net/loss/tests/test_loss.py \
  dg_net/engine/tests/test_gradient_flow.py \
  dg_net/dataset/tests/test_dataset.py -q
```

baseline 테스트가 통과한 뒤 Phase 2를 시작한다.

---

## 6. Phase 2 — Cosine/InfoNCE 이중 objective 구현

이 절의 설계는 확정 사항이다. 다른 InfoNCE 정의를 선택하지 않는다.

### 6.1 representation pair 정의

positive pair는 augmentation view 간 pair가 아니라 동일 sample의 다음 pair이다.

```text
positive: z_original[i] <-> z_degraded[i]
negative: 같은 process batch의 다른 sample z_degraded[j], j != i
```

현재 SSL loader가 반환하는 여러 augmentation view 중 engine은 기존처럼 첫 view만 사용한다.
이번 작업에서 multi-view augmentation contrastive loss를 추가하지 않는다.

### 6.2 InfoNCE 수식

입력:

```text
z_original: [B,D]
z_degraded: [B,D]
```

두 tensor를 마지막 차원으로 L2 normalize한다.

```text
logits = z_original @ z_degraded.T / temperature
labels = arange(B)
loss_forward = cross_entropy(logits, labels)
loss_reverse = cross_entropy(logits.T, labels)
info_nce = 0.5 * (loss_forward + loss_reverse)
```

결정:

- symmetric InfoNCE를 사용한다.
- default temperature는 `0.2`이다.
- batch size가 2 미만이면 명확한 `ValueError`를 발생시킨다.
- negatives는 **local process batch만** 사용한다.
- DDP global gather는 구현하지 않는다.
- 이 결정은 loss 패키지를 tensor-only/process-agnostic으로 유지하기 위한 것이다.
- DDP API 문서에 effective negative count가 per-rank batch size임을 명시한다.

### 6.3 I/M objective와 gradient

I phase:

```text
inference_rep = objective(z_original, z_degraded)
I는 inference_rep를 minimize한다.
```

M phase:

```text
dg_rep = objective(z_original.detach(), z_degraded)
degradation loss에는 -weight * dg_rep로 포함한다.
M은 dg_rep를 maximize한다.
```

M phase에서 inference network parameter는 freeze하지만 `z_degraded` 계산 graph는 유지한다.
`z_original`은 M과 무관한 branch이므로 DG representation term 계산에서 detach한다.

이번 리팩터링에서는 representation threshold term을 다시 도입하지 않는다. 기존 `tau_deg`,
`tau_budget`은 유지하지만 `rep_threshold`, `mask_threshold`, binary mask loss는 추가하지 않는다.
cosine과 InfoNCE 자체의 효과를 먼저 분리 비교하기 위한 결정이다.

### 6.4 Config/API 변경

`LossConfig`에 additive field를 추가한다.

```python
representation_distance: Literal["cosine", "infonce"] = "cosine"
infonce_temperature: float = 0.2
```

호환성을 위해 기존 field 이름 `representation_distance`를 유지한다.

검증:

- `infonce_temperature > 0`
- CLI `--representation-distance` choices를 `("cosine", "infonce")`로 확장
- CLI `--infonce-temperature` 추가
- `TrainConfig.infonce_temperature: float = 0.2` 추가
- `TrainConfig.build_loss_config()`가 값을 전달

`DGLossOutput`의 기존 필드는 유지한다. `inference_rep`와 `dg_rep`는 선택된 objective scalar를 담는다.
새 output field는 추가하지 않는다.

### 6.5 구현 위치

```text
dg_net/loss/loss.py
  - InfoNCELoss(nn.Module) 추가
  - DGLoss가 config에 따라 cosine 또는 InfoNCE 선택
  - DG representation term은 z_original.detach()를 reference로 사용

dg_net/loss/loss_config.py
  - infonce_temperature field와 validation 추가

dg_net/loss/__init__.py
  - InfoNCELoss를 public export에 추가

dg_net/engine/train_config.py
  - config/CLI 연결
```

### 6.6 InfoNCE 필수 테스트

`dg_net/loss/tests/test_loss.py`에 추가:

1. matching pair logits가 강할수록 mismatched pair보다 loss가 작다.
2. symmetric 계산이 수식과 일치한다.
3. batch size 1은 실패한다.
4. temperature 0 이하는 config 생성 단계에서 실패한다.
5. cosine preset과 InfoNCE preset 모두 backward 가능하다.

`dg_net/engine/tests/test_gradient_flow.py`에 추가:

- I InfoNCE phase의 update isolation
- M InfoNCE phase의 update isolation
- M InfoNCE gradient가 frozen encoder/projection 연산을 통과해 degradation에 도달함

---

## 7. Phase 3 — 전역성 진단과 optional spatial refinement

### 7.1 진단 fixture

`dg_net/model/tests/test_globality.py`에서 random image 대신 다음 deterministic image를 사용한다.

```text
horizontal gradient
vertical gradient
center square
left/right two-tone image
```

zero-init residual은 항상 0이므로 진단 전 residual head weight를 고정 seed의 작은 normal 값
`std=0.02`로 테스트 내부에서만 초기화한다. production initialization은 변경하지 않는다.

### 7.2 Cross-patch influence metric

기준 입력 `x`와, 중앙 patch 하나에 `+0.1` perturbation을 적용한 `x'`를 만든다.

```text
delta = abs(M(x') - M(x))
source = perturbation patch 영역
outside = source 외 영역
cross_patch_ratio = outside.mean() / (source.mean() + 1e-8)
```

판정:

- VIT/HYBRID에서 `cross_patch_ratio > 1e-4`이면 전역 의존성이 존재한다.
- CNN-only는 이 기준을 필수 통과할 필요가 없다.
- VIT/HYBRID 둘 다 `<= 1e-4`이면 구현 bug로 판정하고 attention/token flow를 수정한다.

### 7.3 Patch boundary seam metric

residual image에서 patch 경계 양옆 pixel 차이 평균과 비경계 인접 pixel 차이 평균을 계산한다.

```text
boundary_jump = patch 경계의 horizontal/vertical 인접 pixel absolute difference 평균
interior_jump = patch 내부의 horizontal/vertical 인접 pixel absolute difference 평균
seam_ratio = boundary_jump / (interior_jump + 1e-8)
```

VIT/HYBRID deterministic fixture 평균 `seam_ratio`가 `1.25`를 초과하면 patch seam 문제가
확인된 것으로 판정한다.

### 7.4 진단 판정

다음 조건 중 하나라도 참이면 spatial refinement 사용을 권장한다.

```text
VIT/HYBRID 평균 seam_ratio > 1.25
또는
실제 checkpoint sample에서 patch 경계가 시각적으로 반복되고 report.md에 [확인됨]으로 기록됨
```

조건이 모두 거짓이면 기본 patch head 사용을 권장한다. 어느 경우든 아래 optional spatial
refinement 구현은 추가한다. 이는 default-off additive 기능이므로 기존 동작을 변경하지 않는다.

### 7.5 Spatial refinement 확정 설계

`DgNetConfig`에 다음 additive field를 추가한다.

```python
DG_SPATIAL_REFINEMENT: bool = False
DG_REFINEMENT_CHANNELS: int = 32
```

`DegradationGenerator`의 기존 patch residual을 unpatchify한 후 다음 refinement를 적용한다.

```text
residual image [B,C,H,W]
Conv2d(C, refinement_channels, 3, padding=1)
GELU
Conv2d(refinement_channels, refinement_channels, 3, padding=1)
GELU
Conv2d(refinement_channels, C, 3, padding=1)
```

마지막 Conv2d만 zero-init한다. forward 결과:

```text
final_residual = patch_residual + refinement(patch_residual)
```

결정:

- 기존 default는 `DG_SPATIAL_REFINEMENT=False`로 유지한다.
- 기존 checkpoint/config는 field가 없어도 load되어야 한다.
- 새 architecture 실험은 config에서만 켜고 끈다.
- full-image flatten Linear head, overlapping patch tokenizer, 새 dependency는 구현하지 않는다.

`TrainConfig`와 CLI 연결:

```python
dg_spatial_refinement: bool = False
dg_refinement_channels: int = 32
```

```text
--dg-spatial-refinement / --no-dg-spatial-refinement
--dg-refinement-channels 32
```

`TrainConfig.build_model_config()`가 두 값을 `DgNetConfig`에 전달해야 한다.

### 7.6 Spatial refinement 테스트

다음을 검사한다.

1. refinement on/off 모두 기존 output shape를 유지한다.
2. zero-init 시 residual=0과 degraded=original을 유지한다.
3. refinement 마지막 conv에 gradient가 도달한다.
4. save/load round trip이 동작한다.
5. 고정 test fixture에서 refinement를 켠 결과의 seam ratio가 끈 결과보다 작거나 같다.

---

## 8. Phase 4 — ImageNet 지원

### 8.1 이름과 mode

canonical 이름은 `"ImageNet"`이며 `"imagenet"` 입력도 허용한다.

```text
SUPPORTED_DATASETS에 ImageNet 추가
SSL mode 허용: STL10, ImageNet, Synthetic
classification mode 허용: 기존 dataset + ImageNet
ImageNet class count: 1000
```

### 8.2 root/split 규칙

torchvision `datasets.ImageFolder`를 사용한다.

```text
cfg.data_path/train이 존재하고 split=train이면 data_path/train 사용
cfg.data_path/val이 존재하고 split=val 또는 validation이면 data_path/val 사용
위 하위 디렉토리가 없으면 cfg.data_path 자체를 ImageFolder root로 사용
```

ImageNet SSL은 label을 학습에 사용하지 않지만 loader 반환 계약은 다른 torchvision dataset과
동일하게 `(views, labels)`를 유지한다.

### 8.3 subset/filter

기존 `_apply_subset_options()`를 재사용한다. ImageFolder의 `.targets`를 통해 `classes`와
`max_samples`가 동작해야 한다.

### 8.4 테스트

실제 ImageNet을 다운로드하지 않는다. `tmp_path` 아래에 다음 fixture를 만든다.

```text
imagenet/
  train/class0/*.png
  train/class1/*.png
  val/class0/*.png
  val/class1/*.png
```

필수 테스트:

1. classification train/val root dispatch
2. SSL multi-view 반환
3. class filter와 max_samples
4. distributed sampler
5. `SUPPORTED_DATASETS`와 case-insensitive alias

---

## 9. Phase 5 — Visualization, API, Script

### 9.1 Visualization 계약

기존 함수 signature를 유지한다.

```python
save_sample_visualization(output: DgNetOutput, output_path: str | Path) -> Path
```

한 PNG에 첫 sample의 다음 7개 panel을 저장한다.

```text
original
degraded
reconstruction
absolute residual |x - x_dg|
normalized residual heatmap
reconstruction error |x - reconstruction|
degraded reconstruction error |x_dg - reconstruction|
```

각 panel 아래 또는 별도 footer에 다음 수치를 기록한다.

```text
residual mean/std/max
reconstruction MSE
cosine similarity/distance
patch seam ratio
```

명시적 mask가 없으므로 binary `mask_x`를 새로 만들지 않는다. normalized absolute residual
heatmap을 mask-like diagnostic으로 사용한다.

checkpoint의 기존 `{tag}.png` 이름을 유지한다. 추가 artifact가 필요하면 동일 디렉토리에
`{tag}_metrics.txt`를 저장한다.

### 9.2 API canonical 문서

다음을 실제 코드와 일치하도록 갱신한다.

```text
dg_net/model/model_api.md
dg_net/loss/loss_api.md
dg_net/dataset/dataset_api.md
dg_net/engine/engine_api.md
```

공통 필수 내용:

1. 패키지 역할
2. 공개 API
3. 최소 실행 예제
4. config 필드
5. tensor 계약
6. smoke test
7. output artifact
8. train script 연결
9. DDP 주의사항

추가 필수 내용:

- model: globality metric, conditional spatial refinement, visualization
- loss: cosine/InfoNCE 수식, local negatives 결정, I/M 부호와 detach
- dataset: ImageNet root/split, SSL/classification 반환, sampler
- engine: gradient phase, single/DDP/transfer, checkpoint/resume, CLI

### 9.3 Script

최신 single/DDP pipeline script에 다음 환경변수를 추가한다.

```text
REPRESENTATION_DISTANCE=${REPRESENTATION_DISTANCE:-cosine}
INFONCE_TEMPERATURE=${INFONCE_TEMPERATURE:-0.2}
DG_SPATIAL_REFINEMENT=${DG_SPATIAL_REFINEMENT:-false}
DG_REFINEMENT_CHANNELS=${DG_REFINEMENT_CHANNELS:-32}
```

script는 값을 올바른 CLI 인자로 전달해야 한다. 기본 동작은 기존 cosine/no-refinement이다.

각 script는 다음 방식으로 refinement argument를 구성한다.

```bash
REFINEMENT_ARGS=(--no-dg-spatial-refinement)
if [[ "${DG_SPATIAL_REFINEMENT}" == "true" ]]; then
  REFINEMENT_ARGS=(--dg-spatial-refinement)
fi
REFINEMENT_ARGS+=(--dg-refinement-channels "${DG_REFINEMENT_CHANNELS:-32}")
```

loss argument에는 항상 다음을 전달한다.

```bash
--representation-distance "${REPRESENTATION_DISTANCE}"
--infonce-temperature "${INFONCE_TEMPERATURE}"
```

---

## 10. Phase 6 — 통합 검증

### 10.1 전체 pytest

저장소 루트에서 실행:

```bash
PYTHONPATH=dg_net python -m pytest dg_net -q
python -m compileall -q dg_net
git diff --check
```

### 10.2 Loss/model self-test

```bash
PYTHONPATH=dg_net python -m loss.loss
python dg_net/model/dg_model.py
```

### 10.3 CPU Synthetic pretrain smoke

저장소 루트에서 실행:

```bash
cd dg_net
python engine/dg_train.py \
  --task dg_pretrain \
  --dataset Synthetic \
  --device cpu \
  --loop-mode step \
  --total-steps 2 \
  --validate-every-steps 1 \
  --save-every-steps 1 \
  --output-root output/harness_refactor \
  --run-id cosine-pretrain \
  --img-size 16 \
  --patch-size 8 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --max-samples 16 \
  --num-workers 0 \
  --embed-dim 16 \
  --depth 1 \
  --num-heads 4 \
  --decoder-embed-dim 16 \
  --decoder-num-heads 4 \
  --projection-dim 8 \
  --loss-mode hybrid \
  --representation-distance cosine

python engine/dg_train.py \
  --task dg_pretrain \
  --dataset Synthetic \
  --device cpu \
  --loop-mode step \
  --total-steps 2 \
  --validate-every-steps 1 \
  --save-every-steps 1 \
  --output-root output/harness_refactor \
  --run-id infonce-pretrain \
  --img-size 16 \
  --patch-size 8 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --max-samples 16 \
  --num-workers 0 \
  --embed-dim 16 \
  --depth 1 \
  --num-heads 4 \
  --decoder-embed-dim 16 \
  --decoder-num-heads 4 \
  --projection-dim 8 \
  --loss-mode hybrid \
  --representation-distance infonce \
  --infonce-temperature 0.2
cd ..
```

두 run 모두 checkpoint, config, CSV, loss log, visualization PNG를 생성해야 한다.

### 10.4 Transfer smoke

cosine checkpoint로 linear probe, finetune, kNN을 각각 최소 실행한다.

```bash
cd dg_net
CKPT=output/harness_refactor/cosine-pretrain/step000002

python engine/linear_probe_train.py \
  --task linear_probe --checkpoint "$CKPT" --dataset Synthetic --device cpu \
  --loop-mode step --total-steps 1 --validate-every-steps 1 --save-every-steps 1 \
  --output-root output/harness_refactor --run-id linear-smoke \
  --img-size 16 --patch-size 8 --batch-size 4 --eval-batch-size 4 \
  --max-samples 16 --num-workers 0 --embed-dim 16 --depth 1 --num-heads 4 \
  --decoder-embed-dim 16 --decoder-num-heads 4 --projection-dim 8

python engine/finetune_train.py \
  --task finetune --checkpoint "$CKPT" --dataset Synthetic --device cpu \
  --loop-mode step --total-steps 1 --validate-every-steps 1 --save-every-steps 1 \
  --output-root output/harness_refactor --run-id finetune-smoke \
  --img-size 16 --patch-size 8 --batch-size 4 --eval-batch-size 4 \
  --max-samples 16 --num-workers 0 --embed-dim 16 --depth 1 --num-heads 4 \
  --decoder-embed-dim 16 --decoder-num-heads 4 --projection-dim 8

python engine/knn_eval.py \
  --task knn_eval --checkpoint "$CKPT" --dataset Synthetic --device cpu \
  --output-root output/harness_refactor --run-id knn-smoke \
  --img-size 16 --patch-size 8 --batch-size 4 --eval-batch-size 4 \
  --max-samples 16 --num-workers 0 --embed-dim 16 --depth 1 --num-heads 4 \
  --decoder-embed-dim 16 --decoder-num-heads 4 --projection-dim 8 --knn-k 3
cd ..
```

실제 CLI가 위 명령과 충돌하면 backward compatibility를 해치지 않는 범위에서 engine을 수정하여
명령을 동작시킨다. 명령을 문서에서 임의로 삭제하지 않는다.

### 10.5 DDP smoke

CUDA device가 2개 이상이면 실행한다.

```bash
cd dg_net
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 engine/dg_train.py \
  --task dg_pretrain \
  --distributed \
  --dataset Synthetic \
  --device cuda \
  --loop-mode step \
  --total-steps 2 \
  --validate-every-steps 1 \
  --save-every-steps 1 \
  --output-root output/harness_refactor \
  --run-id ddp-infonce-smoke \
  --img-size 16 \
  --patch-size 8 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --max-samples 16 \
  --num-workers 0 \
  --embed-dim 16 \
  --depth 1 \
  --num-heads 4 \
  --decoder-embed-dim 16 \
  --decoder-num-heads 4 \
  --projection-dim 8 \
  --loss-mode hybrid \
  --representation-distance infonce \
  --infonce-temperature 0.2
cd ..
```

CUDA device가 2개 미만이면 DDP smoke를 생략하고 `report.md`에 `[미확인]`으로 명시한다.
DDP 미실행만으로 다른 완료 항목을 막지 않는다.

---

## 11. `report.md` 최종 갱신 규칙

구현 완료 후 최초 report에 다음을 추가한다.

```markdown
## 11. 구현 변경 내역
## 12. Cosine/InfoNCE 비교 결과
## 13. 전역성 및 seam metric 결과
## 14. Gradient flow 검증 결과
## 15. ImageNet loader 검증 결과
## 16. Single/DDP/Transfer smoke 결과
## 17. 남은 위험과 미검증 항목
```

성능 우월성을 2-step smoke로 주장하지 않는다. smoke는 실행 가능성과 gradient/API 정확성만
증명한다. 실제 학습 품질 비교가 없으면 `[미확인]`으로 남긴다.

---

## 12. 완료 조건

아래 항목을 모두 만족하면 작업을 종료한다.

- [ ] 최초 production 수정 전에 `report.md`가 작성되었다.
- [ ] 기존 dataset 테스트와 새 model/loss/engine 테스트가 통과한다.
- [ ] cosine/InfoNCE 선택이 config, CLI, API 문서에 연결된다.
- [ ] InfoNCE는 symmetric local in-batch objective로 구현되었다.
- [ ] I/M phase gradient/update isolation 테스트가 통과한다.
- [ ] M phase gradient가 frozen inference 연산을 통해 degradation에 도달한다.
- [ ] VIT/HYBRID cross-patch influence와 seam ratio가 측정되어 report에 기록된다.
- [ ] optional spatial refinement가 default-off로 구현되고 진단에 따른 사용 권장 여부가 기록된다.
- [ ] ImageNet ImageFolder SSL/classification/DDP sampler 테스트가 통과한다.
- [ ] visualization PNG가 확장된 panel과 통계를 포함한다.
- [ ] canonical package API 문서가 실제 코드와 일치한다.
- [ ] single CPU cosine/InfoNCE pretrain smoke가 통과한다.
- [ ] linear probe, finetune, kNN smoke가 통과한다.
- [ ] CUDA 2개 이상이면 DDP smoke가 통과한다. 아니면 미실행 사유가 기록된다.
- [ ] `python -m compileall -q dg_net`과 `git diff --check`가 통과한다.
- [ ] 최종 `report.md`에 변경, 검증, 남은 위험이 기록된다.

---

## 13. 작업 종료 보고 형식

후속 agent의 최종 응답은 다음 형식을 사용한다.

```text
완료한 phase:
변경한 파일:
핵심 설계 결정:
테스트/검증 결과:
생성된 artifact:
남은 미검증 항목:
```

테스트 실패나 미검증 항목을 숨기지 않는다. 복구 가능한 실패가 남아 있으면 종료하지 말고 수정과
검증을 계속한다.
