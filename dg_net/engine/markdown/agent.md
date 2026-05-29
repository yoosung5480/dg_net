# DGNet engine 구현 에이전트 지침

이 문서는 `dg_net/engine` 패키지의 실제 훈련·평가 루프를 구현할 에이전트에게 전달할 harness용 요구사항이다. 현재 engine의 Python 파일들은 구현 전/초기 상태이며, 구현자는 이 문서를 기준으로 `dg_train.py`, `eval.py`, `linear_probe_train.py`, `knn_eval.py`, `finetune_train.py`, `train_config.py`를 완성한다.


## 0. 현재 프로젝트 상태 요약

- 작업 위치: `dg_net/engine/`
- 현재 engine 파일:
  - `dg_train.py`
  - `eval.py`
  - `finetune_train.py`
  - `knn_eval.py`
  - `linear_probe_train.py`
  - `train_config.py`
- 현재 위 Python 파일들은 비어 있으므로, 구현자는 API 문서와 본 지침을 근거로 새로 정의한다.
- 참고 문서:
  - `api/dataset_api.md`: dataset 패키지 공개 API와 `DataConfig` 계약
  - `api/loss_api.md`: loss 패키지 공개 API와 `LossConfig` / DGNet loss 계약
  - `api/model_api.md`: model 패키지 공개 API와 `DGNet` / `DgNetConfig` / 저장·시각화 계약
  - `markdown/env.md`: 실행 환경
  - `markdown/train_DGnet.md`: DGNet 학습 목표와 수식

## 1. engine 패키지의 역할

`dg_net/engine`은 이미 구현된 세 독립 패키지(`dataset`, `loss`, `model`)를 조립하여 실제 학습·평가·시각화·checkpoint 저장을 수행하는 실행 루프 패키지이다.

engine은 다음을 책임진다.

1. CLI/config 입력을 `TrainConfig` 하나로 수집한다.
2. `DataConfig`, `LossConfig`, `DgNetConfig`를 생성한다.
3. DGNet self-supervised pretraining loop를 실행한다.
4. linear probe, kNN eval, fine-tuning 루프를 실행한다.
5. validation loss/score와 sample forward 결과를 주기적으로 저장한다.
6. 재현 가능한 output 디렉터리와 로그를 남긴다.
7. 최소 smoke 조건인 `step=100` 실행이 성공하도록 한다.

## 2. 외부 의존성 / import 경계

### 2.1 engine이 fan-in할 수 있는 패키지

engine은 외부 프로젝트 패키지 중 아래 세 패키지만 직접 사용한다.

1. `dataset`
   - 문서: `api/dataset_api.md`
   - 핵심 API: `DataConfig`, `prepare_dataloader`, `inspect_loader`
2. `loss`
   - 문서: `api/loss_api.md`
   - 핵심 API: `LossConfig`, `ScheduleConfig`, `prepare_dgloss`, `write_loss_report`
3. `model`
   - 문서: `api/model_api.md`
   - 핵심 API: `DGNet`, `DgNetConfig`, `save_sample_visualization`, `write_model_report`, `parameter_summary`

그 외 sibling 패키지나 임의 내부 구현 파일에 직접 의존하지 않는다. 필요한 기능이 공개 API에 없으면 먼저 문서와 공개 심볼을 확인하고, engine 내부에서 얇은 adapter를 작성한다.

### 2.2 engine 내부 fan-out 규칙

engine 내부에서 다른 engine 모듈이 import해도 되는 파일은 다음 두 개뿐이다.

- `train_config.py`
- `eval.py`

나머지 실행 파일은 독립 executable script로 유지한다.

허용 예:

```python
# dg_train.py, linear_probe_train.py, finetune_train.py 등
from train_config import TrainConfig, get_trainer
from eval import evaluate_dgnet, save_eval_artifacts
```

금지 예:

```python
# 금지: train script끼리 서로 import하지 않는다.
from dg_train import DGTrainer
from linear_probe_train import LinearProbeTrainer
```

## 3. 구현 대상 파일별 책임

### 3.1 `train_config.py`

`train_config.py`는 engine의 유일한 설정 조립 표면이다. 모든 train/eval script는 이 파일의 config만 보고 실행되어야 한다.

필수 책임:

- `TrainConfig` dataclass 정의
- CLI `argparse.Namespace`, dict, 또는 명시 인자에서 `TrainConfig` 생성
- `DataConfig`, `LossConfig`, `DgNetConfig` 생성 메서드 제공
- train mode, loop mode, output policy, optimizer policy, device policy 보관
- `get_trainer(train_cfg)` factory 제공
- 과거 문서의 오타 호환이 필요하면 `TrinConfig = TrainConfig` alias 제공 가능

권장 public surface:

```python
@dataclass
class TrainConfig:
    task: str                  # "dg_pretrain" | "linear_probe" | "knn_eval" | "finetune" | "eval"
    loop_mode: str             # "step" | "epoch"
    total_steps: int
    total_epochs: int
    validate_every_steps: int
    validate_every_epochs: int
    save_every_steps: int
    save_every_epochs: int
    output_root: str
    run_name: str | None
    seed: int
    device: str                # "auto" | "cpu" | "cuda" | "cuda:0" ...

    # dataset/model/loss/optimizer hyperparameters도 보관

    def build_data_config(self, *, split: str | None = None, mode: str | None = None) -> DataConfig: ...
    def build_loss_config(self) -> LossConfig: ...
    def build_model_config(self) -> DgNetConfig: ...
```

필수 factory 계약:

```python
train_cfg = TrainConfig(args)
dg_trainer = get_trainer(train_cfg)
dg_trainer.run()
```

또는 dataclass 생성 관례상 `TrainConfig.from_args(args)`를 추가해도 된다. 단, 모든 script에서 하나의 관례로 통일한다.

### 3.2 `dg_train.py`

DGNet self-supervised pretraining 실행 파일이다.

필수 동작:

- `TrainConfig` 하나로 모든 설정 결정
- `dataset.prepare_dataloader(DataConfig(..., mode="ssl"))` 사용
- STL10 SSL loader의 반환 형식인 `(views, labels)`에서 학습 입력 view를 명시적으로 선택
- `DGNet(DgNetConfig)` 생성
- `prepare_dgloss(LossConfig)` 생성
- DGNet의 두 objective를 교대 최적화
  - I phase: `model.degradation` freeze, `encoder/decoder/projection` train
  - M phase: `model.degradation` train, `encoder/decoder/projection` freeze
- 각 phase는 별도 optimizer 사용
- `values.inference_loss`와 `values.degradation_loss`를 각 phase에서 분리해 backward
- `values.combined_loss`는 로그/smoke용으로만 사용하고 단일 optimizer objective로 쓰지 않음
- `step=100` smoke run이 CPU 또는 CUDA 환경에서 완료되어야 함

교대 최적화의 기준 코드는 `api/model_api.md` 9장과 `api/loss_api.md`의 DGNet 교대 최적화 예시를 따른다.

### 3.3 `eval.py`

`eval.py`는 engine 내부에서 fan-out이 허용되는 평가 유틸리티 모듈이다.

필수 책임:

- validation loss 계산
- validation score 계산
- sample forward 시각화 저장
- loss/score curve 저장
- `train.csv`로부터 시각화를 재생성할 수 있게 metric record append 또는 write 지원
- train script에서 호출 가능한 함수 제공

필수 결과:

- validation loss: 수치로 반환하고 로그/CSV에 기록
- validation score: task별 적절한 score를 반환하고 로그/CSV에 기록
  - DGNet pretrain: reconstruction metric, residual budget metric, degradation target metric 등
  - classification 계열: accuracy/top-k 등
  - kNN eval: kNN classification accuracy
- 이미지 시각화:
  - DGNet: original / degraded / reconstruction 포함
  - classification 계열: 필요 시 prediction summary 또는 confusion/accuracy figure

권장 public surface:

```python
@torch.no_grad()
def evaluate_dgnet(model, loss_fn, loader, device, *, step: int | None = None) -> dict[str, float]: ...

def save_eval_artifacts(...): ...
def append_metrics_csv(csv_path, row: dict): ...
def plot_loss_curve(csv_path, output_path): ...
def plot_accuracy_curve(csv_path, output_path): ...
```

### 3.4 `linear_probe_train.py`

DGNet encoder representation을 고정하고 linear classifier만 학습한다.

필수 동작:

- classification mode `DataConfig` 사용
- pretrained DGNet checkpoint load 지원
- DGNet encoder/backbone은 freeze
- linear head만 optimizer로 업데이트
- validation accuracy와 loss를 주기적으로 저장
- `accuracy.png`, `loss_curve.png`, `train.csv` 생성

### 3.5 `knn_eval.py`

DGNet representation에 대한 kNN 평가 실행 파일이다.

필수 동작:

- classification train split에서 feature bank 생성
- classification test/val split에서 query feature 생성
- kNN accuracy 계산
- checkpoint 또는 현재 run output을 입력으로 받을 수 있음
- 결과를 `.log`와 `train.csv` 또는 별도 eval csv에 기록

### 3.6 `finetune_train.py`

DGNet encoder와 classification head를 함께 fine-tuning하는 실행 파일이다.

필수 동작:

- classification mode `DataConfig` 사용
- pretrained DGNet checkpoint load 지원
- encoder를 포함한 일부/전체 module unfreeze 정책 제공
- classification loss로 학습
- validation accuracy/loss, checkpoint, plot 저장

## 4. Config 요구사항

### 4.1 `TrainConfig`가 반드시 결정해야 하는 항목

`TrainConfig`는 아래 항목을 모두 보관하거나 생성할 수 있어야 한다.

- task 종류
  - `dg_pretrain`
  - `linear_probe`
  - `knn_eval`
  - `finetune`
  - `eval`
- loop 제어 방식
  - `loop_mode="step"`
  - `loop_mode="epoch"`
- 총 실행량
  - step mode: `total_steps`
  - epoch mode: `total_epochs`
- 저장/검증 주기
  - step mode: `validate_every_steps`, `save_every_steps`
  - epoch mode: `validate_every_epochs`, `save_every_epochs`
- output 경로
  - `output_root`
  - `run_name` 또는 실행시간 기반 run id
- 재현성
  - seed
  - deterministic 옵션이 필요하면 명시
- device
  - `auto`일 때 CUDA 가능하면 CUDA, 아니면 CPU
- dataset 설정
  - `DataConfig`의 모든 주요 필드
- model 설정
  - `DgNetConfig`의 주요 필드
- loss 설정
  - `LossConfig`, `ScheduleConfig`의 주요 필드
- optimizer/scheduler 설정
  - inference optimizer
  - degradation optimizer
  - classifier optimizer
- checkpoint 설정
  - resume path
  - pretrained path
  - save/load map_location

### 4.2 step mode와 epoch mode

#### step mode

- 훈련 루프는 오직 global step 수로 종료된다.
- epoch 수는 종료 조건에 영향을 주지 않는다.
- DataLoader가 끝나면 iterator를 다시 생성해 다음 step을 계속 진행한다.
- `validate_every_steps`마다 validation 실행
- `save_every_steps`마다 checkpoint와 sample artifact 저장
- 예: `total_steps=100`이면 정확히 optimizer step 기준 100 step 수행

#### epoch mode

- 훈련 루프는 오직 epoch 수로 종료된다.
- step 총량은 종료 조건에 영향을 주지 않는다.
- `validate_every_epochs`마다 validation 실행
- `save_every_epochs`마다 checkpoint와 sample artifact 저장
- DDP sampler를 쓰면 epoch 시작 시 `loader.sampler.set_epoch(epoch)` 호출

### 4.3 dataset / model pixel 계약

시각화와 DGNet degradation 해석을 위해 다음 계약을 명확히 해야 한다.

- `DataConfig.img_size == DgNetConfig.IMG_SIZE` 보장
- `DgNetConfig.IMG_SIZE % DgNetConfig.PATCH_SIZE == 0` 보장
- `DgNetConfig.CLAMP_DEGRADED=True`를 사용한다면 dataset normalization은 보통 `mean=(0,0,0)`, `std=(1,1,1)`로 두어 `[0,1]` pixel 의미를 유지한다.
- ImageNet normalization을 사용하는 경우 `CLAMP_DEGRADED=False`를 기본으로 두고, sample visualization 해석에 주석/로그를 남긴다.

## 5. 외부 패키지 API 사용 계약

### 5.1 dataset

공개 API만 사용한다.

```python
from dataset import DataConfig, prepare_dataloader, inspect_loader
```

SSL pretraining loader:

```python
cfg = DataConfig(
    dataset="STL10",
    mode="ssl",
    data_path="/home/jeongyuseong/바탕화면/datasets",
    ssl_num_views=2,
    img_size=32,
    batch_size=4,
    num_workers=0,
    drop_last=True,
    pin_memory=False,
    max_samples=400,
    use_augmentation=False,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
)
loader = prepare_dataloader(cfg)
views, labels = next(iter(loader))
images = views[0]
```

classification loader:

```python
cfg = DataConfig(
    dataset="CIFAR100",
    mode="classification",
    split="test",
    img_size=32,
    use_augmentation=False,
    batch_size=128,
    drop_last=False,
)
loader = prepare_dataloader(cfg)
images, labels = next(iter(loader))
```

### 5.2 model

공개 API만 사용한다.

```python
from model.dg_model import (
    DGNet,
    DgNetConfig,
    save_sample_visualization,
    write_model_report,
    parameter_summary,
)
```

필수 사용 계약:

- checkpoint 저장은 가능하면 `model.save(output_dir)` 사용
- load는 `DGNet.load(saved_dir, map_location=...)` 사용
- sample forward 이미지는 `save_sample_visualization(output, path)` 사용
- 학습 시작 시 첫 batch sample에 대해 `write_model_report(model, sample, output_dir)` 실행
- 동일한 첫 batch sample을 주기적 checkpoint마다 다시 forward하여 변화 추적 이미지 저장

### 5.3 loss

공개 API만 사용한다.

```python
from loss.loss_config import LossConfig, ScheduleConfig
from loss.loss import prepare_dgloss, write_loss_report
```

필수 사용 계약:

- `loss_fn = prepare_dgloss(loss_cfg)`
- forward 결과는 `values = loss_fn(output, step=global_step)`
- I phase는 `values.inference_loss.backward()`
- M phase는 `values.degradation_loss.backward()`
- `values.metrics()`와 `values.effective_weights`를 CSV/log에 기록
- 필요 시 `write_loss_report(values, path)`로 step별 loss report 저장

## 6. 출력 디렉터리와 artifact 규칙

모든 실행은 `output/{run_id}/` 아래에 재현 가능한 산출물을 남긴다. `run_id`는 기본적으로 실행시간 기반 문자열이며, 사용자가 `run_name`을 주면 함께 포함해도 된다.

### 6.1 step mode 구조

```text
output/
└── {run_id}/
    ├── config.json
    ├── train.csv
    ├── loss_curve.png
    ├── accuracy.png              # classification/linear probe/fine-tune/kNN에서 생성, DG pretrain에서는 optional
    ├── run.log
    ├── step000000/
    │   ├── model.pt
    │   ├── config.json
    │   ├── step000000.png        # original / degraded / reconstruction
    │   ├── dg_model.log
    │   └── loss.log
    ├── step000100/
    │   ├── model.pt
    │   ├── config.json
    │   ├── step000100.png
    │   ├── dg_model.log
    │   └── loss.log
    └── ...
```

### 6.2 epoch mode 구조

```text
output/
└── {run_id}/
    ├── config.json
    ├── train.csv
    ├── loss_curve.png
    ├── accuracy.png
    ├── run.log
    ├── epoch000001/
    │   ├── model.pt
    │   ├── config.json
    │   ├── epoch000001.png
    │   ├── dg_model.log
    │   └── loss.log
    └── ...
```

### 6.3 `train.csv` 최소 컬럼

DGNet pretraining 최소 컬럼:

```text
run_id,task,loop_mode,epoch,step,split,phase,train_loss,validation_loss,validation_score,
inference_loss,degradation_loss,reconstruction_distance,target_distance,mask_budget_distance,damage_kl,
alpha_inference,alpha_target,lambda_budget,lambda_reg,lr_inference,lr_degradation,elapsed_sec
```

classification 계열 최소 컬럼:

```text
run_id,task,loop_mode,epoch,step,split,train_loss,validation_loss,validation_score,accuracy,top1,top5,lr,elapsed_sec
```

없는 값은 빈 문자열 또는 `nan`으로 남기되, plot 함수가 실패하지 않도록 처리한다.

### 6.4 로그 요구사항

- `run.log`는 전체 실행 요약과 환경, config, dataset inspect 결과, checkpoint 경로를 기록한다.
- 각 `step*/` 또는 `epoch*/` 하위 `.log`는 해당 시점의 sample forward 결과, validation 결과, loss metrics를 기록한다.
- 로그는 auto-research/harness가 원인을 추적할 수 있을 정도로 구체적이어야 한다.
- 예외 발생 시 stack trace와 마지막 성공 step/epoch를 남긴다.

## 7. 훈련 루프 세부 요구사항

### 7.1 공통 trainer 인터페이스

각 trainer는 최소한 다음 인터페이스를 갖는다.

```python
class BaseTrainer:
    def __init__(self, cfg: TrainConfig): ...
    def run(self) -> None: ...
    def train_step(self, batch) -> dict[str, float]: ...
    def validate(self) -> dict[str, float]: ...
    def save_checkpoint(self, tag: str) -> Path: ...
```

구현 위치는 각 script 내부여도 되지만, 다른 script에서 import하지 않도록 한다. 공유가 꼭 필요한 함수는 `train_config.py` 또는 `eval.py`에만 둔다.

### 7.2 첫 batch 고정 sample 추적

모든 train script는 학습 시작 시 첫 번째 batch에서 sample batch를 하나 고정한다.

요구사항:

1. 첫 batch를 device에 올려 `fixed_sample`로 보관한다.
2. 학습 전 `step000000/` 또는 `epoch000000/`에 forward 결과 저장
3. 이후 save 주기마다 같은 `fixed_sample`을 forward
4. `save_sample_visualization()` 또는 `write_model_report()`로 변화를 저장
5. 파일명은 현재 tag와 일치시킨다. 예: `step000100.png`, `epoch000005.png`

### 7.3 checkpoint 저장

- DGNet은 `model.save(tag_dir)`를 우선 사용한다.
- classifier head 등 engine에서 추가한 module이 있으면 별도 `checkpoint.pt`에 함께 저장한다.
- 최소 저장 항목:
  - model weights
  - model config
  - train config
  - optimizer state
  - scheduler state가 있으면 포함
  - global step / epoch
  - best validation metric
- load/resume 시 config mismatch를 감지하고 명확한 에러를 낸다.

### 7.4 metric plotting

- `loss_curve.png`는 `train.csv`로부터 재생성 가능해야 한다.
- classification 계열은 `accuracy.png`를 생성한다.
- DGNet pretraining은 accuracy가 없으므로 `accuracy.png`는 생략 가능하지만, harness 호환을 위해 빈 plot 대신 `validation_score` curve를 저장해도 된다.
- plot 실패가 학습 전체를 중단시키지 않도록 로그를 남기고 계속 진행할 수 있다. 단, 최종 smoke test에서는 plot 생성 여부를 검증한다.

## 8. CLI 요구사항

각 실행 파일은 `python <script>.py --...` 형태로 직접 실행 가능해야 한다.

필수 공통 인자:

```text
--task
--loop-mode {step,epoch}
--total-steps
--total-epochs
--validate-every-steps
--validate-every-epochs
--save-every-steps
--save-every-epochs
--output-root
--run-name
--seed
--device
--data-path
--dataset
--img-size
--batch-size
--num-workers
--max-samples
--use-augmentation / --no-use-augmentation
--checkpoint
--resume
```

DGNet pretrain 추가 인자:

```text
--patch-size
--embed-dim
--depth
--num-heads
--decoder-embed-dim
--decoder-num-heads
--projection-dim
--dg-architect {VIT,CNN,HYBRID}
--cnn-architect {RESNET,CONVNEXT}
--clamp-degraded
--lr-inference
--lr-degradation
--weight-decay
--tau-deg
--beta-mask
--lambda-budget
--lambda-reg
```

classification 계열 추가 인자:

```text
--num-classes
--lr
--classifier-lr
--encoder-lr
--freeze-encoder
--knn-k
--temperature
```

## 9. smoke / harness 성공 기준

구현 완료 판정은 최소 다음 명령 또는 동등한 harness로 확인한다.

### 9.1 DGNet pretrain 100 step

```bash
python dg_train.py \
  --loop-mode step \
  --total-steps 100 \
  --validate-every-steps 50 \
  --save-every-steps 100 \
  --dataset STL10 \
  --data-path /home/jeongyuseong/바탕화면/datasets \
  --img-size 32 \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 400 \
  --no-use-augmentation \
  --output-root output
```

성공 조건:

- exit code 0
- `output/{run_id}/config.json` 존재
- `output/{run_id}/train.csv` 존재
- `output/{run_id}/loss_curve.png` 존재
- `output/{run_id}/run.log` 존재
- `output/{run_id}/step000100/model.pt` 존재
- `output/{run_id}/step000100/config.json` 존재
- `output/{run_id}/step000100/step000100.png` 존재
- `train.csv`에 step 100 row 존재
- validation loss와 validation score가 finite numeric

### 9.2 epoch mode smoke

```bash
python dg_train.py \
  --loop-mode epoch \
  --total-epochs 1 \
  --validate-every-epochs 1 \
  --save-every-epochs 1 \
  --dataset STL10 \
  --data-path /home/jeongyuseong/바탕화면/datasets \
  --img-size 32 \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 32 \
  --no-use-augmentation \
  --output-root output
```

성공 조건:

- exit code 0
- `epoch000001/` artifact 존재
- epoch mode에서 step 종료 조건을 사용하지 않음

### 9.3 import boundary smoke

- `dg_train.py`, `linear_probe_train.py`, `finetune_train.py`, `knn_eval.py`는 서로 import하지 않는다.
- engine 내부 import는 `train_config.py`, `eval.py`만 허용한다.
- dataset/loss/model은 공개 API만 사용한다.

## 10. 구현 우선순위

1. `train_config.py`
   - `TrainConfig`
   - CLI parser helper
   - `build_data_config()`, `build_loss_config()`, `build_model_config()`
   - `get_trainer()`
2. `eval.py`
   - DGNet validation
   - CSV append
   - plot 저장
   - sample artifact 저장 helper
3. `dg_train.py`
   - step mode 100-step smoke 우선
   - epoch mode 추가
   - checkpoint/resume 추가
4. `linear_probe_train.py`
5. `knn_eval.py`
6. `finetune_train.py`

최초 구현은 DGNet pretraining smoke가 통과하는 최소 기능을 우선 완성한다. 그 후 classification transfer 루프를 확장한다.

## 11. 주의사항과 금지사항
- train script끼리 서로 import하지 않는다.
- `combined_loss` 하나로 DGNet의 I/M objective를 동시에 최적화하지 않는다.
- I 모델 그래디언트 backward에서는 M 모델에는 그래디언트 플로우가 영향을 주지 않아야한다.
- M 모델 그래디언트 backward에서는 I 모델에는 그래디언트 플로우가 영향을 주지 않아야한다.
- step mode에서 epoch 수로 종료하지 않는다.
- epoch mode에서 total step 수로 종료하지 않는다.
- sample visualization 없이 checkpoint만 저장하지 않는다.
- `train.csv` 없이 png만 저장하지 않는다.
- output artifact 경로를 임의로 흩뿌리지 않는다.
- CUDA가 없을 때 실패하지 않도록 `device=auto`는 CPU fallback을 지원한다.
- `DataConfig.img_size`와 `DgNetConfig.IMG_SIZE` mismatch를 조용히 허용하지 않는다.
- `DgNetConfig.CLAMP_DEGRADED=True`와 normalized input 조합은 명시적으로 경고하거나 config에서 막는다.

## 12. 환경 정보

현재 문서 기준 환경(`markdown/env.md`)