# agent.md

## 0. 작업 목적

현재 DGNet 훈련 프레임워크를 새 loss 정의와 term-wise scheduler 구조에 맞춰 재구현한다.

이번 작업의 목표는 단순히 `loss.py`만 수정하는 것이 아니다. 아래 항목이 모두 연결되어야 한다.

1. 새로운 loss objective 적용
2. 각 loss 단항식별 scheduler 적용
3. representation only / reconstruction only / hybrid loss ablation 지원
4. DDP 기반 multi-GPU 학습 지원
5. pretrain 이후 linear probe 전이학습까지 bash script 하나로 실행 지원
6. 각 패키지별 `api.md` 갱신
7. 모든 smoke test와 실행 결과를 `output/`에 남김

---

## 1. 연구/훈련 목표 요약

DGNet은 Self-Supervised Learning을 위한 adversarial problem generator이다.

전체 네트워크는 두 모델로 구성된다.

- `I`: Inference Network / AutoEncoder / Representation Learner
- `M`: Degradation Network / DGNet / Problem Generator

기본 흐름은 다음과 같다.

$$
x \in \mathbb{R}^{B \times C \times H \times W}
$$

$$
r = M(x)
$$

$$
x_{\text{dg}} = x - r
$$

$$
z_x = I_{\text{proj}}(I_{\text{enc}}(x))
$$

$$
z_{\text{dg}} = I_{\text{proj}}(I_{\text{enc}}(x_{\text{dg}}))
$$

$$
\hat{x}_{\text{dg}} = I_{\text{dec}}(I_{\text{enc}}(x_{\text{dg}}))
$$

핵심 연구 질문은 다음이다.

```text
SSL에서 좋은 representation은 좋은 learner만으로 만들어지는가,
아니면 learner가 푸는 문제를 만드는 problem generator의 품질도 중요한가?
```

따라서 DGNet은 단순 noise generator가 아니라 `I`를 훈련시키기 위한 문제생성기이다.

---

## 2. 프로젝트 환경

기준 환경은 다음과 같다.

```text
environment: pretext_task_gen
python: 3.10.20
torch: 2.2.2+cu121
torchvision: 0.17.2+cu121
cuda_available: True
```

작업자는 위 환경에서 동작하는 코드를 작성해야 한다.

---

## 3. 최종 디렉토리 구조 목표

```text
root/
    src/
        model/
            cnn.py
            vit.py
            hybrid.py
            dg_model.py
            model_api.md
            output/
                *.log
                *.png

        loss/
            loss.py
            loss_config.py
            loss_api.md
            output/
                loss_config_summary.log
                loss_term_onoff_test.log
                loss_scheduler_test.log
                loss_forward_shape_test.log
                loss_ablation_mode_test.log

        dataset/
            pretrain_datasets.py
            cls_datasets.py
            dataset_api.md
            output/
                *.txt
                *.log

        engine/
            train_config.py
            eval.py
            dg_train.py
            linear_probe_train.py
            finetune_train.py
            knn_eval.py
            engine_api.md
            output/
                *.log

        scripts/
            run_dgnet_pipeline_single_gpu.bash
            run_dgnet_pipeline_ddp.bash
            smoke_loss_scheduler.bash

    output/
        runs...
```

---

## 4. Fan-out / Fan-in 규칙

`loss.py`와 `loss_config.py`는 독립 모듈이어야 한다.

허용되는 의존 방향:

```text
engine -> loss
engine -> model
engine -> dataset
```

금지되는 의존 방향:

```text
loss -> engine
loss -> model
loss -> dataset
```

loss 패키지는 오직 tensor를 입력받아 loss를 계산해야 한다.

---

## 5. 최종 loss objective

최종 프레임워크는 다음 세 가지 실험 모드를 모두 지원한다.

1. representation only
2. reconstruction only
3. representation + reconstruction hybrid

이를 위해 모든 loss term은 독립적인 coefficient와 scheduler를 가져야 한다.

---

## 6. Inference Network loss

Inference Network loss는 다음과 같다.

$$
\mathcal{L}_{I}
=
\alpha_{\text{rec}}^{I}
D_{\text{rec}}^{I}
+
\alpha_{\text{rep}}^{I}
D_{\text{rep}}^{I}
$$

### 6.1 Inference reconstruction loss

$$
D_{\text{rec}}^{I}
=
\frac{1}{BCHW}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{\text{dg},bchw}-x_{bchw})^2
$$

### 6.2 Inference representation loss

$$
D_{\text{rep}}^{I}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{rep}}^{(b)}
$$

cosine distance를 사용할 경우:

$$
D_{\text{rep}}^{(b)}
=
1
-
\frac{
\langle z_{x,b}, z_{\text{dg},b} \rangle
}{
\|z_{x,b}\|_2 \|z_{\text{dg},b}\|_2 + \epsilon
}
$$

`I`는 이 값을 줄이는 방향으로 학습된다.

---

## 7. Degradation Network loss

DGNet `M`의 전체 loss는 다음과 같다.

$$
\mathcal{L}_{M}
=
\alpha_{\text{target}}^{M}
D_{\text{target}}^{M}
+
\lambda_{\text{budget}}^{M}
D_{\text{budget}}^{M}
-
\lambda_{\text{KL}}^{M}
D_{\text{KL}}^{M}
-
\alpha_{\text{rep}}^{M}
D_{\text{rep}}^{M}
+
\lambda_{\text{sparse}}^{M}
D_{\text{sparse}}^{M}
$$

optimizer는 minimize를 수행한다.

$$
\min_M \mathcal{L}_M
$$

따라서 DGNet이 키워야 하는 항은 음수 부호로 들어간다.

```text
- lambda_KL * D_KL
- alpha_rep * D_rep
```

---

## 8. image-wise scalar threshold 규칙

threshold는 tensor가 아니라 image-wise scalar로 사용한다.

금지되는 방식:

$$
\left|
\operatorname{mean}_{b,c,h,w}(\cdot)-\tau
\right|
$$

사용해야 하는 방식:

$$
\operatorname{mean}_{b}
\left(
\left|
\operatorname{mean}_{c,h,w}(\cdot)-\tau
\right|
\right)
$$

이 규칙은 reconstruction target과 budget 모두에 적용한다.

---

## 9. DGNet target distance

이미지별 reconstruction distance:

$$
D_{\text{rec}}^{M,(b)}
=
\frac{1}{CHW}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
(\hat{x}_{\text{dg},bchw}-x_{bchw})^2
$$

target distance:

$$
D_{\text{target}}^{M}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left|
D_{\text{rec}}^{M,(b)}
-
\tau_{\text{deg}}
\right|
$$

DGNet은 각 이미지마다 reconstruction difficulty를 `tau_deg` 근처로 맞춰야 한다.

---

## 10. DGNet budget distance

이미지별 residual energy:

$$
D_{\text{budget-energy}}^{M,(b)}
=
\frac{1}{CHW}
\sum_{c=1}^{C}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
r_{bchw}^{2}
$$

budget distance:

$$
D_{\text{budget}}^{M}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left|
D_{\text{budget-energy}}^{M,(b)}
-
\tau_{\text{budget}}
\right|
$$

DGNet은 각 이미지마다 residual energy budget을 `tau_budget` 근처로 맞춰야 한다.

---

## 11. channel-wise spatial KL

channel 평균 damage map을 사용하지 않는다.

금지되는 방식:

$$
d_{bhw}
=
\frac{1}{C}
\sum_c |r_{bchw}|
$$

사용해야 하는 방식:

$$
d_{bc,hw}
=
|r_{bchw}|
$$

flatten:

$$
d_{bc} \in \mathbb{R}^{HW}
$$

채널별 damage probability:

$$
p_{bc,i}
=
\frac{d_{bc,i}}
{\sum_{j=1}^{HW}d_{bc,j}+\epsilon}
$$

uniform spatial probability:

$$
u_i = \frac{1}{HW}
$$

채널별 이미지별 KL:

$$
D_{\text{KL}}^{(b,c)}
=
\sum_{i=1}^{HW}
p_{bc,i}
\log
\frac{p_{bc,i}}{u_i}
$$

최종 KL:

$$
D_{\text{KL}}^{M}
=
\frac{1}{BC}
\sum_{b=1}^{B}
\sum_{c=1}^{C}
D_{\text{KL}}^{(b,c)}
$$

DGNet loss에는 다음처럼 들어간다.

$$
-\lambda_{\text{KL}}^{M}D_{\text{KL}}^{M}
$$

주의:

```text
D_KL은 semantic mask를 보장하지 않는다.
D_KL은 uniform brightness degradation shortcut 방지용 regularizer이다.
반드시 ablation 가능해야 한다.
```

---

## 12. representation loss

DGNet과 Inference Network는 동일한 representation distance를 사용한다.

$$
D_{\text{rep}}^{M}
=
\frac{1}{B}
\sum_{b=1}^{B}
D_{\text{rep}}^{(b)}
$$

`I`는 이를 줄인다.

$$
\min_I D_{\text{rep}}^{I}
$$

`M`은 이를 키운다.

$$
\max_M D_{\text{rep}}^{M}
$$

minimize loss에서는 다음과 같이 구현한다.

$$
-\alpha_{\text{rep}}^{M}D_{\text{rep}}^{M}
$$

---

## 13. sparsity penalty

ADIOS-style sparsity penalty는 명시적 mask가 있을 때만 사용한다.

현재 DGNet이 residual만 출력한다면 기본적으로 꺼둔다.

$$
\lambda_{\text{sparse}}^{M}=0
$$

명시적 mask가 있는 경우:

$$
p^{(b)}
=
\sin
\left(
\frac{\pi}{HW}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
m_{bhw}
\right)^{-1}
$$

$$
D_{\text{sparse}}^M
=
\frac{1}{B}
\sum_{b=1}^{B}p^{(b)}
$$

---

## 14. LossConfig 설계

기존처럼 각 weight를 float로만 두면 안 된다.

각 term은 반드시 `TermWeightConfig`로 관리한다.

```python
from dataclasses import dataclass
from typing import Literal

ScheduleKind = Literal["none", "linear", "cosine"]

@dataclass(frozen=True)
class TermWeightConfig:
    """
    단일 loss term의 coefficient와 scheduler 정책.

    weight:
        base coefficient.
        0.0이면 해당 term은 완전히 비활성화된다.

    schedule:
        none / linear / cosine.

    start:
        scheduler multiplier 시작값.

    end:
        scheduler multiplier 종료값.

    total_steps:
        end 값에 도달하는 step 수.
    """
    weight: float = 1.0
    schedule: ScheduleKind = "none"
    start: float = 1.0
    end: float = 1.0
    total_steps: int = 10000
```

`LossConfig` 예시:

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class LossConfig:
    # Inference terms
    inference_rec: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=1.0, schedule="none"))
    inference_rep: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=0.0, schedule="none"))

    # DGNet terms
    dg_target: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=1.0, schedule="none"))
    dg_budget: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=1.0, schedule="none"))
    dg_kl: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=0.0, schedule="none"))
    dg_rep: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=0.0, schedule="none"))
    dg_sparse: TermWeightConfig = field(default_factory=lambda: TermWeightConfig(weight=0.0, schedule="none"))

    # Thresholds
    tau_deg: float = 0.05
    tau_budget: float = 0.01

    # Representation distance
    representation_distance: str = "cosine"
    eps: float = 1e-8

    # Reduction
    inference_reduction: str = "mean"
    dg_reduction: str = "mean"
```

중요: dataclass에서 mutable/default object 문제를 피하기 위해 `field(default_factory=...)`를 사용한다.

---

## 15. scheduler 계산 규칙

실제 weight는 다음과 같이 계산한다.

$$
w_t = w_{\text{base}} \cdot s(t)
$$

단, base weight가 0이면 scheduler 설정과 무관하게 항상 0이어야 한다.

$$
w_{\text{base}}=0 \Rightarrow w_t=0
$$

linear:

$$
s(t)
=
s_{\text{start}}
+
(s_{\text{end}}-s_{\text{start}})
\frac{t}{T}
$$

cosine:

$$
s(t)
=
s_{\text{end}}
+
\frac{1}{2}(s_{\text{start}}-s_{\text{end}})
\left(1+\cos\frac{\pi t}{T}\right)
$$

`t`는 `[0, T]`로 clamp한다.

---

## 16. weight가 0인 term 처리 규칙

권장 구현:

```python
w = effective_weight(self.config.dg_kl, step)

if w == 0.0:
    dg_kl = original.new_zeros(())
    weighted_dg_kl = original.new_zeros(())
else:
    dg_kl = self.channel_wise_spatial_kl(residual)
    weighted_dg_kl = -w * dg_kl
```

금지 구현:

```python
dg_kl = self.channel_wise_spatial_kl(residual)
weighted_dg_kl = 0.0 * dg_kl
```

계수가 0인 term은 계산 자체를 하지 않아야 한다.

---

## 17. loss.py 구현 구조

권장 class 구조:

```python
class InferenceReconstructionLoss(nn.Module):
    ...

class RepresentationDistance(nn.Module):
    ...

class DGTargetDistance(nn.Module):
    ...

class DGBudgetDistance(nn.Module):
    ...

class ChannelWiseSpatialKL(nn.Module):
    ...

class SparsePenalty(nn.Module):
    ...

class DGLoss(nn.Module):
    ...

def prepare_dgloss(config: LossConfig) -> DGLoss:
    ...
```

---

## 18. DGLoss 입력 인터페이스

`DGLoss`는 최소한 다음 입력을 받아야 한다.

```python
loss_out = loss_fn(
    original=x,
    degraded=x_dg,
    residual=residual,
    reconstruction=x_dg_recon,
    z_original=z_x,
    z_degraded=z_dg,
    mask=None,
    step=global_step,
)
```

모델 output 객체를 받는 adapter도 제공해도 된다.

```python
loss_out = loss_fn.from_output(model_output, step=global_step)
```

---

## 19. DGLoss 출력 인터페이스

loss는 scalar만 반환하면 안 된다.

각 raw term, weighted term, 현재 effective weight를 모두 확인할 수 있어야 한다.

```python
@dataclass
class DGLossOutput:
    inference_loss: torch.Tensor
    degradation_loss: torch.Tensor
    combined_loss: torch.Tensor

    inference_rec: torch.Tensor
    inference_rep: torch.Tensor

    dg_target: torch.Tensor
    dg_budget: torch.Tensor
    dg_kl: torch.Tensor
    dg_rep: torch.Tensor
    dg_sparse: torch.Tensor

    weighted_inference_rec: torch.Tensor
    weighted_inference_rep: torch.Tensor

    weighted_dg_target: torch.Tensor
    weighted_dg_budget: torch.Tensor
    weighted_dg_kl: torch.Tensor
    weighted_dg_rep: torch.Tensor
    weighted_dg_sparse: torch.Tensor

    effective_weights: dict

    def metrics(self) -> dict:
        ...
```

`combined_loss`는 logging/smoke test용이다. 실제 학습에서는 phase별 loss를 사용한다.

---

## 20. 실험 모드 preset

`LossConfig`는 최소한 다음 preset 생성자를 제공해야 한다.

```python
LossConfig.representation_only(...)
LossConfig.reconstruction_only(...)
LossConfig.hybrid(...)
```

### 20.1 representation only

```python
loss_cfg = LossConfig(
    inference_rec=TermWeightConfig(weight=0.0, schedule="none"),
    inference_rep=TermWeightConfig(weight=1.0, schedule="none"),

    dg_target=TermWeightConfig(weight=0.0, schedule="none"),
    dg_budget=TermWeightConfig(weight=0.0, schedule="none"),
    dg_kl=TermWeightConfig(weight=0.0, schedule="none"),
    dg_rep=TermWeightConfig(weight=1.0, schedule="none"),
    dg_sparse=TermWeightConfig(weight=0.0, schedule="none"),
)
```

### 20.2 reconstruction only

```python
loss_cfg = LossConfig(
    inference_rec=TermWeightConfig(weight=1.0, schedule="none"),
    inference_rep=TermWeightConfig(weight=0.0, schedule="none"),

    dg_target=TermWeightConfig(weight=1.0, schedule="none"),
    dg_budget=TermWeightConfig(weight=1.0, schedule="none"),
    dg_kl=TermWeightConfig(weight=0.01, schedule="none"),
    dg_rep=TermWeightConfig(weight=0.0, schedule="none"),
    dg_sparse=TermWeightConfig(weight=0.0, schedule="none"),
)
```

### 20.3 hybrid with scheduled terms

```python
loss_cfg = LossConfig(
    inference_rec=TermWeightConfig(weight=1.0, schedule="none"),
    inference_rep=TermWeightConfig(weight=1.0, schedule="cosine", start=0.0, end=1.0, total_steps=10000),

    dg_target=TermWeightConfig(weight=1.0, schedule="none"),
    dg_budget=TermWeightConfig(weight=1.0, schedule="linear", start=0.0, end=1.0, total_steps=5000),
    dg_kl=TermWeightConfig(weight=0.01, schedule="cosine", start=0.0, end=1.0, total_steps=10000),
    dg_rep=TermWeightConfig(weight=1.0, schedule="none"),
    dg_sparse=TermWeightConfig(weight=0.0, schedule="none"),
)
```

---

## 21. CLI 요구사항

기존 CLI는 다음 구형 인자를 지원했다.

```text
--alpha-inference
--alpha-target
--lambda-budget
--lambda-reg
--tau-deg
--beta-mask
--loss-reduction
```

새 CLI는 term-wise config를 지원해야 한다.

최소 지원해야 할 인자:

```text
--loss-mode {representation_only,reconstruction_only,hybrid,custom}

--inference-rec-weight FLOAT
--inference-rec-schedule {none,linear,cosine}
--inference-rec-start FLOAT
--inference-rec-end FLOAT
--inference-rec-total-steps INT

--inference-rep-weight FLOAT
--inference-rep-schedule {none,linear,cosine}
--inference-rep-start FLOAT
--inference-rep-end FLOAT
--inference-rep-total-steps INT

--dg-target-weight FLOAT
--dg-target-schedule {none,linear,cosine}
--dg-target-start FLOAT
--dg-target-end FLOAT
--dg-target-total-steps INT

--dg-budget-weight FLOAT
--dg-budget-schedule {none,linear,cosine}
--dg-budget-start FLOAT
--dg-budget-end FLOAT
--dg-budget-total-steps INT

--dg-kl-weight FLOAT
--dg-kl-schedule {none,linear,cosine}
--dg-kl-start FLOAT
--dg-kl-end FLOAT
--dg-kl-total-steps INT

--dg-rep-weight FLOAT
--dg-rep-schedule {none,linear,cosine}
--dg-rep-start FLOAT
--dg-rep-end FLOAT
--dg-rep-total-steps INT

--dg-sparse-weight FLOAT
--dg-sparse-schedule {none,linear,cosine}
--dg-sparse-start FLOAT
--dg-sparse-end FLOAT
--dg-sparse-total-steps INT

--tau-deg FLOAT
--tau-budget FLOAT
--representation-distance {cosine}
--loss-reduction {mean,sum}
```

Backward compatibility를 위해 기존 인자는 alias로 남겨도 된다.

```text
--alpha-inference -> --inference-rec-weight
--alpha-target -> --dg-target-weight
--lambda-budget -> --dg-budget-weight
--lambda-reg -> --dg-kl-weight
--beta-mask -> --tau-budget
```

단, 새 문서와 새 script에서는 새 인자명을 사용한다.

---

## 22. Engine 수정 요구사항

`engine/train_config.py`는 새 `LossConfig`를 생성할 수 있어야 한다.

필수 기능:

1. 새 CLI 인자 parsing
2. `TrainConfig.build_loss_config()`에서 `TermWeightConfig` 구성
3. `loss_mode` preset 지원
4. DDP 인자 parsing
5. model/data/loss config를 `config.json`에 저장
6. checkpoint 저장 시 optimizer, scheduler, global_step, epoch 저장
7. resume 시 동일 상태 복원

---

## 23. DDP 지원 요구사항

DDP는 `torchrun` 기반으로 지원한다.

실행 예:

```bash
torchrun --standalone --nproc_per_node=4 ../engine/dg_train.py ...
```

엔진은 다음 환경변수를 읽어야 한다.

```text
LOCAL_RANK
RANK
WORLD_SIZE
```

필수 구현:

1. `torch.distributed.init_process_group(backend="nccl")`
2. `torch.cuda.set_device(local_rank)`
3. 모델을 `DistributedDataParallel(model, device_ids=[local_rank])`로 wrapping
4. `DataConfig(distributed=True, rank=rank, world_size=world_size)` 전달
5. epoch마다 `sampler.set_epoch(epoch)` 호출
6. logging, checkpoint, visualization은 rank 0만 수행
7. barrier가 필요한 위치에서 `dist.barrier()` 사용
8. 종료 시 `dist.destroy_process_group()` 호출

Single GPU 실행도 계속 지원해야 한다.

---

## 24. DGNet alternating training loop

DGNet pretraining은 한 batch에서 두 phase로 나뉜다.

### 24.1 I phase

```text
M freeze
I_encoder / I_decoder / I_projection train
objective = loss_out.inference_loss
```

### 24.2 M phase

```text
M train
I_encoder / I_decoder / I_projection freeze
objective = loss_out.degradation_loss
```

절대 `combined_loss` 하나로 단일 optimizer step을 수행하지 않는다.

---

## 25. output artifact 요구사항

각 run은 다음 구조를 가져야 한다.

```text
output/{run_id}/
    config.json
    train.csv
    loss_curve.png
    accuracy.png
    run.log

    epoch000001/ 또는 step000100/
        model.pt
        config.json
        train_config.json
        checkpoint.pt
        dg_model.log
        loss.log
        sample.png
```

loss log에는 최소한 다음을 남긴다.

```text
inference_loss
degradation_loss
combined_loss
inference_rec
inference_rep
dg_target
dg_budget
dg_kl
dg_rep
dg_sparse
weighted_* terms
effective_weights
```

---

## 26. 각 패키지별 api.md 작성 요구사항

반드시 다음 파일을 작성 또는 갱신한다.

```text
src/model/model_api.md
src/loss/loss_api.md
src/dataset/dataset_api.md
src/engine/engine_api.md
```

각 API 문서는 다음을 포함해야 한다.

1. 가장 짧은 실행 예제
2. 공개 API 목록
3. Config 필드 설명
4. smoke test 코드
5. train script 연결 예시
6. output artifact 설명
7. DDP 사용 시 주의사항

특히 `loss_api.md`는 반드시 다음을 포함한다.

1. `TermWeightConfig` 사용법
2. representation only 설정
3. reconstruction only 설정
4. hybrid 설정
5. term-wise scheduler 설정
6. term off 설정
7. `DGLossOutput.metrics()` logging 예시

`engine_api.md`는 반드시 다음을 포함한다.

1. single GPU pretrain + linear probe 실행 예시
2. DDP pretrain + linear probe 실행 예시
3. resume 예시
4. checkpoint 구조
5. linear probe checkpoint 경로 규칙

---

## 27. bash script 요구사항

`scripts/` 아래에 바로 실행 가능한 bash script를 제공한다.

필수 파일:

```text
scripts/run_dgnet_pipeline_single_gpu.bash
scripts/run_dgnet_pipeline_ddp.bash
scripts/smoke_loss_scheduler.bash
```

각 script는 다음을 만족해야 한다.

1. `set -euo pipefail` 사용
2. repo root 또는 src root로 이동
3. 환경변수를 통해 dataset path, output root, GPU 지정 가능
4. pretrain 후 linear probe를 자동 실행
5. linear probe checkpoint는 방금 생성한 pretrain run의 최신 checkpoint를 사용
6. 실행 종료 후 pretrain output과 linear probe output 경로 출력

---

## 28. 테스트 요구사항

필수 테스트:

1. `python -m loss.loss` tensor-only smoke test
2. LossConfig serialization test
3. term on/off test
4. scheduler test
5. representation only loss forward test
6. reconstruction only loss forward test
7. hybrid loss forward test
8. model forward shape test
9. zero-init DGNet test
10. save/load round trip test
11. single GPU 2-step smoke pretrain
12. DDP 2-GPU smoke pretrain
13. pretrain checkpoint -> linear probe smoke test

모든 결과는 `output/`에 남긴다.

---

## 29. 완료 기준

작업 완료 기준은 다음과 같다.

1. 새 loss objective가 구현되어 있다.
2. 모든 loss term은 `TermWeightConfig`로 제어된다.
3. 각 term별 none/linear/cosine scheduler가 독립적으로 적용된다.
4. weight가 0인 term은 계산되지 않는다.
5. image-wise scalar target이 구현되어 있다.
6. image-wise scalar budget이 구현되어 있다.
7. channel-wise spatial KL이 구현되어 있다.
8. representation only / reconstruction only / hybrid 설정이 동작한다.
9. DDP pretrain이 torchrun으로 동작한다.
10. pretrain 후 linear probe가 bash script 하나로 이어진다.
11. 각 패키지별 api.md가 갱신되어 있다.
12. `output/`에 smoke test와 실행 로그가 남는다.
13. 개발자는 `output/`과 api 문서만 보고 구현 정합성을 판단할 수 있다.
