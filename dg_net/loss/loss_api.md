# Loss API

`loss`는 model/dataset/engine을 import하지 않는 tensor-only 패키지다.

## 공개 API / 최소 예제

```python
from loss.loss import prepare_dgloss
from loss.loss_config import LossConfig, TermWeightConfig

cfg = LossConfig.hybrid()
loss_fn = prepare_dgloss(cfg)
values = loss_fn(
    original=x, degraded=x_dg, residual=r, reconstruction=x_hat,
    z_original=z_x, z_degraded=z_dg, mask=None, step=10,
)
values.inference_loss.backward()    # I phase
values.degradation_loss.backward()  # M phase
print(values.metrics())
```

공개 config는 `TermWeightConfig(weight, schedule, start, end, total_steps)`와
`LossConfig`다. schedule은 `none`, `linear`, `cosine`이며 base weight가 0이면
term 계산 자체가 생략된다. `LossConfig` threshold는 `tau_deg`, `tau_budget`,
`representation_distance="cosine"`, 두 reduction을 제공한다.

## Ablation / scheduler

```python
rep = LossConfig.representation_only()
rec = LossConfig.reconstruction_only()
hybrid = LossConfig.hybrid()
scheduled = LossConfig(
    inference_rep=TermWeightConfig(1, "cosine", 0, 1, 10000),
    dg_budget=TermWeightConfig(1, "linear", 0, 1, 5000),
)
off = TermWeightConfig(weight=0)
```

`DGLossOutput`은 raw term `inference_rec/inference_rep/dg_*`, `weighted_*`,
`effective_weights`, 세 objective를 반환한다. `combined_loss`는 logging 전용이다.

## Smoke / artifact / train 연결

```bash
bash scripts/smoke_loss_scheduler.bash
# loss/output/loss_*_test.log
python engine/dg_train.py --loss-mode hybrid --dg-kl-weight 0.01
```

DDP에서도 loss는 rank/process 상태를 갖지 않으며 `global_step`만 전달받는다.
