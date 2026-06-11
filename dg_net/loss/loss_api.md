# Loss API

> Canonical loss package reference. All commands in this document assume the current working directory is `dg_net/`.

## 1. Package role

`loss/` is a tensor-only package. It does not import the dataset or engine packages and can consume either a `DgNetOutput` object or explicit tensors.

## 2. Public API

```python
from loss import (
    ChannelWiseSpatialKL,
    DGBudgetDistance,
    DGLoss,
    DGLossOutput,
    DGTargetDistance,
    DGTargetRepresentationDistance,
    InfoNCELoss,
    InferenceReconstructionLoss,
    LossConfig,
    RepresentationDistance,
    SparsePenalty,
    TermWeightConfig,
    effective_weight,
    prepare_dgloss,
    write_loss_report,
)
```

## 3. Objectives and signs

`DGLoss` returns two optimizer-facing objectives:

- `inference_loss`: minimized in the I phase
- `degradation_loss`: minimized in the M phase

With step-dependent weights `w_*`, the implementation is:

- `inference_rec = mean((reconstruction - original)^2)` or sum, depending on `inference_reduction`
- `inference_rep = 1 - cosine(z_original, z_degraded)` **or** symmetric local-batch InfoNCE
- `dg_target = | per_image_mse(reconstruction, original) - tau_deg |`
- `dg_budget = | per_image_mse(residual, 0) - tau_budget |`
- `dg_kl = KL(P_damage || U)` channel-wise over spatial positions
- `dg_rep = mean(|(1 - cosine(z_original.detach(), z_degraded)) - tau_rep|)`
- `dg_sparse = sparse penalty over an explicit mask`

The final scalar objectives are:

```text
inference_loss   = + w_inference_rec * inference_rec
                 + w_inference_rep * inference_rep

degradation_loss = + w_dg_target * dg_target
                 + w_dg_budget * dg_budget
                 - w_dg_kl     * dg_kl
                 + w_dg_rep    * dg_rep
                 + w_dg_sparse * dg_sparse
```

Important current behavior:

- `dg_rep` is a targeted cosine-distance constraint: minimizing it moves each sample's distance toward `tau_rep`
- `dg_rep` detaches `z_original` and keeps gradient only through `z_degraded`
- `weighted_dg_kl` carries a minus sign; `weighted_dg_rep` is added with a positive sign
- `combined_loss` is returned for logging, not for the alternating optimizer step in `engine/dg_train.py`
- `dg_sparse` is disabled by default and raises `ValueError` if enabled without `mask`

### InfoNCE local negatives

`InfoNCELoss` builds a `B_local x B_local` similarity matrix from the current process batch only:

```text
logits = normalize(z_original) @ normalize(z_degraded)^T / temperature
loss   = 0.5 * (CE(logits, labels) + CE(logits^T, labels))
```

So in DDP:

- negatives come from the local rank only
- local batch size must be at least `2`

## 4. `TermWeightConfig` and `LossConfig`

```python
from dataclasses import replace
from loss import LossConfig, TermWeightConfig

scheduled = TermWeightConfig(weight=1.0, schedule="cosine", start=0.0, end=1.0, total_steps=100)
config = replace(
    LossConfig.hybrid(),
    inference_rep=scheduled,
    dg_budget=TermWeightConfig(weight=0.5, schedule="linear", start=0.0, end=1.0, total_steps=50),
)
print(config.weights_at(step=0))
print(config.weights_at(step=100))
```

Preset helpers (`representation_only()`, `reconstruction_only()`, `hybrid()`) are convenient for default term bundles. When you need to swap individual term configs, use `dataclasses.replace(...)` or construct `LossConfig(...)` directly.

### Config fields

| field | type / default | notes |
| --- | --- | --- |
| `inference_rec`, `inference_rep`, `dg_target`, `dg_budget`, `dg_kl`, `dg_rep`, `dg_sparse` | `TermWeightConfig` | one schedule per term |
| `tau_deg` | `0.05` | reconstruction target level |
| `tau_budget` | `0.01` | residual budget target |
| `tau_rep` | `0.5` | target cosine representation distance; must be in `(0, 2]` |
| `representation_distance` | `"cosine"` | `"cosine"` or `"infonce"` |
| `infonce_temperature` | `0.2` | must be positive |
| `eps` | `1e-8` | numerical stability |
| `inference_reduction` | `"mean"` | `"mean"` or `"sum"` |
| `dg_reduction` | `"mean"` | `"mean"` or `"sum"` |

`TermWeightConfig` fields:

| field | default |
| --- | ---: |
| `weight` | `1.0` |
| `schedule` | `"none"` |
| `start` | `1.0` |
| `end` | `1.0` |
| `total_steps` | `10000` |

## 5. Shortest runnable example

```python
import torch
from loss import LossConfig, prepare_dgloss

loss_fn = prepare_dgloss(LossConfig.hybrid())
values = loss_fn(
    original=torch.rand(4, 3, 8, 8),
    degraded=torch.rand(4, 3, 8, 8),
    residual=torch.rand(4, 3, 8, 8) * 0.1,
    reconstruction=torch.rand(4, 3, 8, 8),
    z_original=torch.rand(4, 8),
    z_degraded=torch.rand(4, 8),
    step=17,
)
print(values.inference_loss.shape, values.degradation_loss.shape)
print(sorted(values.metrics().keys())[:5])
```

## 6. Presets, scheduler, and term-off examples

```python
from dataclasses import replace
from loss import LossConfig, TermWeightConfig

rep_only = LossConfig.representation_only(representation_distance="infonce")
rec_only = LossConfig.reconstruction_only()
hybrid = LossConfig.hybrid()
term_off = replace(hybrid, dg_kl=TermWeightConfig(weight=0.0), dg_rep=TermWeightConfig(weight=0.0))
custom = LossConfig(
    inference_rec=TermWeightConfig(weight=1.0),
    inference_rep=TermWeightConfig(weight=1.0, schedule="cosine", start=0.0, end=1.0, total_steps=100),
    dg_target=TermWeightConfig(weight=0.1),
    dg_budget=TermWeightConfig(weight=1.0, schedule="linear", start=0.0, end=1.0, total_steps=50),
    dg_kl=TermWeightConfig(weight=0.01),
    dg_rep=TermWeightConfig(weight=1.0),
    dg_sparse=TermWeightConfig(weight=0.0),
)

print(rep_only.representation_distance)
print(rec_only.dg_budget.weight)
print(hybrid.dg_rep.weight)
print(term_off.dg_kl.weight, term_off.dg_rep.weight)
print(custom.weights_at(25)["dg_budget"])
```

## 7. Alternating I/M phase usage

```python
import torch
from loss import LossConfig, prepare_dgloss
from model.dg_model import DGNet, DgNetConfig

model = DGNet(DgNetConfig(
    IMG_SIZE=16,
    PATCH_SIZE=8,
    EMBED_DIM=16,
    DEPTH=1,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=16,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=8,
)).train()
loss_fn = prepare_dgloss(LossConfig.hybrid())
images = torch.rand(4, 3, 16, 16)
output = model(images)
values = loss_fn(output, step=3)

print(values.inference_loss.requires_grad)
print(values.degradation_loss.requires_grad)
print(values.effective_weights["dg_rep"])
```

This mirrors the engine contract:

- I phase backpropagates `values.inference_loss`
- M phase backpropagates `values.degradation_loss`

## 8. Smoke test, logging, and output artifacts

```bash
bash scripts/smoke_loss_scheduler.bash
```

```python
from pathlib import Path
from loss import LossConfig, prepare_dgloss, write_loss_report
import torch

values = prepare_dgloss(LossConfig.hybrid())(
    original=torch.rand(4, 3, 8, 8),
    degraded=torch.rand(4, 3, 8, 8),
    residual=torch.rand(4, 3, 8, 8) * 0.1,
    reconstruction=torch.rand(4, 3, 8, 8),
    z_original=torch.rand(4, 8),
    z_degraded=torch.rand(4, 8),
)
path = write_loss_report(values, Path("loss/output/api_loss.log"))
print(path)
```

Artifacts written by the built-in smoke script:

```text
loss/output/
├── loss_ablation_mode_test.log
├── loss_config_summary.log
├── loss_forward_shape_test.log
├── loss_scheduler_test.log
├── loss_term_onoff_test.log
└── api_loss.log   # when write_loss_report(...) is called
```

## 9. Train script connection

- `engine/dg_train.py` creates the loss module with `prepare_dgloss(cfg.build_loss_config())`
- `TrainConfig.loss_mode` presets: `representation_only`, `reconstruction_only`, `hybrid`, `custom`
- CLI aliases kept for compatibility:
  - `--alpha-inference -> inference_rec_weight`
  - `--alpha-target -> dg_target_weight`
  - `--lambda-budget -> dg_budget_weight`
  - `--lambda-reg -> dg_kl_weight`
  - `--tau-budget` / `--beta-mask` share the same destination
- current engine CLI exposes per-term schedules and supports `cosine` / `infonce` for the inference representation objective
- `--dg-rep-target-weight` configures the targeted representation term while the existing `--dg-rep-schedule` family configures its schedule
- `--tau-rep` configures its target cosine distance

## 10. DDP cautions

- the loss object itself is stateless across ranks; only `step` changes weights over time
- InfoNCE negatives are local-rank only because there is no feature all-gather
- if you enable `representation_distance="infonce"` programmatically under DDP, keep each rank's batch size `>= 2`
- `DGLossOutput.metrics()` is safe for rank-local logging because it detaches scalars before converting to Python floats
