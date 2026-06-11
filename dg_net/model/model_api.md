# Model API

> Canonical model package reference. All commands in this document assume the current working directory is `dg_net/`.

## 1. Package role

`model/dg_model.py` assembles DGNet from a single config surface:

- `M`: selectable degradation generator (`VIT`, `CNN`, `HYBRID`)
- `I`: fixed ViT encoder/decoder for reconstruction and representation learning
- diagnostics: globality metrics, optional spatial refinement, visualization writers

## 2. Public API

```python
from model.dg_model import (
    DGNet,
    DgNet,
    DgNetConfig,
    DGNetConfig,
    DgNetOutput,
    compute_cross_patch_influence,
    compute_patch_seam_ratio,
    parameter_summary,
    save_sample_visualization,
    write_model_report,
)
```

## 3. Shortest runnable example

```python
import torch
from model.dg_model import DGNet, DgNetConfig

cfg = DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=64,
    DEPTH=2,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=64,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=32,
    DG_ARCHITECT="HYBRID",
)
model = DGNet(cfg).eval()

images = torch.rand(2, 3, 32, 32)
with torch.no_grad():
    output = model(images)

print(output.degraded.shape)
print(output.reconstruction.shape)
print(output.representation.shape)
```

Expected tensor contract:

| field | shape |
| --- | --- |
| `original` | `[B, C, H, W]` |
| `residual` | `[B, C, H, W]` |
| `degraded` | `[B, C, H, W]` |
| `reconstruction` | `[B, C, H, W]` |
| `original_reconstruction` | `[B, C, H, W]` |
| `representation` | `[B, PROJECTION_DIM]` |
| `degraded_representation` | `[B, PROJECTION_DIM]` |

`DgNetOutput._fields` is intentionally stable at these seven entries.

## 4. `DgNetConfig` fields

### Core image/token fields

| field | default | notes |
| --- | ---: | --- |
| `IMG_SIZE` | `224` | must be divisible by `PATCH_SIZE` |
| `PATCH_SIZE` | `16` | patch side length |
| `IN_CHANS` | `3` | input/output channels |
| `PROJECTION_DIM` | `128` | representation output dim |

### Shared model sizes

| field | default | notes |
| --- | ---: | --- |
| `EMBED_DIM` | `512` | fallback dim for encoder / decoder / degradation |
| `DEPTH` | `6` | fallback block depth |
| `NUM_HEADS` | `8` | fallback attention heads |
| `MLP_RATIO` | `4.0` | ViT/Hybrid FFN expansion |

### Architecture selection

| field | default | notes |
| --- | --- | --- |
| `ENCODER_ARCHITECT` | `"VIT"` | fixed to `VIT` |
| `DECODER_ARCHITECT` | `"VIT"` | fixed to `VIT` |
| `DG_ARCHITECT` | `"HYBRID"` | `VIT`, `CNN`, or `HYBRID` |
| `CNN_ARCHITECT` | `"CONVNEXT"` | used only when `DG_ARCHITECT="CNN"` |

### Component overrides

All of these default to `None` and fall back to the shared values above:

- `ENCODER_EMBED_DIM`, `ENCODER_DEPTH`, `ENCODER_NUM_HEADS`
- `DECODER_EMBED_DIM`, `DECODER_DEPTH`, `DECODER_NUM_HEADS`
- `DG_EMBED_DIM`, `DG_DEPTH`, `DG_NUM_HEADS`

### Output/control fields

| field | default | notes |
| --- | --- | --- |
| `USE_POS_EMBED` | `True` | learned positional embeddings |
| `POS_EMBED_INIT_ZERO` | `True` | zero-init instead of truncated normal |
| `CLAMP_DEGRADED` | `False` | clamps `x - M(x)` to `[0, 1]` |
| `DG_SPATIAL_REFINEMENT` | `False` | enables full-image residual refinement head |
| `DG_REFINEMENT_CHANNELS` | `32` | refinement hidden channels; must be positive |

### Regularization / block detail

| field | default |
| --- | ---: |
| `ATTN_DROPOUT` | `0.0` |
| `PROJ_DROPOUT` | `0.0` |
| `DROP_PATH` | `0.0` |
| `CNN_KERNEL_SIZE` | `3` |
| `CNN_DROPOUT` | `0.0` |
| `FFN_DROPOUT` | `0.0` |

## 5. Architecture ablation and zero-init behavior

`DG_ARCHITECT` swaps only the degradation branch. The encoder/decoder stay ViT-based in every case.

| degradation mode | block type | behavior |
| --- | --- | --- |
| `VIT` | `ViTBlock` stack | strongest explicit global mixing |
| `CNN` | `TokenCNNBlock` stack | local patch-grid processing |
| `HYBRID` | `HybridBlock` stack | attention + local CNN branch |

Both `DegradationGenerator.residual_head` and the optional spatial-refinement final conv are zero-initialized. A fresh model therefore starts with:

- `residual == 0`
- `degraded == original` when `CLAMP_DEGRADED=False`
- identical original/degraded forward paths before optimization changes the residual branch

## 6. Globality metrics, spatial refinement, and visualization

```python
import torch
from model.dg_model import (
    DGNet,
    DgNetConfig,
    compute_cross_patch_influence,
    compute_patch_seam_ratio,
    save_sample_visualization,
)

cfg = DgNetConfig(
    IMG_SIZE=32,
    PATCH_SIZE=8,
    EMBED_DIM=32,
    DEPTH=2,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=32,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=16,
    DG_ARCHITECT="HYBRID",
    DG_SPATIAL_REFINEMENT=True,
    DG_REFINEMENT_CHANNELS=12,
)
model = DGNet(cfg).eval()
image = torch.rand(1, 3, 32, 32)

with torch.no_grad():
    output = model(image)

cross_patch = compute_cross_patch_influence(model.degradation, image, patch_size=cfg.PATCH_SIZE)
seam_ratio = compute_patch_seam_ratio(output.residual, patch_size=cfg.PATCH_SIZE)
path = save_sample_visualization(output, "model/output/api_sample.png")

print(round(cross_patch, 6), round(seam_ratio, 6), path)
```

`save_sample_visualization()` always writes:

- `{tag}.png` with seven panels
- `{tag}_metrics.txt` with
  - residual mean/std/max
  - reconstruction MSE
  - degraded reconstruction MSE
  - cosine similarity/distance
  - patch seam ratio

The seven panels are:

1. `original`
2. `degraded`
3. `reconstruction`
4. `absolute residual |x - x_dg|`
5. `normalized residual heatmap`
6. `reconstruction error |x - reconstruction|`
7. `degraded reconstruction error |x_dg - reconstruction|`

## 7. Save/load and parameter summary

```python
import tempfile
import torch
from model.dg_model import DGNet, DgNetConfig, parameter_summary

cfg = DgNetConfig(
    IMG_SIZE=16,
    PATCH_SIZE=8,
    EMBED_DIM=16,
    DEPTH=1,
    NUM_HEADS=4,
    DECODER_EMBED_DIM=16,
    DECODER_NUM_HEADS=4,
    PROJECTION_DIM=8,
    DG_ARCHITECT="HYBRID",
)
model = DGNet(cfg).eval()
sample = torch.rand(2, 3, 16, 16)

with torch.no_grad():
    before = model(sample)

with tempfile.TemporaryDirectory() as tmpdir:
    saved = model.save(tmpdir)
    loaded = DGNet.load(saved, map_location="cpu", verbose=False).eval()
    with torch.no_grad():
        after = loaded(sample)

count, size_mib = parameter_summary(model)
print(saved)
print(torch.allclose(before.reconstruction, after.reconstruction), count, round(size_mib, 4))
```

Checkpoint layout for `model.save(path)`:

```text
path/
├── config.json
└── model.pt
```

`DGNet.load()` ignores unknown keys in older configs through `DgNetConfig.from_dict()`, so legacy checkpoints without spatial-refinement fields still load with defaults.

## 8. Smoke test and output artifacts

```bash
python -m model.dg_model
```

Artifacts:

```text
model/output/
├── dg_model.log
├── dg_model_sample.png
├── dg_model_sample_metrics.txt
└── saved_dgnet/
    ├── config.json
    └── model.pt
```

## 9. Train script connection

- `engine/dg_train.py` constructs the model through `TrainConfig.build_model_config()`.
- `scripts/run_dgnet_pipeline_single_gpu.bash` and `scripts/run_dgnet_pipeline_ddp.bash` both use `DG_ARCHITECT=HYBRID` by default.
- current engine CLI exposes `--dg-architect`, `--cnn-architect`, dropout, depth, dim, and clamp controls
- current engine CLI does **not** expose `DG_SPATIAL_REFINEMENT` or `DG_REFINEMENT_CHANNELS`; the scripts therefore run with refinement disabled unless you instantiate `DGNet(DgNetConfig(...))` directly or extend `TrainConfig`

## 10. DDP / checkpoint cautions

- wrap the model with DDP **after** optional checkpoint loading/resume setup, as `engine/dg_train.py` already does
- save checkpoints from the raw module (`trainer.raw_model.save(...)` or `ddp_model.module.save(...)`), not from the wrapper object
- rank 0 is the only writer for checkpoints, logs, PNG artifacts, and CSV rows
- `compute_cross_patch_influence()` expects batch size `1`; use per-sample diagnostics when comparing ranks
