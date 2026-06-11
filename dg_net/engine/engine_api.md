# Engine API

> Canonical engine package reference. All commands in this document assume the current working directory is `dg_net/`.

## 1. Package role

`engine/` composes the public `dataset`, `model`, and `loss` APIs into executable training and evaluation flows:

- DGNet alternating pretraining
- linear probing
- classifier fine-tuning
- kNN representation evaluation
- checkpointing, CSV logging, plots, and sample artifact generation

## 2. Public modules and APIs

| module | role |
| --- | --- |
| `train_config.py` | `TrainConfig`, CLI parser, trainer factory |
| `eval.py` | validation helpers, CSV appenders, plots, sample artifacts |
| `dg_train.py` | alternating DGNet pretraining |
| `linear_probe_train.py` | frozen-representation classifier training |
| `finetune_train.py` | encoder/projection + classifier fine-tuning |
| `knn_eval.py` | cosine kNN evaluation |

Programmatic entry points:

```python
from engine.train_config import TrainConfig, build_arg_parser, get_trainer
from engine.eval import (
    DG_COLUMNS,
    CLASS_COLUMNS,
    append_metrics_csv,
    evaluate_dgnet,
    evaluate_classifier,
    plot_accuracy_curve,
    plot_loss_curve,
    save_eval_artifacts,
)
```

## 3. Shortest runnable DG pretrain smoke

```bash
python engine/dg_train.py   --task dg_pretrain   --device cpu   --dataset Synthetic   --data-path ./tmp-data   --img-size 16   --patch-size 8   --batch-size 2   --eval-batch-size 2   --num-workers 0   --max-samples 8   --no-use-augmentation   --mean 0.0,0.0,0.0   --std 1.0,1.0,1.0   --loop-mode step   --total-steps 2   --validate-every-steps 1   --save-every-steps 2   --eval-max-batches 1   --output-root output/api_smoke
```

This exercises the real alternating I/M loop without requiring an external dataset.

## 4. `TrainConfig`

```python
from engine.train_config import TrainConfig

cfg = TrainConfig(
    task="dg_pretrain",
    device="cpu",
    dataset="Synthetic",
    data_path="./tmp-data",
    img_size=16,
    patch_size=8,
    batch_size=2,
    eval_batch_size=2,
    num_workers=0,
    max_samples=8,
    use_augmentation=False,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    loop_mode="step",
    total_steps=2,
)
print(cfg.build_model_config())
print(cfg.build_loss_config())
print(cfg.build_data_config(mode="ssl", train=True))
```

Important field groups:

### Runtime / loop

- `task`: `dg_pretrain`, `linear_probe`, `knn_eval`, `finetune`, `eval`
- `loop_mode`: `step` or `epoch`
- `total_steps`, `total_epochs`
- `validate_every_steps`, `validate_every_epochs`
- `save_every_steps`, `save_every_epochs`
- `output_root`, `run_name`, `run_id`, `seed`, `deterministic`, `device`

### Dataset / DDP

- all dataset-facing fields from `TrainConfig` feed `build_data_config(...)`
- `distributed`, `local_rank`, `rank`, `world_size` are consumed by the DG pretrain path
- `clamp_degraded=True` requires identity normalization (`mean=0`, `std=1`)

### Model / loss / optimizer

- model: patch size, dim/depth/head groups, `dg_architect`, `cnn_architect`, dropout, clamp
- loss: preset + per-term weight/schedule overrides
- optimizer: `lr_inference`, `lr_degradation`, `classifier_lr`, `encoder_lr`, `weight_decay`

## 5. Gradient phases and trainer behavior

`DGTrainer.train_step()` always runs two optimizer phases on the same batch:

1. **I phase**
   - freeze `degradation`
   - train `encoder`, `decoder`, `projection`
   - backward target: `values.inference_loss`
2. **M phase**
   - freeze `encoder`, `decoder`, `projection`
   - train `degradation`
   - backward target: `values.degradation_loss`

After both phases, all model parameters are re-enabled so evaluation and checkpoint restore operate on the full model state.

## 6. Single-process pretrain, resume, and transfer commands

### Pretrain

```bash
python engine/dg_train.py   --task dg_pretrain   --device cpu   --dataset Synthetic   --data-path ./tmp-data   --img-size 16   --patch-size 8   --batch-size 2   --eval-batch-size 2   --num-workers 0   --max-samples 8   --no-use-augmentation   --mean 0.0,0.0,0.0   --std 1.0,1.0,1.0   --loop-mode step   --total-steps 1   --validate-every-steps 1   --save-every-steps 1   --eval-max-batches 1   --output-root output/api_resume   --run-id api-resume
```

### Resume

```bash
python engine/dg_train.py   --task dg_pretrain   --device cpu   --dataset Synthetic   --data-path ./tmp-data   --img-size 16   --patch-size 8   --batch-size 2   --eval-batch-size 2   --num-workers 0   --max-samples 8   --no-use-augmentation   --mean 0.0,0.0,0.0   --std 1.0,1.0,1.0   --loop-mode step   --total-steps 2   --validate-every-steps 1   --save-every-steps 1   --eval-max-batches 1   --output-root output/api_resume   --run-id api-resume   --resume output/api_resume/api-resume/step000001
```

### Linear probe after pretrain

```bash
python engine/linear_probe_train.py   --task linear_probe   --device cpu   --dataset Synthetic   --data-path ./tmp-data   --img-size 16   --patch-size 8   --batch-size 2   --eval-batch-size 2   --num-workers 0   --max-samples 8   --no-use-augmentation   --mean 0.0,0.0,0.0   --std 1.0,1.0,1.0   --loop-mode step   --total-steps 1   --validate-every-steps 1   --save-every-steps 1   --eval-max-batches 1   --output-root output/api_linear   --run-id api-linear   --checkpoint output/api_resume/api-resume/step000002
```

### Other transfer tasks

- `python engine/finetune_train.py ... --checkpoint <pretrain_tag_dir>`
- `python engine/knn_eval.py ... --checkpoint <pretrain_tag_dir>`

`linear_probe` freezes the DGNet backbone and trains only `Linear(PROJECTION_DIM, num_classes)`.
`finetune` trains encoder/projection + classifier by default while keeping degradation/decoder frozen.
`knn_eval` builds a feature bank from the classification train split and evaluates cosine kNN on the validation/test split.

## 7. One-shot pipeline scripts

### Single-process pipeline

```bash
CUDA_DEVICE='' DEVICE=cpu DATASET=Synthetic DATA_PATH=./tmp-data IMG_SIZE=16 PATCH_SIZE=8 BATCH_SIZE=2 EVAL_BATCH_SIZE=2 NUM_WORKERS=0 MAX_SAMPLES=8 PRETRAIN_EPOCHS=1 LINEAR_EPOCHS=1 SAVE_EVERY_EPOCHS=1 EVAL_MAX_BATCHES=1 OUTPUT_ROOT=output/api_pipeline_single bash scripts/run_dgnet_pipeline_single_gpu.bash
```

### DDP pipeline

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 DATA_PATH=/path/to/dataset OUTPUT_ROOT=output/dgnet_hybrid_ddp bash scripts/run_dgnet_pipeline_ddp.bash
```

The DDP script requires CUDA + NCCL + `torchrun`. In this documentation pass it was syntax-checked with `bash -n`, not executed end-to-end because the current environment has no multi-GPU runtime.

## 8. Checkpoint structure and artifact layout

Run root:

```text
output/{run_id}/
├── config.json
├── run.log
├── train.csv
├── loss_curve.png
├── accuracy.png
├── dataset_reports/
└── step000000/ or epoch000000/
```

Per-tag checkpoint directory for DG pretrain:

```text
{tag}/
├── config.json
├── model.pt
├── train_config.json
├── checkpoint.pt
├── {tag}.png
├── {tag}_metrics.txt
├── dg_model.log
└── loss.log
```

`checkpoint.pt` contains:

- `global_step`
- `epoch`
- serialized `train_config`
- optimizer state(s)
- scheduler state(s)
- best validation score / best accuracy

Linear-probe and fine-tune checkpoints also save the classifier state in `checkpoint.pt`.

## 9. CSV columns

DG pretrain rows use `DG_COLUMNS` from `engine/eval.py`:

```text
run_id, task, loop_mode, epoch, step, split, phase,
train_loss, validation_loss, validation_score,
inference_loss, degradation_loss, combined_loss,
inference_rec, inference_rep, dg_target, dg_budget, dg_kl, dg_rep, dg_sparse,
weighted_inference_rec, weighted_inference_rep, weighted_dg_target, weighted_dg_budget,
weighted_dg_kl, weighted_dg_rep, weighted_dg_sparse,
weight_inference_rec, weight_inference_rep, weight_dg_target, weight_dg_budget,
weight_dg_kl, weight_dg_rep, weight_dg_sparse,
lr_inference, lr_degradation, elapsed_sec
```

Classification / kNN rows use `CLASS_COLUMNS`:

```text
run_id, task, loop_mode, epoch, step, split,
train_loss, validation_loss, validation_score, accuracy, top1, top5, lr, elapsed_sec
```

## 10. Output artifact helpers

`save_eval_artifacts(model, loss_fn, sample, output_dir, tag, step=...)` writes:

- `{tag}.png` and `{tag}_metrics.txt` through `save_sample_visualization()`
- `dg_model.log` through `write_model_report()`
- `loss.log` when a loss module is provided

`plot_loss_curve()` and `plot_accuracy_curve()` read `train.csv` and regenerate `loss_curve.png` / `accuracy.png`.

## 11. Rank-0 and DDP cautions

- DDP pretraining lives only in `dg_train.py`; it initializes NCCL and wraps the model with `DistributedDataParallel`
- rank 0 alone writes `config.json`, `run.log`, `train.csv`, checkpoints, PNGs, and dataset reports
- non-rank-0 workers return from `save_checkpoint()` after a barrier without writing files
- `DistributedSampler` is used for train and validation loaders when `distributed=True`
- `sampler.set_epoch(epoch)` is called before each epoch in distributed epoch-mode loops
- current transfer scripts (`linear_probe_train.py`, `finetune_train.py`, `knn_eval.py`) are single-process flows in the checked-in pipeline scripts
