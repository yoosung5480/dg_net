#!/usr/bin/env bash
set -euo pipefail

# Single-GPU DGNet pretrain -> linear probe one-shot script.
# Usage:
#   CUDA_DEVICE=4 DATA_PATH=/home/cv202355545/dataset bash scripts/run_dgnet_pipeline_single_gpu.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${SRC_ROOT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
DATA_PATH="${DATA_PATH:-/home/jeongyuseong/바탕화면/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/dgnet_hybrid}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-20}"
DATASET="${DATASET:-STL10}"
IMG_SIZE="${IMG_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-192}"
DEPTH="${DEPTH:-4}"
NUM_HEADS="${NUM_HEADS:-3}"
PROJECTION_DIM="${PROJECTION_DIM:-192}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-1}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-1}"

PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-hybrid-pretrain-${STAMP}}"
LINEAR_RUN_ID="${LINEAR_RUN_ID:-hybrid-linear-probe-${STAMP}}"

COMMON_DATA_ARGS=(
  --dataset "${DATASET}"
  --data-path "${DATA_PATH}"
  --img-size "${IMG_SIZE}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --use-augmentation
  --crop-scale 0.2,1.0
  --hflip-prob 0.5
  --mean 0.0,0.0,0.0
  --std 1.0,1.0,1.0
  --eval-max-batches "${EVAL_MAX_BATCHES}"
  --device "${DEVICE}"
)

if [[ -n "${MAX_SAMPLES:-}" && "${MAX_SAMPLES}" != "0" ]]; then
  COMMON_DATA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

COMMON_MODEL_ARGS=(
  --patch-size "${PATCH_SIZE}"
  --dg-embed-dim "${EMBED_DIM}"
  --dg-depth "${DEPTH}"
  --dg-num-heads "${NUM_HEADS}"
  --encoder-embed-dim "${EMBED_DIM}"
  --encoder-depth "${DEPTH}"
  --encoder-num-heads "${NUM_HEADS}"
  --decoder-embed-dim "${EMBED_DIM}"
  --decoder-depth "${DEPTH}"
  --decoder-num-heads "${NUM_HEADS}"
  --embed-dim "${EMBED_DIM}"
  --depth "${DEPTH}"
  --num-heads "${NUM_HEADS}"
  --projection-dim "${PROJECTION_DIM}"
  --dg-architect HYBRID
  --cnn-architect CONVNEXT
  --no-clamp-degraded
  --attn-dropout 0.0
  --proj-dropout 0.0
  --drop-path 0.1
)

# Reconstruction-only baseline with term-wise schedules.
LOSS_ARGS=(
  --loss-mode reconstruction_only
  --inference-rec-weight 1.0
  --inference-rec-schedule none
  --inference-rep-weight 0.0
  --inference-rep-schedule none
  --dg-target-weight 0.1
  --dg-target-schedule none
  --dg-budget-weight 0.0
  --dg-budget-schedule none
  --dg-kl-weight 0.0
  --dg-kl-schedule none
  --dg-rep-weight 0.0
  --dg-rep-schedule none
  --dg-sparse-weight 0.0
  --dg-sparse-schedule none
  --tau-deg 0.5
  --tau-budget 0.0
  --loss-reduction mean
)

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python engine/dg_train.py \
  --task dg_pretrain \
  --loop-mode epoch \
  --total-epochs "${PRETRAIN_EPOCHS}" \
  --validate-every-epochs 1 \
  --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${PRETRAIN_RUN_ID}" \
  --run-name hybrid-pretrain \
  --seed 42 \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  --lr-inference 1.5e-4 \
  --lr-degradation 1.0e-4 \
  --weight-decay 5.0e-2

CHECKPOINT_DIR="${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}/epoch$(printf "%06d" "${PRETRAIN_EPOCHS}")"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="$(find "${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}" -maxdepth 1 -type d -name 'epoch*' | sort | tail -n 1)"
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python engine/linear_probe_train.py \
  --task linear_probe \
  --checkpoint "${CHECKPOINT_DIR}" \
  --loop-mode epoch \
  --total-epochs "${LINEAR_EPOCHS}" \
  --validate-every-epochs 1 \
  --save-every-epochs 10 \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${LINEAR_RUN_ID}" \
  --run-name hybrid-linear-probe \
  --seed 42 \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  --classifier-lr 1.0e-3 \
  --weight-decay 0.0

echo "pretrain_output=${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}"
echo "linear_probe_output=${OUTPUT_ROOT}/${LINEAR_RUN_ID}"
echo "linear_probe_checkpoint=${CHECKPOINT_DIR}"
