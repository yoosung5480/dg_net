#!/usr/bin/env bash
set -euo pipefail

# DDP DGNet pretrain -> linear probe one-shot script.
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/run_dgnet_pipeline_ddp.bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${SRC_ROOT}"

STAMP="$(date +%Y%m%d-%H%M%S)"
DATA_PATH="${DATA_PATH:-/home/jeongyuseong/바탕화면/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/rec_only}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-20}"
DATASET="${DATASET:-STL10}"
IMG_SIZE="${IMG_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-192}"
DEPTH="${DEPTH:-4}"
NUM_HEADS="${NUM_HEADS:-3}"
PROJECTION_DIM="${PROJECTION_DIM:-192}"
REPRESENTATION_DISTANCE="${REPRESENTATION_DISTANCE:-cosine}"
INFONCE_TEMPERATURE="${INFONCE_TEMPERATURE:-0.2}"
DG_SPATIAL_REFINEMENT="${DG_SPATIAL_REFINEMENT:-false}"
DG_REFINEMENT_CHANNELS="${DG_REFINEMENT_CHANNELS:-32}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-200}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-50}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"

PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-ddp-hybrid-pretrain-${STAMP}}"
LINEAR_RUN_ID="${LINEAR_RUN_ID:-ddp-hybrid-linear-probe-${STAMP}}"

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
  --device cuda
  --distributed
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

REFINEMENT_ARGS=(--no-dg-spatial-refinement)
if [[ "${DG_SPATIAL_REFINEMENT}" == "true" ]]; then
  REFINEMENT_ARGS=(--dg-spatial-refinement)
fi
REFINEMENT_ARGS+=(--dg-refinement-channels "${DG_REFINEMENT_CHANNELS}")

LOSS_ARGS=(
  --loss-mode hybrid
  --representation-distance "${REPRESENTATION_DISTANCE}"
  --infonce-temperature "${INFONCE_TEMPERATURE}"
  --inference-rec-weight 1.0
  --inference-rec-schedule none
  --inference-rep-weight 0.0
  --inference-rep-schedule none
  --dg-target-weight 0.1
  --dg-target-schedule cosine
  --dg-target-start 0.001
  --dg-target-end 0.1
  --dg-target-total-steps 10000
  --dg-budget-weight 0.05
  --dg-budget-schedule none
  --dg-kl-weight 0.0
  --dg-kl-schedule none 
  --dg-rep-weight 0.0
  --dg-rep-schedule none
  --dg-sparse-weight 0.0
  --dg-sparse-schedule none
  --tau-deg 0.3
  --tau-budget 0.3
  --loss-reduction mean
)

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 engine/dg_train.py \
  --task dg_pretrain \
  --loop-mode epoch \
  --total-epochs "${PRETRAIN_EPOCHS}" \
  --validate-every-epochs 1 \
  --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${PRETRAIN_RUN_ID}" \
  --run-name ddp-hybrid-pretrain \
  --seed 42 \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  "${REFINEMENT_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  --lr-inference 1.5e-4 \
  --lr-degradation 1.0e-4 \
  --weight-decay 5.0e-2

CHECKPOINT_DIR="${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}/epoch$(printf "%06d" "${PRETRAIN_EPOCHS}")"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_DIR="$(find "${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}" -maxdepth 1 -type d -name 'epoch*' | sort | tail -n 1)"
fi

LINEAR_DATA_ARGS=()
for arg in "${COMMON_DATA_ARGS[@]}"; do
  [[ "${arg}" == "--distributed" ]] || LINEAR_DATA_ARGS+=("${arg}")
done

# Linear probe can run single GPU by default, because the classifier is lightweight.
CUDA_VISIBLE_DEVICES=0 python engine/linear_probe_train.py \
  --task linear_probe \
  --checkpoint "${CHECKPOINT_DIR}" \
  --loop-mode epoch \
  --total-epochs "${LINEAR_EPOCHS}" \
  --validate-every-epochs 1 \
  --save-every-epochs ${SAVE_EVERY_EPOCHS} \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${LINEAR_RUN_ID}" \
  --run-name ddp-hybrid-linear-probe \
  --seed 42 \
  "${LINEAR_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  "${REFINEMENT_ARGS[@]}" \
  --classifier-lr 1.0e-3 \
  --weight-decay 0.0

echo "pretrain_output=${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}"
echo "linear_probe_output=${OUTPUT_ROOT}/${LINEAR_RUN_ID}"
echo "linear_probe_checkpoint=${CHECKPOINT_DIR}"
