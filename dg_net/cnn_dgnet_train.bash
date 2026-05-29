#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

STAMP="$(date +%Y%m%d-%H%M%S)"
DATA_PATH="${DATA_PATH:-/home/jeongyuseong/바탕화면/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-20}"

PRETRAIN_RUN_ID="case2_2-cnn-dgnet-pretrain-${STAMP}"
LINEAR_RUN_ID="case2_2-cnn-dgnet-linear-probe-${STAMP}"

COMMON_DATA_ARGS=(
  --dataset STL10
  --data-path "${DATA_PATH}"
  --img-size 32
  --batch-size 16
  --eval-batch-size 64
  --num-workers "${NUM_WORKERS}"
  --use-augmentation
  --crop-scale 0.2,1.0
  --hflip-prob 0.5
  --mean 0.0,0.0,0.0
  --std 1.0,1.0,1.0
  --eval-max-batches "${EVAL_MAX_BATCHES}"
  --device "${DEVICE}"
)

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  COMMON_DATA_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

COMMON_MODEL_ARGS=(
  --patch-size 4

  # DGNet: 1
  --dg-embed-dim 384
  --dg-depth 4
  --dg-num-heads 6

  # Encoder: 3
  --encoder-embed-dim 384
  --encoder-depth 12
  --encoder-num-heads 6

  # Decoder: 1
  --decoder-embed-dim 384
  --decoder-depth 4
  --decoder-num-heads 6

  --embed-dim 384
  --depth 4
  --num-heads 6
  --projection-dim 384

  --dg-architect HYBRID
  --cnn-architect CONVNEXT
  --no-clamp-degraded
  --attn-dropout 0.0
  --proj-dropout 0.0
  --drop-path 0.1
)

LOSS_ARGS=(
  --alpha-inference 1.0
  --alpha-target 1.0
  --lambda-budget 2.0
  --lambda-reg 0.1
  --tau-deg 0.5
  --beta-mask 0.5
  --loss-reduction mean
)

CUDA_VISIBLE_DEVICES=0 python engine/dg_train.py \
  --task dg_pretrain \
  --loop-mode epoch \
  --total-epochs 10 \
  --validate-every-epochs 1 \
  --save-every-epochs 1 \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${PRETRAIN_RUN_ID}" \
  --run-name case2_2-cnn-dgnet-pretrain \
  --seed 42 \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  "${LOSS_ARGS[@]}" \
  --lr-inference 3.0e-4 \
  --lr-degradation 1.0e-4 \
  --weight-decay 5.0e-2

CUDA_VISIBLE_DEVICES=0 python engine/linear_probe_train.py \
  --task linear_probe \
  --checkpoint "${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}/epoch000010" \
  --loop-mode epoch \
  --total-epochs 10 \
  --validate-every-epochs 1 \
  --save-every-epochs 1 \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${LINEAR_RUN_ID}" \
  --run-name case2_2-cnn-dgnet-linear-probe \
  --seed 42 \
  "${COMMON_DATA_ARGS[@]}" \
  "${COMMON_MODEL_ARGS[@]}" \
  --classifier-lr 1.0e-3 \
  --weight-decay 1.0e-4

echo "pretrain_output=${OUTPUT_ROOT}/${PRETRAIN_RUN_ID}"
echo "linear_probe_output=${OUTPUT_ROOT}/${LINEAR_RUN_ID}"
