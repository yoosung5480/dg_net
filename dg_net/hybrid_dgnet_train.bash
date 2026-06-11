#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"

DATA_PATH="${DATA_PATH:-/home/cv202355581/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/hybrid}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-20}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-500}"
LINEAR_EPOCHS="${LINEAR_EPOCHS:-100}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"

PRETRAIN_RUN_ID="${PRETRAIN_RUN_ID:-case1-hybrid-dgnet-pretrain-${STAMP}}"
LINEAR_RUN_ID="${LINEAR_RUN_ID:-case1-hybrid-dgnet-linear-probe-${STAMP}}"

exec bash "${SCRIPT_DIR}/scripts/run_dgnet_pipeline_ddp.bash" "$@"
