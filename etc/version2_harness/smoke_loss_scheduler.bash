#!/usr/bin/env bash
set -euo pipefail

# Smoke test for loss config, term on/off, term-wise scheduler, and tensor-only forward.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="${SRC_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${SRC_ROOT}"

mkdir -p loss/output
python -m loss.loss

echo "loss smoke output=loss/output"
