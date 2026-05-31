#!/usr/bin/env bash
#
# smoke_test.sh -- tiny end-to-end check of the full NeuralKNN-VC pipeline.
#
# Parameterised to finish fast on an RTX 3060 (12 GB). It:
#   1. Runs a PURE kNN-VC conversion (--backend knn) -- proves SOTA-quality
#      audio with NO training (pretrained WavLM-Large + prematched HiFi-GAN).
#   2. Distils the pool-free neural converter for a handful of steps (tiny).
#   3. Runs inference with the trained neural converter (--backend neural).
#   4. Benchmarks both backends on a couple of held-out pairs.
#
# This is a SMOKE test (correctness/plumbing), not a quality run. For real
# quality, bump --steps (e.g. 50000) and drop the --max-* caps.
#
# Usage:
#   bash scripts/smoke_test.sh
# Override knobs via env vars, e.g.:
#   STEPS=50 MAX_SPK=6 bash scripts/smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PY="${ROOT_DIR}/.venv/bin/python"
if [ ! -x "${PY}" ]; then
    PY="python"
fi

# ---- tiny smoke knobs (override via env) ----
DATA_DIR="${DATA_DIR:-data/librispeech/LibriSpeech/dev-clean}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke}"
STEPS="${STEPS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_SPK="${MAX_SPK:-4}"
MAX_FILES="${MAX_FILES:-8}"
POOL_UTTS="${POOL_UTTS:-4}"
CROP_SEC="${CROP_SEC:-2.0}"
TOPK="${TOPK:-4}"
NUM_PAIRS="${NUM_PAIRS:-2}"

CKPT="${OUTPUT_DIR}/converter.pt"

echo "========================================================"
echo "NeuralKNN-VC :: SMOKE TEST"
echo "  python     : ${PY}"
echo "  data-dir   : ${DATA_DIR}"
echo "  output-dir : ${OUTPUT_DIR}"
echo "  steps      : ${STEPS}  batch ${BATCH_SIZE}  crop ${CROP_SEC}s"
echo "  caps       : max-speakers ${MAX_SPK}  max-files ${MAX_FILES}"
echo "========================================================"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: data dir '${DATA_DIR}' not found."
    echo "Run: bash scripts/download_data.sh"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo ""
echo "[1/4] Pure kNN-VC conversion (NO training, SOTA-quality backbone) ..."
"${PY}" infer.py \
    --backend knn \
    --data-dir "${DATA_DIR}" \
    --topk "${TOPK}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "[2/4] Distilling pool-free neural converter (${STEPS} tiny steps) ..."
"${PY}" -m distill.train \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --crop-sec "${CROP_SEC}" \
    --max-speakers "${MAX_SPK}" \
    --max-files-per-speaker "${MAX_FILES}" \
    --pool-utts "${POOL_UTTS}" \
    --topk "${TOPK}" \
    --no-wandb

echo ""
echo "[3/4] Neural-converter inference (pool-free, single forward pass) ..."
"${PY}" infer.py \
    --backend neural \
    --converter "${CKPT}" \
    --data-dir "${DATA_DIR}" \
    --topk "${TOPK}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "[4/4] Benchmark: kNN vs neural (ECAPA sim/leak + RTF) ..."
"${PY}" benchmark.py \
    --converter "${CKPT}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-pairs "${NUM_PAIRS}" \
    --topk "${TOPK}"

echo ""
echo "========================================================"
echo "SMOKE TEST COMPLETE"
echo "  Outputs   : ${OUTPUT_DIR}/"
echo "  Benchmark : ${OUTPUT_DIR}/benchmark.md"
echo "========================================================"
