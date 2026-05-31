#!/usr/bin/env bash
#
# download_data.sh -- fetch the LibriSpeech dev-clean corpus for NeuralKNN-VC.
#
# Idempotent: safe to re-run. Skips download/extraction if already present.
#
# Result layout (matches Config.data_dir default):
#   data/librispeech/LibriSpeech/dev-clean/<speaker>/<chapter>/*.flac
#
# dev-clean is ~337 MB, 16 kHz FLAC, 40 speakers. Perfect for the 16 kHz
# kNN-VC pipeline and for distilling the pool-free neural converter on a
# single 12 GB GPU.
#
# ----------------------------------------------------------------------------
# VCTK ALTERNATIVE (not downloaded here):
#   VCTK is ~11 GB at 48 kHz (data/vctk/wav48_silence_trimmed/p###/*_mic1.flac).
#   AudioFolderDataset already globs that layout, but you MUST keep everything
#   at 16 kHz (utils.load_audio resamples on the fly). To use it instead:
#       bash download_vctk.sh
#       ... --data-dir data/vctk/wav48_silence_trimmed
#   dev-clean is recommended for fast iteration; VCTK gives more speakers.
# ----------------------------------------------------------------------------

set -euo pipefail

# Resolve repo root (this script lives in <root>/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${ROOT_DIR}/data/librispeech"
URL="https://www.openslr.org/resources/12/dev-clean.tar.gz"
TARBALL="${DATA_DIR}/dev-clean.tar.gz"
EXTRACT_DIR="${DATA_DIR}/LibriSpeech/dev-clean"

mkdir -p "${DATA_DIR}"

echo "========================================================"
echo "NeuralKNN-VC :: LibriSpeech dev-clean download"
echo "========================================================"
echo "  Target : ${EXTRACT_DIR}"
echo ""

if [ -d "${EXTRACT_DIR}" ] && [ -n "$(ls -A "${EXTRACT_DIR}" 2>/dev/null)" ]; then
    echo "Already extracted -- skipping download/extract."
else
    if [ -f "${TARBALL}" ]; then
        echo "Tarball already present -- skipping download."
    else
        echo "Downloading dev-clean (~337 MB) from:"
        echo "  ${URL}"
        if command -v wget >/dev/null 2>&1; then
            wget -c -O "${TARBALL}" "${URL}"
        else
            curl -L -C - -o "${TARBALL}" "${URL}"
        fi
    fi

    echo "Extracting ..."
    tar -xzf "${TARBALL}" -C "${DATA_DIR}"
fi

# Report speaker / file counts.
NUM_SPEAKERS=$(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
NUM_FILES=$(find "${EXTRACT_DIR}" -name "*.flac" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "========================================================"
echo "Done."
echo "  Location : ${EXTRACT_DIR}"
echo "  Speakers : ${NUM_SPEAKERS}"
echo "  Files    : ${NUM_FILES}"
echo "========================================================"
echo ""
echo "Next: instant SOTA-quality demo with NO training:"
echo "  .venv/bin/python infer.py --backend knn \\"
echo "      --data-dir data/librispeech/LibriSpeech/dev-clean"
