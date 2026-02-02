#!/usr/bin/env bash
###############################################################################
# Voice Conversion Flow Matching – uv-based bootstrap
###############################################################################

set -Eeuo pipefail
trap 'echo -e "\033[1;31m[ERROR]\033[0m Setup failed at line $LINENO"; exit 1' ERR

# ── Config ───────────────────────────────────────────────────────────────────
PYTHON_VERSION="3.10"
# LibriTTS dev-clean: ~1.2GB, ~40 speakers, perfect for quick experiments
LIBRITTS_URL="https://www.openslr.org/resources/60/dev-clean.tar.gz"
DATA_DIR="./data"
TAR_NAME="dev-clean.tar.gz"

# ── Pretty print ─────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m    $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m    $*"; }

###############################################################################
# 0. System packages (espeak-ng, libsndfile, aria2)
###############################################################################
pkg_install() {
  local sudo_prefix=("$@")           # empty when already root
  if command -v apt-get &>/dev/null; then
      "${sudo_prefix[@]}" apt-get update -qq
      "${sudo_prefix[@]}" apt-get install -y build-essential espeak-ng libsndfile1 wget unzip curl aria2
  elif command -v yum &>/dev/null; then
      "${sudo_prefix[@]}" yum install -y gcc gcc-c++ make espeak-ng libsndfile wget unzip curl aria2
  else
      warn "No apt-get or yum found – install build tools, espeak-ng and libsndfile manually."
  fi
}
info "Installing system dependencies..."
if (( EUID == 0 )); then pkg_install; else pkg_install sudo; fi
success "System dependencies installed"

###############################################################################
# 1. Install uv (fast Python package manager)
###############################################################################
info "Checking for uv..."
if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell config for persistence
    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi
    success "uv installed"
else
    success "uv already installed"
fi

# Ensure uv is in PATH for this session
export PATH="$HOME/.local/bin:$PATH"

###############################################################################
# 2. Create virtual environment and install dependencies
###############################################################################
info "Creating Python $PYTHON_VERSION virtual environment..."
if [[ -d ".venv" ]]; then
    warn ".venv already exists – reusing"
else
    uv venv .venv --python "$PYTHON_VERSION"
    success "Virtual environment created"
fi

info "Activating virtual environment..."
source .venv/bin/activate

info "Installing Python dependencies with uv..."
uv pip install -e .
success "Python dependencies installed"

###############################################################################
# 3. Pre-trained weights
###############################################################################
info "Fetching model checkpoints (if needed)..."
if python download_models.py; then
    success "Models downloaded successfully"
else
    warn "Model download encountered issues - models will be downloaded automatically when first used"
fi

###############################################################################
# 4. LibriTTS dev-clean corpus (~1.2 GB, ~40 speakers)
###############################################################################
mkdir -p "$DATA_DIR"
pushd "$DATA_DIR" >/dev/null

download_libritts() {
    info "Downloading LibriTTS dev-clean (~1.2 GB)..."
    if command -v aria2c &>/dev/null; then
        aria2c -x 16 -s 16 -o "$TAR_NAME" "$LIBRITTS_URL"
    else
        wget -c "$LIBRITTS_URL" -O "$TAR_NAME"
    fi
}

if [[ ! -f $TAR_NAME ]]; then
    download_libritts
else
    success "LibriTTS tar.gz already exists – skipping download"
fi

if [[ ! -d LibriTTS/dev-clean ]]; then
    info "Extracting LibriTTS dev-clean..."
    tar -xzf "$TAR_NAME"
    # Move to expected location
    mkdir -p LibriTTS
    if [[ -d dev-clean ]]; then
        mv dev-clean LibriTTS/
    fi
    success "Extraction complete"
else
    success "LibriTTS dev-clean already extracted"
fi
popd >/dev/null

###############################################################################
# 5. Project dirs
###############################################################################
mkdir -p checkpoints logs preprocessed results
success "checkpoints/, logs/, preprocessed/, and results/ directories ready"

###############################################################################
# 6. Recap
###############################################################################
cat <<EOF

─────────────────────────────  SETUP COMPLETE  ─────────────────────────────
Environment : .venv (uv-managed)
Python      : $PYTHON_VERSION
Data root   : \${PWD}/${DATA_DIR}/LibriTTS/dev-clean
Checkpts    : ./checkpoints
Logs        : ./logs
Preprocessed: ./preprocessed

Dataset Info
────────────
LibriTTS dev-clean: ~40 speakers, ~1.2 GB
Perfect for quick A/B testing of Baseline CFM vs SG-Flow

Next steps
──────────
1)  source .venv/bin/activate
2)  python data/preprocess.py --data_root ./data/LibriTTS/dev-clean --output_dir ./preprocessed
3)  python utils/svd_projection.py --data_dir ./preprocessed --output ./preprocessed/projection_matrix.pt
4)  python train.py --mode baseline --data_dir ./preprocessed
5)  python train.py --mode sg_flow --data_dir ./preprocessed

Happy training!
EOF
