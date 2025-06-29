#!/usr/bin/env bash
###############################################################################
# RC-OP Voice Conversion – robust bootstrap
###############################################################################

set -Eeuo pipefail
trap 'echo -e "\033[1;31m[ERROR]\033[0m Setup failed at line $LINENO"; exit 1' ERR

# ── Config ───────────────────────────────────────────────────────────────────
ENV_NAME="rcop"
PYTHON_VERSION="3.10"
VCTK_URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
DATA_DIR="./data"
ZIP_NAME="VCTK-Corpus-0.92.zip"
CONDA_ROOT="$HOME/miniconda"         # single source of truth

# ── Pretty print ─────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m    $*"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m   $*"; }

###############################################################################
# 0. System packages (compiler, espeak-ng, libsndfile, wget, unzip, curl, aria2)
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
if (( EUID == 0 )); then pkg_install; else pkg_install sudo; fi

###############################################################################
# 1. Miniconda / Mambaforge bootstrap or repair
###############################################################################
info "Checking Conda…"
if ! command -v conda &>/dev/null; then
    if [[ -d "$CONDA_ROOT" ]]; then
        warn "$CONDA_ROOT exists but Conda not on PATH – repairing installation."
        INSTALL_FLAG="-u"                          # update in-place
    else
        info "Conda not installed – installing fresh copy."
        INSTALL_FLAG=""
    fi
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b $INSTALL_FLAG -p "$CONDA_ROOT"
    rm miniconda.sh
fi
export PATH="$CONDA_ROOT/bin:$PATH"
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"

###############################################################################
# 2. Environment creation / activation
###############################################################################
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    success "Conda env '$ENV_NAME' already exists"
else
    info "Creating env '$ENV_NAME' (Python $PYTHON_VERSION)…"
    if command -v mamba &>/dev/null; then
        mamba create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    else
        conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
    fi
fi
conda activate "$ENV_NAME"
success "Activated env '$ENV_NAME'"

###############################################################################
# 3. Python dependencies
###############################################################################
info "Installing Python requirements…"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
success "Python deps installed"

###############################################################################
# 4. Pre-trained weights
###############################################################################
info "Fetching model checkpoints (if needed)…"
if python download_models.py; then
    success "Models downloaded successfully"
else
    warn "Model download encountered issues - models will be downloaded automatically when first used"
fi

###############################################################################
# 5. VCTK corpus
###############################################################################
mkdir -p "$DATA_DIR"
pushd "$DATA_DIR" >/dev/null

if [[ ! -f $ZIP_NAME ]]; then
    info "Downloading VCTK (~22 GB)…"
    if command -v aria2c &>/dev/null; then
        aria2c -x 16 -s 16 -o "$ZIP_NAME" "$VCTK_URL"
    else
        wget -c "$VCTK_URL" -O "$ZIP_NAME"
    fi
else
    success "VCTK zip already exists – skipping download"
fi

if [[ ! -d VCTK-Corpus-0.92 ]]; then
    info "Extracting VCTK corpus…"
    unzip -q "$ZIP_NAME"
    success "Extraction complete"
else
    success "VCTK corpus already extracted"
fi
popd >/dev/null

###############################################################################
# 6. Project dirs
###############################################################################
mkdir -p checkpoints logs
success "checkpoints/ and logs/ ready"

###############################################################################
# 7. Recap
###############################################################################
cat <<EOF

─────────────────────────────  SETUP COMPLETE 🎉  ─────────────────────────────
Conda env : $ENV_NAME
Python    : $PYTHON_VERSION
Data root : \${PWD}/${DATA_DIR}/VCTK-Corpus-0.92
Checkpts  : ./checkpoints
Logs      : ./logs

Next steps
──────────
1)  conda activate $ENV_NAME
2)  export VCTK_ROOT=\${PWD}/${DATA_DIR}/VCTK-Corpus-0.92
3)  python train.py --data_root \$VCTK_ROOT --epochs 20

Happy training!
EOF
