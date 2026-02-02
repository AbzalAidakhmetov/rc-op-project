# Voice Conversion with Rectified Flow Matching

A Voice Conversion system comparing two flow matching approaches:

1. **Baseline CFM:** Standard Conditional Flow Matching starting from Gaussian Noise
2. **SG-Flow:** Rectified Flow starting from Orthogonally Projected Content Subspace

## Core Hypothesis

The SG-Flow method removes speaker information via SVD projection before the flow process, potentially leading to:
- Better content preservation
- Cleaner speaker conversion
- More interpretable latent space

## Architecture

- **Backbone:** `microsoft/wavlm-base-plus` (Frozen, 768-dim)
- **Flow Target:** Native WavLM features (768-dim)
- **Conditioning:** Target Speaker Embedding (ECAPA-TDNN, 192-dim)
- **Decoder:** 1D ResNet mapping WavLM → Mel-Spectrogram
- **Vocoder:** Pre-trained HiFi-GAN

## Project Structure

```
voice-conversion-flow/
├── config.py                 # Configuration (SAMPLE_RATE, WAVLM_DIM, etc.)
├── train.py                  # Training with phased learning + AMP
├── inference.py              # Inference with Euler ODE solver
├── playground.ipynb          # Quick experimentation notebook
├── setup.sh                  # One-command setup (uv-based)
├── pyproject.toml            # Dependencies
│
├── data/
│   ├── preprocess.py         # Feature extraction (LibriTTS/VCTK)
│   ├── dataset.py            # PyTorch Dataset for precomputed features
│   └── vctk.py               # Legacy VCTK loader
│
├── models/
│   ├── flow_network.py       # Transformer-based velocity prediction
│   ├── flow_matching.py      # BaselineCFM and SGFlow implementations
│   ├── decoder.py            # 1D ResNet decoder (WavLM → Mel)
│   ├── projection.py         # Orthogonal projection module
│   └── system.py             # VoiceConversionSystem wrapper
│
└── utils/
    ├── svd_projection.py     # SVD computation for speaker subspace
    ├── checkpoint.py         # Model checkpointing
    └── logging.py            # Logging utilities
```

## Quick Start

### 1. Setup Environment

```bash
# One-command setup (downloads LibriTTS dev-clean ~1.2GB)
./setup.sh

# Activate environment
source .venv/bin/activate
```

### 2. Preprocess Data

```bash
# Extract WavLM features, speaker embeddings, and mel spectrograms
python data/preprocess.py \
    --data_root ./data/LibriTTS/dev-clean \
    --output_dir ./preprocessed

# Compute SVD projection matrix
python utils/svd_projection.py \
    --data_dir ./preprocessed \
    --output ./preprocessed/projection_matrix.pt
```

### 3. Train Models

```bash
# Train Baseline CFM (from Gaussian noise)
python train.py --mode baseline --data_dir ./preprocessed

# Train SG-Flow (from content subspace)
python train.py --mode sg_flow --data_dir ./preprocessed
```

### 4. Inference

```bash
python inference.py \
    --checkpoint checkpoints/sg_flow_best.pt \
    --source_wav path/to/source.wav \
    --ref_wav path/to/reference.wav \
    --output_dir results/
```

## Training Details

### Phased Training

- **Phase A (Steps 1-2000):** Train Decoder only (Flow frozen)
  - Ensures mel reconstruction works before training flow
- **Phase B (Steps 2001-20000):** Train Flow + Decoder jointly

### Key Parameters

```python
SAMPLE_RATE = 16000
WAVLM_DIM = 768        # wavlm-base-plus hidden size
BATCH_SIZE = 8         # RTX 3090 friendly
LR = 1e-4
NUM_STEPS = 20000      # Total training steps
svd_rank = 64          # Speaker subspace dimensions
```

## Dataset

**LibriTTS dev-clean** (recommended for quick experiments):
- Size: ~1.2 GB
- Speakers: ~40 speakers
- Format: 24kHz WAV, sentence-level segments
- Download: Automatic via `setup.sh`

Also supports VCTK (auto-detected by preprocessing script).

## Playground Notebook

For quick experimentation without full training:

```bash
jupyter notebook playground.ipynb
```

The notebook includes:
- Model architecture verification
- Loss computation comparison
- Quick training on synthetic data
- ODE solver visualization

## Flow Matching Methods

### Baseline CFM
```
x_0 ~ N(0, I)                    # Start from Gaussian noise
x_t = (1-t)*x_0 + t*x_1          # Linear interpolation
v_target = x_1 - x_0             # Target velocity
Loss = MSE(v_pred, v_target)     # Flow matching loss
```

### SG-Flow
```
x_0 = P_content @ x_1            # Start from content projection
x_t = (1-t)*x_0 + t*x_1          # Linear interpolation
v_target = x_1 - x_0             # Target velocity (smaller!)
Loss = MSE(v_pred, v_target)     # Flow matching loss
```

The key insight: SG-Flow starts closer to the target, so the flow only needs to learn the speaker-specific transformation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (RTX 3090 recommended)
- ~4GB GPU memory for training

## License

MIT
