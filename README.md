# RC-OP: Reference-Conditioned Orthogonal Projection

A clean, modular implementation of RC-OP for voice conversion using WavLM-Large SSL features and orthogonal projection. This implementation is designed to run on VAST.ai GPU instances without Docker or Colab dependencies.

## Overview

RC-OP (Reference-Conditioned Orthogonal Projection) is a voice conversion method that:
- Uses WavLM-Large for self-supervised speech representations
- Employs a pre-trained SpeechBrain model for speaker embeddings
- Applies orthogonal projection to explicitly remove speaker information from content features by projecting each frame onto the subspace orthogonal to a learned speaker axis
- Uses gradient reversal to ensure speaker-agnostic content representations

## Project Structure

```
rc-op/
├── README.md
├── requirements.txt
├── setup.sh                    # Single comprehensive setup script
├── config.py                   # Configuration settings
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script  
├── infer.py                    # Inference script
├── data/
│   ├── __init__.py
│   └── vctk.py                # VCTK dataset handling
├── models/
│   ├── __init__.py
│   ├── rcop.py                # Main RC-OP model
│   ├── grad_reverse.py        # Gradient reversal layer
│   └── projection.py          # Orthogonal projection
└── utils/
    ├── __init__.py
    ├── phonemes.py            # Phoneme utilities
    └── logging.py             # Logging utilities
```

## Quick Setup

### One-Command Installation

```bash
# Run the comprehensive setup script
bash setup.sh
```

This single script will:
1. ✅ Install Miniconda if not present
2. ✅ Create and activate the `rcop` conda environment
3. ✅ Install all Python dependencies
4. ✅ Download pre-trained models (WavLM-Large, etc.)
5. ✅ Download and extract the VCTK dataset
6. ✅ Create necessary directories (checkpoints, logs)
7. ✅ Verify the complete setup

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create Python environment
conda create -n rcop python=3.10 -y && conda activate rcop
# OR
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Download VCTK dataset
bash download_vctk.sh ./data

# Create directories
mkdir -p checkpoints logs
```

## Usage

### Training

```bash
# Activate environment (if not already active)
conda activate rcop

# Set VCTK data path
export VCTK_ROOT=./data/VCTK-Corpus-0.92

# Start training
python train.py \
  --data_root $VCTK_ROOT \
  --save_dir checkpoints \
  --epochs 20 \
  --subset 500 \
  --batch_size 16
```

### Evaluation

```bash
python evaluate.py \
  --ckpt checkpoints/rcop_epoch20.pt \
  --data_root $VCTK_ROOT \
  --subset 100
```

### Inference

```bash
python infer.py \
  --ckpt checkpoints/rcop_epoch20.pt \
  --source_wav /path/to/source.wav \
  --ref_wav /path/to/reference.wav \
  --out_wav output.wav \
  --use_vocoder
```

## Configuration

The `config.py` file contains all hyperparameters:

```python
@dataclass
class Config:
    device: str = "cuda"
    seed: int = 42
    
    # Data
    target_sr: int = 16_000
    subset: int = 500
    batch_size: int = 16
    
    # Model dimensions
    d_ssl: int = 1024      # WavLM-Large hidden size
    d_spk: int = 192       # SpeechBrain ECAPA-TDNN size
    
    # Training
    epochs: int = 20
    lr: float = 1e-4
    lambda_ramp: bool = True
```

## Model Architecture

### RC-OP Model
- **Speaker Projection Layer**: Linear layer that learns speaker-specific axes
- **Phoneme Classifier**: Predicts phonemes from projected features
- **Speaker Classifier**: Adversarial classifier with gradient reversal
- **Orthogonal Projection**: Removes speaker information from SSL features

### Key Components

1. **Gradient Reversal**: Ensures content features are speaker-agnostic
2. **Orthogonal Projection**: Mathematical removal of speaker components
3. **SSL Features**: WavLM-Large provides rich content representations
4. **Speaker Embeddings**: SpeechBrain `spkrec-ecapa-voxceleb` provides speaker characteristics

## Dependencies

Key packages:
- `torch>=2.2`: PyTorch framework
- `transformers>=4.41`: WavLM and HiFi-GAN models
- `speechbrain>=0.5.16`: Speaker embedding extraction
- `timm>=0.9.16`: Required by SpeechBrain models
- `soundfile`: Audio I/O
- `resampy`: Audio resampling

## Troubleshooting

### Common Issues

1. **Conda not found**: The setup script will automatically install Miniconda
2. **CUDA out of memory**: Reduce batch size or use CPU training
3. **VCTK download fails**: Check internet connection and try again
4. **Model download fails**: Ensure transformers cache directory is writable

### WavLM Model Download Issues

If you encounter errors during model download like:
```
OSError: Can't load tokenizer for 'microsoft/wavlm-large'
```

This is usually due to:
- **Network connectivity issues**: Check your internet connection
- **Transformers version compatibility**: The setup uses transformers>=4.41,<4.45
- **Model availability**: The model should be available on HuggingFace

**Solutions:**
1. **Automatic retry**: The setup script will continue even if model download fails. Models will be downloaded automatically when first used.
2. **Manual download**: After setup, run:
   ```bash
   conda activate rcop
   python download_models.py
   ```
3. **Test setup**: After setup, verify everything works:
   ```bash
   conda activate rcop
   python test_setup.py
   ```

### Environment Variables

```bash
# Set these for optimal performance
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
export VCTK_ROOT=./data/VCTK-Corpus-0.92  # VCTK dataset path
``` 