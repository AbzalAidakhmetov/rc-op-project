# RC-OP: Representation Controllable Orthogonal Projection

A clean, modular implementation of RC-OP for voice conversion using WavLM-Large SSL features and orthogonal projection. This implementation is designed to run on VAST.ai GPU instances without Docker or Colab dependencies.

## Overview

RC-OP (Representation Controllable Orthogonal Projection) is a voice conversion method that:
- Uses WavLM-Large for self-supervised speech representations
- Employs Resemblyzer for speaker embeddings  
- Applies orthogonal projection to remove speaker information from content features
- Uses gradient reversal to ensure speaker-agnostic content representations

## Project Structure

```
rc-op/
├── README.md
├── requirements.txt
├── config.py                 # Configuration settings
├── train.py                  # Training script
├── evaluate.py               # Evaluation script  
├── infer.py                  # Inference script
├── data/
│   ├── __init__.py
│   └── vctk.py              # VCTK dataset handling
├── models/
│   ├── __init__.py
│   ├── rcop.py              # Main RC-OP model
│   ├── grad_reverse.py      # Gradient reversal layer
│   └── projection.py        # Orthogonal projection
└── utils/
    ├── __init__.py
    ├── phonemes.py          # Phoneme utilities
    └── logging.py           # Logging utilities
```

## Quick Start (VAST.ai)

### 1. Setup Environment

```bash
# Launch a VAST.ai instance (Ubuntu 22.04, CUDA ≥ 12) and SSH in
sudo apt update && sudo apt install -y git

# Clone or copy the project
git clone <your-repo-url> rc-op
cd rc-op

# Create Python environment
conda create -n rcop python=3.10 -y && conda activate rcop
# OR
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Set VCTK data path
export VCTK_ROOT=/path/to/VCTK-Corpus-0.92

# Create necessary directories
mkdir -p checkpoints logs
```

### 3. Train Model

```bash
# Quick training run (500 samples, 20 epochs)
python train.py \
  --data_root $VCTK_ROOT \
  --save_dir checkpoints \
  --epochs 20 \
  --subset 500 \
  --batch_size 1

# Full training (all VCTK data)
python train.py \
  --data_root $VCTK_ROOT \
  --save_dir checkpoints \
  --epochs 50 \
  --subset 44000 \
  --batch_size 1
```

### 4. Evaluate Model

```bash
python evaluate.py \
  --ckpt checkpoints/rcop_epoch20.pt \
  --data_root $VCTK_ROOT \
  --subset 100
```

### 5. Voice Conversion Inference

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
    batch_size: int = 1
    
    # Model dimensions
    d_ssl: int = 1024      # WavLM-Large hidden size
    d_spk: int = 256       # Resemblyzer size
    
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
4. **Speaker Embeddings**: Resemblyzer provides speaker characteristics

## Training Details

- **Lambda Scheduling**: Gradually increases adversarial loss weight
- **Frozen Models**: WavLM and Resemblyzer remain frozen during training
- **Content Loss**: Cross-entropy on phoneme predictions
- **Speaker Loss**: Cross-entropy on speaker predictions (reversed gradients)

## Evaluation Metrics

1. **Speaker Classification Accuracy**: How well the model can predict speakers
2. **Content Feature Consistency**: Similarity of content features within speakers
3. **Speaker Embedding Consistency**: Validation of speaker representation quality

## Tips for VAST.ai

- **Disk Speed**: Use NVMe instances for better I/O performance with large datasets
- **Memory**: Ensure sufficient GPU memory for WavLM-Large (requires ~4GB)
- **Checkpointing**: Model saves checkpoints every epoch for recovery
- **Subset Training**: Use `--subset` for quick experimentation

### Cost Optimization

```bash
# Smoke test (very fast, low cost)
python train.py --data_root $VCTK_ROOT --epochs 1 --subset 100

# Development run (moderate cost)
python train.py --data_root $VCTK_ROOT --epochs 10 --subset 1000

# Full training (production quality)
python train.py --data_root $VCTK_ROOT --epochs 50 --subset 44000
```

## Dependencies

Key packages:
- `torch>=2.2`: PyTorch framework
- `transformers>=4.41`: WavLM model
- `resemblyzer`: Speaker embedding extraction
- `vocoder==0.1.3`: HiFi-GAN vocoder for synthesis
- `soundfile`: Audio I/O
- `resampy`: Audio resampling

## Inference Options

### Basic Inference
```bash
python infer.py --ckpt model.pt --source_wav src.wav --ref_wav ref.wav --out_wav out.wav
```

### With HiFi-GAN Vocoder
```bash
python infer.py --ckpt model.pt --source_wav src.wav --ref_wav ref.wav --out_wav out.wav --use_vocoder
```

## Resuming Training

```bash
python train.py \
  --data_root $VCTK_ROOT \
  --resume checkpoints/rcop_epoch10.pt \
  --epochs 20
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size (keep at 1) or use smaller subset
2. **Audio Loading Errors**: Ensure VCTK data is properly extracted
3. **Model Loading**: Check checkpoint path and model compatibility
4. **Slow Training**: Use NVMe storage and ensure proper CUDA setup

### Performance Tips

- Use `--subset` for development and testing
- Monitor GPU memory usage during training
- Use mixed precision training for larger models (future enhancement)

## Future Enhancements

- Multi-GPU training support
- Mixed precision training
- Advanced vocoder integration
- Real-time inference optimization
- Support for other SSL models (Wav2Vec2, HuBERT)

## Citation

If you use this implementation, please cite the original RC-OP work and relevant model papers:

```bibtex
@article{wavlm,
  title={WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing},
  author={Chen, Sanyuan and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2022}
}

@misc{resemblyzer,
  title={Resemblyzer},
  author={Титов, Corentin},
  howpublished={\url{https://github.com/resemble-ai/Resemblyzer}},
  year={2019}
}
```

## License

This implementation is provided for research and educational purposes. Please respect the licenses of the underlying models (WavLM, Resemblyzer, etc.). 