from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    """Configuration for Voice Conversion with Rectified Flow Matching."""
    
    # Device and reproducibility
    device: str = "cuda"
    seed: int = 42
    run_name: Optional[str] = None  # For wandb tracking
    
    # Core parameters (from prompt requirements)
    SAMPLE_RATE: int = 16000
    WAVLM_DIM: int = 768           # wavlm-base-plus hidden size
    LATENT_DIM: int = 768          # Flow in native WavLM space
    BATCH_SIZE: int = 8            # RTX 3090 friendly
    LR: float = 1e-4
    NUM_STEPS: int = 20000         # Training steps (not epochs)
    
    # Flow matching specific
    num_timesteps: int = 1000      # Number of timesteps for flow
    sigma_min: float = 1e-4        # Minimum noise level
    
    # Speaker embedding
    d_spk: int = 192               # ECAPA-TDNN embedding dimension
    
    # Mel spectrogram parameters (for decoder target)
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 8000
    
    # SVD projection parameters
    svd_rank: int = 64             # Top-k singular vectors for speaker subspace
    svd_num_samples: int = 10000   # Number of random frames for SVD computation
    
    # Flow network architecture
    flow_hidden_dim: int = 512     # Hidden dimension for flow network
    flow_num_layers: int = 6       # Number of transformer layers
    flow_num_heads: int = 8        # Number of attention heads
    flow_dropout: float = 0.1     # Dropout rate
    
    # Decoder architecture
    decoder_hidden_dim: int = 512  # Hidden dimension for decoder
    decoder_num_layers: int = 4    # Number of conv layers
    
    # Training settings
    grad_clip: float = 1.0         # Gradient clipping norm
    warmup_steps: int = 1000       # Learning rate warmup steps
    log_interval: int = 100        # Log every N steps
    save_interval: int = 2000      # Save checkpoint every N steps
    eval_interval: int = 1000      # Evaluate every N steps
    
    # Data
    max_duration_s: float = 10.0   # Maximum audio duration in seconds
    num_workers: int = 4           # DataLoader workers
    
    # Vocoder
    vocoder_name: str = "microsoft/speecht5_hifigan"
    
    # WavLM model
    wavlm_name: str = "microsoft/wavlm-base-plus"
    wavlm_layer: int = 6           # Which layer to extract (6 or -1 for last)
    
    # Speaker model
    speaker_model_name: str = "speechbrain/spkrec-ecapa-voxceleb"
