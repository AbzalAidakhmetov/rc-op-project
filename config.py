from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class Config:
    device: str         = "cuda"          # "cuda" or "cpu"
    seed:   int         = 42
    run_name: Optional[str] = None # For wandb tracking

    # data
    target_sr:  int     = 16_000
    subset:     int     = 500             # training subset size
    batch_size: int     = 16

    # model dims
    d_ssl: int = 1024                      # WavLM-Large hidden size
    d_spk: int = 192                       # SpeechBrain ECAPA-TDNN size
    
    # mel spectrogram
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    f_min: int = 0
    f_max: int = 8000

    # training
    epochs:        int     = 20
    lr:            float   = 1e-4
    lambda_ramp:   bool    = True 
    finetune_layers: int   = 8               # Num final layers to finetune in vocoder head
    
    # loss weights
    lambda_ph:     float   = 1.0
    lambda_sp:     float   = 1.0
    lambda_recon:  float   = 1.0
    lambda_spectral: float = 1.0
    
    # multi-resolution stft loss
    stft_loss_fft_sizes: List[int] = field(default_factory=lambda: [1024, 2048, 512])
    stft_loss_hop_sizes: List[int] = field(default_factory=lambda: [120, 240, 50])
    stft_loss_win_sizes: List[int] = field(default_factory=lambda: [600, 1200, 240]) 