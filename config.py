from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    device: str         = "cuda"          # "cuda" or "cpu"
    seed:   int         = 42
    run_name: Optional[str] = None # For wandb tracking

    # data
    target_sr:  int     = 16_000
    subset:     int     = 500             # training subset size
    batch_size: int     = 1

    # model dims
    d_ssl: int = 1024                      # WavLM-Large hidden size
    d_spk: int = 256                       # Resemblyzer size

    # training
    epochs:        int     = 20
    lr:            float   = 1e-4
    lambda_ramp:   bool    = True 
    
    # loss weights
    lambda_ph:     float   = 1.0
    lambda_sp:     float   = 1.0
    lambda_recon:  float   = 2.5 