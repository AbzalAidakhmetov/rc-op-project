"""Central configuration for NeuralKNN-VC.

A single dataclass holds every tunable knob for the kNN-VC quality backbone and
the novel pool-free neural converter. `from_args` lets any CLI (built with
argparse) override fields by attribute name.
"""

from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class Config:
    # ---- WavLM / feature space (kNN-VC backbone) ----
    wavlm_sr: int = 16000          # WavLM-Large + HiFi-GAN operate at 16 kHz
    wavlm_layer: int = 6           # layer 6 (1-indexed, as in the kNN-VC paper) = VC sweet spot
    wavlm_dim: int = 1024          # WavLM-Large hidden size
    ecapa_dim: int = 192           # ECAPA-TDNN speaker embedding size

    # ---- kNN matcher ----
    knn_topk: int = 4              # k for cosine-kNN averaging
    knn_pool_seconds: Optional[float] = None  # None = use whole reference for the pool

    # ---- converter network (pool-free neural converter) ----
    hidden_dim: int = 512          # ResNet hidden channels
    num_res_blocks: int = 8        # number of dilated FiLM residual blocks

    # ---- distillation training ----
    batch_size: int = 8
    lr: float = 2e-4
    steps: int = 50000
    warmup_steps: int = 500
    grad_clip: float = 1.0
    crop_sec: float = 3.0
    use_cosine_loss: bool = True

    # ---- data caps (smoke knobs; None = use everything) ----
    data_dir: str = "data/librispeech/LibriSpeech/dev-clean"
    max_speakers: Optional[int] = None
    max_files_per_speaker: Optional[int] = None

    # ---- runtime ----
    output_dir: str = "outputs"
    device: str = "cuda"


def from_args(args) -> "Config":
    """Build a Config, overriding any field whose name matches an attribute on
    `args` (an argparse.Namespace or any object). Only non-None overrides are
    applied for optional fields so a missing CLI flag keeps the default.

    argparse stores dashed flags (``--max-speakers``) as underscored attrs
    (``max_speakers``), so attribute names line up with Config field names.
    """
    cfg = Config()
    for f in fields(Config):
        if hasattr(args, f.name):
            val = getattr(args, f.name)
            if val is not None:
                setattr(cfg, f.name, val)
    return cfg
