"""Neural model components for NeuralKNN-VC.

Exposes the pool-free NeuralConverter (the novel contribution) and the frozen
ECAPA SpeakerEncoder used to condition it.
"""

from models.converter import NeuralConverter, ResidualBlock1D, SinusoidalPosEmb
from models.speaker import SpeakerEncoder

__all__ = [
    "NeuralConverter",
    "ResidualBlock1D",
    "SinusoidalPosEmb",
    "SpeakerEncoder",
]
