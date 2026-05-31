"""NeuralKNN-VC quality backbone.

This package wraps the pretrained kNN-VC teacher (Baas, van Niekerk, Kamper,
"Voice Conversion With Just Nearest Neighbors", Interspeech 2023): WavLM-Large
for feature extraction (layer 6) plus a prematched HiFi-GAN vocoder, both loaded
from the official ``bshall/knn-vc`` torch.hub repo. Everything is frozen and
runs under ``torch.no_grad`` -- this is the teacher we distil the pool-free
neural converter from.
"""

from .knnvc import KNNVC

__all__ = ["KNNVC"]
