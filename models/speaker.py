"""Frozen ECAPA-TDNN speaker encoder (192-dim embeddings).

Ported from the proven SpeakerEncoder in the legacy main.py. Produces a single
192-dim speaker embedding per utterance, used to condition the pool-free
NeuralConverter.
"""

# --- torchaudio compat shim (MUST run before speechbrain import) ---
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier


class SpeakerEncoder(nn.Module):
    """Frozen ECAPA-TDNN producing 192-dim speaker embeddings.

    Args:
        device: torch device string ('cuda' or 'cpu').
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa_voxceleb",
            run_opts={"device": device},
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Encode 16 kHz audio into a 192-dim speaker embedding.

        Args:
            audio_16k: (B, T) or (T,) waveform tensor at 16 kHz.

        Returns:
            (B, 192) speaker embeddings.
        """
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        audio_16k = audio_16k.to(self.device)
        # encode_batch -> (B, 1, 192); squeeze the singleton frame dim.
        emb = self.encoder.encode_batch(audio_16k)
        return emb.squeeze(1)
