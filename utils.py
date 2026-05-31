"""Shared utilities for NeuralKNN-VC.

Everything operates at 16 kHz (the rate of WavLM-Large and the prematched
HiFi-GAN vocoder). Features are (T, D) for a single utterance; the kNN matcher
is the non-parametric core shared by the backbone teacher and the distillation
target generator.
"""

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def peak_norm(wav: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    """Peak-normalise a waveform to +/- `peak` (default 0.95)."""
    return wav / (wav.abs().max() + 1e-9) * peak


def load_audio(path, target_sr: int = 16000, max_sec: float | None = None) -> torch.Tensor:
    """Load an audio file as a 1D float32 tensor.

    mono (channel mean) -> peak-normalised to 0.95 -> resampled to target_sr ->
    optionally truncated to `max_sec` seconds.
    """
    wav_np, sr = sf.read(str(path))
    wav = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
    if wav.dim() > 1:
        wav = wav.mean(-1)
    wav = peak_norm(wav)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_sec is not None:
        wav = wav[: int(max_sec * target_sr)]
    return wav


def save_wav(path, wav: torch.Tensor, sr: int = 16000) -> None:
    """Write a 1D (or (1, T)) waveform tensor to `path` as a wav at `sr`."""
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu().float()
        if wav.dim() > 1:
            wav = wav.squeeze()
        wav = wav.numpy()
    else:
        wav = np.asarray(wav, dtype=np.float32).squeeze()
    sf.write(str(path), wav, sr)


def batched_cosine_knn(
    query: torch.Tensor,
    pool: torch.Tensor,
    topk: int = 4,
    chunk: int = 4096,
) -> torch.Tensor:
    """Core kNN matcher used by the backbone teacher and distillation.

    For every query frame, find the `topk` most cosine-similar frames in `pool`
    and return the MEAN of those neighbour feature vectors.

    Args:
        query: (Tq, D) source feature frames.
        pool:  (Np, D) target-speaker feature pool.
        topk:  number of nearest neighbours to average.
        chunk: query frames processed per block (bounds memory).

    Returns:
        (Tq, D) converted features (averaged neighbours), same dtype/device as query.
    """
    device = query.device
    pool = pool.to(device)
    # L2-normalise so dot product == cosine similarity.
    q = F.normalize(query.float(), dim=-1)
    p = F.normalize(pool.float(), dim=-1)
    pool_f = pool.float()

    k = min(topk, p.shape[0])
    out = torch.empty_like(query, dtype=torch.float32)

    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        sims = q[start:end] @ p.t()              # (c, Np) cosine similarities
        _, idx = sims.topk(k, dim=-1)            # (c, k) neighbour indices
        neigh = pool_f[idx]                      # (c, k, D)
        out[start:end] = neigh.mean(dim=1)       # average neighbour vectors

    return out.to(dtype=query.dtype)
