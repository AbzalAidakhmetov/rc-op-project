"""Pool-free neural converter (the project's novel contribution).

Distilled from the kNN-VC teacher: a parametric, speaker-conditioned network
that maps source WavLM-Large features to target-speaker features in a single
forward pass -- with NO target feature pool at inference, only one ECAPA
speaker embedding.

Reuses the FiLM-conditioned 1D ResNet from the legacy main.py
(SinusoidalPosEmb, ResidualBlock1D), adapted from velocity prediction to a
feature->feature delta regressor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional / timestep embedding.

    Ported verbatim from main.py. Not used by the NeuralConverter forward pass
    (there is no flow timestep), but provided per the interface spec so other
    code can reuse it from this module.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar timesteps -> (B, dim)"""
        half = self.dim // 2
        freq = math.log(10_000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -freq)
        emb = t.unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the CHANNEL dimension only, for (B, C, T) tensors.

    Unlike ``nn.GroupNorm`` (which normalizes over channels x time per sample),
    this normalizes each time frame independently using only its own channels.
    That makes it PADDING-INVARIANT: the statistics of a valid frame do not
    depend on how many zero-padded frames sit beside it in the batch, so the
    train-time (padded batch) and inference-time (single un-padded sequence)
    activations match exactly. This is the fix for the train/inference
    normalization mismatch reported by review (issue 1).
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> normalize over C per (B, T) frame.
        x = x.transpose(1, 2)        # (B, T, C)
        x = self.norm(x)
        return x.transpose(1, 2)     # (B, C, T)


class ResidualBlock1D(nn.Module):
    """Dilated 1D residual block with FiLM (Feature-wise Linear Modulation).

    Path:  ChannelLayerNorm -> FiLM(scale, shift) -> GELU -> DilatedConv1d
        -> ChannelLayerNorm -> GELU -> Conv1x1 -> + skip

    A global conditioning vector (here: the speaker embedding projection) is
    mapped to per-channel scale/shift applied after the first norm.

    Normalization is per-frame channel-only (padding-invariant). An optional
    ``mask`` (B, 1, T) zeros padded frames after each conv so the dilated
    convolution cannot leak zero-padded positions into valid frames near the
    right boundary.
    """

    def __init__(self, channels: int, dilation: int = 1, cond_dim: int = 512):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.dilated_conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        # FiLM: global cond -> scale + shift
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x   : (B, C, T)
        cond: (B, cond_dim)  global conditioning (speaker)
        mask: (B, 1, T) float, 1 = valid frame, 0 = padded (optional)
        """
        h = self.norm1(x)

        # FiLM conditioning
        ss = self.film(cond)                        # (B, 2C)
        scale, shift = ss.chunk(2, dim=-1)          # each (B, C)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = F.gelu(h)
        # Zero padded frames before the dilated conv so they contribute nothing
        # to valid frames at the right boundary.
        if mask is not None:
            h = h * mask
        h = self.dilated_conv(h)

        h = self.norm2(h)
        h = F.gelu(h)
        h = self.pointwise(h)

        out = x + h  # skip connection
        if mask is not None:
            out = out * mask
        return out


class NeuralConverter(nn.Module):
    """Pool-free, speaker-conditioned feature converter.

    Maps source WavLM-Large features to target-speaker features, learning a
    DELTA over the source (output = source_feats + net(...)). Globally
    conditioned on a single ECAPA speaker embedding via FiLM -- no flow
    timestep, no target feature pool.

    Args:
        feat_dim:   WavLM-Large feature dim (1024).
        hidden_dim: ResNet hidden channels (512).
        spk_dim:    ECAPA speaker embedding dim (192).
        num_blocks: number of dilated residual blocks (8).

    forward(source_feats:(B,T,1024), spk_emb:(B,192)) -> converted_feats:(B,T,1024)
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_dim: int = 512,
        spk_dim: int = 192,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Input projection: feature -> hidden (channel-first Conv1d).
        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, kernel_size=3, padding=1)

        # Global conditioning from the speaker embedding (no timestep).
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Dilated residual backbone (repeated dilation schedule).
        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:num_blocks]
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, dilation=d, cond_dim=hidden_dim)
            for d in dilations
        ])

        # Output head: hidden -> feature (padding-invariant per-frame norm).
        self.out_norm = ChannelLayerNorm(hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, feat_dim, kernel_size=1)

    def forward(
        self,
        source_feats: torch.Tensor,
        spk_emb: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        source_feats: (B, T, 1024)
        spk_emb     : (B, 192)
        lengths     : (B,) valid (unpadded) time length per item (optional).
                      When given, padded frames are masked to zero before every
                      dilated conv so they cannot leak into valid frames; the
                      per-frame channel norm is padding-invariant regardless.
        returns     : (B, T, 1024)  == source_feats + learned delta
        """
        B, T, _ = source_feats.shape

        # Build a (B, 1, T) frame mask from lengths (None -> all valid).
        mask = None
        if lengths is not None:
            ar = torch.arange(T, device=source_feats.device).unsqueeze(0)  # (1, T)
            mask = (ar < lengths.unsqueeze(1)).to(source_feats.dtype)       # (B, T)
            mask = mask.unsqueeze(1)                                        # (B, 1, T)

        # Channel-first for Conv1d.
        x = source_feats.transpose(1, 2)          # (B, feat_dim, T)
        if mask is not None:
            x = x * mask

        h = self.input_proj(x)                    # (B, hidden, T)
        if mask is not None:
            h = h * mask
        g = self.spk_mlp(spk_emb)                 # (B, hidden)

        for block in self.blocks:
            h = block(h, g, mask)

        h = self.out_norm(h)
        h = F.gelu(h)
        delta = self.out_proj(h)                  # (B, feat_dim, T)

        delta = delta.transpose(1, 2)             # (B, T, feat_dim)
        if mask is not None:
            delta = delta * mask.transpose(1, 2)
        return source_feats + delta
