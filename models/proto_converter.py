"""Prototype-conditioned neural converter (experiment toward SOTA).

The pool-free NeuralConverter conditions only on a single 192-d ECAPA vector,
which is an information bottleneck: it cannot specify a target voice well enough
to match kNN-VC's full-pool matching, so it over-smooths and plateaus ~0.39
zero-shot target-similarity.

This variant restores target information WITHOUT the full pool: it cross-attends
the source frames to a small set of M target "prototype" feature vectors sampled
(or clustered) from a few seconds of reference audio. This is between pool-free
(1 vector) and full kNN-VC (thousands of pool frames + per-frame search). It is
still parametric (one forward pass, no growing pool search) and needs only a
short reference, but gives the network enough target detail to reproduce -- and
potentially sharpen -- the kNN target features.

Reuses ResidualBlock1D / ChannelLayerNorm from models.converter (padding-invariant).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.converter import ResidualBlock1D, ChannelLayerNorm


class ProtoCrossAttn(nn.Module):
    """Cross-attention from source frames (queries) to target prototypes (keys/values)."""

    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.norm = ChannelLayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x, proto_kv, mask=None):
        # x: (B, C, T) ; proto_kv: (B, M, C) already in hidden dim.
        h = self.norm(x).transpose(1, 2)                # (B, T, C)
        out, _ = self.attn(h, proto_kv, proto_kv, need_weights=False)
        out = out.transpose(1, 2)                       # (B, C, T)
        if mask is not None:
            out = out * mask
        return x + out


class PrototypeConverter(nn.Module):
    """Source features + target prototypes (+ ECAPA) -> target features (delta).

    forward(source_feats:(B,T,1024), protos:(B,M,1024), spk_emb:(B,192),
            lengths:(B,)|None) -> (B,T,1024)
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        hidden_dim: int = 512,
        spk_dim: int = 192,
        num_blocks: int = 8,
        n_heads: int = 8,
        in_instance_norm: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        # When True, the source features are instance-normalised (per-utterance,
        # per-dim, over valid time) before the network AND used as the residual
        # base -- so source-speaker statistics are stripped and identity must be
        # rebuilt from the prototypes. Output stays real-scale (delta is learned
        # against the real kNN target).
        self.in_instance_norm = in_instance_norm

        self.input_proj = nn.Conv1d(feat_dim, hidden_dim, kernel_size=3, padding=1)
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Project target prototypes to hidden dim (+ norm for stable attention).
        self.proto_proj = nn.Linear(feat_dim, hidden_dim)
        self.proto_norm = nn.LayerNorm(hidden_dim)

        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:num_blocks]
        self.res_blocks = nn.ModuleList(
            [ResidualBlock1D(hidden_dim, dilation=d, cond_dim=hidden_dim) for d in dilations]
        )
        self.attn_blocks = nn.ModuleList(
            [ProtoCrossAttn(hidden_dim, n_heads=n_heads) for _ in dilations]
        )

        self.out_norm = ChannelLayerNorm(hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, feat_dim, kernel_size=1)

    def forward(self, source_feats, protos, spk_emb, lengths=None):
        B, T, _ = source_feats.shape
        mask = None
        if lengths is not None:
            ar = torch.arange(T, device=source_feats.device).unsqueeze(0)
            mask = (ar < lengths.unsqueeze(1)).to(source_feats.dtype).unsqueeze(1)  # (B,1,T)

        # Residual base: raw source, or instance-normalised source (strips
        # source-speaker stats) computed over VALID frames only.
        base = source_feats
        if self.in_instance_norm:
            if mask is not None:
                m = mask.transpose(1, 2)                                  # (B,T,1)
                denom = m.sum(1, keepdim=True).clamp_min(1.0)
                mean = (source_feats * m).sum(1, keepdim=True) / denom
                var = ((source_feats - mean) ** 2 * m).sum(1, keepdim=True) / denom
                base = (source_feats - mean) / (var.sqrt() + 1e-5)
                base = base * m
            else:
                mean = source_feats.mean(1, keepdim=True)
                std = source_feats.std(1, keepdim=True) + 1e-5
                base = (source_feats - mean) / std

        x = base.transpose(1, 2)                       # (B, feat, T)
        if mask is not None:
            x = x * mask
        h = self.input_proj(x)
        if mask is not None:
            h = h * mask

        g = self.spk_mlp(spk_emb)                      # (B, hidden)
        pk = self.proto_norm(self.proto_proj(protos))  # (B, M, hidden)

        for res, attn in zip(self.res_blocks, self.attn_blocks):
            h = res(h, g, mask)
            h = attn(h, pk, mask)

        h = self.out_norm(h)
        h = F.gelu(h)
        delta = self.out_proj(h).transpose(1, 2)       # (B, T, feat)
        if mask is not None:
            delta = delta * mask.transpose(1, 2)
        return base + delta
