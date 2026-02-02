"""
Velocity prediction network for Rectified Flow Matching.

Uses a Transformer architecture to predict the velocity field v(x_t, t, spk)
that transports samples from x_0 to x_1.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on timestep and speaker."""
    
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, d_model * 2)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerBlock(nn.Module):
    """Transformer block with adaptive normalization."""
    
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = AdaLayerNorm(d_model, d_cond)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = AdaLayerNorm(d_model, d_cond)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Self-attention with adaptive norm
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=mask)
        x = x + attn_out
        
        # Feed-forward with adaptive norm
        x = x + self.ff(self.norm2(x, cond))
        
        return x


class FlowNetwork(nn.Module):
    """
    Transformer-based velocity prediction network for flow matching.
    
    Predicts v(x_t, t, spk) where:
    - x_t: noisy WavLM features at time t
    - t: timestep in [0, 1]
    - spk: speaker embedding
    """
    
    def __init__(
        self,
        d_input: int = 768,
        d_model: int = 512,
        d_spk: int = 192,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Speaker embedding projection
        self.spk_proj = nn.Sequential(
            nn.Linear(d_spk, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Combined conditioning dimension
        d_cond = d_model * 2  # time + speaker
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_cond, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_input)
        
        # Initialize output projection to zero for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        spk_embed: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict velocity at time t.
        
        Args:
            x_t: (B, T, D) noisy features at time t
            t: (B,) timestep values in [0, 1]
            spk_embed: (B, d_spk) speaker embeddings
            mask: (B, T) padding mask (True = padded)
            
        Returns:
            (B, T, D) predicted velocity
        """
        B, T, D = x_t.shape
        
        # Project input
        h = self.input_proj(x_t)
        
        # Get conditioning
        t_emb = self.time_embed(t)
        spk_emb = self.spk_proj(spk_embed)
        cond = torch.cat([t_emb, spk_emb], dim=-1)
        
        # Invert mask for attention (True = ignore)
        if mask is not None:
            attn_mask = ~mask
        else:
            attn_mask = None
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h, cond, attn_mask)
        
        # Output
        h = self.output_norm(h)
        v = self.output_proj(h)
        
        return v
