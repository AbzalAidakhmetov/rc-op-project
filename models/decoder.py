"""
Decoder module: WavLM features -> Mel spectrogram.

Implements a 1D ResNet Decoder as specified in the architecture requirements.
Uses 3-4 Residual Conv1D blocks to learn the texture mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualConv1dBlock(nn.Module):
    """
    Residual 1D Convolutional Block for the decoder.
    
    Structure:
        x -> Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> (+x) -> ReLU
    
    Includes speaker conditioning via FiLM (Feature-wise Linear Modulation).
    """
    
    def __init__(
        self,
        channels: int,
        d_cond: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, 
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # FiLM conditioning (scale and shift from speaker embedding)
        self.film = nn.Linear(d_cond, channels * 2)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input features
            cond: (B, d_cond) conditioning vector (speaker embedding)
        """
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # FiLM conditioning
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        
        # Residual connection
        out = out + residual
        out = F.relu(out)
        
        return out


class WavLMToMelDecoder(nn.Module):
    """
    1D ResNet Decoder that maps WavLM features (768) to Mel Spectrogram (80).
    
    Uses 3-4 Residual Conv1D blocks to learn the texture mapping,
    with speaker conditioning via FiLM.
    """
    
    def __init__(
        self,
        d_wavlm: int = 768,
        d_spk: int = 192,
        d_hidden: int = 512,
        n_mels: int = 80,
        num_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
        upsample_factor: float = 1.25,
    ):
        """
        Args:
            d_wavlm: WavLM feature dimension (768)
            d_spk: speaker embedding dimension (192)
            d_hidden: hidden channel dimension
            n_mels: number of mel bins (80)
            num_layers: number of residual blocks (3-4 recommended)
            kernel_size: convolution kernel size
            dropout: dropout rate
            upsample_factor: ratio of mel frames to WavLM frames (~1.25)
        """
        super().__init__()
        
        self.d_wavlm = d_wavlm
        self.n_mels = n_mels
        self.upsample_factor = upsample_factor
        
        # Speaker embedding projection
        self.spk_proj = nn.Sequential(
            nn.Linear(d_spk, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        
        # Input projection: WavLM dim -> hidden channels
        self.input_conv = nn.Conv1d(d_wavlm, d_hidden, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(d_hidden)
        
        # Stack of Residual Conv1D blocks (3-4 blocks as per spec)
        self.res_blocks = nn.ModuleList([
            ResidualConv1dBlock(
                channels=d_hidden,
                d_cond=d_hidden,
                kernel_size=kernel_size,
                dilation=2**i,  # Increasing dilation for larger receptive field
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        
        # Output projection: hidden channels -> mel bins
        self.output_conv = nn.Conv1d(d_hidden, n_mels, kernel_size=1)
    
    def forward(
        self,
        wavlm_features: torch.Tensor,
        spk_embed: torch.Tensor,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decode WavLM features to mel spectrogram.
        
        Args:
            wavlm_features: (B, T_wavlm, d_wavlm) WavLM features
            spk_embed: (B, d_spk) speaker embedding
            target_length: optional target mel length for upsampling
            
        Returns:
            (B, T_mel, n_mels) mel spectrogram (log-scale)
        """
        B, T, D = wavlm_features.shape
        
        # Project speaker embedding for conditioning
        cond = self.spk_proj(spk_embed)
        
        # Input projection: (B, T, D) -> (B, D, T) -> Conv -> (B, C, T)
        h = wavlm_features.transpose(1, 2)
        h = self.input_conv(h)
        h = self.input_bn(h)
        h = F.relu(h)
        
        # Apply residual blocks
        for block in self.res_blocks:
            h = block(h, cond)
        
        # Output projection
        mel = self.output_conv(h)  # (B, n_mels, T)
        mel = mel.transpose(1, 2)  # (B, T, n_mels)
        
        # Upsample to match mel frame rate if needed
        if target_length is not None:
            mel = F.interpolate(
                mel.transpose(1, 2),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        elif self.upsample_factor != 1.0:
            target_len = int(T * self.upsample_factor)
            mel = F.interpolate(
                mel.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        
        return mel


# Note: VoiceConversionSystem (the main wrapper) is in models/system.py
