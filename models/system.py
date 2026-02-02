"""
Voice Conversion System - Main wrapper combining Flow and Decoder.

This module holds:
- The Flow Network
- The Decoder
- The pre-computed Projection Matrix (buffer, not parameter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .flow_matching import RectifiedFlowMatching, BaselineCFM, SGFlow
from .decoder import WavLMToMelDecoder
from .projection import OrthogonalProjection


class VoiceConversionSystem(nn.Module):
    """
    Complete Voice Conversion System combining flow matching and decoder.
    
    Pipeline:
    1. Source WavLM -> Content projection (remove speaker) [SG-Flow only]
    2. Flow matching: Transport to target speaker distribution
    3. Decoder: Convert WavLM features to mel spectrogram
    4. Vocoder (external): Mel -> waveform
    """
    
    def __init__(
        self,
        flow_model: RectifiedFlowMatching,
        decoder: WavLMToMelDecoder,
        projection: Optional[OrthogonalProjection] = None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.decoder = decoder
        
        # Store projection as buffer (not parameter)
        if projection is not None:
            if hasattr(projection, 'P_content') and projection.P_content is not None:
                self.register_buffer('P_content', projection.P_content)
            else:
                self.P_content = None
            if hasattr(projection, 'P_speaker') and projection.P_speaker is not None:
                self.register_buffer('P_speaker', projection.P_speaker)
            else:
                self.P_speaker = None
            if hasattr(projection, 'mean') and projection.mean is not None:
                self.register_buffer('proj_mean', projection.mean)
            else:
                self.proj_mean = None
        else:
            self.P_content = None
            self.P_speaker = None
            self.proj_mean = None
    
    def compute_loss(
        self,
        x1: torch.Tensor,
        target_spk: torch.Tensor,
        target_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "sg_flow",
        train_flow: bool = True,
        train_decoder: bool = True,
    ) -> dict:
        """
        Compute training loss.
        
        Forward Pass Logic:
        - Sample t ~ U[0,1]
        - If mode == 'baseline': x_0 = noise ~ N(0,I)
        - If mode == 'sg_flow': x_0 = P_content @ x_1 (Projected start)
        - v_target = x_1 - x_0
        - Interpolate x_t = (1-t)*x_0 + t*x_1
        - Predict v = flow_network(x_t, t, spk)
        - Loss = MSE(v, v_target) (Flow Matching Loss)
        - Auxiliary Loss: Decoder Loss (L1 on mel)
        
        Args:
            x1: (B, T, D) target WavLM features
            target_spk: (B, d_spk) target speaker embedding
            target_mel: (B, T_mel, n_mels) target mel spectrogram
            mask: (B, T) valid frame mask
            mode: "baseline" or "sg_flow"
            train_flow: whether to compute flow loss
            train_decoder: whether to compute decoder loss
        """
        B, T, D = x1.shape
        device = x1.device
        losses = {}
        
        # Flow matching loss
        if train_flow:
            flow_loss, flow_info = self.flow_model(x1, target_spk, mask)
            losses["flow_loss"] = flow_loss
            losses.update(flow_info)
        else:
            losses["flow_loss"] = torch.tensor(0.0, device=device)
        
        # Decoder loss (reconstruction: WavLM -> Mel)
        if train_decoder:
            pred_mel = self.decoder(x1, target_spk, target_mel.shape[1])
            
            if mask is not None:
                # Create mel-aligned mask
                mel_mask = F.interpolate(
                    mask.float().unsqueeze(1),
                    size=target_mel.shape[1],
                    mode="nearest",
                ).squeeze(1).bool()
                mel_mask_expanded = mel_mask.unsqueeze(-1)
                decoder_loss = F.l1_loss(
                    pred_mel * mel_mask_expanded,
                    target_mel * mel_mask_expanded,
                )
            else:
                decoder_loss = F.l1_loss(pred_mel, target_mel)
            
            losses["decoder_loss"] = decoder_loss
            losses["pred_mel"] = pred_mel
        else:
            losses["decoder_loss"] = torch.tensor(0.0, device=device)
        
        # Total loss
        losses["loss"] = losses["flow_loss"] + losses["decoder_loss"]
        
        return losses
    
    def forward(
        self,
        x1: torch.Tensor,
        target_spk: torch.Tensor,
        target_mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "sg_flow",
    ) -> dict:
        """Training forward pass (trains both flow and decoder)."""
        return self.compute_loss(x1, target_spk, target_mel, mask, mode, True, True)
    
    @torch.no_grad()
    def convert(
        self,
        source_wavlm: torch.Tensor,
        target_spk: torch.Tensor,
        num_steps: int = 20,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Voice conversion inference using Euler ODE solver.
        
        Pipeline:
        1. Take Source WavLM features
        2. If SG-Flow: Project to Content Subspace (x_0 = P_content @ x)
           If Baseline: Sample Noise (x_0 ~ N(0,I))
        3. Solve ODE: x_1 = x_0 + integral(v dt)
        4. Decoder(x_1) -> Mel
        5. (External) HiFi-GAN(Mel) -> Audio
        
        Args:
            source_wavlm: (B, T, D) source WavLM features
            target_spk: (B, d_spk) target speaker embedding
            num_steps: ODE solver steps (10-20 recommended)
            mask: (B, T) valid frame mask
            
        Returns:
            converted_wavlm: (B, T, D) converted WavLM features
            mel: (B, T_mel, n_mels) predicted mel spectrogram
        """
        # Use flow model's sample method (handles baseline vs sg_flow internally)
        converted_wavlm = self.flow_model.sample(
            source_wavlm, target_spk, num_steps, mask
        )
        
        # Decode to mel
        mel = self.decoder(converted_wavlm, target_spk)
        
        return converted_wavlm, mel
    
    def euler_solve(
        self,
        x0: torch.Tensor,
        target_spk: torch.Tensor,
        num_steps: int = 20,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Explicit Euler ODE solver.
        
        Solves: dx/dt = v(x_t, t, spk) from t=0 to t=1
        
        x_{t+dt} = x_t + v(x_t, t, spk) * dt
        """
        device = x0.device
        B = x0.shape[0]
        
        dt = 1.0 / num_steps
        x_t = x0.clone()
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            v = self.flow_model.velocity_net(x_t, t, target_spk, mask=mask)
            x_t = x_t + v * dt
        
        return x_t
