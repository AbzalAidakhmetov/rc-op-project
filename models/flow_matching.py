"""
Rectified Flow Matching implementation.

Implements two methods:
1. Baseline CFM: Standard Conditional Flow Matching from Gaussian noise
2. SG-Flow: Rectified Flow from orthogonally projected content subspace

Reference: Flow Matching for Generative Modeling (Lipman et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .flow_network import FlowNetwork
from .projection import OrthogonalProjection


class RectifiedFlowMatching(nn.Module):
    """
    Base class for Rectified Flow Matching.
    
    The flow interpolates between x_0 (start) and x_1 (target):
        x_t = (1 - t) * x_0 + t * x_1
        
    The velocity field is:
        v = dx_t/dt = x_1 - x_0
        
    We train a network to predict this velocity given x_t, t, and conditioning.
    """
    
    def __init__(
        self,
        velocity_net: FlowNetwork,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min
    
    def get_x0(
        self,
        x1: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Get starting point x_0. Override in subclasses."""
        raise NotImplementedError
    
    def forward(
        self,
        x1: torch.Tensor,
        spk_embed: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute flow matching loss.
        
        Args:
            x1: (B, T, D) target WavLM features
            spk_embed: (B, d_spk) target speaker embedding
            mask: (B, T) valid frame mask (True = valid)
            
        Returns:
            loss: scalar loss value
            info: dict with additional info
        """
        B, T, D = x1.shape
        device = x1.device
        
        # Sample random timesteps
        t = torch.rand(B, device=device)
        
        # Sample noise for x_0
        noise = torch.randn_like(x1)
        
        # Get starting point (method-specific)
        x0 = self.get_x0(x1, noise)
        
        # Interpolate: x_t = (1-t)*x_0 + t*x_1
        t_expanded = t[:, None, None]
        x_t = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Add small noise for numerical stability
        x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        
        # Target velocity: v = x_1 - x_0
        v_target = x1 - x0
        
        # Predict velocity
        v_pred = self.velocity_net(x_t, t, spk_embed, mask=mask)
        
        # Compute MSE loss (masked if mask provided)
        loss_per_frame = F.mse_loss(v_pred, v_target, reduction="none")
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            loss = (loss_per_frame * mask_expanded).sum() / mask_expanded.sum()
        else:
            loss = loss_per_frame.mean()
        
        info = {
            "v_pred_norm": v_pred.norm(dim=-1).mean().item(),
            "v_target_norm": v_target.norm(dim=-1).mean().item(),
            "t_mean": t.mean().item(),
        }
        
        return loss, info
    
    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        spk_embed: torch.Tensor,
        num_steps: int = 50,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Euler integration.
        
        Args:
            x0: (B, T, D) starting point
            spk_embed: (B, d_spk) target speaker embedding
            num_steps: number of integration steps
            mask: (B, T) valid frame mask
            
        Returns:
            (B, T, D) generated features
        """
        device = x0.device
        B = x0.shape[0]
        
        dt = 1.0 / num_steps
        x_t = x0.clone()
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            v = self.velocity_net(x_t, t, spk_embed, mask=mask)
            x_t = x_t + v * dt
        
        return x_t


class BaselineCFM(RectifiedFlowMatching):
    """
    Baseline Conditional Flow Matching.
    
    Starts from Gaussian noise: x_0 ~ N(0, I)
    """
    
    def __init__(self, velocity_net: FlowNetwork, sigma_min: float = 1e-4):
        super().__init__(velocity_net, sigma_min)
    
    def get_x0(self, x1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Start from Gaussian noise."""
        return noise
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int],
        spk_embed: torch.Tensor,
        num_steps: int = 50,
        mask: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate from Gaussian noise."""
        x0 = torch.randn(shape, device=device)
        return super().sample(x0, spk_embed, num_steps, mask)


class SGFlow(RectifiedFlowMatching):
    """
    SG-Flow: Rectified Flow from Orthogonally Projected Content Subspace.
    
    Starts from content-only projection of target: x_0 = P_content @ x_1
    
    This removes speaker information from the starting point, making the
    flow's job to add back the target speaker characteristics.
    """
    
    def __init__(
        self,
        velocity_net: FlowNetwork,
        projection: OrthogonalProjection,
        sigma_min: float = 1e-4,
        noise_scale: float = 0.1,
    ):
        super().__init__(velocity_net, sigma_min)
        self.projection = projection
        self.noise_scale = noise_scale
    
    def get_x0(self, x1: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Start from content-only projection of x1, with optional noise."""
        x0_content = self.projection.project_content(x1)
        # Add small noise to prevent trivial solution when t=0
        x0 = x0_content + self.noise_scale * noise
        return x0
    
    @torch.no_grad()
    def sample(
        self,
        source_features: torch.Tensor,
        spk_embed: torch.Tensor,
        num_steps: int = 50,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate from source features (voice conversion).
        
        Args:
            source_features: (B, T, D) source WavLM features
            spk_embed: (B, d_spk) target speaker embedding
            num_steps: number of integration steps
            mask: (B, T) valid frame mask
            
        Returns:
            (B, T, D) converted features with target speaker characteristics
        """
        # Project source to content subspace
        x0 = self.projection.project_content(source_features)
        return super().sample(x0, spk_embed, num_steps, mask)


def create_flow_model(
    method: str = "sg_flow",
    d_input: int = 768,
    d_model: int = 512,
    d_spk: int = 192,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    projection_path: Optional[str] = None,
    sigma_min: float = 1e-4,
) -> RectifiedFlowMatching:
    """
    Factory function to create flow matching model.
    
    Args:
        method: "baseline_cfm" or "sg_flow"
        d_input: WavLM feature dimension
        d_model: transformer hidden dimension
        d_spk: speaker embedding dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        dropout: dropout rate
        projection_path: path to SVD projection matrix (required for sg_flow)
        sigma_min: minimum noise level
        
    Returns:
        Flow matching model
    """
    velocity_net = FlowNetwork(
        d_input=d_input,
        d_model=d_model,
        d_spk=d_spk,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    
    if method == "baseline_cfm":
        return BaselineCFM(velocity_net, sigma_min=sigma_min)
    elif method == "sg_flow":
        if projection_path is None:
            raise ValueError("projection_path required for sg_flow")
        projection = OrthogonalProjection(projection_path=projection_path)
        return SGFlow(velocity_net, projection, sigma_min=sigma_min)
    else:
        raise ValueError(f"Unknown method: {method}")
