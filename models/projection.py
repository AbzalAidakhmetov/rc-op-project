"""
Orthogonal projection module for speaker/content separation.
"""

import torch
import torch.nn as nn
from typing import Optional


class OrthogonalProjection(nn.Module):
    """
    Module for projecting WavLM features onto content or speaker subspaces.
    
    Uses precomputed SVD projection matrices to separate speaker and content
    information in the WavLM feature space.
    """
    
    def __init__(
        self,
        projection_path: Optional[str] = None,
        P_content: Optional[torch.Tensor] = None,
        P_speaker: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
    ):
        """
        Initialize projection module.
        
        Args:
            projection_path: Path to saved projection matrices (.pt file)
            P_content: (D, D) content projection matrix (alternative to path)
            P_speaker: (D, D) speaker projection matrix (alternative to path)
            mean: (D,) mean vector for centering
        """
        super().__init__()
        
        if projection_path is not None:
            data = torch.load(projection_path, map_location="cpu")
            P_content = data["P_content"]
            P_speaker = data["P_speaker"]
            mean = data["mean"]
        
        if P_content is not None:
            self.register_buffer("P_content", P_content)
        else:
            self.P_content = None
            
        if P_speaker is not None:
            self.register_buffer("P_speaker", P_speaker)
        else:
            self.P_speaker = None
            
        if mean is not None:
            self.register_buffer("mean", mean)
        else:
            self.mean = None
    
    def project_content(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features onto content subspace (remove speaker info).
        
        Args:
            x: (*, D) input features
            
        Returns:
            (*, D) content-only features
        """
        if self.P_content is None:
            return x
        
        original_shape = x.shape
        D = original_shape[-1]
        x_flat = x.reshape(-1, D)
        
        if self.mean is not None:
            x_flat = x_flat - self.mean
        
        projected = x_flat @ self.P_content.T
        
        if self.mean is not None:
            projected = projected + self.mean
        
        return projected.reshape(original_shape)
    
    def project_speaker(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features onto speaker subspace (keep only speaker info).
        
        Args:
            x: (*, D) input features
            
        Returns:
            (*, D) speaker-only features
        """
        if self.P_speaker is None:
            return torch.zeros_like(x)
        
        original_shape = x.shape
        D = original_shape[-1]
        x_flat = x.reshape(-1, D)
        
        if self.mean is not None:
            x_flat = x_flat - self.mean
        
        projected = x_flat @ self.P_speaker.T
        
        return projected.reshape(original_shape)
    
    def forward(self, x: torch.Tensor, mode: str = "content") -> torch.Tensor:
        """
        Project features.
        
        Args:
            x: (*, D) input features
            mode: "content" or "speaker"
            
        Returns:
            Projected features
        """
        if mode == "content":
            return self.project_content(x)
        elif mode == "speaker":
            return self.project_speaker(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
