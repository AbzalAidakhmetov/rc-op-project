import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalProjection(nn.Module):
    """
    Projects a batch of sequences of vectors onto the subspace
    orthogonal to a corresponding batch of axes.
    """
    def forward(self, features: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Input features, shape (N, T, D)
            axes (torch.Tensor): Axes to project against, shape (N, D)
        
        Returns:
            torch.Tensor: Orthogonally projected features, shape (N, T, D)
        """
        # Ensure axis is a unit vector for correct projection
        axes = F.normalize(axes, p=2, dim=-1)
        
        # Add a time dimension to the axis for broadcasting
        axes_expanded = axes.unsqueeze(1) # (N, 1, D)
        
        # Project features onto the axis: b = (v . u)
        # This is the component of the features that lies *along* the axis
        proj_component = torch.sum(features * axes_expanded, dim=-1, keepdim=True)
        
        # Subtract the projection from the original features to get the orthogonal component
        # v_ortho = v - (v . u) * u
        ortho_features = features - proj_component * axes_expanded
        
        return ortho_features 