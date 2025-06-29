import torch
import torch.nn.functional as F

def orthogonal_project(features: torch.Tensor, axis: torch.Tensor):
    """Remove the component of `features` along `axis` (frame-wise)."""
    axis = F.normalize(axis, dim=-1)
    alpha = torch.sum(features * axis.unsqueeze(0), dim=-1, keepdim=True)
    return features - alpha * axis.unsqueeze(0) 