import torch
import torch.nn.functional as F

def orthogonal_project(features: torch.Tensor, axis: torch.Tensor):
    """Remove the component of `features` along `axis` (batch-wise or frame-wise)."""
    # features: (N, T, D) or (T, D)
    # axis: (N, D) or (D)
    
    axis = F.normalize(axis, dim=-1)

    is_batch = features.dim() == 3
    if is_batch:
        # axis is (N, D), unsqueeze for broadcasting over T
        axis_unsq = axis.unsqueeze(1) # (N, 1, D)
    else:
        # axis is (D), unsqueeze for broadcasting over T
        axis_unsq = axis.unsqueeze(0) # (1, D)

    # Project features onto the axis
    alpha = torch.sum(features * axis_unsq, dim=-1, keepdim=True)

    # Subtract the projection
    return features - alpha * axis_unsq 