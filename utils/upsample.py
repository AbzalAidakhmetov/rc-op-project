import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedUpsample(nn.Module):
    """Learned 1.25× time-axis up-sampling (50 Hz → 62.5 Hz).
    1.  Linear interpolation to the target length.
    2.  Two lightweight Conv1D layers to refine / de-blur the spectrogram.
    Input / output shape: (N, T, C) where C = n_mels.
    """

    def __init__(self, channel_dim: int):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv1d(channel_dim, channel_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channel_dim, channel_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, T, C)
        if x.size(1) == 0:
            return x
        target_len = int(round(x.size(1) * 1.25))  # 50 Hz → 62.5 Hz
        # Step 1: Linear up-sampling (operate on (N, C, T))
        x_t = x.permute(0, 2, 1)
        x_up = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
        # Step 2: Refinement convs
        x_refined = self.refine(x_up)
        return x_refined.permute(0, 2, 1) 