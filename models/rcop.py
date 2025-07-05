import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .grad_reverse import grad_reverse
from .projection   import orthogonal_project

class ConvNeXtBlock(nn.Module):
    """A simplified ConvNeXt-style block for 1D sequences."""
    def __init__(self, d_model, kernel_size=7, expand_ratio=2):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            groups=d_model, padding="same"
        )
        self.norm = nn.LayerNorm(d_model)
        self.pw_conv1 = nn.Linear(d_model, d_model * expand_ratio)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(d_model * expand_ratio, d_model)

    def forward(self, x):
        # Input x has shape (N, T, C)
        residual = x
        x = x.permute(0, 2, 1) # (N, C, T)
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1) # (N, T, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        return residual + x

class VocoderHead(nn.Module):
    """
    A more sophisticated vocoder head that converts SSL features to mel-spectrograms.
    It handles the time dimension upsampling from SSL feature rate to mel-spectrogram rate.
    """
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, upsample_factor):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(hidden_dim) for _ in range(n_blocks)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, target_len=None):
        # x: (N, T_in, C_in)
        x = self.input_proj(x) # (N, T_in, H)

        # Upsample time dimension
        x = x.permute(0, 2, 1) # (N, H, T_in)
        if target_len:
             x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        else:
             x = F.interpolate(x, scale_factor=self.upsample_factor, mode='linear', align_corners=False)
        x = x.permute(0, 2, 1) # (N, T_out, H)

        for block in self.blocks:
            x = block(x)

        x = self.output_proj(x) # (N, T_out, C_out)
        return x

class RCOP(nn.Module):
    def __init__(self, d_spk, d_ssl, n_phones, n_spk, n_mels=80, n_res_blocks=3):
        super().__init__()
        self.W_proj       = nn.Linear(d_spk, d_ssl, bias=False)  # learns speaker axis
        self.ph_clf       = nn.Linear(d_ssl, n_phones)
        self.sp_clf       = nn.Linear(d_ssl, n_spk)
        
        # More sophisticated vocoder head
        # WavLM features are at 50Hz (stride 320), Mels are at ~62.5Hz (hop 256)
        # Upsampling factor is target_sr / hop_length / 50 = 16000 / 256 / 50 = 1.25
        self.vocoder_head = VocoderHead(
            input_dim=d_ssl,
            output_dim=n_mels,
            hidden_dim=d_ssl, # internal dimension
            n_blocks=n_res_blocks,
            upsample_factor=1.25 
        )

    def project_orthogonally(self, features, axis):
        """Helper method to call the projection function."""
        return orthogonal_project(features, axis)

    def forward(self, ssl_features, spk_embed, lambd=0.0, target_mel_len=None):
        # 1. Learn speaker projection axis
        spk_axis = self.W_proj(spk_embed)
        
        # 2. Project to get speaker-agnostic content features
        proj_feats = self.project_orthogonally(ssl_features, spk_axis)
        
        # 3. Phoneme classification on content features
        ph_logits = self.ph_clf(proj_feats)
        
        # 4. Adversarial speaker classification on content features
        rev_feats = grad_reverse(proj_feats, lambd)
        sp_logits_frame = self.sp_clf(rev_feats)
        sp_logits = sp_logits_frame.mean(dim=1)
        
        # 5. Reconstruct mel-spectrogram
        pred_mels = self.vocoder_head(proj_feats, target_len=target_mel_len)
        
        return ph_logits, sp_logits, pred_mels 