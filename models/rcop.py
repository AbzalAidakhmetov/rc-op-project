import torch
import torch.nn as nn
from typing import List
from .grad_reverse import grad_reverse
from .projection   import orthogonal_project

class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        return x + self.block(x)

class RCOP(nn.Module):
    def __init__(self, d_spk, d_ssl, n_phones, n_spk, n_mels=80, n_res_blocks=3):
        super().__init__()
        self.W_proj       = nn.Linear(d_spk, d_ssl, bias=False)  # learns speaker axis
        self.ph_clf       = nn.Linear(d_ssl, n_phones)
        self.sp_clf       = nn.Linear(d_ssl, n_spk)
        
        # Decoder with residual blocks for improved gradient flow
        decoder_layers: List[nn.Module] = [nn.Linear(d_ssl, d_ssl)]
        for _ in range(n_res_blocks):
            decoder_layers.append(ResidualBlock(d_ssl))
        decoder_layers.append(nn.Linear(d_ssl, n_mels))
        
        self.vocoder_head = nn.Sequential(*decoder_layers)

    def forward(self, ssl_frames, spk_embed, lambd=1.):
        axis       = self.W_proj(spk_embed)                 # (d_ssl)
        proj_feats = orthogonal_project(ssl_frames, axis)   # (T, d_ssl)

        ph_logits  = self.ph_clf(proj_feats)                # (T, n_phones)
        
        # Vocoder head produces the mel-spectrogram for reconstruction
        pred_mels = self.vocoder_head(proj_feats)           # (T, n_mels)

        mean_feat  = proj_feats.mean(0)                     # (d_ssl)
        rev_feat   = grad_reverse(mean_feat, lambd)
        sp_logits  = self.sp_clf(rev_feat.unsqueeze(0))     # (1, n_spk)

        return ph_logits, sp_logits, pred_mels 