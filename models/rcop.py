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
        # The decoder now receives both content features and speaker information
        decoder_input_dim = d_ssl + d_spk
        decoder_layers: List[nn.Module] = [nn.Linear(decoder_input_dim, d_ssl)]
        for _ in range(n_res_blocks):
            decoder_layers.append(ResidualBlock(d_ssl))
        decoder_layers.append(nn.Linear(d_ssl, n_mels))
        
        self.vocoder_head = nn.Sequential(*decoder_layers)

    def project_orthogonally(self, features, axis):
        """Helper method to call the projection function."""
        return orthogonal_project(features, axis)

    def forward(self, ssl_frames, spk_embed, lambd=1.):
        # ssl_frames: (N, T, d_ssl) or (T, d_ssl)
        # spk_embed: (N, d_spk) or (d_spk)
        is_batch = ssl_frames.dim() == 3

        axis       = self.W_proj(spk_embed)                 # (N, d_ssl) or (d_ssl)
        proj_feats = self.project_orthogonally(ssl_frames, axis)   # (N, T, d_ssl) or (T, d_ssl)

        ph_logits  = self.ph_clf(proj_feats)                # (N, T, n_phones) or (T, n_phones)

        # Expand speaker embedding to match time dimension for vocoder
        if is_batch:
            T = proj_feats.size(1)
            spk_embed_expanded = spk_embed.unsqueeze(1).expand(-1, T, -1) # (N, T, d_spk)
        else:
            T = proj_feats.size(0)
            spk_embed_expanded = spk_embed.unsqueeze(0).expand(T, -1) # (T, d_spk)

        # Concatenate content and speaker info for reconstruction
        vocoder_input = torch.cat([proj_feats, spk_embed_expanded], dim=-1)
        pred_mels = self.vocoder_head(vocoder_input)           # (N, T, n_mels) or (T, n_mels)

        # Adversarial speaker classification is now frame-wise for robustness
        rev_feats   = grad_reverse(proj_feats, lambd)       # (N, T, d_ssl) or (T, d_ssl)
        sp_logits_per_frame = self.sp_clf(rev_feats)        # (N, T, n_spk) or (T, n_spk)
        
        # Average over time dimension to get a single prediction per utterance
        if is_batch:
            sp_logits = sp_logits_per_frame.mean(dim=1) # (N, n_spk)
        else:
            sp_logits = sp_logits_per_frame.mean(dim=0, keepdim=True) # (1, n_spk)

        return ph_logits, sp_logits, pred_mels 