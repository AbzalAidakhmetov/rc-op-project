import torch
import torch.nn as nn
import torch.nn.functional as F
from .grad_reverse import GradientReversal
from .projection import OrthogonalProjection
from .decoder import CustomDecoder
from .hires_features import HiResFeatureExtractor
from utils.upsample import LearnedUpsample

HIRES_DIM = 64 # Dimension for the high-resolution feature stream

class RCOP(nn.Module):
    """
    Reference-Conditioned Orthogonal Projection (RC-OP) for Voice Conversion.
    This model disentangles speaker and content information from speech.
    """
    def __init__(self, d_spk, d_ssl, n_phones, n_spk, n_mels: int = 80, hop_length: int = 256):
        super().__init__()
        self.d_spk = d_spk
        self.d_ssl = d_ssl

        # 1. Speaker axis projection: Learns to map a speaker embedding to a D-dimensional axis.
        self.spk_proj = nn.Linear(d_spk, d_ssl)
        
        # 2. Orthogonal projection module.
        self.ortho_proj = OrthogonalProjection()

        # 3. Phoneme classifier: Predicts phonemes from the content-only features.
        #    We use a small MLP instead of a single linear layer to give the model
        #    more capacity to learn the complex mapping from features to phonemes.
        classifier_hidden_dim = d_ssl // 2
        self.ph_clf = nn.Sequential(
            nn.Linear(d_ssl, classifier_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(classifier_hidden_dim),
            nn.Linear(classifier_hidden_dim, n_phones)
        )

        # 4. Speaker classifier: Adversarially trained to predict the speaker from content features.
        # The Gradient Reversal Layer ensures the main model learns to produce features
        # from which the speaker *cannot* be identified.
        self.spk_grl = GradientReversal()
        self.spk_clf = nn.Linear(d_ssl, n_spk)
        
        # 5. High-resolution feature extractor
        self.hires_net = HiResFeatureExtractor(output_dim=HIRES_DIM, upsample_factor=hop_length)

        # 6. Custom decoder: Synthesizes a mel-spectrogram from the combined
        #    content features, high-resolution features, and a target speaker embedding.
        self.vocoder_head = CustomDecoder(d_ssl=d_ssl + HIRES_DIM, d_spk=d_spk, n_mels=n_mels)

        # Learned up-sampler: 50 Hz → 62.5 Hz (factor 1.25)
        # This will now operate on the high-dimensional features before the decoder.
        self.upsample = LearnedUpsample(d_ssl)

    def forward(self, ssl_features, spk_embed, raw_audio, lambd=1.0, bypass_projection=False):
        """
        Args:
            ssl_features (torch.Tensor): SSL features from WavLM, shape (N, T_ssl, D_ssl)
            spk_embed (torch.Tensor): Speaker embedding, shape (N, D_spk)
            raw_audio (torch.Tensor): Raw audio waveform for the hires stream, shape (N, T_audio)
            lambd (float): Lambda for the Gradient Reversal Layer.
            bypass_projection (bool): If True, bypass the orthogonal projection for debugging.
        
        Returns:
            ph_logits (torch.Tensor): Phoneme predictions, shape (N, T_ssl, n_phones)
            sp_logits_agg (torch.Tensor): Aggregated speaker predictions, shape (N, n_spk)
            pred_mels (torch.Tensor): Predicted mel-spectrogram, shape (N, T_mel, n_mels)
        """
        # 1. Project speaker embedding to get the speaker's characteristic axis in the feature space.
        spk_axis = self.spk_proj(spk_embed)

        if bypass_projection:
            # For debugging reconstruction, bypass the projection to see if the model can overfit.
            content_features = ssl_features
        else:
            # 2. Orthogonally project the SSL features to remove the speaker component,
            #    leaving only the content component.
            content_features = self.ortho_proj(ssl_features, spk_axis)
        
        # 3. Predict phonemes from the ORIGINAL SSL features.
        ph_logits = self.ph_clf(ssl_features)
        
        # 4. Adversarially predict the speaker from the purified content features.
        reversed_content = self.spk_grl(content_features, lambd)
        sp_logits = self.spk_clf(reversed_content)
        
        # Aggregate speaker logits over the time dimension for a single classification per utterance.
        sp_logits_agg = sp_logits.mean(dim=1)

        # 5. Upsample the low-resolution content features.
        upsampled_content_features = self.upsample(content_features)
        
        # 6. Extract high-resolution features from the raw audio.
        hires_features = self.hires_net(raw_audio)

        # Align temporal dimensions before concatenation
        target_len = min(upsampled_content_features.size(1), hires_features.size(1))
        upsampled_content_features = upsampled_content_features[:, :target_len, :]
        hires_features = hires_features[:, :target_len, :]
        
        # 7. Combine the feature streams
        combined_content_features = torch.cat([upsampled_content_features, hires_features], dim=-1)

        # 8. Synthesize a mel-spectrogram from the combined features
        pred_mels = self.vocoder_head(combined_content_features, spk_embed)
        
        return ph_logits, sp_logits_agg, pred_mels