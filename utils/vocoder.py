from transformers import SpeechT5ForTextToSpeech
from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaIN(nn.Module):
    """
    Adaptive Layer Normalization for styling content features with a speaker embedding.
    This layer normalizes the content input and then applies a learned scale and bias
    derived from the speaker embedding. This provides fine-grained control over the
    synthesis style, which is crucial for overcoming "robotic" audio quality.
    """
    def __init__(self, style_dim, content_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(content_dim, affine=False)
        self.fc = nn.Linear(style_dim, content_dim * 2)

    def forward(self, content, style):
        # Normalize the content features.
        # Permute to (N, C, T) for instance normalization.
        normalized_content = self.norm(content.permute(0, 2, 1))
        normalized_content = normalized_content.permute(0, 2, 1)

        # Predict scale and bias from the speaker embedding.
        h = self.fc(style)
        gamma, beta = torch.chunk(h, chunks=2, dim=-1)

        # Apply the learned style.
        return (1 + gamma.unsqueeze(1)) * normalized_content + beta.unsqueeze(1)

class PretrainedVocoderHead(nn.Module):
    """
    High-quality mel generator using the decoder & post-net of a pre-trained
    SpeechT5 model. It adapts RC-OP content features and a speaker embedding
    into the SpeechT5 hidden space, then runs them through the frozen (or
    partially fine-tuned) decoder to generate an 80-dimension mel spectrogram.
    """
    def __init__(self, d_ssl: int, d_spk: int):
        super().__init__()
        cfg = Config()

        # Load the base SpeechT5 model.
        tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", cache_dir="./models"
        )
        
        # Accommodate different 'transformers' versions by finding the correct
        # decoder and postnet attributes before assigning to self.
        decoder_module = None
        postnet_module = None
        self.is_unified_decoder = False

        if hasattr(tts_model, "speech_decoder"):
            decoder_module = tts_model.speech_decoder
            self.is_unified_decoder = True
        elif hasattr(tts_model, "speech_decoder_postnet"):
            decoder_module = tts_model.speech_decoder_postnet
            self.is_unified_decoder = True
        elif hasattr(tts_model, "decoder"):
            decoder_module = getattr(tts_model, "decoder", None)
            postnet_module = getattr(tts_model, "postnet", None)
            self.is_unified_decoder = False
        
        if decoder_module is None:
            raise AttributeError(
                "Could not find a 'speech_decoder', 'speech_decoder_postnet', or 'decoder' "
                "in the SpeechT5 model. Please check your 'transformers' library version."
            )
        
        self.decoder = decoder_module
        self.postnet = postnet_module

        # Freeze all parameters by default.
        self.decoder.requires_grad_(False)
        if self.postnet:
            self.postnet.requires_grad_(False)
        
        # --- Fine-tuning Policy ---
        # A more robust way to unfreeze parameters, avoiding direct calls
        # to requires_grad_() on potentially non-module attributes.

        # Unfreeze the final projection.
        if hasattr(self.decoder, 'feat_out'):
            for param in self.decoder.feat_out.parameters():
                param.requires_grad = True
        
        # Unfreeze the post-net.
        if self.is_unified_decoder:
            # For unified decoders, find postnet parameters by name. This is the
            # most robust method as it doesn't depend on accessing a 'postnet' attribute.
            for name, param in self.decoder.named_parameters():
                if 'postnet' in name:
                    param.requires_grad = True
        elif self.postnet:
            # For separate postnets, unfreeze the whole module.
            for param in self.postnet.parameters():
                param.requires_grad = True

        # Optionally unfreeze the last N decoder layers.
        if cfg.finetune_layers > 0:
            layer_container = None
            if hasattr(self.decoder, 'layers'): # Older structure
                layer_container = self.decoder.layers
            elif hasattr(self.decoder, 'decoder') and hasattr(self.decoder.decoder, 'layers'): # Newer structure
                layer_container = self.decoder.decoder.layers
            
            if layer_container:
                num_layers = len(layer_container)
                start_layer = max(0, num_layers - cfg.finetune_layers)
                for i in range(start_layer, num_layers):
                    # Unfreeze all parameters within the target layer.
                    for param in layer_container[i].parameters():
                        param.requires_grad = True
            else:
                print("Warning: Could not find decoder layers to fine-tune.")

        # --- Feature Adaptation ---
        hidden_size = tts_model.config.hidden_size
        self.content_proj = nn.Linear(d_ssl, hidden_size)
        
        # Adaptive Layer Normalization (AdaIN) for speaker conditioning.
        # This replaces the previous additive fusion with a more powerful styling mechanism.
        self.adain = AdaIN(style_dim=d_spk, content_dim=hidden_size)

    def forward(self, content_features: torch.Tensor, spk_embed: torch.Tensor) -> torch.Tensor:
        """
        Generate a mel-spectrogram from content and speaker features.
        """
        # Project content features into the model's hidden dimension.
        projected_content = self.content_proj(content_features)

        # Apply AdaIN to style the content features using the speaker embedding.
        hidden_states = self.adain(projected_content, spk_embed)

        if self.is_unified_decoder:
            # Newer versions: decoder output is a tuple, second element is the final mel.
            decoder_output = self.decoder(hidden_states=hidden_states)
            mel_spectrogram = decoder_output[1]
        else:
            # Older versions: manual forward pass through decoder, projection, and postnet.
            decoder_output = self.decoder(hidden_states=hidden_states)
            # Project to mel dimension
            mel_before_postnet = self.decoder.feat_out(decoder_output.last_hidden_state)
            # Apply postnet for refinement
            if self.postnet:
                mel_spectrogram = mel_before_postnet + self.postnet(mel_before_postnet.transpose(1, 2)).transpose(1, 2)
            else:
                mel_spectrogram = mel_before_postnet

        return mel_spectrogram 