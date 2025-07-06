import torch
from transformers import AutoFeatureExtractor, AutoModel
from speechbrain.inference import EncoderClassifier

def extract_features(audio, wavlm_processor, wavlm_model, speaker_model, device):
    """Extract SSL features and speaker embeddings."""
    
    # Ensure audio is a tensor for processing
    if not isinstance(audio, torch.Tensor):
        audio = torch.from_numpy(audio)
    
    # Ensure audio is the right shape and on CPU for WavLM processor
    audio_np = audio.squeeze().cpu().numpy()
    
    # Extract SSL features with WavLM
    inputs = wavlm_processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        ssl_outputs = wavlm_model(**inputs)
        ssl_features = ssl_outputs.last_hidden_state.squeeze(0)  # (T, d_ssl)
    
    # Extract speaker embedding with SpeechBrain
    with torch.no_grad():
        # SpeechBrain model expects a batch on the correct device
        audio_device = audio.to(device)
        spk_embed = speaker_model.encode_batch(audio_device.unsqueeze(0)).squeeze() # (d_spk,)
    
    return ssl_features, spk_embed 