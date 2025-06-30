#!/usr/bin/env python3

import argparse
import os

# Mitigate threading issues with BLAS libraries (for resource-limited environments)
# This must be done BEFORE importing torch, numpy, etc.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import soundfile as sf
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, SpeechT5HifiGan
from resemblyzer import VoiceEncoder

from config import Config
from models.rcop import RCOP
from utils.logging import setup_logger
from utils.phonemes import get_num_phones

def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_audio(file_path, target_sr=16000):
    """Load and preprocess audio file."""
    try:
        audio, orig_sr = sf.read(file_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Take first channel if stereo
        
        # Resample if needed
        if orig_sr != target_sr:
            from resampy import resample
            audio = resample(audio, orig_sr, target_sr)
        
        return torch.from_numpy(audio).float()
    except Exception as e:
        raise ValueError(f"Failed to load audio from {file_path}: {e}")

def load_model_from_checkpoint(checkpoint_path, num_speakers, num_phones, device):
    """Load RCOP model from checkpoint."""
    config = Config()
    
    # Load checkpoint first to get metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to extract num_speakers from checkpoint metadata
    if 'num_speakers' in checkpoint:
        num_speakers = checkpoint['num_speakers']
    elif 'model_state_dict' in checkpoint:
        # Infer from model state dict if available
        state_dict = checkpoint['model_state_dict']
        if 'sp_clf.weight' in state_dict:
            num_speakers = state_dict['sp_clf.weight'].size(0)
    
    # Initialize model
    model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_speakers
    )
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, num_speakers

def extract_features(audio, wavlm_processor, wavlm_model, voice_encoder, device):
    """Extract SSL features and speaker embeddings."""
    
    # Ensure audio is the right shape and on CPU for processing
    if audio.dim() == 1:
        audio_np = audio.numpy()
    else:
        audio_np = audio.squeeze().numpy()
    
    # Extract SSL features with WavLM
    inputs = wavlm_processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        ssl_outputs = wavlm_model(**inputs)
        ssl_features = ssl_outputs.last_hidden_state.squeeze(0)  # (T, d_ssl)
    
    # Extract speaker embedding with Resemblyzer
    with torch.no_grad():
        # Resemblyzer expects numpy array
        spk_embed = voice_encoder.embed_utterance(audio_np)
        spk_embed = torch.from_numpy(spk_embed).float().to(device)  # (d_spk,)
    
    return ssl_features, spk_embed

def convert_voice(rcop_model, source_ssl_features, ref_spk_embed, device):
    """Convert voice by projecting source content to reference speaker."""
    
    with torch.no_grad():
        # Get the speaker axis for reference speaker
        ref_axis = rcop_model.W_proj(ref_spk_embed)
        
        # Project source SSL features to remove speaker info
        from models.projection import orthogonal_project
        content_features = orthogonal_project(source_ssl_features, ref_axis)
        
        # In a complete implementation, you would:
        # 1. Use the content features to synthesize speech with the reference speaker
        # 2. This typically involves a vocoder or TTS system
        # For now, we return the content features
        
        return content_features

def simple_vocoder_synthesis(features, target_length=None):
    """Simple placeholder vocoder - converts features back to audio."""
    # This is a very simplified vocoder implementation
    # In practice, you'd use a proper vocoder like HiFi-GAN
    
    # Simple approach: use the mean of features as a "harmonic profile"
    # and generate audio through basic synthesis
    
    if target_length is None:
        target_length = features.size(0) * 320  # Rough frame-to-sample ratio
    
    # Use a subset of features as harmonic content
    harmonic_content = features[:, :64].mean(dim=1)  # (T,)
    
    # Generate simple audio by repeating and modulating the harmonic content
    samples_per_frame = target_length // features.size(0)
    audio = []
    
    for i, harm in enumerate(harmonic_content):
        # Simple sine wave generation based on feature values
        t = torch.linspace(0, 2 * np.pi, samples_per_frame)
        freq = 100 + torch.clamp(harm * 200, 0, 400)  # Map to reasonable freq range
        frame_audio = 0.1 * torch.sin(freq * t)
        audio.append(frame_audio)
    
    audio = torch.cat(audio)
    
    # Ensure correct length
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - len(audio)))
    
    return audio

def inference(checkpoint_path, source_wav, ref_wav, output_wav, device, logger, use_vocoder=True):
    """Perform voice conversion inference."""
    
    logger.info(f"Loading audio files...")
    logger.info(f"Source: {source_wav}")
    logger.info(f"Reference: {ref_wav}")
    
    # Load audio files
    source_audio = load_audio(source_wav)
    ref_audio = load_audio(ref_wav)
    
    logger.info(f"Source audio length: {len(source_audio) / 16000:.2f}s")
    logger.info(f"Reference audio length: {len(ref_audio) / 16000:.2f}s")
    
    # Load models
    logger.info("Loading models...")
    config = Config()
    
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm_model.eval()
    wavlm_model.to(device)
    
    voice_encoder = VoiceEncoder()
    voice_encoder.eval()
    
    # For inference, we use dummy values for num_speakers and num_phones
    # In practice, these should match the training setup
    num_speakers = 100  # Adjust based on your training data
    num_phones = get_num_phones()
    
    rcop_model, num_speakers = load_model_from_checkpoint(checkpoint_path, num_speakers, num_phones, device)
    
    # Extract features
    logger.info("Extracting features...")
    source_ssl_features, source_spk_embed = extract_features(
        source_audio, wavlm_processor, wavlm_model, voice_encoder, device
    )
    ref_ssl_features, ref_spk_embed = extract_features(
        ref_audio, wavlm_processor, wavlm_model, voice_encoder, device
    )
    
    logger.info(f"Source SSL features shape: {source_ssl_features.shape}")
    logger.info(f"Reference speaker embedding shape: {ref_spk_embed.shape}")
    
    # Perform voice conversion
    logger.info("Performing voice conversion...")
    converted_features = convert_voice(rcop_model, source_ssl_features, ref_spk_embed, device)
    
    logger.info(f"Converted features shape: {converted_features.shape}")
    
    # Synthesize audio
    if use_vocoder:
        try:
            # Try to use SpeechT5 HiFi-GAN vocoder
            logger.info("Using HiFi-GAN vocoder...")
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            vocoder.to(device)
            vocoder.eval()
            
            # Convert features to mel-spectrogram format expected by vocoder
            # This is a simplified conversion - in practice you'd need proper feature processing
            with torch.no_grad():
                # Project to mel dimensions (80 dims typically)
                if converted_features.size(-1) != 80:
                    mel_proj = torch.nn.Linear(converted_features.size(-1), 80).to(device)
                    mel_features = mel_proj(converted_features)
                else:
                    mel_features = converted_features
                
                # Generate audio
                converted_audio = vocoder(mel_features.unsqueeze(0))
                converted_audio = converted_audio.squeeze().cpu()
            
        except Exception as e:
            logger.warning(f"Vocoder failed, using simple synthesis: {e}")
            converted_audio = simple_vocoder_synthesis(converted_features.cpu(), len(source_audio))
    else:
        logger.info("Using simple synthesis...")
        converted_audio = simple_vocoder_synthesis(converted_features.cpu(), len(source_audio))
    
    # Save output
    logger.info(f"Saving converted audio to {output_wav}")
    sf.write(output_wav, converted_audio.numpy(), 16000)
    
    logger.info("Voice conversion completed!")
    
    # Return some statistics
    return {
        'source_length': len(source_audio),
        'output_length': len(converted_audio),
        'feature_shape': converted_features.shape
    }

def main():
    parser = argparse.ArgumentParser(description="RC-OP Voice Conversion Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--source_wav", type=str, required=True, help="Source audio file")
    parser.add_argument("--ref_wav", type=str, required=True, help="Reference speaker audio file")
    parser.add_argument("--out_wav", type=str, required=True, help="Output audio file")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_vocoder", action="store_true", help="Use HiFi-GAN vocoder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.source_wav):
        raise FileNotFoundError(f"Source audio file not found: {args.source_wav}")
    if not os.path.exists(args.ref_wav):
        raise FileNotFoundError(f"Reference audio file not found: {args.ref_wav}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.out_wav) if os.path.dirname(args.out_wav) else ".", exist_ok=True)
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("infer", args.log_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(Config().seed)
    logger.info(f"Using random seed: {Config().seed}")

    # Perform inference
    try:
        results = inference(
            args.ckpt, args.source_wav, args.ref_wav, args.out_wav, 
            device, logger, args.use_vocoder
        )
        logger.info(f"Inference results: {results}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main() 