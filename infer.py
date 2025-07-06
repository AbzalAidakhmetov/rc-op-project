#!/usr/bin/env python3

import argparse
import os

# --- Environment Setup ---
# This must be done BEFORE importing torch, numpy, etc. to mitigate threading issues.
from utils.environment import setup_environment
setup_environment()
# -------------------------

import torch
torch.set_num_threads(1)
import soundfile as sf
import numpy as np
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, SpeechT5HifiGan, AutoFeatureExtractor, AutoModel
from speechbrain.inference import EncoderClassifier
import torchaudio.transforms as T
import librosa

from config import Config
from models.rcop import RCOP
from utils.logging import setup_logger
from utils.phonemes import get_num_phones
from utils.environment import set_seed
from utils.features import extract_features
from utils.checkpoint import load_model_from_checkpoint

MODEL_CACHE_DIR = "./models"

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

def griffin_lim_synthesis(mel_spectrogram, config, target_length=None):
    """
    Synthesize audio from a mel-spectrogram using the Griffin-Lim algorithm.
    This serves as a proper, functional placeholder for a neural vocoder.
    """
    # Inverse Mel-scale transformation
    inverse_mel_scaler = T.InverseMelScale(
        n_stft=config.n_fft // 2 + 1,
        n_mels=config.n_mels,
        sample_rate=config.target_sr,
        f_min=config.f_min,
        f_max=config.f_max,
    )
    
    # The model predicts mel-spectrograms on a natural log scale.
    # Griffin-Lim requires linear magnitude, so we exponentiate.
    linear_spectrogram = torch.exp(mel_spectrogram)
    
    # Invert mel scale to get linear frequency spectrogram
    linear_spectrogram = inverse_mel_scaler(linear_spectrogram.T).T

    # Convert to numpy for librosa
    spec_np = linear_spectrogram.cpu().numpy()
    
    # Griffin-Lim algorithm
    audio = librosa.griffinlim(
        spec_np,
        n_iter=32, # Number of iterations for phase recovery
        hop_length=config.hop_length,
        win_length=config.win_length
    )
    
    audio = torch.from_numpy(audio)

    # Ensure correct length
    if target_length is not None:
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = torch.nn.functional.pad(audio, (0, target_length - len(audio)))
    
    return audio

def inference(checkpoint_path, source_wav, ref_wav, output_wav, device, logger, use_vocoder=True, no_speaker_fusion=False):
    """Perform voice conversion inference."""
    
    logger.info(f"Loading audio files...")
    logger.info(f"Source: {source_wav}")
    logger.info(f"Reference: {ref_wav}")
    
    # Load audio files
    source_audio = load_audio(source_wav)
    ref_audio = load_audio(ref_wav)
    
    logger.info(f"Source audio length: {len(source_audio) / 16000:.2f}s")
    logger.info(f"Reference audio length: {len(ref_audio) / 16000:.2f}s")
    
    # Load config for synthesis parameters
    config = Config()

    # Load models
    logger.info("Loading models...")
    
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model.eval()
    wavlm_model.to(device)
    
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(MODEL_CACHE_DIR, "spkrec-ecapa-voxceleb"),
        run_opts={"device": device}
    )
    if not speaker_model:
        logger.error("Fatal: SpeechBrain speaker model failed to load.")
        raise RuntimeError("Could not load SpeechBrain speaker model.")
    speaker_model.eval()
    
    # The number of speakers is determined by the CHECKPOINT.
    rcop_model, num_speakers = load_model_from_checkpoint(checkpoint_path, 0, get_num_phones(), device)
    logger.info(f"Loaded model trained on {num_speakers} speakers.")
    
    # Extract features
    logger.info("Extracting features...")
    source_ssl_features, source_spk_embed = extract_features(
        source_audio, wavlm_processor, wavlm_model, speaker_model, device
    )
    _, ref_spk_embed = extract_features(
        ref_audio, wavlm_processor, wavlm_model, speaker_model, device
    )
    
    if no_speaker_fusion:
        logger.warning("--- SPEAKER FUSION DISABLED ---")
        logger.warning("Generating audio from content features with a ZERO speaker embedding.")
        ref_spk_embed = torch.zeros_like(ref_spk_embed)
    
    logger.info(f"Source SSL features shape: {source_ssl_features.shape}")
    logger.info(f"Reference speaker embedding shape: {ref_spk_embed.shape}")
    
    # Perform voice conversion
    logger.info("Performing voice conversion...")
    
    with torch.no_grad():
        # The new model forward pass handles the entire conversion.
        # We don't need the adversarial component for inference, so lambda is 0.
        _, _, pred_mels = rcop_model(
            source_ssl_features.unsqueeze(0), 
            ref_spk_embed.unsqueeze(0),
            source_audio.unsqueeze(0).to(device),
            lambd=0.0
        )
        pred_mels = pred_mels.squeeze(0)
    
    logger.info(f"Predicted mel-spectrogram shape: {pred_mels.shape}")
    
    # Safety: if for any reason there is a 1-frame mismatch (rounding), fix it.
    target_mel_len = int(np.ceil(len(source_audio) / config.hop_length))
    if pred_mels.size(0) != target_mel_len:
        logger.warning(f"Frame-length mismatch (pred={pred_mels.size(0)}, target={target_mel_len}). Applying final interpolation.")
        pred_mels = torch.nn.functional.interpolate(
            pred_mels.unsqueeze(0).permute(0, 2, 1),
            size=target_mel_len,
            mode="linear",
            align_corners=False
        ).permute(0, 2, 1).squeeze(0)
    
    # Synthesize audio
    if use_vocoder:
        try:
            # Try to use SpeechT5 HiFi-GAN vocoder
            logger.info("Using HiFi-GAN vocoder...")
            vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan", cache_dir=MODEL_CACHE_DIR
            )
            vocoder.to(device)
            vocoder.eval()
            
            with torch.no_grad():
                # The model's vocoder_head is trained to output 80-dim mels,
                # which is what the HiFi-GAN vocoder expects.
                if pred_mels.size(-1) != 80:
                    logger.warning(f"Mel-spectrogram dimension is {pred_mels.size(-1)}, but vocoder expects 80. This may lead to poor results.")

                # Generate audio (mel: B,n_mels,T)
                # The HiFi-GAN vocoder expects (B, T, n_mels)
                converted_audio = vocoder(pred_mels.unsqueeze(0))
                converted_audio = converted_audio.squeeze().cpu()
            
        except Exception as e:
            logger.critical("="*50)
            logger.critical("HIGH-QUALITY VOCODER FAILED TO LOAD.")
            logger.critical(f"Reason: {e}")
            logger.critical("FALLING BACK TO PLACEHOLDER AUDIO SYNTHESIS.")
            logger.critical("The resulting audio quality will be EXTREMELY POOR.")
            logger.critical("="*50)
            converted_audio = griffin_lim_synthesis(pred_mels.cpu(), config, len(source_audio))
    else:
        logger.warning("="*50)
        logger.warning("HiFi-GAN vocoder was not enabled (`--use_vocoder` flag missing).")
        logger.warning("Using placeholder audio synthesis. Quality will be low but functional.")
        logger.warning("="*50)
        converted_audio = griffin_lim_synthesis(pred_mels.cpu(), config, len(source_audio))
    
    # Save output
    logger.info(f"Saving converted audio to {output_wav}")
    sf.write(output_wav, converted_audio.numpy(), 16000)
    
    # --- Save original and reference audio for comparison ---
    base, ext = os.path.splitext(output_wav)
    
    source_output_path = f"{base}_source{ext}"
    logger.info(f"Saving original source audio for comparison to {source_output_path}")
    sf.write(source_output_path, source_audio.numpy(), 16000)

    ref_output_path = f"{base}_reference{ext}"
    logger.info(f"Saving original reference audio for comparison to {ref_output_path}")
    sf.write(ref_output_path, ref_audio.numpy(), 16000)
    
    logger.info("Voice conversion completed!")
    
    # Return some statistics
    return {
        'source_length': len(source_audio),
        'output_length': len(converted_audio),
        'feature_shape': pred_mels.shape
    }

def main():
    parser = argparse.ArgumentParser(description="RC-OP Voice Conversion Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--source_wav", type=str, default=None, help="Source audio file (optional; random if not provided)")
    parser.add_argument("--ref_wav", type=str, default=None, help="Reference speaker audio file (optional; random if not provided)")
    parser.add_argument("--data_root", type=str, default="./data/VCTK-Corpus-0.92", help="Path to VCTK dataset (required for random mode)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_vocoder", action="store_true", help="Use HiFi-GAN vocoder")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--no_speaker_fusion", action="store_true", help="Generate audio from content features without speaker fusion.")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("infer", args.log_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(Config().seed)
    logger.info(f"Using random seed: {Config().seed}")

    source_wav_path = args.source_wav
    ref_wav_path = args.ref_wav

    # --- Automatic File Selection ---
    if not source_wav_path or not ref_wav_path:
        logger.info("Source or reference WAV not provided. Selecting random files from VCTK...")
        from pathlib import Path
        import random

        if not os.path.isdir(args.data_root):
            raise FileNotFoundError(f"VCTK data root not found for random selection: {args.data_root}")
        
        wav_parent_dir = Path(args.data_root) / "wav48_silence_trimmed"
        if not wav_parent_dir.exists():
             wav_parent_dir = Path(args.data_root) / "wav48"

        speakers = [p for p in wav_parent_dir.iterdir() if p.is_dir()]
        if len(speakers) < 2:
            raise ValueError("Need at least two speakers in the dataset for random conversion.")

        speaker1, speaker2 = random.sample(speakers, 2)
        
        source_wav_path = str(random.choice(list(speaker1.glob("*_mic1.flac"))))
        ref_wav_path = str(random.choice(list(speaker2.glob("*_mic1.flac"))))

        logger.info(f"Randomly selected source: {source_wav_path}")
        logger.info(f"Randomly selected reference: {ref_wav_path}")
    
    # Validate input files
    if not os.path.exists(source_wav_path):
        raise FileNotFoundError(f"Source audio file not found: {source_wav_path}")
    if not os.path.exists(ref_wav_path):
        raise FileNotFoundError(f"Reference audio file not found: {ref_wav_path}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt}")

    # --- Generate descriptive output filename ---
    from pathlib import Path
    source_name = Path(source_wav_path).stem.replace("_mic1", "")
    ref_name = Path(ref_wav_path).stem.replace("_mic1", "")
    output_filename = f"converted_{source_name}_to_{ref_name}.wav"
    output_wav_path = os.path.join(args.output_dir, output_filename)
    
    # Perform inference
    try:
        results = inference(
            checkpoint_path=args.ckpt, 
            source_wav=source_wav_path,
            ref_wav=ref_wav_path,
            output_wav=output_wav_path, 
            device=device, 
            logger=logger, 
            use_vocoder=args.use_vocoder, 
            no_speaker_fusion=args.no_speaker_fusion
        )
        logger.info(f"Inference results: {results}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    main() 