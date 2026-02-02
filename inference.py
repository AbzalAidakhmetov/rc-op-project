#!/usr/bin/env python3
"""
Inference script for Voice Conversion with Rectified Flow Matching.

Pipeline:
1. Take Source Audio -> Extract WavLM features
2. If SG-Flow: Project to Content Subspace (x_0 = P_content @ x)
   If Baseline: Sample Noise (x_0 ~ N(0,I))
3. Solve ODE using Euler solver (10-20 steps)
4. Decoder(x_1) -> Mel spectrogram
5. HiFi-GAN(Mel) -> Audio

Usage:
    python inference.py --checkpoint checkpoints/sg_flow_best.pt \
        --source_wav source.wav --ref_wav reference.wav \
        --output_dir results/
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, SpeechT5HifiGan
from speechbrain.inference import EncoderClassifier
from resampy import resample
from tqdm import tqdm

from config import Config
from models.flow_matching import create_flow_model
from models.decoder import WavLMToMelDecoder
from models.system import VoiceConversionSystem
from models.projection import OrthogonalProjection
from utils.logging import setup_logger


MODEL_CACHE_DIR = "./models"


def load_audio(file_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio file."""
    audio, orig_sr = sf.read(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if orig_sr != target_sr:
        audio = resample(audio, orig_sr, target_sr)
    return torch.from_numpy(audio).float()


class FeatureExtractor:
    """Extract WavLM features and speaker embeddings for inference."""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        print("Loading WavLM model...")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.wavlm_name, cache_dir=MODEL_CACHE_DIR
        )
        self.wavlm_model = WavLMModel.from_pretrained(
            config.wavlm_name, cache_dir=MODEL_CACHE_DIR
        )
        self.wavlm_model.eval()
        self.wavlm_model.requires_grad_(False)
        self.wavlm_model.to(device)
        
        print("Loading speaker encoder...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source=config.speaker_model_name,
            savedir=os.path.join(MODEL_CACHE_DIR, "spkrec-ecapa-voxceleb"),
            run_opts={"device": device}
        )
        self.speaker_model.eval()
    
    @torch.no_grad()
    def extract_wavlm(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract WavLM features."""
        inputs = self.wavlm_processor(
            audio.cpu().numpy(),
            sampling_rate=self.config.SAMPLE_RATE,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.wavlm_model(**inputs, output_hidden_states=True)
        
        if self.config.wavlm_layer == -1:
            features = outputs.last_hidden_state.squeeze(0)
        else:
            features = outputs.hidden_states[self.config.wavlm_layer].squeeze(0)
        
        return features
    
    @torch.no_grad()
    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding."""
        audio_device = audio.to(self.device)
        embedding = self.speaker_model.encode_batch(audio_device.unsqueeze(0))
        return embedding.squeeze()


def euler_ode_solve(
    flow_model,
    x0: torch.Tensor,
    target_spk: torch.Tensor,
    num_steps: int = 20,
) -> torch.Tensor:
    """
    Euler ODE solver for flow matching.
    
    Solves: dx/dt = v(x_t, t, spk) from t=0 to t=1
    Using: x_{t+dt} = x_t + v(x_t, t, spk) * dt
    """
    device = x0.device
    B = x0.shape[0]
    
    dt = 1.0 / num_steps
    x_t = x0.clone()
    
    for i in tqdm(range(num_steps), desc="ODE Solving", leave=False):
        t = torch.full((B,), i / num_steps, device=device)
        v = flow_model.velocity_net(x_t, t, target_spk)
        x_t = x_t + v * dt
    
    return x_t


def load_model(checkpoint_path: str, config: Config, device: str) -> VoiceConversionSystem:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    mode = checkpoint.get("mode", "sg_flow")
    ckpt_config = checkpoint.get("config", {})
    
    # Get model parameters from checkpoint config
    d_model = ckpt_config.get("d_model", 512)
    num_layers = ckpt_config.get("num_layers", 6)
    num_heads = ckpt_config.get("num_heads", 8)
    dropout = ckpt_config.get("dropout", 0.1)
    projection_path = ckpt_config.get("projection_path", None)
    
    # Create model
    flow_model = create_flow_model(
        method=mode,
        d_input=config.WAVLM_DIM,
        d_model=d_model,
        d_spk=config.d_spk,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        projection_path=projection_path,
    )
    
    decoder = WavLMToMelDecoder(
        d_wavlm=config.WAVLM_DIM,
        d_spk=config.d_spk,
        d_hidden=config.decoder_hidden_dim,
        n_mels=config.n_mels,
        num_layers=config.decoder_num_layers,
    )
    
    projection = None
    if projection_path and os.path.exists(projection_path):
        projection = OrthogonalProjection(projection_path=projection_path)
    
    model = VoiceConversionSystem(flow_model, decoder, projection)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, mode


def convert_voice(
    model: VoiceConversionSystem,
    source_wavlm: torch.Tensor,
    target_spk: torch.Tensor,
    mode: str,
    num_steps: int = 20,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Perform voice conversion.
    
    Args:
        model: Trained VoiceConversionSystem
        source_wavlm: (T, D) source WavLM features
        target_spk: (D_spk,) target speaker embedding
        mode: "baseline" or "sg_flow"
        num_steps: ODE solver steps
        device: compute device
    
    Returns:
        mel: (T_mel, n_mels) converted mel spectrogram
    """
    source_wavlm = source_wavlm.unsqueeze(0).to(device)
    target_spk = target_spk.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get starting point based on mode
        if mode == "sg_flow":
            # Project source to content subspace
            x0 = model.flow_model.sample.__self__.projection.project_content(source_wavlm)
        else:
            # Baseline: start from Gaussian noise
            x0 = torch.randn_like(source_wavlm)
        
        # Solve ODE: x_0 -> x_1
        x1 = euler_ode_solve(model.flow_model, x0, target_spk, num_steps)
        
        # Decode to mel spectrogram
        mel = model.decoder(x1, target_spk)
    
    return mel.squeeze(0)


def mel_to_audio(mel: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Convert mel spectrogram to audio using HiFi-GAN."""
    print("Loading HiFi-GAN vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", cache_dir=MODEL_CACHE_DIR
    )
    vocoder.to(device)
    vocoder.eval()
    
    with torch.no_grad():
        # HiFi-GAN expects (B, T, n_mels)
        audio = vocoder(mel.unsqueeze(0).to(device))
        audio = audio.squeeze().cpu()
    
    return audio


def main():
    parser = argparse.ArgumentParser(description="Voice Conversion Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--source_wav", type=str, required=True, help="Source audio file")
    parser.add_argument("--ref_wav", type=str, required=True, help="Reference speaker audio file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--num_steps", type=int, default=20, help="ODE solver steps (10-20)")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = setup_logger("inference", args.log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Device: {device}")
    logger.info(f"Source: {args.source_wav}")
    logger.info(f"Reference: {args.ref_wav}")
    logger.info(f"ODE steps: {args.num_steps}")
    
    config = Config()
    
    # Load feature extractor
    logger.info("Loading feature extractor...")
    extractor = FeatureExtractor(config, device)
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model, mode = load_model(args.checkpoint, config, device)
    logger.info(f"Model mode: {mode}")
    
    # Load audio files
    logger.info("Loading audio files...")
    source_audio = load_audio(args.source_wav, config.SAMPLE_RATE)
    ref_audio = load_audio(args.ref_wav, config.SAMPLE_RATE)
    
    logger.info(f"Source duration: {len(source_audio)/config.SAMPLE_RATE:.2f}s")
    logger.info(f"Reference duration: {len(ref_audio)/config.SAMPLE_RATE:.2f}s")
    
    # Extract features
    logger.info("Extracting features...")
    source_wavlm = extractor.extract_wavlm(source_audio)
    target_spk = extractor.extract_speaker_embedding(ref_audio)
    
    logger.info(f"Source WavLM shape: {source_wavlm.shape}")
    logger.info(f"Target speaker embedding shape: {target_spk.shape}")
    
    # Voice conversion
    logger.info("Performing voice conversion...")
    converted_mel = convert_voice(
        model, source_wavlm, target_spk, mode,
        num_steps=args.num_steps, device=device
    )
    logger.info(f"Converted mel shape: {converted_mel.shape}")
    
    # Synthesize audio
    logger.info("Synthesizing audio with HiFi-GAN...")
    converted_audio = mel_to_audio(converted_mel, device)
    logger.info(f"Converted audio length: {len(converted_audio)/config.SAMPLE_RATE:.2f}s")
    
    # Save outputs
    source_name = Path(args.source_wav).stem
    ref_name = Path(args.ref_wav).stem
    
    output_path = os.path.join(
        args.output_dir, 
        f"converted_{source_name}_to_{ref_name}_{mode}.wav"
    )
    sf.write(output_path, converted_audio.numpy(), config.SAMPLE_RATE)
    logger.info(f"Saved converted audio to {output_path}")
    
    # Also save source and reference for comparison
    source_copy_path = os.path.join(args.output_dir, f"source_{source_name}.wav")
    ref_copy_path = os.path.join(args.output_dir, f"reference_{ref_name}.wav")
    
    sf.write(source_copy_path, source_audio.numpy(), config.SAMPLE_RATE)
    sf.write(ref_copy_path, ref_audio.numpy(), config.SAMPLE_RATE)
    
    logger.info(f"Saved source copy to {source_copy_path}")
    logger.info(f"Saved reference copy to {ref_copy_path}")
    logger.info("Voice conversion complete!")


if __name__ == "__main__":
    main()
