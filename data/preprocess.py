#!/usr/bin/env python3
"""
Preprocessing script for Voice Conversion with Rectified Flow Matching.

Supports both LibriTTS and VCTK datasets.

Extracts and caches:
1. WavLM features (from wavlm-base-plus, layer 6 or last hidden state)
2. Speaker embeddings (ECAPA-TDNN via SpeechBrain)
3. Mel spectrograms (aligned with WavLM frame rate)

Usage:
    # LibriTTS (recommended for quick experiments)
    python data/preprocess.py --data_root ./data/LibriTTS/dev-clean --output_dir ./preprocessed
    
    # VCTK
    python data/preprocess.py --data_root ./data/VCTK-Corpus-0.92 --output_dir ./preprocessed
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from resampy import resample
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from speechbrain.inference import EncoderClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


def load_audio(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    try:
        audio, orig_sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio[:, 0]
    except Exception:
        audio, orig_sr = torchaudio.load(audio_path)
        audio = audio[0].numpy()
    
    if orig_sr != target_sr:
        audio = resample(audio, orig_sr, target_sr)
    
    return torch.from_numpy(audio).float()


def detect_dataset_type(data_root: Path) -> str:
    """Auto-detect whether the dataset is LibriTTS or VCTK."""
    # Check for LibriTTS structure: speaker_id/chapter_id/*.wav
    for speaker_dir in data_root.iterdir():
        if speaker_dir.is_dir():
            for subdir in speaker_dir.iterdir():
                if subdir.is_dir():
                    # Found nested directory structure -> LibriTTS
                    wav_files = list(subdir.glob("*.wav"))
                    if wav_files:
                        return "libritts"
            # Check for VCTK structure: speaker_id/*.flac
            flac_files = list(speaker_dir.glob("*.flac"))
            if flac_files:
                return "vctk"
    
    # Fallback checks
    if (data_root / "wav48_silence_trimmed").exists() or (data_root / "wav48").exists():
        return "vctk"
    
    return "libritts"  # Default to LibriTTS


def get_libritts_files(data_root: Path, max_duration_s: float = 10.0) -> List[Tuple[str, str, str]]:
    """
    Scan LibriTTS directory for audio files.
    
    LibriTTS structure: {speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utterance_id}.wav
    """
    all_files = []
    
    print("Scanning LibriTTS directory...")
    for speaker_dir in data_root.iterdir():
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        
        # Iterate through chapter directories
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            
            # Find all .wav files in this chapter
            for wav_path in chapter_dir.glob("*.wav"):
                utt_id = wav_path.stem  # e.g., "1089_134686_000001_000001"
                all_files.append((str(wav_path), speaker_id, utt_id))
    
    # Filter by duration
    filtered_files = []
    print(f"Found {len(all_files)} files, filtering by duration...")
    for wav_path, speaker_id, utt_id in tqdm(all_files, desc="Filtering"):
        try:
            info = sf.info(wav_path)
            duration = info.frames / info.samplerate
            if duration <= max_duration_s and duration >= 0.5:  # At least 0.5s
                filtered_files.append((wav_path, speaker_id, utt_id))
        except Exception:
            continue
    
    print(f"Kept {len(filtered_files)}/{len(all_files)} files (0.5s - {max_duration_s}s)")
    return filtered_files


def get_vctk_files(data_root: Path, max_duration_s: float = 10.0) -> List[Tuple[str, str, str]]:
    """
    Scan VCTK directory for audio files.
    
    VCTK structure: {speaker_id}/{speaker_id}_{utterance_id}_mic1.flac
    """
    wav_parent_dir = data_root
    if (data_root / "wav48_silence_trimmed").exists():
        wav_parent_dir = data_root / "wav48_silence_trimmed"
    elif (data_root / "wav48").exists():
        wav_parent_dir = data_root / "wav48"
    
    all_files = []
    
    print("Scanning VCTK directory...")
    for speaker_dir in wav_parent_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            for wav_path in speaker_dir.glob("*_mic1.flac"):
                utt_id = wav_path.stem.replace("_mic1", "")
                all_files.append((str(wav_path), speaker_id, utt_id))
    
    filtered_files = []
    print(f"Found {len(all_files)} files, filtering by duration...")
    for wav_path, speaker_id, utt_id in tqdm(all_files, desc="Filtering"):
        try:
            info = sf.info(wav_path)
            duration = info.frames / info.samplerate
            if duration <= max_duration_s and duration >= 0.5:
                filtered_files.append((wav_path, speaker_id, utt_id))
        except Exception:
            continue
    
    print(f"Kept {len(filtered_files)}/{len(all_files)} files (0.5s - {max_duration_s}s)")
    return filtered_files


def get_audio_files(data_root: Path, max_duration_s: float = 10.0) -> Tuple[List[Tuple[str, str, str]], str]:
    """Auto-detect dataset type and get audio files."""
    dataset_type = detect_dataset_type(data_root)
    print(f"Detected dataset type: {dataset_type.upper()}")
    
    if dataset_type == "libritts":
        files = get_libritts_files(data_root, max_duration_s)
    else:
        files = get_vctk_files(data_root, max_duration_s)
    
    return files, dataset_type


class FeatureExtractor:
    """Extract WavLM features, speaker embeddings, and mel spectrograms."""
    
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        
        print("Loading WavLM model...")
        self.wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
            config.wavlm_name, cache_dir="./models"
        )
        self.wavlm_model = WavLMModel.from_pretrained(
            config.wavlm_name, cache_dir="./models"
        )
        self.wavlm_model.eval()
        self.wavlm_model.requires_grad_(False)
        self.wavlm_model.to(device)
        
        print("Loading speaker encoder...")
        self.speaker_model = EncoderClassifier.from_hparams(
            source=config.speaker_model_name,
            savedir="./models/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        self.speaker_model.eval()
        
        print("Setting up mel spectrogram transform...")
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
        ).to(device)
    
    @torch.no_grad()
    def extract_wavlm(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract WavLM features from audio."""
        audio_np = audio.cpu().numpy()
        inputs = self.wavlm_processor(
            audio_np, 
            sampling_rate=self.config.SAMPLE_RATE, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.wavlm_model(**inputs, output_hidden_states=True)
        
        if self.config.wavlm_layer == -1:
            features = outputs.last_hidden_state.squeeze(0)
        else:
            features = outputs.hidden_states[self.config.wavlm_layer].squeeze(0)
        
        return features.cpu()
    
    @torch.no_grad()
    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding using ECAPA-TDNN."""
        audio_device = audio.to(self.device)
        embedding = self.speaker_model.encode_batch(audio_device.unsqueeze(0))
        return embedding.squeeze().cpu()
    
    @torch.no_grad()
    def extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram."""
        audio_device = audio.to(self.device)
        mel = self.mel_transform(audio_device.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0).permute(1, 0).cpu()


def preprocess_dataset(
    data_root: str,
    output_dir: str,
    config: Config,
    device: str = "cuda",
    max_files: Optional[int] = None,
):
    """Preprocess entire dataset and save features."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wavlm_dir = output_dir / "wavlm"
    mel_dir = output_dir / "mel"
    spk_dir = output_dir / "speaker_embeddings"
    
    wavlm_dir.mkdir(exist_ok=True)
    mel_dir.mkdir(exist_ok=True)
    spk_dir.mkdir(exist_ok=True)
    
    # Auto-detect dataset type and get files
    files, dataset_type = get_audio_files(data_root, config.max_duration_s)
    if max_files:
        files = files[:max_files]
    
    extractor = FeatureExtractor(config, device)
    
    metadata = {
        "utterances": {},
        "speakers": {},
        "dataset_type": dataset_type,
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "wavlm_dim": config.WAVLM_DIM,
            "n_mels": config.n_mels,
            "hop_length": config.hop_length,
            "wavlm_layer": config.wavlm_layer,
        }
    }
    
    speaker_audios: Dict[str, List[torch.Tensor]] = {}
    
    print("Extracting features...")
    for wav_path, speaker_id, utt_id in tqdm(files, desc="Processing"):
        try:
            audio = load_audio(wav_path, config.SAMPLE_RATE)
            
            if len(audio) < config.SAMPLE_RATE * 0.5:
                continue
            
            wavlm_features = extractor.extract_wavlm(audio)
            mel_features = extractor.extract_mel(audio)
            
            wavlm_path = wavlm_dir / f"{speaker_id}_{utt_id}_wavlm.pt"
            mel_path = mel_dir / f"{speaker_id}_{utt_id}_mel.pt"
            
            torch.save(wavlm_features, wavlm_path)
            torch.save(mel_features, mel_path)
            
            if speaker_id not in speaker_audios:
                speaker_audios[speaker_id] = []
            if len(speaker_audios[speaker_id]) < 10:
                speaker_audios[speaker_id].append(audio)
            
            metadata["utterances"][f"{speaker_id}_{utt_id}"] = {
                "speaker_id": speaker_id,
                "wavlm_path": str(wavlm_path.relative_to(output_dir)),
                "mel_path": str(mel_path.relative_to(output_dir)),
                "wavlm_frames": wavlm_features.shape[0],
                "mel_frames": mel_features.shape[0],
                "duration_s": len(audio) / config.SAMPLE_RATE,
            }
            
            if speaker_id not in metadata["speakers"]:
                metadata["speakers"][speaker_id] = {"utterance_count": 0}
            metadata["speakers"][speaker_id]["utterance_count"] += 1
            
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            continue
    
    print("Extracting speaker embeddings...")
    for speaker_id, audios in tqdm(speaker_audios.items(), desc="Speaker embeddings"):
        embeddings = []
        for audio in audios:
            emb = extractor.extract_speaker_embedding(audio)
            embeddings.append(emb)
        
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding = avg_embedding / avg_embedding.norm()
        
        spk_path = spk_dir / f"{speaker_id}_spkemb.pt"
        torch.save(avg_embedding, spk_path)
        
        metadata["speakers"][speaker_id]["embedding_path"] = str(spk_path.relative_to(output_dir))
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nPreprocessing complete!")
    print(f"  Total utterances: {len(metadata['utterances'])}")
    print(f"  Total speakers: {len(metadata['speakers'])}")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for Voice Conversion (supports LibriTTS and VCTK)")
    parser.add_argument("--data_root", type=str, required=True, 
                       help="Path to dataset root (e.g., ./data/LibriTTS/dev-clean or ./data/VCTK-Corpus-0.92)")
    parser.add_argument("--output_dir", type=str, default="./preprocessed", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process (for quick testing)")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max audio duration in seconds")
    
    args = parser.parse_args()
    
    config = Config()
    config.max_duration_s = args.max_duration
    config.device = args.device
    
    preprocess_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
