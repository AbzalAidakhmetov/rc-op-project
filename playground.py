#!/usr/bin/env python3
"""
SG-Flow Hypothesis Test: Content-Start vs Source-Start
======================================================

This script tests whether starting a flow from a content-only projection
("projected start") makes conversion to a target speaker easier than starting
from the full source mel ("source start").

Training:
    - Text-matched cross-speaker pairs (same transcript, different speaker)
    - Flow-matching velocity MSE in mel space
    - Auxiliary losses: content preservation + target speaker alignment

Main comparison:
    - Projected-Start: x0 = project_content(source_mel)
    - Source-Start baseline:     x0 = source_mel

Pre-trained components:
    - SVD projection for content start
    - Pre-trained speaker encoder (SUPERB_SV)
    - Pre-trained vocoder (Vocos)

Usage:
    python playground.py

    # After training, use for voice conversion:
    from playground import load_trained_models, convert_audio_file, MelExtractor
    
    models = load_trained_models('outputs/trained_models.pt')
    mel_extractor = MelExtractor(models['cfg'], device)
    
    convert_audio_file(
        models['projected'],
        models['speaker_encoder'],
        mel_extractor,
        source_path='source.wav',       # Content to preserve
        target_ref_path='target_ref.wav', # Voice to imitate
        output_path='converted_projected.wav',
        cfg=models['cfg'],
        device=device
    )
    
    convert_audio_file(
        models['source'],
        models['speaker_encoder'],
        mel_extractor,
        source_path='source.wav',
        target_ref_path='target_ref.wav',
        output_path='converted_source.wav',
        cfg=models['cfg'],
        device=device
    )

Outputs:
    - outputs/trained_models.pt: Saved model checkpoint
    - outputs/source.wav: Source audio
    - outputs/target.wav: Target audio
    - outputs/source_start_converted.wav: Source-start conversion output
    - outputs/projected_start_converted.wav: Projected-start conversion output
    - outputs/training_curves.png: Training loss comparison
    - outputs/conversion_comparison.png: Mel spectrogram comparison
    - outputs/results.txt: Summary of results
"""

# ==============================================================================
# Imports
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import time
import random
import re

# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class Config:
    """Config for real-data mel-spectrogram experiments."""
    # Audio - LibriTTS is 24kHz
    sample_rate: int = 24000
    
    # Mel spectrogram
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    f_min: int = 0
    f_max: int = 12000
    
    # Speaker embedding dimension
    d_spk: int = 192
    
    # Model architecture
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    
    # SVD projection
    svd_rank: int = 40
    
    # Training (3-day-friendly on 3090)
    batch_size: int = 2
    lr: float = 1e-4
    num_steps: int = 20000
    max_frames: int = 160
    aux_content_weight: float = 0.5
    aux_speaker_weight: float = 0.5
    
    # ODE
    ode_steps: int = 25

    # Projection + experiment sizing (smaller = faster)
    min_samples_per_speaker: int = 10
    max_speakers: int = 24
    max_samples_per_speaker: int = 10
    eval_pairs: int = 16
    
    # Vocoder: "vocos" (24kHz)
    vocoder_type: str = "vocos"
    
    # Data
    data_dir: str = "./data/LibriTTS/dev-clean"
    min_duration: float = 1.0
    max_duration: float = 8.0
    
    # Output
    output_dir: str = "./outputs"


# ==============================================================================
# Audio Processing
# ==============================================================================
class MelExtractor:
    """Extract mel spectrograms from audio using soundfile for I/O."""
    
    def __init__(self, cfg: Config, device='cpu'):
        self.cfg = cfg
        self.device = device
        
        self.mel_transform = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        ).to(device)
        
        self.resample_cache = {}
    
    def load_audio(self, path: str) -> torch.Tensor:
        """Load and resample audio to target sample rate using soundfile."""
        audio_np, sr = sf.read(path)
        
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.mean(axis=1)).float().unsqueeze(0)
        
        if sr != self.cfg.sample_rate:
            if sr not in self.resample_cache:
                self.resample_cache[sr] = T.Resample(sr, self.cfg.sample_rate)
            waveform = self.resample_cache[sr](waveform)
        
        return waveform.squeeze(0)
    
    @torch.no_grad()
    def extract_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract log-mel spectrogram. Returns (T, n_mels)."""
        audio = audio.to(self.device)
        mel = self.mel_transform(audio.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0).permute(1, 0).cpu()


# ==============================================================================
# Speaker Embedding
# ==============================================================================
class SimpleSpeakerEncoder(nn.Module):
    """Simple speaker embedding based on mel-spectrogram statistics."""
    
    def __init__(self, n_mels, d_spk=192):
        super().__init__()
        self.n_mels = n_mels
        self.d_spk = d_spk
        self.proj = nn.Linear(n_mels * 4, d_spk)
        nn.init.orthogonal_(self.proj.weight)
    
    @torch.no_grad()
    def encode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from mel spectrogram (T, n_mels)."""
        mel_mean = mel.mean(dim=0)
        mel_std = mel.std(dim=0) + 1e-6
        
        if mel.shape[0] > 1:
            delta = mel[1:] - mel[:-1]
            delta_mean = delta.mean(dim=0)
            delta_std = delta.std(dim=0) + 1e-6
        else:
            delta_mean = torch.zeros_like(mel_mean)
            delta_std = torch.ones_like(mel_std)
        
        stats = torch.cat([mel_mean, mel_std, delta_mean, delta_std])
        embedding = self.proj(stats)
        return F.normalize(embedding, dim=-1)

    def encode_batch(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from a batch of mels (B, T, n_mels)."""
        mel_mean = mel.mean(dim=1)
        mel_std = mel.std(dim=1) + 1e-6
        
        if mel.shape[1] > 1:
            delta = mel[:, 1:] - mel[:, :-1]
            delta_mean = delta.mean(dim=1)
            delta_std = delta.std(dim=1) + 1e-6
        else:
            delta_mean = torch.zeros_like(mel_mean)
            delta_std = torch.ones_like(mel_std)
        
        stats = torch.cat([mel_mean, mel_std, delta_mean, delta_std], dim=-1)
        embedding = self.proj(stats)
        return F.normalize(embedding, dim=-1)


class PretrainedSpeakerEncoder:
    """Pretrained speaker encoder using WavLM embeddings."""
    
    def __init__(self, device: torch.device, input_sr: int):
        bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
        self.model = bundle.get_model().to(device).eval()
        self.model_sr = bundle.sample_rate
        self.input_sr = input_sr
        self.device = device
        self.resampler = None
        if input_sr != self.model_sr:
            self.resampler = T.Resample(input_sr, self.model_sr)
        
        with torch.no_grad():
            dummy = torch.zeros(1, self.model_sr, device=device)
            feats, _ = self.model.extract_features(dummy)
        self.embedding_dim = feats[-1].shape[-1]
    
    @torch.no_grad()
    def encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if self.resampler is not None:
            waveform = self.resampler(waveform.cpu()).to(self.device)
        else:
            waveform = waveform.to(self.device)
        feats, _ = self.model.extract_features(waveform)
        emb = feats[-1].mean(dim=1)
        return F.normalize(emb.squeeze(0), dim=-1)


# ==============================================================================
# Data Scanning
# ==============================================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scan_libritts(data_dir: str, cfg: Config) -> Dict[str, List[Dict]]:
    """Scan LibriTTS directory and organize by speaker."""
    data_path = Path(data_dir)
    speaker_data = {}
    total_wavs = 0
    
    for speaker_dir in tqdm(list(data_path.iterdir()), desc="Scanning speakers"):
        if not speaker_dir.is_dir():
            continue
        
        speaker_id = speaker_dir.name
        speaker_data[speaker_id] = []
        
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            
            for wav_path in chapter_dir.glob("*.wav"):
                total_wavs += 1
                text_path = wav_path.with_suffix(".normalized.txt")
                if not text_path.exists():
                    continue
                text = normalize_text(text_path.read_text().strip())
                if not text:
                    continue
                
                info = sf.info(str(wav_path))
                duration = info.duration
                
                if cfg.min_duration <= duration <= cfg.max_duration:
                    speaker_data[speaker_id].append({
                        "path": str(wav_path),
                        "duration": duration,
                        "text": text,
                    })
    
    if total_wavs == 0:
        raise ValueError(f"No .wav files found under {data_dir}. Download LibriTTS dev-clean first.")
    
    speaker_data = {k: v for k, v in speaker_data.items() if len(v) >= cfg.min_samples_per_speaker}
    total_samples = sum(len(v) for v in speaker_data.values())
    print(f"Found {total_samples} samples from {len(speaker_data)} speakers")
    
    return speaker_data


# ==============================================================================
# SVD Projection
# ==============================================================================
def compute_svd_projection(speaker_mels: Dict[str, List[torch.Tensor]], svd_rank: int) -> torch.Tensor:
    """Compute a content projection matrix from between-speaker variance."""
    speaker_means = []
    for mels in speaker_mels.values():
        all_frames = torch.cat(mels, dim=0)
        speaker_means.append(all_frames.mean(dim=0))
    
    speaker_means = torch.stack(speaker_means)
    centered = speaker_means - speaker_means.mean(dim=0)
    
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    V_k = Vh[:svd_rank].T
    
    P_speaker = V_k @ V_k.T
    P_content = torch.eye(V_k.shape[0]) - P_speaker
    
    total_var = (S ** 2).sum()
    speaker_var = (S[:svd_rank] ** 2).sum()
    print(f"Speaker subspace ({svd_rank} dims) captures {100 * speaker_var / total_var:.1f}% of between-speaker variance")
    
    return P_content




# ==============================================================================
# Dataset
# ==============================================================================
class RealVCDataset:
    """Voice conversion dataset with text-matched cross-speaker pairs."""
    
    def __init__(
        self,
        speaker_mels: Dict[str, List[torch.Tensor]],
        speaker_texts: Dict[str, List[str]],
        speaker_embeddings: Dict[str, torch.Tensor],
        max_frames: int = 200,
    ):
        self.speaker_mels = speaker_mels
        self.speaker_texts = speaker_texts
        self.speaker_embeddings = speaker_embeddings
        self.max_frames = max_frames
        self.speakers = list(speaker_mels.keys())
        self.num_samples = sum(len(v) for v in speaker_mels.values())
        
        text_groups = {}
        for spk, mels in speaker_mels.items():
            texts = speaker_texts[spk]
            for mel, text in zip(mels, texts):
                text_groups.setdefault(text, []).append((spk, mel))
        
        self.text_groups = {}
        for text, items in text_groups.items():
            if len({spk for spk, _ in items}) >= 2:
                self.text_groups[text] = items
        
        self.texts = list(self.text_groups.keys())
        self.use_text_pairs = len(self.texts) > 0
    
    def generate_batch(self, batch_size: int, device: torch.device):
        """Generate a batch for voice conversion training (same text)."""
        source_mels = []
        target_mels = []
        source_spks = []
        target_spks = []
        
        for _ in range(batch_size):
            if self.use_text_pairs:
                text = random.choice(self.texts)
                items = self.text_groups[text]
                src_spk, src_mel = random.choice(items)
                tgt_spk, tgt_mel = random.choice([it for it in items if it[0] != src_spk])
            else:
                src_spk = random.choice(self.speakers)
                tgt_spk = random.choice([s for s in self.speakers if s != src_spk])
                src_mel = random.choice(self.speaker_mels[src_spk])
                tgt_mel = random.choice(self.speaker_mels[tgt_spk])
            
            src_mel = self._truncate(src_mel)
            tgt_mel = self._truncate(tgt_mel)
            if src_mel.shape[0] != tgt_mel.shape[0]:
                pair_len = min(src_mel.shape[0], tgt_mel.shape[0])
                src_mel = src_mel[:pair_len]
                tgt_mel = tgt_mel[:pair_len]
            
            source_mels.append(src_mel)
            target_mels.append(tgt_mel)
            source_spks.append(self.speaker_embeddings[src_spk])
            target_spks.append(self.speaker_embeddings[tgt_spk])
        
        max_len = max(m.shape[0] for m in target_mels)
        
        source_batch = torch.zeros(batch_size, max_len, source_mels[0].shape[1])
        target_batch = torch.zeros(batch_size, max_len, target_mels[0].shape[1])
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        for i, (src, tgt) in enumerate(zip(source_mels, target_mels)):
            source_batch[i, :src.shape[0]] = src
            target_batch[i, :tgt.shape[0]] = tgt
            mask[i, :tgt.shape[0]] = True
        
        return {
            "source_mel": source_batch.to(device),
            "target_mel": target_batch.to(device),
            "source_spk": torch.stack(source_spks).to(device),
            "target_spk": torch.stack(target_spks).to(device),
            "mask": mask.to(device),
        }
    
    def _truncate(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.shape[0] > self.max_frames:
            start = random.randint(0, mel.shape[0] - self.max_frames)
            return mel[start:start + self.max_frames]
        return mel


# ==============================================================================
# Model Components
# ==============================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class VelocityNet(nn.Module):
    """Transformer-based velocity network."""
    
    def __init__(self, d_input, d_model, d_spk, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        self.spk_proj = nn.Linear(d_spk, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, d_input)
    
    def forward(self, x_t, t, spk_emb):
        h = self.input_proj(x_t)
        h = h + self.time_emb(t).unsqueeze(1)
        h = h + self.spk_proj(spk_emb).unsqueeze(1)
        h = self.transformer(h)
        return self.output_proj(h)


class SGFlow(nn.Module):
    """Start-conditioned flow: projected or source start."""
    
    def __init__(
        self,
        velocity_net,
        P_content: torch.Tensor,
        start_mode: str = "projected",
    ):
        """
        Args:
            velocity_net: VelocityNet instance
            P_content: (D, D) fixed content projection matrix
            start_mode: "projected" or "source"
        """
        super().__init__()
        self.velocity_net = velocity_net
        self.start_mode = start_mode
        self.register_buffer('P_content', P_content)
    
    def project_content(self, x):
        """Project to content subspace (remove speaker info)."""
        return x @ self.P_content
    
    def _start_state(self, x_src: torch.Tensor) -> torch.Tensor:
        if self.start_mode == "projected":
            return self.project_content(x_src)
        if self.start_mode == "source":
            return x_src
        raise ValueError(f"Unknown start_mode: {self.start_mode}")
    
    @torch.no_grad()
    def sample(self, x_src, spk_emb, num_steps=10):
        device = x_src.device
        B = x_src.shape[0]
        x_t = self._start_state(x_src)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device)
            v = self.velocity_net(x_t, t, spk_emb)
            x_t = x_t + v * dt
        
        return x_t


# ==============================================================================
# Training
# ==============================================================================
def train_model(
    model, 
    dataset, 
    cfg, 
    device, 
    model_name="Model", 
    proxy_speaker_encoder: Optional[SimpleSpeakerEncoder] = None,
):
    """
    Train a flow matching model on real data.
    
    Args:
        model: flow model to train
        dataset: RealVCDataset instance
        cfg: Config instance
        device: torch device
        model_name: name for progress bar
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.num_steps)
    
    losses = []
    transport_dists = []
    time_to_threshold = None
    loss_threshold = 0.3
    
    model.train()
    pbar = tqdm(range(cfg.num_steps), desc=f"Training {model_name}")
    start_time = time.time()
    
    for step in pbar:
        batch = dataset.generate_batch(cfg.batch_size, device)
        
        optimizer.zero_grad()
        
        # Start-conditioned flow: source -> target with speaker conditioning
        x0 = model._start_state(batch['source_mel'])
        x1 = batch['target_mel']
        
        t = torch.rand(x1.shape[0], device=device)
        t_expand = t[:, None, None]
        
        x_t = (1 - t_expand) * x0 + t_expand * x1
        v_target = x1 - x0
        v_pred = model.velocity_net(x_t, t, batch['target_spk'])
        
        loss = ((v_pred - v_target) ** 2 * batch['mask'].unsqueeze(-1)).sum() / batch['mask'].sum() / x1.shape[-1]
        transport_dists.append((x1 - x0).norm(dim=-1).mean().item())
        
        # Auxiliary losses on estimated endpoint
        x1_hat = x_t + (1 - t_expand) * v_pred
        
        aux_loss = 0.0
        if cfg.aux_content_weight > 0:
            diff = (x1_hat - batch['source_mel']).abs() * batch['mask'].unsqueeze(-1)
            content_loss = diff.sum() / batch['mask'].sum() / batch['source_mel'].shape[-1]
            aux_loss = aux_loss + cfg.aux_content_weight * content_loss
        
        if cfg.aux_speaker_weight > 0 and proxy_speaker_encoder is not None:
            with torch.no_grad():
                target_spk_proxy = proxy_speaker_encoder.encode_batch(batch['target_mel'])
            pred_spk_proxy = proxy_speaker_encoder.encode_batch(x1_hat)
            spk_loss = 1 - F.cosine_similarity(pred_spk_proxy, target_spk_proxy, dim=-1).mean()
            aux_loss = aux_loss + cfg.aux_speaker_weight * spk_loss
        
        loss = loss + aux_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if time_to_threshold is None and loss.item() < loss_threshold:
            time_to_threshold = step
        
        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    train_time = time.time() - start_time
    
    return losses, {
        "final_loss": np.mean(losses[-100:]),
        "min_loss": min(losses),
        "time_to_threshold": time_to_threshold,
        "train_time": train_time,
        "avg_transport_dist": np.mean(transport_dists[-100:]) if transport_dists else 0,
    }


# ==============================================================================
# Audio Reconstruction
# ==============================================================================

# Global vocoder cache to avoid reloading
_vocoder_cache = {}

def get_vocoder(vocoder_type: str = "vocos", device: str = "cpu"):
    """Load and cache vocoder model."""
    global _vocoder_cache
    
    cache_key = f"{vocoder_type}_{device}"
    if cache_key in _vocoder_cache:
        return _vocoder_cache[cache_key]
    
    if vocoder_type == "vocos":
        from vocos import Vocos
        vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        vocoder.to(device)
        vocoder.eval()
        _vocoder_cache[cache_key] = ("vocos", vocoder)
        print("✓ Loaded Vocos vocoder (24kHz)")
        return _vocoder_cache[cache_key]
    
    raise ValueError(f"Unknown vocoder_type: {vocoder_type}")


def mel_to_audio(mel: torch.Tensor, cfg: Config, vocoder_type: str = "vocos") -> torch.Tensor:
    """
    Convert log-mel spectrogram back to audio.
    
    Args:
        mel: (T, n_mels) log-mel spectrogram
        cfg: Config object
        vocoder_type: "vocos"
    
    Returns:
        audio: (num_samples,) waveform tensor
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voc_type, vocoder = get_vocoder(vocoder_type, device)
    
    # Vocos expects (B, n_mels, T) in log scale
    mel_input = mel.T.unsqueeze(0).to(device)  # (1, n_mels, T)
    with torch.no_grad():
        audio = vocoder.decode(mel_input)
    return audio.squeeze().cpu()


# ==============================================================================
# Evaluation Metrics
# ==============================================================================
def compute_content_preservation(
    source_mel: torch.Tensor, 
    converted_mel: torch.Tensor,
) -> Dict[str, float]:
    """
    Measure content preservation using frame-level similarity metrics.
    
    Since we don't have ASR, we use proxy metrics:
    1. DTW-aligned frame similarity (shape preservation)
    2. Delta feature correlation (dynamics preservation)
    3. Spectral envelope correlation
    
    Args:
        source_mel: (T1, D) source mel spectrogram
        converted_mel: (T2, D) converted mel spectrogram
        
    Returns:
        dict with content preservation metrics
    """
    source_np = source_mel.cpu().numpy()
    converted_np = converted_mel.cpu().numpy()
    
    # 1. Global spectral envelope similarity
    source_envelope = source_np.mean(axis=0)
    converted_envelope = converted_np.mean(axis=0)
    envelope_corr = np.corrcoef(source_envelope, converted_envelope)[0, 1]
    
    # 2. Delta (dynamics) correlation
    source_delta = np.diff(source_np, axis=0)
    converted_delta = np.diff(converted_np, axis=0)
    
    # Truncate to same length
    min_len = min(source_delta.shape[0], converted_delta.shape[0])
    if min_len > 0:
        source_delta = source_delta[:min_len]
        converted_delta = converted_delta[:min_len]
        
        # Flatten and compute correlation
        delta_corr = np.corrcoef(
            source_delta.flatten(), 
            converted_delta.flatten()
        )[0, 1]
    else:
        delta_corr = 0.0
    
    # 3. Frame-wise MSE (after length normalization)
    # Simple resampling to same length
    from scipy.interpolate import interp1d
    
    if source_np.shape[0] != converted_np.shape[0]:
        x_src = np.linspace(0, 1, source_np.shape[0])
        x_conv = np.linspace(0, 1, converted_np.shape[0])
        
        interp_func = interp1d(x_conv, converted_np, axis=0, kind='linear')
        converted_resampled = interp_func(x_src)
    else:
        converted_resampled = converted_np
    
    frame_mse = np.mean((source_np - converted_resampled) ** 2)
    
    return {
        "envelope_corr": float(envelope_corr) if not np.isnan(envelope_corr) else 0.0,
        "delta_corr": float(delta_corr) if not np.isnan(delta_corr) else 0.0,
        "frame_mse": float(frame_mse),
    }


def compute_speaker_similarity(
    converted_mel: torch.Tensor,
    reference_mel: torch.Tensor,
    speaker_encoder: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute similarity between converted output and reference speaker.
    """
    with torch.no_grad():
        converted_mel = converted_mel.to(device)
        reference_mel = reference_mel.to(device)
        
        converted_spk_emb = speaker_encoder.encode_mel(converted_mel)
        reference_spk_emb = speaker_encoder.encode_mel(reference_mel)
        
        similarity = F.cosine_similarity(
            converted_spk_emb.unsqueeze(0),
            reference_spk_emb.unsqueeze(0),
            dim=-1
        ).item()
        
    return similarity


# ==============================================================================
# Voice Conversion Test
# ==============================================================================
def test_voice_conversion(
    model: nn.Module,
    source_mel: torch.Tensor,
    target_mel: torch.Tensor,
    target_spk_emb: torch.Tensor,
    speaker_encoder: nn.Module,
    device: torch.device,
    num_steps: int = 50,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Test voice conversion from source to target speaker.
    
    Args:
        model: start-conditioned flow
        source_mel: (T, D) source mel spectrogram
        target_mel: (T, D) target mel spectrogram (for metrics)
        target_spk_emb: (D_spk,) target speaker embedding (for conditioning)
        speaker_encoder: speaker embedding extractor
        device: torch device
        num_steps: ODE integration steps
    Returns:
        converted_mel: (T, D) converted mel spectrogram
        metrics: dict with evaluation metrics
    """
    model.eval()
    source_mel = source_mel.to(device).unsqueeze(0)  # (1, T, D)
    target_spk_emb = target_spk_emb.to(device).unsqueeze(0)  # (1, D_spk)
    
    with torch.no_grad():
        converted = model.sample(source_mel, target_spk_emb, num_steps=num_steps)
    
    converted_mel = converted.squeeze(0).cpu()  # (T, D)
    source_mel_cpu = source_mel.squeeze(0).cpu()
    
    # Compute metrics
    spk_sim = compute_speaker_similarity(
        converted_mel, target_mel.cpu(), speaker_encoder, device
    )
    
    content_metrics = compute_content_preservation(source_mel_cpu, converted_mel)
    
    # Check if speaker was changed (source vs converted)
    source_spk_sim = compute_speaker_similarity(
        converted_mel, source_mel_cpu, speaker_encoder, device
    )
    
    metrics = {
        "target_spk_sim": spk_sim,
        "source_spk_sim": source_spk_sim,
        "spk_change": source_spk_sim - spk_sim,  # Negative = moved toward target
        **content_metrics,
    }
    
    return converted_mel, metrics


# ==============================================================================
# Visualization
# ==============================================================================
def visualize_conversion_comparison(
    source_mel: torch.Tensor,
    target_mel: torch.Tensor,
    source_out: torch.Tensor,
    projected_out: torch.Tensor,
    output_path: Optional[Path] = None,
) -> None:
    """
    Comprehensive comparison of all models' outputs.
    
    Args:
        source_mel: (T, D) source mel
        target_mel: (T, D) target mel (ground truth)
        source_out: (T, D) source-start output
        projected_out: (T, D) projected-start output
        output_path: path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Top row: Mel spectrograms
    axes[0, 0].imshow(source_mel.T, aspect='auto', origin='lower')
    axes[0, 0].set_title('Source')
    axes[0, 0].set_ylabel('Mel bin')
    
    axes[0, 1].imshow(target_mel.T, aspect='auto', origin='lower')
    axes[0, 1].set_title('Target (GT)')
    
    axes[0, 2].imshow(source_out.T, aspect='auto', origin='lower')
    axes[0, 2].set_title('Source-Start')
    
    axes[0, 3].imshow(projected_out.T, aspect='auto', origin='lower')
    axes[0, 3].set_title('Projected-Start')
    
    # Bottom row: Error maps
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Error Maps\n(vs Target)', ha='center', va='center', fontsize=12)
    
    axes[1, 1].axis('off')
    
    # Truncate to minimum length for error computation (handles different length samples)
    min_len = min(source_out.shape[0], target_mel.shape[0])
    target_mel_trunc = target_mel[:min_len]
    source_out_trunc = source_out[:min_len]
    projected_out_trunc = projected_out[:min_len]
    
    source_err = (source_out_trunc - target_mel_trunc).abs()
    im_b = axes[1, 2].imshow(source_err.T, aspect='auto', origin='lower', cmap='Reds')
    axes[1, 2].set_title(f'Source-Start Error\nMSE: {(source_err**2).mean():.4f}')
    axes[1, 2].set_ylabel('Mel bin')
    axes[1, 2].set_xlabel('Time')
    
    projected_err = (projected_out_trunc - target_mel_trunc).abs()
    im_s = axes[1, 3].imshow(projected_err.T, aspect='auto', origin='lower', cmap='Reds')
    axes[1, 3].set_title(f'Projected-Start Error\nMSE: {(projected_err**2).mean():.4f}')
    axes[1, 3].set_xlabel('Time')
    
    plt.suptitle('Voice Conversion Comparison', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved {output_path}")
    
    plt.close()


# ==============================================================================
# Experiment Metrics
# ==============================================================================
def run_comparison_experiments(
    baseline_model: nn.Module,
    projected_model: nn.Module,
    dataset: RealVCDataset,
    speaker_encoder: nn.Module,
    cfg: Config,
    device: torch.device,
    num_test_samples: int = 20,
) -> Dict[str, Dict]:
    """
    Run comprehensive comparison experiments between start modes.
    
    Experiments:
    1. Reconstruction (same speaker): How well does each method reconstruct?
    2. Voice Conversion (different speakers): How well does each convert voice?
    3. Content Preservation: Is linguistic content maintained?
    4. Speaker Transfer: Does output match target speaker?
    
    Returns:
        Dict with experiment results for each model
    """
    results = {
        'source_start': {'recon': [], 'conversion': []},
        'projected_start': {'recon': [], 'conversion': []},
    }
    
    baseline_model.eval()
    projected_model.eval()
    
    print(f"\nRunning {num_test_samples} test samples...")
    
    for i in tqdm(range(num_test_samples), desc="Experiments"):
        if dataset.use_text_pairs:
            text = random.choice(dataset.texts)
            items = dataset.text_groups[text]
            src_spk, src_mel = random.choice(items)
            tgt_spk, tgt_mel = random.choice([it for it in items if it[0] != src_spk])
        else:
            src_spk = random.choice(dataset.speakers)
            tgt_spk = random.choice([s for s in dataset.speakers if s != src_spk])
            src_mel = random.choice(dataset.speaker_mels[src_spk])
            tgt_mel = random.choice(dataset.speaker_mels[tgt_spk])
        
        src_mel = dataset._truncate(src_mel).to(device).unsqueeze(0)
        tgt_mel = dataset._truncate(tgt_mel).to(device).unsqueeze(0)
        
        src_spk_emb = dataset.speaker_embeddings[src_spk].to(device).unsqueeze(0)
        tgt_spk_emb = dataset.speaker_embeddings[tgt_spk].to(device).unsqueeze(0)
        
        with torch.no_grad():
            # === Experiment 1: Reconstruction (same speaker) ===
            source_recon = baseline_model.sample(src_mel, src_spk_emb, num_steps=cfg.ode_steps)
            projected_recon = projected_model.sample(src_mel, src_spk_emb, num_steps=cfg.ode_steps)
            
            results['source_start']['recon'].append(F.mse_loss(source_recon, src_mel).item())
            results['projected_start']['recon'].append(F.mse_loss(projected_recon, src_mel).item())
        
        # === Experiment 2: Voice Conversion (different speakers) ===
        _, source_metrics = test_voice_conversion(
            baseline_model, src_mel.squeeze(0), tgt_mel.squeeze(0), tgt_spk_emb.squeeze(0),
            speaker_encoder, device, num_steps=cfg.ode_steps
        )
        _, projected_metrics = test_voice_conversion(
            projected_model, src_mel.squeeze(0), tgt_mel.squeeze(0), tgt_spk_emb.squeeze(0),
            speaker_encoder, device, num_steps=cfg.ode_steps
        )
        
        results['source_start']['conversion'].append(source_metrics)
        results['projected_start']['conversion'].append(projected_metrics)
    
    # Aggregate results
    summary = {}
    for model_name, model_results in results.items():
        conv = model_results['conversion']
        speaker_shift = [r['target_spk_sim'] - r['source_spk_sim'] for r in conv]
        summary[model_name] = {
            'recon_mse_mean': np.mean(model_results['recon']),
            'recon_mse_std': np.std(model_results['recon']),
            'tgt_spk_sim_mean': np.mean([r['target_spk_sim'] for r in conv]),
            'tgt_spk_sim_std': np.std([r['target_spk_sim'] for r in conv]),
            'src_spk_sim_mean': np.mean([r['source_spk_sim'] for r in conv]),
            'src_spk_sim_std': np.std([r['source_spk_sim'] for r in conv]),
            'speaker_shift_mean': np.mean(speaker_shift),
            'speaker_shift_std': np.std(speaker_shift),
            'delta_corr_mean': np.mean([r['delta_corr'] for r in conv]),
            'delta_corr_std': np.std([r['delta_corr'] for r in conv]),
            'frame_mse_mean': np.mean([r['frame_mse'] for r in conv]),
            'frame_mse_std': np.std([r['frame_mse'] for r in conv]),
        }
    
    return summary


def print_experiment_results(summary: Dict[str, Dict], model_labels: Optional[Dict[str, str]] = None) -> str:
    """Print formatted experiment results."""
    model_labels = model_labels or {}
    output = []
    output.append("\n" + "="*70)
    output.append("EXPERIMENT RESULTS: Start-Mode Comparison")
    output.append("="*70)
    
    # Table header
    models = list(summary.keys())
    col_width = 18
    output.append(f"\n{'Metric':<25} " + " ".join(f"{model_labels.get(m, m):>{col_width}}" for m in models))
    output.append("-" * (25 + (col_width + 1) * len(models)))
    
    # Reconstruction MSE
    row = f"{'Recon MSE ↓':<25}"
    for m in models:
        row += f" {summary[m]['recon_mse_mean']:>11.4f}±{summary[m]['recon_mse_std']:.2f}"
    output.append(row)
    
    # Target Speaker Similarity
    row = f"{'Target Spk Sim ↑':<25}"
    for m in models:
        row += f" {summary[m]['tgt_spk_sim_mean']:>11.4f}±{summary[m]['tgt_spk_sim_std']:.2f}"
    output.append(row)
    
    # Source Speaker Similarity
    row = f"{'Source Spk Sim ↓':<25}"
    for m in models:
        row += f" {summary[m]['src_spk_sim_mean']:>11.4f}±{summary[m]['src_spk_sim_std']:.2f}"
    output.append(row)
    
    # Speaker Shift
    row = f"{'Speaker Shift ↑':<25}"
    for m in models:
        row += f" {summary[m]['speaker_shift_mean']:>11.4f}±{summary[m]['speaker_shift_std']:.2f}"
    output.append(row)
    
    # Content Preservation
    row = f"{'Content Delta Corr ↑':<25}"
    for m in models:
        row += f" {summary[m]['delta_corr_mean']:>11.4f}±{summary[m]['delta_corr_std']:.2f}"
    output.append(row)
    
    row = f"{'Content Frame MSE ↓':<25}"
    for m in models:
        row += f" {summary[m]['frame_mse_mean']:>11.4f}±{summary[m]['frame_mse_std']:.2f}"
    output.append(row)
    
    output.append("-" * (25 + (col_width + 1) * len(models)))
    
    # Analysis
    output.append("\nAnalysis:")
    
    source_start = summary.get('source_start')
    projected_start = summary.get('projected_start')
    if source_start and projected_start:
        if projected_start['speaker_shift_mean'] > source_start['speaker_shift_mean']:
            output.append(f"  ✓ Projected start shifts speaker toward target more")
        else:
            output.append(f"  ✗ Source-start shifts speaker more")
        
        if projected_start['delta_corr_mean'] >= source_start['delta_corr_mean']:
            output.append(f"  ✓ Projected start preserves content dynamics better")
        else:
            output.append(f"  ✗ Source-start preserves content dynamics better")
        
        if projected_start['recon_mse_mean'] <= source_start['recon_mse_mean']:
            output.append(f"  ✓ Projected start reconstructs as well or better")
        else:
            output.append(f"  ? Projected start reconstruction is worse")
    
    result_str = "\n".join(output)
    print(result_str)
    return result_str


# ==============================================================================
# Voice Conversion Function
# ==============================================================================
def convert_voice(
    model: nn.Module,
    source_mel: torch.Tensor,
    target_spk_emb: torch.Tensor,
    device: torch.device,
    num_steps: int = 50,
) -> torch.Tensor:
    """
    Convert voice from source to target speaker.
    
    Args:
        model: Trained flow model (start-conditioned)
        source_mel: (T, D) source mel spectrogram
        target_spk_emb: (D_spk,) target speaker embedding
        device: torch device
        num_steps: ODE integration steps
        
    Returns:
        converted_mel: (T, D) converted mel spectrogram
    """
    model.eval()
    source_mel = source_mel.to(device).unsqueeze(0)  # (1, T, D)
    target_spk_emb = target_spk_emb.to(device).unsqueeze(0)  # (1, D_spk)
    
    with torch.no_grad():
        converted = model.sample(source_mel, target_spk_emb, num_steps=num_steps)
    
    return converted.squeeze(0).cpu()  # (T, D)


def convert_audio_file(
    model: nn.Module,
    speaker_encoder: nn.Module,
    mel_extractor,
    source_path: str,
    target_ref_path: str,
    output_path: str,
    cfg: Config,
    device: torch.device,
) -> str:
    """
    Convert voice from source audio file to target speaker.
    
    Args:
        model: Trained flow model
        speaker_encoder: Speaker embedding extractor
        mel_extractor: MelExtractor instance
        source_path: Path to source audio (content to preserve)
        target_ref_path: Path to target speaker reference audio
        output_path: Output path for converted audio
        cfg: Config instance
        device: torch device
    Returns:
        output_path: Path to saved converted audio
    """
    # Load and extract features
    source_audio = mel_extractor.load_audio(source_path)
    target_audio = mel_extractor.load_audio(target_ref_path)
    
    source_mel = mel_extractor.extract_mel(source_audio).to(device)
    target_mel = mel_extractor.extract_mel(target_audio).to(device)
    
    # Get target speaker embedding
    target_spk_emb = speaker_encoder.encode_waveform(target_audio)
    
    # Convert
    converted_mel = convert_voice(
        model, source_mel, target_spk_emb, device, 
        num_steps=cfg.ode_steps
    )
    
    # Vocode to audio
    converted_audio = mel_to_audio(converted_mel, cfg, cfg.vocoder_type)
    
    # Save
    sf.write(output_path, converted_audio.numpy(), cfg.sample_rate)
    print(f"Saved converted audio to {output_path}")
    
    return output_path


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("="*70)
    print("SG-Flow Hypothesis Test: Content-Start vs Source-Start")
    print("="*70)
    
    cfg = Config()
    
    # Experiment configuration
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"torchaudio version: {torchaudio.__version__}")
    print(f"\nExperiment Settings:")
    print(f"  - Baseline start: source")
    print(f"  - SVD rank: {cfg.svd_rank}")
    
    # Set seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Check data
    data_path = Path(cfg.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"LibriTTS data not found at {data_path}. Run setup.sh first.")
    print(f"\n✓ Data found at {data_path}")
    
    # Initialize mel extractor
    print("\n" + "="*70)
    print("Initializing audio processing...")
    print("="*70)
    mel_extractor = MelExtractor(cfg, device)
    speaker_encoder = PretrainedSpeakerEncoder(device, cfg.sample_rate)
    cfg.d_spk = speaker_encoder.embedding_dim
    cfg.d_spk = speaker_encoder.embedding_dim
    proxy_speaker_encoder = SimpleSpeakerEncoder(cfg.n_mels, cfg.d_spk).to(device)
    for param in proxy_speaker_encoder.parameters():
        param.requires_grad = False
    proxy_speaker_encoder.eval()
    
    # Scan data
    print("\n" + "="*70)
    print("Scanning LibriTTS data...")
    print("="*70)
    speaker_data = scan_libritts(cfg.data_dir, cfg)
    
    # Extract features
    print("\n" + "="*70)
    print("Extracting mel spectrograms and speaker embeddings...")
    print("="*70)
    
    MAX_SAMPLES_PER_SPEAKER = cfg.max_samples_per_speaker
    MAX_SPEAKERS = cfg.max_speakers
    
    speaker_mels = {}
    speaker_texts = {}
    speaker_embeddings = {}
    
    speaker_list = list(speaker_data.keys())[:MAX_SPEAKERS]
    
    for speaker_id in tqdm(speaker_list, desc="Processing speakers"):
        samples = speaker_data[speaker_id]
        speaker_mels[speaker_id] = []
        speaker_texts[speaker_id] = []
        
        for sample in samples[:MAX_SAMPLES_PER_SPEAKER]:
            audio = mel_extractor.load_audio(sample["path"])
            mel = mel_extractor.extract_mel(audio)
            
            if mel.shape[0] >= 50:
                speaker_mels[speaker_id].append(mel)
                speaker_texts[speaker_id].append(sample["text"])
                if speaker_id not in speaker_embeddings:
                    speaker_embeddings[speaker_id] = speaker_encoder.encode_waveform(audio).cpu()
        
    speaker_mels = {k: v for k, v in speaker_mels.items() if len(v) >= 1}
    speaker_texts = {k: v for k, v in speaker_texts.items() if k in speaker_mels}
    speaker_embeddings = {k: v for k, v in speaker_embeddings.items() if k in speaker_mels}
    
    print(f"\nProcessed {len(speaker_mels)} speakers")
    print(f"Total samples: {sum(len(v) for v in speaker_mels.values())}")
    if not speaker_mels:
        raise ValueError(f"No usable speakers found in {cfg.data_dir}. Check transcripts and filtering.")
    
    # Create dataset
    dataset = RealVCDataset(speaker_mels, speaker_texts, speaker_embeddings, max_frames=cfg.max_frames)
    pair_mode = "text-matched" if dataset.use_text_pairs else "unpaired"
    print(f"\nDataset: {dataset.num_samples} samples, {len(dataset.speakers)} speakers ({pair_mode})")
    
    # Setup projection (SVD only)
    print("\n" + "="*70)
    print("Computing SVD projection matrix...")
    print("="*70)
    P_content = compute_svd_projection(speaker_mels, cfg.svd_rank).to(device)
    
    # Create models
    print("\n" + "="*70)
    print("Creating models...")
    print("="*70)
    
    # Source-start baseline
    torch.manual_seed(42)
    source_vnet = VelocityNet(cfg.n_mels, cfg.d_model, cfg.d_spk, cfg.num_layers, cfg.num_heads).to(device)
    source_start_model = SGFlow(
        source_vnet,
        P_content=P_content,
        start_mode="source",
    )
    
    # Projected-start
    torch.manual_seed(42)
    projected_vnet = VelocityNet(cfg.n_mels, cfg.d_model, cfg.d_spk, cfg.num_layers, cfg.num_heads).to(device)
    projected_model = SGFlow(
        projected_vnet, 
        P_content=P_content,
        start_mode="projected",
    )
    
    n_params = sum(p.numel() for p in projected_model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # Train Source-Start baseline
    print("\n" + "="*70)
    print("Training Source-Start baseline...")
    print("="*70)
    source_losses, source_metrics = train_model(
        source_start_model, dataset, cfg, device, 
        model_name="Source-Start",
        proxy_speaker_encoder=proxy_speaker_encoder,
    )
    print(f"\nSource-Start metrics: {source_metrics}")
    
    # Clear GPU cache between training phases to prevent OOM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Train Projected-Start
    print("\n" + "="*70)
    print("Training Projected-Start...")
    print("="*70)
    projected_losses, projected_metrics = train_model(
        projected_model, dataset, cfg, device, 
        model_name="Projected-Start",
        proxy_speaker_encoder=proxy_speaker_encoder,
    )
    print(f"\nProjected-Start metrics: {projected_metrics}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(source_losses, alpha=0.5, label='Source-Start')
    axes[0].plot(projected_losses, alpha=0.5, label='Projected-Start')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss (raw)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    window = 50
    source_smooth = np.convolve(source_losses, np.ones(window)/window, mode='valid')
    projected_smooth = np.convolve(projected_losses, np.ones(window)/window, mode='valid')
    axes[1].plot(source_smooth, label='Source-Start', linewidth=2)
    axes[1].plot(projected_smooth, label='Projected-Start', linewidth=2)
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'Smoothed Loss (window={window})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].semilogy(source_smooth, label='Source-Start', linewidth=2)
    axes[2].semilogy(projected_smooth, label='Projected-Start', linewidth=2)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss (log)')
    axes[2].set_title('Log Scale')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"\n✓ Saved {output_dir}/training_curves.png")
    
    # ==================================================================
    # Run Structured Comparison Experiments
    # ==================================================================
    print("\n" + "="*70)
    print("Running Comparison Experiments...")
    print("="*70)
    
    experiment_results = run_comparison_experiments(
        baseline_model=source_start_model,
        projected_model=projected_model,
        dataset=dataset,
        speaker_encoder=proxy_speaker_encoder,
        cfg=cfg,
        device=device,
        num_test_samples=cfg.eval_pairs,
    )
    
    experiment_summary = print_experiment_results(
        experiment_results,
        model_labels={
            "source_start": "Source-Start",
            "projected_start": "Projected-Start",
        },
    )
    
    # ==================================================================
    # Single Sample Voice Conversion Demo
    # ==================================================================
    print("\n" + "="*70)
    print("Voice Conversion Demo (Single Sample)...")
    print("="*70)
    
    source_start_model.eval()
    projected_model.eval()
    
    if dataset.use_text_pairs:
        demo_text = random.choice(dataset.texts)
        demo_items = dataset.text_groups[demo_text]
        source_spk, source_mel = random.choice(demo_items)
        target_spk, target_mel = random.choice([it for it in demo_items if it[0] != source_spk])
    else:
        source_spk = random.choice(dataset.speakers)
        target_spk = random.choice([s for s in dataset.speakers if s != source_spk])
        source_mel = random.choice(speaker_mels[source_spk])
        target_mel = random.choice(speaker_mels[target_spk])
    
    target_spk_emb = speaker_embeddings[target_spk]
    
    print(f"\nDemo: Speaker {source_spk} -> Speaker {target_spk}")
    
    # Projected-start conversion
    projected_converted, projected_conv_metrics = test_voice_conversion(
        projected_model, source_mel, target_mel, target_spk_emb,
        proxy_speaker_encoder, device, num_steps=cfg.ode_steps
    )
    
    # Source-start conversion
    source_converted, source_conv_metrics = test_voice_conversion(
        source_start_model, source_mel, target_mel, target_spk_emb,
        proxy_speaker_encoder, device, num_steps=cfg.ode_steps
    )
    
    # Content-only conversion
    # Visualize conversion comparison
    visualize_conversion_comparison(
        source_mel, 
        target_mel,
        source_converted.cpu(),
        projected_converted.cpu(),
        output_path=output_dir / "conversion_comparison.png"
    )
    
    # ==================================================================
    # Save audio samples (conversion-focused)
    print("\n" + "="*70)
    print("Saving conversion audio...")
    print("="*70)
    
    source_audio = mel_to_audio(source_mel, cfg, cfg.vocoder_type)
    target_audio = mel_to_audio(target_mel, cfg, cfg.vocoder_type)
    source_conv_audio = mel_to_audio(source_converted.cpu(), cfg, cfg.vocoder_type)
    projected_conv_audio = mel_to_audio(projected_converted.cpu(), cfg, cfg.vocoder_type)
    
    sf.write(output_dir / "source.wav", source_audio.numpy(), cfg.sample_rate)
    sf.write(output_dir / "target.wav", target_audio.numpy(), cfg.sample_rate)
    sf.write(output_dir / "source_start_converted.wav", source_conv_audio.numpy(), cfg.sample_rate)
    sf.write(output_dir / "projected_start_converted.wav", projected_conv_audio.numpy(), cfg.sample_rate)
    
    print(f"✓ Saved audio files to {output_dir}/")
    print(f"  - source.wav ({source_audio.shape[0] / cfg.sample_rate:.2f}s)")
    print(f"  - target.wav ({target_audio.shape[0] / cfg.sample_rate:.2f}s)")
    print(f"  - source_start_converted.wav ({source_conv_audio.shape[0] / cfg.sample_rate:.2f}s)")
    print(f"  - projected_start_converted.wav ({projected_conv_audio.shape[0] / cfg.sample_rate:.2f}s)")
    
    # Save results summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    loss_improve = (source_metrics['final_loss'] - projected_metrics['final_loss']) / source_metrics['final_loss'] * 100
    
    # Get experiment metrics
    exp_source = experiment_results['source_start']
    exp_projected = experiment_results['projected_start']
    
    exp_summary_clean = experiment_summary.strip()
    results = f"""
SG-Flow Hypothesis Test Results
===============================

Setup:
- Data: LibriTTS dev-clean (24kHz)
- Speakers: {len(dataset.speakers)}
- Samples: {dataset.num_samples}
- Mel dims: {cfg.n_mels}
- Training steps: {cfg.num_steps}
- Projection type: SVD (rank={cfg.svd_rank})
- Training task: Voice Conversion

Training Metrics:
- Source-start final loss: {source_metrics['final_loss']:.4f}
- Projected-start final loss:  {projected_metrics['final_loss']:.4f}
- Loss improvement: {loss_improve:.1f}%

Experiment Results (averaged over {cfg.eval_pairs} test samples):
{exp_summary_clean}

Hypothesis Validation:
{"✓ Projected-start converges faster (lower training loss)" if projected_metrics['final_loss'] < source_metrics['final_loss'] else "✗ Projected-start did not converge faster"}
{"✓ Projected start shifts speaker toward target more" if exp_projected['speaker_shift_mean'] > exp_source['speaker_shift_mean'] else "✗ Source-start shifts speaker more"}
{"✓ Projected-start preserves content dynamics better" if exp_projected['delta_corr_mean'] > exp_source['delta_corr_mean'] else "✗ Projected-start content dynamics not better"}

Key Insight:
Projected-start begins from content-only mel (speaker reduced), giving a
structured starting point that should adapt to the target speaker with less
speaker inertia than source-start.
"""
    
    print(results)
    
    with open(output_dir / "results.txt", "w") as f:
        f.write(results)
    print(f"✓ Saved {output_dir}/results.txt")
    
    # Save trained models for later use
    print("\n" + "="*70)
    print("Saving models...")
    print("="*70)
    
    torch.save({
        'projected_state_dict': projected_model.state_dict(),
        'source_state_dict': source_start_model.state_dict(),
        'P_content': P_content.cpu(),
        'config': cfg.__dict__,
        'start_modes': {
            'projected': projected_model.start_mode,
            'source': source_start_model.start_mode,
        },
    }, output_dir / "trained_models.pt")
    print(f"✓ Saved models to {output_dir}/trained_models.pt")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nTo use the trained model for voice conversion:")
    print("  from playground import convert_audio_file, load_trained_models")
    print("  models = load_trained_models('outputs/trained_models.pt')")
    print("  convert_audio_file(models['projected'], ..., source.wav, target_ref.wav, output.wav)")
    print("  convert_audio_file(models['source'], ..., source.wav, target_ref.wav, output.wav)")


def load_trained_models(checkpoint_path: str, device: str = 'cuda'):
    """
    Load trained models from checkpoint.
    
    Args:
        checkpoint_path: Path to trained_models.pt
        device: 'cuda' or 'cpu'
        
    Returns:
        dict with 'projected', 'source', 'speaker_encoder', 'cfg'
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg_dict = checkpoint['config']
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    
    # Rebuild models
    speaker_encoder = PretrainedSpeakerEncoder(device, cfg.sample_rate)
    
    P_content = checkpoint['P_content'].to(device)
    start_modes = checkpoint.get('start_modes', {})
    
    projected_model = None
    if checkpoint.get('projected_state_dict') is not None:
        projected_vnet = VelocityNet(cfg.n_mels, cfg.d_model, cfg.d_spk, cfg.num_layers, cfg.num_heads).to(device)
        projected_model = SGFlow(
            projected_vnet,
            P_content=P_content,
            start_mode=start_modes.get('projected', 'projected'),
        )
        projected_model.load_state_dict(checkpoint['projected_state_dict'])
        projected_model.eval()
    
    source_model = None
    if checkpoint.get('source_state_dict') is not None:
        source_vnet = VelocityNet(cfg.n_mels, cfg.d_model, cfg.d_spk, cfg.num_layers, cfg.num_heads).to(device)
        source_model = SGFlow(
            source_vnet,
            P_content=P_content,
            start_mode=start_modes.get('source', 'source'),
        )
        source_model.load_state_dict(checkpoint['source_state_dict'])
        source_model.eval()
    
    return {
        'projected': projected_model,
        'source': source_model,
        'speaker_encoder': speaker_encoder,
        'cfg': cfg,
    }


if __name__ == "__main__":
    main()