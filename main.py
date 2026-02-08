#!/usr/bin/env python3
"""
Voice Conversion v2 -- Rectified Flow Matching with 1D ResNet backbone.
Single-file implementation: config, data, model, training, validation, inference.

Improvements over v1 (main.py):
  - 1D Convolutional ResNet backbone (vs. frame-wise MLP): smooth spectrograms,
    no jitter/robotic artifacts.
  - Dilated convolutions with FiLM conditioning (Adaptive scale/shift per block).
  - Sinusoidal timestep embeddings (vs. raw scalar).
  - Mode switch: noise / source / svd  (controls the flow starting point z0).
  - Periodic validation: saves converted audio every N steps so you can monitor.
  - Optional wandb logging.

Usage:
    python second.py --mode svd   --steps 100000
    python second.py --mode noise --steps 50000  --no-wandb
    python second.py --mode source --steps 100000 --val-every 2500

    # Inference-only (skip training, load a checkpoint):
    python second.py --inference-only outputs_v2/model_final.pt \\
        --source path/to/source.wav --target path/to/target_ref.wav

Architecture:
  WavLM-base-plus (768-dim, 50 Hz) --> upsample to mel rate (~93 Hz)
  ECAPA-TDNN (192-dim) --> per-block FiLM conditioning
  Sinusoidal timestep --> MLP --> per-block FiLM conditioning
  8x ResidualBlock1D  (dilated conv -> GELU -> conv1x1 -> GroupNorm -> skip)
  Output Conv1d --> 100 mel bins
  Vocoder: Vocos decode(mel_predicted)
"""

import argparse
import enum
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Compatibility shim for SpeechBrain with newer torchaudio versions
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import WavLMModel
from vocos import Vocos
from speechbrain.inference.speaker import EncoderClassifier

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# =====================================================================
# Configuration
# =====================================================================

class FlowMode(enum.Enum):
    NOISE = "noise"    # z0 ~ N(0,1), condition on WavLM
    SOURCE = "source"  # z0 = proj(WavLM), condition on WavLM
    SVD = "svd"        # z0 = proj(SVD-stripped WavLM), condition on SVD-stripped WavLM


# Audio parameters
WAVLM_SR = 16000       # WavLM expects 16 kHz
VOCOS_SR = 24000       # Vocos expects 24 kHz
VOCOS_HOP = 256        # Vocos hop length -> ~93.75 mel frames/sec
MEL_DIM = 100          # Vocos mel bands

# Model dimensions
WAVLM_DIM = 768        # WavLM-base-plus hidden size
ECAPA_DIM = 192        # ECAPA-TDNN speaker embedding size
HIDDEN_DIM = 512       # ResNet hidden channels
NUM_RES_BLOCKS = 8     # Number of residual blocks

# Training defaults
BATCH_SIZE = 16
LR = 1e-4
DEFAULT_STEPS = 100_000
WARMUP_STEPS = 1000
CROP_SEC = 2.0         # Random crop duration (seconds)
CFG_DROPOUT = 0.1      # Probability of dropping speaker embedding for CFG
SVD_K = 2              # Number of top singular vectors to remove
GRAD_CLIP = 1.0

# Inference defaults
ODE_STEPS = 50         # Euler ODE steps (50+ recommended for quality)
GUIDANCE_SCALE = 1.5   # CFG guidance scale (1.0 = no guidance)

# Paths
VCTK_ROOT = Path("data/vctk/wav48_silence_trimmed")
OUTPUT_DIR = Path("outputs_v2")

# Validation defaults
VAL_EVERY = 5000
NUM_VAL_PAIRS = 4


def parse_args():
    p = argparse.ArgumentParser(
        description="Voice Conversion v2 -- Rectified Flow + ResNet-1D"
    )
    # Mode
    p.add_argument("--mode", type=str, default="svd",
                   choices=["noise", "source", "svd"],
                   help="Flow starting-point mode (default: svd)")
    # Data
    p.add_argument("--data-dir", type=str, default=str(VCTK_ROOT),
                   help="Path to VCTK wav48_silence_trimmed")
    # Training
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help=f"Training steps (default: {DEFAULT_STEPS})")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    p.add_argument("--cfg-dropout", type=float, default=CFG_DROPOUT)
    p.add_argument("--svd-k", type=int, default=SVD_K,
                   help="SVD: number of top components to remove")
    p.add_argument("--wavlm-layer", type=int, default=-1,
                   help="WavLM hidden layer to use (-1 = last, 6-7 = more phonetic/less speaker)")
    p.add_argument("--instance-norm", action="store_true",
                   help="Apply instance normalisation to WavLM features (strips speaker statistics)")
    p.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--checkpoint-every", type=int, default=None,
                   help="Save checkpoint every N steps (default: auto)")
    # Validation
    p.add_argument("--val-every", type=int, default=VAL_EVERY,
                   help="Generate validation samples every N steps")
    p.add_argument("--num-val-pairs", type=int, default=NUM_VAL_PAIRS)
    p.add_argument("--val-seed", type=int, default=42,
                   help="RNG seed for validation pair selection (same seed = same pairs)")
    # Inference
    p.add_argument("--ode-steps", type=int, default=ODE_STEPS)
    p.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip the final demo conversion after training")
    # Inference-only mode
    p.add_argument("--inference-only", type=str, default=None, metavar="CKPT",
                   help="Skip training; load this checkpoint and run inference")
    p.add_argument("--source", type=str, default=None,
                   help="Source audio path (inference-only)")
    p.add_argument("--target", type=str, default=None,
                   help="Target speaker reference path (inference-only)")
    # wandb
    p.add_argument("--run-name", type=str, default=None,
                   help="wandb run name (default: auto-generated)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging")
    return p.parse_args()


# =====================================================================
# Dataset
# =====================================================================

class VCTKDataset(Dataset):
    """
    VCTK dataset (wav48_silence_trimmed).
    Loads audio, creates 16 kHz (WavLM) and 24 kHz (Vocos) versions,
    applies proportional random crop so both cover the same time segment.
    """

    def __init__(self, root_dir: Path, crop_sec: float = CROP_SEC):
        self.root_dir = Path(root_dir)
        self.crop_sec = crop_sec

        self.files: list[tuple[str, Path]] = []
        self.speaker_files: dict[str, list[Path]] = {}

        for spk_dir in sorted(self.root_dir.iterdir()):
            if not spk_dir.is_dir() or not spk_dir.name.startswith("p"):
                continue
            spk = spk_dir.name
            # Prefer mic1 for consistency; fall back to any flac/wav
            audio_files = sorted(spk_dir.glob("*_mic1.flac"))
            if not audio_files:
                audio_files = sorted(spk_dir.glob("*.flac"))
            if not audio_files:
                audio_files = sorted(spk_dir.glob("*.wav"))
            if audio_files:
                self.speaker_files[spk] = audio_files
                for f in audio_files:
                    self.files.append((spk, f))

        self.speakers = sorted(self.speaker_files.keys())
        print(f"VCTKDataset: {len(self.files)} files, {len(self.speakers)} speakers")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        spk, path = self.files[idx]

        wav_np, sr = sf.read(path)
        wav = torch.from_numpy(wav_np.astype(np.float32))
        if wav.dim() > 1:
            wav = wav.mean(-1)

        # Skip very short files (< 0.5 s)
        if wav.shape[0] < sr * 0.5:
            return self[random.randint(0, len(self) - 1)]

        # Peak-normalise
        wav = wav / (wav.abs().max() + 1e-9) * 0.95

        # Dual-rate resampling
        audio_16k = torchaudio.functional.resample(wav, sr, WAVLM_SR)
        audio_24k = torchaudio.functional.resample(wav, sr, VOCOS_SR)

        # Proportional random crop (same time window for both rates)
        max_16k = int(self.crop_sec * WAVLM_SR)
        max_24k = int(self.crop_sec * VOCOS_SR)

        if audio_16k.shape[0] > max_16k:
            total_sec = audio_16k.shape[0] / WAVLM_SR
            start_sec = random.random() * (total_sec - self.crop_sec)
            s16 = int(start_sec * WAVLM_SR)
            s24 = int(start_sec * VOCOS_SR)
            audio_16k = audio_16k[s16 : s16 + max_16k]
            audio_24k = audio_24k[s24 : s24 + max_24k]

        return {"audio_16k": audio_16k, "audio_24k": audio_24k, "speaker": spk}


def collate_fn(batch):
    """Pad sequences to max length in batch."""

    def pad(seqs):
        ml = max(s.shape[0] for s in seqs)
        out = torch.zeros(len(seqs), ml)
        lengths = []
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
            lengths.append(s.shape[0])
        return out, torch.tensor(lengths)

    a16, l16 = pad([b["audio_16k"] for b in batch])
    a24, l24 = pad([b["audio_24k"] for b in batch])
    return {
        "audio_16k": a16,
        "audio_24k": a24,
        "lengths_16k": l16,
        "lengths_24k": l24,
        "speakers": [b["speaker"] for b in batch],
    }


# =====================================================================
# Feature Extraction  (all frozen, no gradients)
# =====================================================================

class SpeakerEncoder(nn.Module):
    """Frozen ECAPA-TDNN producing 192-dim speaker embeddings."""

    def __init__(self, device="cuda"):
        super().__init__()
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa_voxceleb",
            run_opts={"device": device},
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """(B, T) or (T,) -> (B, 192)"""
        if audio_16k.dim() == 1:
            audio_16k = audio_16k.unsqueeze(0)
        return self.encoder.encode_batch(audio_16k).squeeze(1)


class FeatureExtractor(nn.Module):
    """Frozen WavLM + Vocos + ECAPA for on-the-fly feature extraction."""

    def __init__(self, device="cuda"):
        super().__init__()

        self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocos.eval()
        for p in self.vocos.parameters():
            p.requires_grad = False

        self.spk_encoder = SpeakerEncoder(device=device)

    @torch.no_grad()
    def extract_wavlm(self, audio_16k: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        (B, T_audio) -> (B, T_feat, 768)
        layer: -1 = last hidden state, 0..N = specific transformer layer.
               Layers 6-7 are more phonetic/content-focused (less speaker info).
        """
        if layer == -1:
            return self.wavlm(audio_16k).last_hidden_state
        out = self.wavlm(audio_16k, output_hidden_states=True)
        # hidden_states is a tuple of (embedding_output, layer_0, layer_1, ...)
        # so index layer+1 to get the output of transformer layer `layer`
        return out.hidden_states[layer + 1]

    @torch.no_grad()
    def extract_mel(self, audio_24k: torch.Tensor) -> torch.Tensor:
        """(B, T_audio) -> (B, T_mel, 100)"""
        return self.vocos.feature_extractor(audio_24k).transpose(1, 2)

    @torch.no_grad()
    def extract_speaker(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """(B, T_audio) -> (B, 192)"""
        return self.spk_encoder.encode(audio_16k)

    def decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, T_mel, 100) -> (B, T_audio)"""
        return self.vocos.decode(mel.transpose(1, 2))


# =====================================================================
# SVD Projection
# =====================================================================

def compute_svd_projection(
    dataloader: DataLoader,
    feat_ext: FeatureExtractor,
    device: str,
    num_samples: int = 500,
    k: int = SVD_K,
    save_path: Path | None = None,
) -> torch.Tensor:
    """
    Collect WavLM features from *num_samples* utterances, run SVD, and return
    a (768, 768) projection matrix P that removes the top-k singular vectors
    (which encode speaker identity).
    """
    print(f"Computing SVD projection (k={k}, up to {num_samples} utterances) ...")
    all_feats = []
    count = 0

    for batch in dataloader:
        if count >= num_samples:
            break
        audio_16k = batch["audio_16k"].to(device)
        with torch.no_grad():
            feats = feat_ext.extract_wavlm(audio_16k)  # (B, T, 768)
        for i in range(feats.shape[0]):
            if count >= num_samples:
                break
            all_feats.append(feats[i].cpu())
            count += 1

    all_feats = torch.cat(all_feats, dim=0)  # (total_frames, 768)
    mean = all_feats.mean(0, keepdim=True)
    centered = all_feats - mean

    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    Vk = Vh[:k].T  # (768, k)
    P = torch.eye(WAVLM_DIM) - Vk @ Vk.T  # (768, 768)

    print(f"  Frames analysed : {all_feats.shape[0]:,}")
    print(f"  Top-{k} singular vals : {[f'{v:.1f}' for v in S[:k].tolist()]}")
    print(f"  Variance removed: {S[:k].pow(2).sum() / S.pow(2).sum() * 100:.1f}%")

    if save_path is not None:
        torch.save({"P": P, "mean": mean}, save_path)
        print(f"  Saved: {save_path}")

    return P


# =====================================================================
# Model -- 1D ResNet with FiLM Conditioning
# =====================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional / timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) scalar timesteps -> (B, dim)"""
        half = self.dim // 2
        freq = math.log(10_000) / (half - 1)
        freq = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -freq)
        emb = t.unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResidualBlock1D(nn.Module):
    """
    Dilated 1D residual block with FiLM (Feature-wise Linear Modulation).

    Path:  GroupNorm -> FiLM(scale, shift) -> GELU -> DilatedConv1d
        -> GroupNorm -> GELU -> Conv1x1 -> + skip

    The global conditioning vector (timestep + speaker) is projected to
    per-channel scale and shift and applied after the first GroupNorm.
    """

    def __init__(self, channels: int, dilation: int = 1, cond_dim: int = HIDDEN_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dilated_conv = nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=dilation, dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        # FiLM: global cond -> scale + shift
        self.film = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, T)
        cond: (B, cond_dim)  global conditioning (timestep + speaker)
        """
        h = self.norm1(x)

        # FiLM conditioning
        ss = self.film(cond)                        # (B, 2C)
        scale, shift = ss.chunk(2, dim=-1)          # each (B, C)
        h = h * (1.0 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = F.gelu(h)
        h = self.dilated_conv(h)

        h = self.norm2(h)
        h = F.gelu(h)
        h = self.pointwise(h)

        return x + h  # skip connection


class FlowMatchingResNet(nn.Module):
    """
    1D-ResNet velocity predictor for Rectified Flow Matching.

    forward(x, t, condition, speaker_emb) -> predicted velocity

        x          : (B, T, mel_dim)    noisy mel at time t
        t          : (B,)               timestep in [0, 1]
        condition  : (B, T, wavlm_dim)  upsampled WavLM features
        speaker_emb: (B, ecapa_dim)     target speaker embedding

    Also provides project_to_mel(wavlm) for building z0 in source/svd modes.
    """

    def __init__(
        self,
        mel_dim: int = MEL_DIM,
        wavlm_dim: int = WAVLM_DIM,
        hidden_dim: int = HIDDEN_DIM,
        spk_dim: int = ECAPA_DIM,
        num_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim

        # --- input projections ---
        self.input_proj = nn.Conv1d(mel_dim, hidden_dim, kernel_size=1)
        self.cond_proj = nn.Conv1d(wavlm_dim, hidden_dim, kernel_size=3, padding=1)

        # --- global conditioning ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spk_mlp = nn.Sequential(
            nn.Linear(spk_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- ResNet backbone (increasing dilations, repeated) ---
        dilations = [1, 2, 4, 8, 1, 2, 4, 8][:num_blocks]
        self.blocks = nn.ModuleList([
            ResidualBlock1D(hidden_dim, dilation=d, cond_dim=hidden_dim)
            for d in dilations
        ])

        # --- output head ---
        self.out_norm = nn.GroupNorm(32, hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, mel_dim, kernel_size=1)

        # --- content -> mel projection (for source / svd z0) ---
        self.content_to_mel = nn.Linear(wavlm_dim, mel_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Transpose to channel-first for Conv1d
        x_cf = x.transpose(1, 2)               # (B, mel, T)
        cond_cf = condition.transpose(1, 2)     # (B, wavlm, T)

        h = self.input_proj(x_cf) + self.cond_proj(cond_cf)  # (B, hidden, T)

        # Global conditioning = timestep + speaker
        g = self.time_mlp(t) + self.spk_mlp(speaker_emb)     # (B, hidden)

        for block in self.blocks:
            h = block(h, g)

        h = self.out_norm(h)
        h = F.gelu(h)
        v = self.out_proj(h)       # (B, mel, T)
        return v.transpose(1, 2)   # (B, T, mel)

    def project_to_mel(self, wavlm_feats: torch.Tensor) -> torch.Tensor:
        """Learnable projection: (B, T, 768) -> (B, T, 100)."""
        return self.content_to_mel(wavlm_feats)


# =====================================================================
# Utilities
# =====================================================================

def upsample_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """(B, T, C) -> (B, target_len, C) via linear interpolation."""
    return F.interpolate(
        x.transpose(1, 2), size=target_len, mode="linear", align_corners=False
    ).transpose(1, 2)


def load_audio(
    path: str, target_sr: int, max_sec: float | None = None
) -> torch.Tensor:
    """Load audio from file, resample, peak-normalise, optionally truncate."""
    wav_np, sr = sf.read(path)
    wav = torch.from_numpy(wav_np.astype(np.float32))
    if wav.dim() > 1:
        wav = wav.mean(-1)
    wav = wav / (wav.abs().max() + 1e-9) * 0.95
    wav = torchaudio.functional.resample(wav, sr, target_sr)
    if max_sec is not None:
        wav = wav[: int(max_sec * target_sr)]
    return wav


def mel_frames_for(num_samples: int, hop: int = VOCOS_HOP) -> int:
    return num_samples // hop


def instance_norm_features(x: torch.Tensor) -> torch.Tensor:
    """
    Per-utterance, per-dimension normalisation: zero mean, unit variance.
    Strips global speaker statistics (pitch bias, formant offsets, etc.).
    x: (B, T, C) -> (B, T, C)
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    return (x - mean) / std


def strip_speaker_from_wavlm(
    wavlm: torch.Tensor,
    svd_proj: torch.Tensor | None,
    use_instance_norm: bool,
) -> torch.Tensor:
    """
    Apply speaker-stripping transforms to WavLM features.
    Order: instance-norm first (removes global stats), then SVD (removes residual directions).
    """
    if use_instance_norm:
        wavlm = instance_norm_features(wavlm)
    if svd_proj is not None:
        wavlm = torch.matmul(wavlm, svd_proj.to(wavlm.device))
    return wavlm


# =====================================================================
# Validation -- periodic sample generation during training
# =====================================================================

class ValidationSet:
    """Small fixed set of (source, target) speaker pairs for monitoring."""

    def __init__(self, dataset: VCTKDataset, num_pairs: int = NUM_VAL_PAIRS, seed: int = 42):
        self.pairs: list[dict] = []
        speakers = dataset.speakers

        # Use a dedicated RNG so the same seed always gives the same pairs,
        # regardless of what happened before this call.
        rng = random.Random(seed)
        used: set[tuple[str, str]] = set()

        for _ in range(num_pairs):
            # Pick a unique (src, tgt) pair
            for _attempt in range(100):
                src = rng.choice(speakers)
                tgt = rng.choice([s for s in speakers if s != src])
                if (src, tgt) not in used:
                    used.add((src, tgt))
                    break

            src_path = rng.choice(dataset.speaker_files[src])
            tgt_path = rng.choice(dataset.speaker_files[tgt])
            self.pairs.append({
                "src_spk": src,
                "tgt_spk": tgt,
                "src_16k": load_audio(str(src_path), WAVLM_SR, max_sec=3.0),
                "src_24k": load_audio(str(src_path), VOCOS_SR, max_sec=3.0),
                "tgt_16k": load_audio(str(tgt_path), WAVLM_SR, max_sec=5.0),
                "tgt_24k": load_audio(str(tgt_path), VOCOS_SR, max_sec=5.0),
            })

        print(f"Validation set: {len(self.pairs)} pairs")
        for i, pair in enumerate(self.pairs):
            print(f"  [{i}] {pair['src_spk']} -> {pair['tgt_spk']}")


@torch.no_grad()
def run_validation(
    model: FlowMatchingResNet,
    feat_ext: FeatureExtractor,
    val_set: ValidationSet,
    svd_proj: torch.Tensor | None,
    mode: FlowMode,
    device,
    step: int,
    output_dir: Path,
    ode_steps: int = ODE_STEPS,
    guidance: float = GUIDANCE_SCALE,
    wavlm_layer: int = -1,
    use_instance_norm: bool = False,
):
    """Generate converted samples for every validation pair and save as wav."""
    model.eval()
    val_dir = output_dir / "val_samples"
    val_dir.mkdir(exist_ok=True)

    # Save source originals and target references once
    ref_dir = val_dir / "references"
    if not ref_dir.exists():
        ref_dir.mkdir()
        for i, pair in enumerate(val_set.pairs):
            sf.write(
                str(ref_dir / f"source_{pair['src_spk']}_{i}.wav"),
                pair["src_24k"].numpy(), VOCOS_SR,
            )
            sf.write(
                str(ref_dir / f"target_{pair['tgt_spk']}_{i}.wav"),
                pair["tgt_24k"].numpy(), VOCOS_SR,
            )

    audios: list[tuple[str, np.ndarray]] = []
    for i, pair in enumerate(val_set.pairs):
        audio_np, dur = convert_voice(
            model, feat_ext,
            pair["src_16k"], pair["src_24k"], pair["tgt_16k"],
            svd_proj, mode, device,
            ode_steps=ode_steps, guidance=guidance,
            wavlm_layer=wavlm_layer, use_instance_norm=use_instance_norm,
        )
        fname = f"step{step:07d}_{pair['src_spk']}_to_{pair['tgt_spk']}.wav"
        out_path = val_dir / fname
        sf.write(str(out_path), audio_np, VOCOS_SR)
        audios.append((fname, audio_np))
        print(f"    val[{i}] {pair['src_spk']}->{pair['tgt_spk']}  {dur:.2f}s")

    model.train()
    return audios


# =====================================================================
# Training
# =====================================================================

def save_checkpoint(model, optimizer, scheduler, svd_proj, step, mode, output_dir):
    path = output_dir / f"checkpoint_{step:07d}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "svd_proj": svd_proj.cpu() if svd_proj is not None else None,
        "step": step,
        "mode": mode.value,
    }, path)
    print(f"  Checkpoint saved: {path}")


def plot_loss(history: list[float], output_dir: Path):
    if not history:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history, alpha=0.25, color="steelblue")
    w = min(200, len(history) // 5) or 1
    if w > 1:
        smooth = np.convolve(history, np.ones(w) / w, mode="valid")
        ax1.plot(range(w - 1, len(history)), smooth, color="crimson", lw=2)
    ax1.set(xlabel="Step", ylabel="Loss", title="Training Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(history, alpha=0.25, color="steelblue")
    if w > 1:
        ax2.plot(range(w - 1, len(history)), smooth, color="crimson", lw=2)
    ax2.set(xlabel="Step", ylabel="Loss (log)", title="Log-scale Loss")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=150)
    plt.close(fig)


def train(
    accelerator: Accelerator,
    model: FlowMatchingResNet,
    feat_ext: FeatureExtractor,
    dataloader: DataLoader,
    svd_proj: torch.Tensor | None,
    mode: str,
    args,
    val_set: ValidationSet,
    output_dir: Path,
):
    mode_enum = FlowMode(mode)
    device = accelerator.device
    use_wandb = HAS_WANDB and not args.no_wandb

    # ---- Optimiser + LR schedule (linear warmup -> cosine decay) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, args.steps - WARMUP_STEPS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Accelerate wrapping
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if svd_proj is not None:
        svd_proj = svd_proj.to(device)
    feat_ext = feat_ext.to(device)

    ckpt_every = args.checkpoint_every or max(args.steps // 5, 5000)

    # ---- wandb ----
    if use_wandb and accelerator.is_main_process:
        wandb.init(
            project="voice-conversion-v2",
            name=args.run_name,
            config={
                "mode": mode, "steps": args.steps,
                "batch_size": args.batch_size, "lr": args.lr,
                "svd_k": args.svd_k, "wavlm_layer": args.wavlm_layer,
                "instance_norm": args.instance_norm,
                "hidden_dim": HIDDEN_DIM,
                "num_blocks": NUM_RES_BLOCKS, "cfg_dropout": args.cfg_dropout,
            },
        )

    # ---- Resume ----
    loss_history: list[float] = []
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from step {start_step}")

    # ---- Training loop ----
    model.train()
    step = start_step
    pbar = tqdm(total=args.steps, initial=start_step, desc=f"Training [{mode}]")

    while step < args.steps:
        for batch in dataloader:
            if step >= args.steps:
                break

            audio_16k = batch["audio_16k"]
            audio_24k = batch["audio_24k"]
            B = audio_16k.shape[0]

            # ---- Feature extraction (frozen) ----
            with torch.no_grad():
                mel_gt = feat_ext.extract_mel(audio_24k)                        # (B, T_mel, 100)
                wavlm = feat_ext.extract_wavlm(audio_16k, layer=args.wavlm_layer)  # (B, T_wlm, 768)
                spk_emb = feat_ext.extract_speaker(audio_16k)                   # (B, 192)

            T_mel = mel_gt.shape[1]
            wavlm_up = upsample_to_length(wavlm, T_mel)        # (B, T_mel, 768)

            # ---- Prepare z0 and condition based on mode ----
            raw = accelerator.unwrap_model(model)

            if mode_enum == FlowMode.NOISE:
                # Condition still benefits from speaker stripping
                condition = strip_speaker_from_wavlm(wavlm_up, svd_proj, args.instance_norm)
                z0 = torch.randn_like(mel_gt)
            elif mode_enum == FlowMode.SOURCE:
                z0 = raw.project_to_mel(wavlm_up)
                condition = wavlm_up
            else:  # SVD
                wavlm_stripped = strip_speaker_from_wavlm(wavlm_up, svd_proj, args.instance_norm)
                z0 = raw.project_to_mel(wavlm_stripped)
                condition = wavlm_stripped

            z1 = mel_gt
            v_target = z1 - z0

            # ---- CFG dropout: randomly zero speaker embedding ----
            if args.cfg_dropout > 0:
                drop_mask = (torch.rand(B, 1, device=device) < args.cfg_dropout).float()
                spk_emb = spk_emb * (1.0 - drop_mask)

            # ---- Flow matching interpolation ----
            t = torch.rand(B, device=device)                     # (B,)
            z_t = (1.0 - t[:, None, None]) * z0 + t[:, None, None] * z1

            # ---- Predict velocity ----
            v_pred = model(z_t, t, condition, spk_emb)
            loss = F.mse_loss(v_pred, v_target)

            # ---- Backward + optimiser ----
            accelerator.backward(loss)
            if args.grad_clip > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # ---- Logging ----
            lv = loss.item()
            loss_history.append(lv)
            avg = sum(loss_history[-200:]) / min(200, len(loss_history))
            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{lv:.4f}", avg=f"{avg:.4f}", lr=f"{lr_now:.2e}")
            pbar.update(1)
            step += 1

            if use_wandb and accelerator.is_main_process and step % 50 == 0:
                wandb.log({"loss": lv, "avg_loss": avg, "lr": lr_now}, step=step)

            # ---- Periodic validation ----
            if step % args.val_every == 0 and accelerator.is_main_process:
                print(f"\n  [Validation @ step {step}]")
                val_audios = run_validation(
                    accelerator.unwrap_model(model), feat_ext, val_set,
                    svd_proj, mode_enum, device, step, output_dir,
                    ode_steps=min(args.ode_steps, 32),  # faster for mid-training
                    guidance=args.guidance_scale,
                    wavlm_layer=args.wavlm_layer,
                    use_instance_norm=args.instance_norm,
                )
                if use_wandb:
                    for fname, anp in val_audios:
                        wandb.log(
                            {f"val/{fname}": wandb.Audio(anp, sample_rate=VOCOS_SR)},
                            step=step,
                        )
                model.train()

            # ---- Periodic checkpoint ----
            if step % ckpt_every == 0 and accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model), optimizer, scheduler,
                    svd_proj, step, mode_enum, output_dir,
                )

    pbar.close()

    # ---- Final stats ----
    final_avg = sum(loss_history[-1000:]) / min(1000, len(loss_history))
    print(f"\nTraining complete ({step:,} steps)")
    print(f"  Final avg loss (last 1k): {final_avg:.4f}")
    print(f"  Min loss: {min(loss_history):.4f}")

    if accelerator.is_main_process:
        plot_loss(loss_history, output_dir)

    if use_wandb and accelerator.is_main_process:
        wandb.finish()

    return model, loss_history


# =====================================================================
# Inference
# =====================================================================

@torch.no_grad()
def euler_ode(
    model: FlowMatchingResNet,
    z0: torch.Tensor,
    condition: torch.Tensor,
    spk_emb: torch.Tensor,
    num_steps: int = ODE_STEPS,
    guidance: float = GUIDANCE_SCALE,
) -> torch.Tensor:
    """
    Euler ODE solver with Classifier-Free Guidance.
    Integrates from z0 (t=0) to z1 (t=1).
    """
    z = z0.clone()
    dt = 1.0 / num_steps
    B = z.shape[0]
    device = z.device
    null_emb = torch.zeros_like(spk_emb)

    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=device)
        if guidance != 1.0:
            v_c = model(z, t, condition, spk_emb)
            v_u = model(z, t, condition, null_emb)
            v = v_u + guidance * (v_c - v_u)
        else:
            v = model(z, t, condition, spk_emb)
        z = z + v * dt

    return z


def convert_voice(
    model: FlowMatchingResNet,
    feat_ext: FeatureExtractor,
    src_16k: torch.Tensor,
    src_24k: torch.Tensor,
    tgt_16k: torch.Tensor,
    svd_proj: torch.Tensor | None,
    mode: FlowMode | str,
    device,
    ode_steps: int = ODE_STEPS,
    guidance: float = GUIDANCE_SCALE,
    wavlm_layer: int = -1,
    use_instance_norm: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Full voice conversion pipeline.

    Returns:
        audio_numpy at 24 kHz, duration in seconds.
    """
    model.eval()

    if isinstance(mode, str):
        mode = FlowMode(mode)

    a16 = src_16k.unsqueeze(0).to(device)
    a24 = src_24k.unsqueeze(0).to(device)
    t16 = tgt_16k.unsqueeze(0).to(device)

    T_mel = mel_frames_for(a24.shape[-1])

    wavlm = feat_ext.extract_wavlm(a16, layer=wavlm_layer)
    wavlm_up = upsample_to_length(wavlm, T_mel)
    spk_emb = feat_ext.extract_speaker(t16)

    if mode == FlowMode.NOISE:
        condition = strip_speaker_from_wavlm(wavlm_up, svd_proj, use_instance_norm)
        z0 = torch.randn(1, T_mel, MEL_DIM, device=device)
    elif mode == FlowMode.SOURCE:
        z0 = model.project_to_mel(wavlm_up)
        condition = wavlm_up
    else:  # SVD
        wavlm_stripped = strip_speaker_from_wavlm(wavlm_up, svd_proj, use_instance_norm)
        z0 = model.project_to_mel(wavlm_stripped)
        condition = wavlm_stripped

    z1 = euler_ode(model, z0, condition, spk_emb, num_steps=ode_steps, guidance=guidance)
    audio = feat_ext.decode_mel(z1)
    audio_np = audio[0].cpu().numpy()
    return audio_np, len(audio_np) / VOCOS_SR


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    mode = args.mode
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("Voice Conversion v2 -- Rectified Flow + ResNet-1D")
    print("=" * 70)
    print(f"  Mode           : {mode}")
    print(f"  Steps          : {args.steps:,}")
    print(f"  Batch size     : {args.batch_size}")
    print(f"  LR             : {args.lr}")
    print(f"  CFG dropout    : {args.cfg_dropout}")
    print(f"  SVD k          : {args.svd_k}")
    print(f"  WavLM layer    : {args.wavlm_layer} ({'last' if args.wavlm_layer == -1 else args.wavlm_layer})")
    print(f"  Instance norm  : {args.instance_norm}")
    print(f"  ODE steps      : {args.ode_steps}")
    print(f"  Guidance       : {args.guidance_scale}")
    print(f"  Val every      : {args.val_every}")
    print(f"  Output         : {output_dir}")
    print(f"  wandb          : {'off' if args.no_wandb or not HAS_WANDB else 'on'}")
    if args.resume:
        print(f"  Resume         : {args.resume}")
    print()

    # ---- Inference-only shortcut ----
    if args.inference_only:
        _run_inference_only(args, output_dir)
        return

    # ---- Setup ----
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    dataset = VCTKDataset(data_dir, crop_sec=CROP_SEC)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
        drop_last=True, persistent_workers=True,
    )

    print("\nLoading frozen models (WavLM, Vocos, ECAPA) ...")
    feat_ext = FeatureExtractor(device=str(device)).to(device)

    # SVD projection (only computed for svd mode)
    svd_proj = None
    if mode == "svd":
        svd_path = output_dir / "svd_proj.pt"
        if svd_path.exists():
            svd_proj = torch.load(svd_path, map_location="cpu")["P"]
            print(f"Loaded SVD projection from {svd_path}")
        else:
            svd_proj = compute_svd_projection(
                dataloader, feat_ext, device,
                num_samples=500, k=args.svd_k, save_path=svd_path,
            )

    model = FlowMatchingResNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlowMatchingResNet: {n_params:,} parameters")

    val_set = ValidationSet(dataset, num_pairs=args.num_val_pairs, seed=args.val_seed)

    # ---- Train ----
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model, loss_history = train(
        accelerator, model, feat_ext, dataloader, svd_proj,
        mode, args, val_set, output_dir,
    )

    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(model)
        torch.save({
            "model": raw.state_dict(),
            "svd_proj": svd_proj.cpu() if svd_proj is not None else None,
            "mode": mode,
            "step": args.steps,
            "wavlm_layer": args.wavlm_layer,
            "instance_norm": args.instance_norm,
            "loss_history": loss_history,
        }, output_dir / "model_final.pt")
        print(f"Saved: {output_dir / 'model_final.pt'}")

    if args.skip_inference:
        print("\nSkipping final inference (--skip-inference)")
        return

    # ---- Final demo conversion ----
    print("\n" + "=" * 70)
    print("FINAL INFERENCE DEMO")
    print("=" * 70)

    raw = accelerator.unwrap_model(model)
    raw.eval()
    mode_enum = FlowMode(mode)

    speakers = dataset.speakers
    src_spk = speakers[0]
    tgt_spk = speakers[min(4, len(speakers) - 1)]

    src_path = dataset.speaker_files[src_spk][0]
    tgt_path = dataset.speaker_files[tgt_spk][0]
    print(f"Source : {src_spk} ({src_path.name})")
    print(f"Target : {tgt_spk} ({tgt_path.name})")

    src_16k = load_audio(str(src_path), WAVLM_SR, max_sec=4.0)
    src_24k = load_audio(str(src_path), VOCOS_SR, max_sec=4.0)
    tgt_16k = load_audio(str(tgt_path), WAVLM_SR, max_sec=5.0)

    conv_np, conv_dur = convert_voice(
        raw, feat_ext, src_16k, src_24k, tgt_16k,
        svd_proj, mode_enum, device,
        ode_steps=args.ode_steps, guidance=args.guidance_scale,
        wavlm_layer=args.wavlm_layer, use_instance_norm=args.instance_norm,
    )
    conv_path = output_dir / f"final_{src_spk}_to_{tgt_spk}.wav"
    sf.write(str(conv_path), conv_np, VOCOS_SR)
    print(f"Converted     : {conv_path} ({conv_dur:.2f}s)")

    recon_np, recon_dur = convert_voice(
        raw, feat_ext, src_16k, src_24k, src_16k,
        svd_proj, mode_enum, device,
        ode_steps=args.ode_steps, guidance=args.guidance_scale,
        wavlm_layer=args.wavlm_layer, use_instance_norm=args.instance_norm,
    )
    recon_path = output_dir / f"final_recon_{src_spk}.wav"
    sf.write(str(recon_path), recon_np, VOCOS_SR)
    print(f"Reconstructed : {recon_path} ({recon_dur:.2f}s)")

    src_dur = len(src_24k) / VOCOS_SR
    orig_path = output_dir / f"original_{src_spk}.wav"
    sf.write(str(orig_path), src_24k.numpy(), VOCOS_SR)

    print(f"\n{'=' * 70}")
    print("DONE!")
    print(f"  Original       : {orig_path} ({src_dur:.2f}s)")
    print(f"  Reconstructed  : {recon_path} ({recon_dur:.2f}s)")
    print(f"  Converted      : {conv_path} ({conv_dur:.2f}s)")
    print(f"  Loss plot      : {output_dir / 'loss.png'}")
    print(f"  Val samples    : {output_dir / 'val_samples/'}")
    print(f"{'=' * 70}")


def _run_inference_only(args, output_dir: Path):
    """Standalone inference: load checkpoint, convert, and save."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"\nLoading checkpoint: {args.inference_only}")
    ckpt = torch.load(args.inference_only, map_location="cpu")

    saved_mode = ckpt.get("mode", args.mode)
    mode_enum = FlowMode(saved_mode)
    svd_proj = ckpt.get("svd_proj")

    model = FlowMatchingResNet()
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (step {ckpt.get('step', '?')}, mode={saved_mode})")

    print("Loading frozen models ...")
    feat_ext = FeatureExtractor(device=device).to(device)

    # Determine source / target audio paths
    if args.source and args.target:
        source_path = args.source
        target_path = args.target
    else:
        # Fall back to first two speakers in VCTK
        data_dir = Path(args.data_dir)
        speakers = sorted(
            d.name for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("p")
        )
        src_spk, tgt_spk = speakers[0], speakers[min(4, len(speakers) - 1)]
        src_dir = data_dir / src_spk
        tgt_dir = data_dir / tgt_spk
        source_path = str(next(src_dir.glob("*_mic1.flac"), next(src_dir.glob("*.flac"))))
        target_path = str(next(tgt_dir.glob("*_mic1.flac"), next(tgt_dir.glob("*.flac"))))

    print(f"Source : {source_path}")
    print(f"Target : {target_path}")

    src_16k = load_audio(source_path, WAVLM_SR, max_sec=10.0)
    src_24k = load_audio(source_path, VOCOS_SR, max_sec=10.0)
    tgt_16k = load_audio(target_path, WAVLM_SR, max_sec=5.0)

    print(f"Converting ({args.ode_steps} ODE steps, guidance={args.guidance_scale}) ...")
    conv_np, dur = convert_voice(
        model, feat_ext, src_16k, src_24k, tgt_16k,
        svd_proj, mode_enum, device,
        ode_steps=args.ode_steps, guidance=args.guidance_scale,
        wavlm_layer=args.wavlm_layer, use_instance_norm=args.instance_norm,
    )

    src_name = Path(source_path).stem
    tgt_name = Path(target_path).stem
    conv_path = output_dir / f"{src_name}_to_{tgt_name}.wav"
    sf.write(str(conv_path), conv_np, VOCOS_SR)
    print(f"Saved: {conv_path} ({dur:.2f}s)")

    # Reconstruction
    recon_np, rdur = convert_voice(
        model, feat_ext, src_16k, src_24k, src_16k,
        svd_proj, mode_enum, device,
        ode_steps=args.ode_steps, guidance=args.guidance_scale,
        wavlm_layer=args.wavlm_layer, use_instance_norm=args.instance_norm,
    )
    recon_path = output_dir / f"{src_name}_reconstructed.wav"
    sf.write(str(recon_path), recon_np, VOCOS_SR)
    print(f"Saved: {recon_path} ({rdur:.2f}s)")

    orig_path = output_dir / f"{src_name}_original.wav"
    sf.write(str(orig_path), src_24k.numpy(), VOCOS_SR)

    print(f"\n{'=' * 50}")
    print("DONE!")
    print(f"  Original      : {orig_path}")
    print(f"  Reconstructed : {recon_path}")
    print(f"  Converted     : {conv_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
