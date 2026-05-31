"""NeuralKNN-VC dataset: speaker-grouped audio folder with random crops.

Everything is 16 kHz (kNN-VC operates entirely at 16 kHz). Audio is grouped by
the TOP-LEVEL speaker directory under ``root_dir`` and globbed recursively so
the same loader covers:

  * LibriSpeech / LibriTTS:  <spk>/<chapter>/*.flac
  * VCTK:                    p###/*_mic1.flac  (or any *.flac)
  * flat layouts:            <spk>/*.wav

``__getitem__`` returns ``{'audio_16k', 'speaker', 'path'}`` where ``audio_16k``
is a random ``crop_sec``-second crop (mono, peak-normalised, 16 kHz).

Used by ``distill/dataset_gen.py`` to build kNN-VC distillation pairs.
"""

import random
from pathlib import Path

import numpy as np  # noqa: F401  (kept available per module contract)
import soundfile as sf  # noqa: F401  (kept available per module contract)
import torch
import torchaudio  # noqa: F401  (kept available per module contract)
from torch.utils.data import DataLoader, Dataset

from utils import load_audio

# Audio extensions searched recursively under each speaker directory.
AUDIO_EXTS = ("*.flac", "*.wav")


class AudioFolderDataset(Dataset):
    """Speaker-grouped audio dataset returning random fixed-length crops.

    Parameters
    ----------
    root_dir : str | Path
        Directory whose immediate sub-directories are speakers.
    crop_sec : float
        Length (seconds) of the random crop returned by ``__getitem__``.
    sr : int
        Sample rate (16000 for kNN-VC; everything stays at 16 kHz).
    max_speakers : int | None
        If set, keep only the first ``max_speakers`` speakers (sorted).
    max_files_per_speaker : int | None
        If set, keep only the first N files per speaker (sorted).
    """

    def __init__(
        self,
        root_dir,
        crop_sec: float = 3.0,
        sr: int = 16000,
        max_speakers: int | None = None,
        max_files_per_speaker: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.crop_sec = float(crop_sec)
        self.sr = int(sr)
        self.max_speakers = max_speakers
        self.max_files_per_speaker = max_files_per_speaker

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"AudioFolderDataset: root_dir does not exist: {self.root_dir}"
            )

        # Group audio by top-level speaker directory (recursive glob).
        self.speaker_files: dict[str, list[Path]] = {}
        for spk_dir in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            files: list[Path] = []
            for ext in AUDIO_EXTS:
                files.extend(spk_dir.rglob(ext))
            files = sorted(set(files))
            if not files:
                continue
            if self.max_files_per_speaker is not None:
                files = files[: self.max_files_per_speaker]
            self.speaker_files[spk_dir.name] = files

        self.speakers = sorted(self.speaker_files.keys())
        if self.max_speakers is not None:
            self.speakers = self.speakers[: self.max_speakers]
            self.speaker_files = {s: self.speaker_files[s] for s in self.speakers}

        if not self.speakers:
            raise RuntimeError(
                f"AudioFolderDataset: no audio (*.flac/*.wav) found under {self.root_dir}"
            )

        # Flat index of (speaker, path) for __getitem__.
        self.files: list[tuple[str, Path]] = []
        for spk in self.speakers:
            for f in self.speaker_files[spk]:
                self.files.append((spk, f))

        print(
            f"AudioFolderDataset: {len(self.files)} files, "
            f"{len(self.speakers)} speakers (sr={self.sr}, crop={self.crop_sec}s)"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        spk, path = self.files[idx]

        # load_audio: mono, peak-normalised to 0.95, resampled to self.sr.
        wav = load_audio(str(path), target_sr=self.sr)

        # Skip very short clips (< 0.5 s) by retrying a random index.
        if wav.numel() < int(self.sr * 0.5):
            return self[random.randint(0, len(self) - 1)]

        crop_len = int(self.crop_sec * self.sr)
        if wav.shape[0] > crop_len:
            start = random.randint(0, wav.shape[0] - crop_len)
            wav = wav[start : start + crop_len]

        return {"audio_16k": wav, "speaker": spk, "path": str(path)}


def collate_fn(batch: list[dict]) -> dict:
    """Pad variable-length ``audio_16k`` to the batch max.

    Returns ``{'audio_16k': (B, T), 'lengths': (B,), 'speakers': list[str],
    'paths': list[str]}``.
    """
    audios = [b["audio_16k"] for b in batch]
    lengths = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    max_len = int(lengths.max().item())

    out = torch.zeros(len(audios), max_len, dtype=torch.float32)
    for i, a in enumerate(audios):
        out[i, : a.shape[0]] = a

    return {
        "audio_16k": out,
        "lengths": lengths,
        "speakers": [b["speaker"] for b in batch],
        "paths": [b["path"] for b in batch],
    }


def build_dataloader(config, shuffle: bool = True, num_workers: int = 0):
    """Build a ``(DataLoader, AudioFolderDataset)`` pair from a ``Config``.

    Reads ``data_dir``, ``crop_sec``, ``wavlm_sr``, ``max_speakers``,
    ``max_files_per_speaker`` and ``batch_size`` from ``config``.

    .. note::
        The actual training path (``distill/train.py``) does NOT use this
        DataLoader -- it calls ``distill.dataset_gen.make_distill_batch``
        directly in the main process, where the frozen WavLM-Large + ECAPA
        teacher already lives. This loader only yields raw audio crops (no
        teacher), so it is safe, but if you wire teacher-based batch generation
        into a DataLoader you MUST keep ``num_workers=0`` (the default here):
        each worker is a separate process and would otherwise replicate the
        315M-param WavLM-Large teacher, blowing past 12 GB of VRAM. For the same
        reason ``persistent_workers`` is left off and ``pin_memory`` is only
        enabled when workers are used.
    """
    dataset = AudioFolderDataset(
        root_dir=config.data_dir,
        crop_sec=config.crop_sec,
        sr=config.wavlm_sr,
        max_speakers=config.max_speakers,
        max_files_per_speaker=config.max_files_per_speaker,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(num_workers > 0),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset
