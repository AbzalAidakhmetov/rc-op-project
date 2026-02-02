#!/usr/bin/env python3
"""
Dataset classes for Voice Conversion with Rectified Flow Matching.

Loads precomputed features from the preprocessing step.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PrecomputedVCDataset(Dataset):
    """
    Dataset that loads precomputed WavLM features, mel spectrograms, and speaker embeddings.
    
    Returns:
        dict with keys:
            - source_wavlm: (T, 768) source utterance WavLM features
            - target_wavlm: (T, 768) target utterance WavLM features
            - target_spk: (192,) target speaker embedding
            - target_mel: (T_mel, 80) target mel spectrogram
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        val_ratio: float = 0.1,
        same_speaker: bool = True,
        max_frames: int = 500,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Path to preprocessed data directory
            split: "train" or "val"
            val_ratio: Ratio of speakers to use for validation
            same_speaker: If True, source=target (reconstruction). If False, cross-speaker.
            max_frames: Maximum number of WavLM frames to return (for batching)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.same_speaker = same_speaker
        self.max_frames = max_frames
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        # Split speakers into train/val
        all_speakers = sorted(self.metadata["speakers"].keys())
        random.seed(seed)
        random.shuffle(all_speakers)
        
        val_count = max(1, int(len(all_speakers) * val_ratio))
        if split == "val":
            self.speakers = all_speakers[:val_count]
        else:
            self.speakers = all_speakers[val_count:]
        
        # Filter utterances by speaker split
        self.utterances = []
        self.speaker_to_utterances: Dict[str, List[str]] = {s: [] for s in self.speakers}
        
        for utt_id, utt_info in self.metadata["utterances"].items():
            if utt_info["speaker_id"] in self.speakers:
                self.utterances.append(utt_id)
                self.speaker_to_utterances[utt_info["speaker_id"]].append(utt_id)
        
        # Create speaker to embedding path mapping
        self.speaker_embeddings = {}
        for speaker_id in self.speakers:
            emb_path = self.data_dir / self.metadata["speakers"][speaker_id]["embedding_path"]
            self.speaker_embeddings[speaker_id] = emb_path
        
        print(f"Loaded {split} set: {len(self.utterances)} utterances from {len(self.speakers)} speakers")
    
    def __len__(self) -> int:
        return len(self.utterances)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        utt_id = self.utterances[idx]
        utt_info = self.metadata["utterances"][utt_id]
        speaker_id = utt_info["speaker_id"]
        
        # Load target features
        target_wavlm = torch.load(self.data_dir / utt_info["wavlm_path"])
        target_mel = torch.load(self.data_dir / utt_info["mel_path"])
        target_spk = torch.load(self.speaker_embeddings[speaker_id])
        
        if self.same_speaker:
            # Reconstruction mode: source = target
            source_wavlm = target_wavlm.clone()
        else:
            # Cross-speaker mode: pick random utterance from different speaker
            other_speakers = [s for s in self.speakers if s != speaker_id]
            if other_speakers:
                source_speaker = random.choice(other_speakers)
                source_utt_id = random.choice(self.speaker_to_utterances[source_speaker])
                source_info = self.metadata["utterances"][source_utt_id]
                source_wavlm = torch.load(self.data_dir / source_info["wavlm_path"])
            else:
                source_wavlm = target_wavlm.clone()
        
        # Truncate to max_frames if needed
        if target_wavlm.shape[0] > self.max_frames:
            start = random.randint(0, target_wavlm.shape[0] - self.max_frames)
            target_wavlm = target_wavlm[start:start + self.max_frames]
            
            # Align mel spectrogram (assuming ~1.25x frame rate difference)
            mel_ratio = target_mel.shape[0] / (target_wavlm.shape[0] + self.max_frames - target_wavlm.shape[0])
            mel_start = int(start * mel_ratio)
            mel_end = int((start + self.max_frames) * mel_ratio)
            target_mel = target_mel[mel_start:mel_end]
        
        if source_wavlm.shape[0] > self.max_frames:
            start = random.randint(0, source_wavlm.shape[0] - self.max_frames)
            source_wavlm = source_wavlm[start:start + self.max_frames]
        
        return {
            "source_wavlm": source_wavlm,      # (T, 768)
            "target_wavlm": target_wavlm,      # (T, 768)
            "target_spk": target_spk,          # (192,)
            "target_mel": target_mel,          # (T_mel, 80)
            "utt_id": utt_id,
            "speaker_id": speaker_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable length sequences."""
    
    source_wavlm = [item["source_wavlm"] for item in batch]
    target_wavlm = [item["target_wavlm"] for item in batch]
    target_mel = [item["target_mel"] for item in batch]
    target_spk = torch.stack([item["target_spk"] for item in batch])
    
    # Pad sequences
    source_wavlm_padded = pad_sequence(source_wavlm, batch_first=True)
    target_wavlm_padded = pad_sequence(target_wavlm, batch_first=True)
    target_mel_padded = pad_sequence(target_mel, batch_first=True)
    
    # Create masks
    source_lengths = torch.tensor([x.shape[0] for x in source_wavlm])
    target_lengths = torch.tensor([x.shape[0] for x in target_wavlm])
    mel_lengths = torch.tensor([x.shape[0] for x in target_mel])
    
    max_source_len = source_wavlm_padded.shape[1]
    max_target_len = target_wavlm_padded.shape[1]
    max_mel_len = target_mel_padded.shape[1]
    
    source_mask = torch.arange(max_source_len)[None, :] < source_lengths[:, None]
    target_mask = torch.arange(max_target_len)[None, :] < target_lengths[:, None]
    mel_mask = torch.arange(max_mel_len)[None, :] < mel_lengths[:, None]
    
    return {
        "source_wavlm": source_wavlm_padded,    # (B, T_src, 768)
        "target_wavlm": target_wavlm_padded,    # (B, T_tgt, 768)
        "target_spk": target_spk,               # (B, 192)
        "target_mel": target_mel_padded,        # (B, T_mel, 80)
        "source_mask": source_mask,             # (B, T_src)
        "target_mask": target_mask,             # (B, T_tgt)
        "mel_mask": mel_mask,                   # (B, T_mel)
        "source_lengths": source_lengths,       # (B,)
        "target_lengths": target_lengths,       # (B,)
        "mel_lengths": mel_lengths,             # (B,)
    }


def create_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 8,
    num_workers: int = 4,
    same_speaker: bool = True,
    max_frames: int = 500,
    **kwargs,
) -> Tuple[DataLoader, PrecomputedVCDataset]:
    """Create a dataloader for the precomputed VC dataset."""
    
    dataset = PrecomputedVCDataset(
        data_dir=data_dir,
        split=split,
        same_speaker=same_speaker,
        max_frames=max_frames,
        **kwargs,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    
    return dataloader, dataset
