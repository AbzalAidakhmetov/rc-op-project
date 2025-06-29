import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import random
from resampy import resample

class VCTKDataset(Dataset):
    """VCTK dataset for voice conversion training."""
    
    def __init__(self, data_root: str, target_sr: int = 16000, subset: Optional[int] = None):
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        
        # Find the directory containing speaker folders (e.g., 'p225', 'p226')
        wav_parent_dir = self.data_root
        if (self.data_root / "wav48_silence_trimmed").exists():
             wav_parent_dir = self.data_root / "wav48_silence_trimmed"
        elif (self.data_root / "wav48").exists():
             wav_parent_dir = self.data_root / "wav48"
        
        # Find all wav files
        wav_files = []
        for speaker_dir in wav_parent_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                for wav_file in speaker_dir.glob("*_mic1.flac"):
                    wav_files.append((str(wav_file), speaker_id))
        
        if not wav_files:
            raise ValueError(f"No wav files found in subdirectories of {wav_parent_dir}")
            
        # Optional subset for faster training
        if subset and subset < len(wav_files):
            wav_files = random.sample(wav_files, subset)
            
        self.wav_files = wav_files
        
        # Create speaker mapping
        speakers = list(set(spk for _, spk in wav_files))
        speakers.sort()
        self.speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
        self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}
        self.num_speakers = len(speakers)
        
        print(f"Loaded {len(self.wav_files)} files from {len(speakers)} speakers")
        
    def __len__(self):
        return len(self.wav_files)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str]:
        wav_path, speaker_id = self.wav_files[idx]
        
        # Load audio
        try:
            # Try soundfile first (handles more formats)
            audio, orig_sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel if stereo
        except:
            # Fallback to torchaudio
            audio, orig_sr = torchaudio.load(wav_path)
            audio = audio[0].numpy()  # Take first channel
            
        # Resample if needed
        if orig_sr != self.target_sr:
            audio = resample(audio, orig_sr, self.target_sr)
            
        # Convert to tensor
        audio = torch.from_numpy(audio).float()
        
        # Get speaker ID
        spk_id = self.speaker_to_id[speaker_id]
        
        return audio, spk_id, wav_path
        
    def get_num_speakers(self):
        return self.num_speakers
        
    def get_speaker_files(self, speaker_id: str) -> List[str]:
        """Get all files for a specific speaker."""
        return [wav_path for wav_path, spk in self.wav_files if spk == speaker_id]

def collate_fn(batch):
    """Custom collate function to handle variable length audio."""
    audios, spk_ids, wav_paths = zip(*batch)
    
    # For now, just return single items (batch_size=1)
    # In a more advanced implementation, you'd pad sequences
    return audios[0], torch.tensor(spk_ids[0]), wav_paths[0]

def create_dataloader(data_root: str, 
                     target_sr: int = 16000,
                     subset: Optional[int] = None,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 0):
    """Create a VCTK dataloader."""
    if batch_size != 1:
        raise ValueError("This implementation currently supports batch_size=1 only. Update collate_fn for larger batches.")
    
    dataset = VCTKDataset(data_root, target_sr, subset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset 