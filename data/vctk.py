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
from tqdm import tqdm

class VCTKDataset(Dataset):
    """VCTK dataset for voice conversion training."""
    
    def __init__(self, data_root: str, target_sr: int = 16000, subset: Optional[int] = None, max_duration_s: int = 20):
        self.data_root = Path(data_root)
        self.target_sr = target_sr
        
        # Find the directory containing speaker folders (e.g., 'p225', 'p226')
        wav_parent_dir = self.data_root
        if (self.data_root / "wav48_silence_trimmed").exists():
             wav_parent_dir = self.data_root / "wav48_silence_trimmed"
        elif (self.data_root / "wav48").exists():
             wav_parent_dir = self.data_root / "wav48"
        
        # Find all wav files and ensure they have a matching transcript
        all_wav_files = []
        txt_root = self.data_root / "txt"
        self.phone_id_cache = {}

        for speaker_dir in wav_parent_dir.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                for wav_path in speaker_dir.glob("*_mic1.flac"):
                    # Check for corresponding text file
                    utt_id = wav_path.stem.replace("_mic1", "")
                    txt_path = txt_root / speaker_id / f"{utt_id}.txt"
                    if txt_path.exists():
                        all_wav_files.append((str(wav_path), speaker_id, txt_path))
        
        # Filter out files that are too long to prevent OOM errors
        wav_files = []
        print("Scanning dataset to filter out long audio files...")
        for wav_path, speaker_id, txt_path in tqdm(all_wav_files, desc="Filtering audio"):
            try:
                info = sf.info(wav_path)
                duration = info.frames / info.samplerate
                if duration <= max_duration_s:
                    wav_files.append((wav_path, speaker_id, txt_path))
            except Exception:
                # If we can't read the file info, it's safer to skip it
                continue
        
        original_count = len(all_wav_files)
        if original_count > 0:
            removed_count = original_count - len(wav_files)
            print(f"Filtered out {removed_count} files (> {max_duration_s}s). Kept {len(wav_files)} files.")

        if not wav_files:
            raise ValueError(f"No valid audio-text pairs found in subdirectories of {wav_parent_dir}")
            
        # Optional subset for faster training (before phoneme processing)
        if subset and subset < len(wav_files):
            wav_files = random.sample(wav_files, subset)

        # Precompute phoneme IDs only for the selected wav_files
        from utils.phonemes import text_to_phones, phones_to_ids, get_num_phones
        for wav_path, speaker_id, txt_path in tqdm(wav_files, desc="Phonemizing"):
            try:
                with open(txt_path, "r") as f_txt:
                    transcript = f_txt.read().strip()
                phone_seq = text_to_phones(transcript)
                phone_ids = phones_to_ids(phone_seq)
                if len(phone_ids) == 0:
                    phone_ids = [get_num_phones() - 1]
            except Exception:
                phone_ids = [get_num_phones() - 1]
            self.phone_id_cache[wav_path] = phone_ids
        
        # Remove txt_path from tuples for __getitem__
        self.wav_files = [(wav, spk) for wav, spk, _ in wav_files]
        
        # Create speaker mapping
        speakers = sorted({spk for _, spk in self.wav_files})
        self.speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
        self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}
        self.num_speakers = len(speakers)
        
        print(f"Loaded {len(self.wav_files)} files from {len(speakers)} speakers")
        
    def __len__(self):
        return len(self.wav_files)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str, list]:
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
        
        phone_ids = self.phone_id_cache[wav_path]
        return audio, spk_id, wav_path, phone_ids
        
    def get_num_speakers(self):
        return self.num_speakers
        
    def get_speaker_files(self, speaker_id: str) -> List[str]:
        """Get all files for a specific speaker."""
        return [wav_path for wav_path, spk in self.wav_files if spk == speaker_id]

def collate_fn(batch):
    """Custom collate function to handle variable length audio."""
    audios, spk_ids, wav_paths, phone_id_lists = zip(*batch)
    
    # Currently batch_size == 1; simply unwrap first elements
    return audios[0], torch.tensor(spk_ids[0]), wav_paths[0], phone_id_lists[0]

def create_dataloader(data_root: str, 
                     target_sr: int = 16000,
                     subset: Optional[int] = None,
                     batch_size: int = 1,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     max_duration_s: int = 20):
    """Create a VCTK dataloader."""
    if batch_size != 1:
        raise ValueError("This implementation currently supports batch_size=1 only. Update collate_fn for larger batches.")
    
    dataset = VCTKDataset(data_root, target_sr, subset, max_duration_s)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset 