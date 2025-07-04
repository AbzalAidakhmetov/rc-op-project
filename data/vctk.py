import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
from resampy import resample
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass

@dataclass
class VCTKArgs:
    data_root: Optional[str] = None # Can be None if file_list is provided
    target_sr: int = 16000
    subset: Optional[int] = None
    max_duration_s: int = 20
    speaker_list: Optional[List[str]] = None
    speaker_to_id: Optional[Dict[str, int]] = None
    file_list: Optional[List[Tuple[str, str, Path]]] = None

def get_vctk_files(data_root: Path, max_duration_s: int) -> List[Tuple[str, str, Path]]:
    """Scans the VCTK directory to find all valid .flac files and their transcripts,
    filtering out audio clips that are too long."""
    
    wav_parent_dir = data_root
    if (data_root / "wav48_silence_trimmed").exists():
        wav_parent_dir = data_root / "wav48_silence_trimmed"
    elif (data_root / "wav48").exists():
        wav_parent_dir = data_root / "wav48"
    
    all_wav_files = []
    txt_root = data_root / "txt"

    for speaker_dir in wav_parent_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            for wav_path in speaker_dir.glob("*_mic1.flac"):
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
            continue
    
    original_count = len(all_wav_files)
    if original_count > 0:
        removed_count = original_count - len(wav_files)
        print(f"Filtered out {removed_count} files (> {max_duration_s}s). Kept {len(wav_files)} files.")
    
    return wav_files

class VCTKDataset(Dataset):
    """VCTK dataset for voice conversion training."""
    
    def __init__(self, args: VCTKArgs):
        self.args = args
        
        if args.file_list is not None:
            # Use the provided file list, no need to scan filesystem
            files_to_process = args.file_list
        else:
            # Fallback to original behavior: scan filesystem
            if not args.data_root:
                raise ValueError("`data_root` must be provided if `file_list` is not.")
            self.data_root = Path(args.data_root)
            files_to_process = get_vctk_files(self.data_root, args.max_duration_s)

        # Filter by speaker_list if provided
        if args.speaker_list:
            files_for_speakers = [f for f in files_to_process if f[1] in args.speaker_list]
        else:
            files_for_speakers = files_to_process
            
        if not files_for_speakers:
            if args.speaker_list:
                raise ValueError(f"No valid audio-text pairs found for the given speakers in the provided file list.")
            else:
                raise ValueError(f"No valid audio-text pairs found in {self.data_root}")
            
        # Optional subset for faster training
        if args.subset and args.subset < len(files_for_speakers):
            wav_files_subset = random.sample(files_for_speakers, args.subset)
        else:
            wav_files_subset = files_for_speakers
            
        # Precompute phoneme IDs
        from utils.phonemes import text_to_phones, phones_to_ids, get_num_phones
        self.phone_id_cache = {}
        for wav_path, speaker_id, txt_path in tqdm(wav_files_subset, desc="Phonemizing"):
            try:
                with open(txt_path, "r") as f_txt:
                    transcript = f_txt.read().strip()
                phone_seq = text_to_phones(transcript)
                phone_ids = phones_to_ids(phone_seq)
                if len(phone_ids) == 0:
                    phone_ids = [get_num_phones() - 1]
            except Exception:
                phone_ids = [get_num_phones() - 1] # Fallback for any error
            self.phone_id_cache[str(wav_path)] = phone_ids
        
        self.wav_files = [(wav, spk) for wav, spk, _ in wav_files_subset]
        
        # Create speaker mapping
        if args.speaker_to_id:
            self.speaker_to_id = args.speaker_to_id
            speakers_in_use = {spk for _, spk in self.wav_files}
            self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items() if spk in speakers_in_use}
        else:
            speakers = sorted({spk for _, spk in self.wav_files})
            self.speaker_to_id = {spk: i for i, spk in enumerate(speakers)}
            self.id_to_speaker = {i: spk for spk, i in self.speaker_to_id.items()}

        if not self.speaker_to_id:
             self.num_speakers = 0
        else:
             self.num_speakers = len(self.speaker_to_id)
        
        print(f"Loaded {len(self.wav_files)} files from {len(set(self.id_to_speaker.values()))} speakers")
        
    def __len__(self):
        return len(self.wav_files)
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, str, list]:
        wav_path, speaker_id = self.wav_files[idx]
        
        # Load audio
        try:
            audio, orig_sr = sf.read(wav_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
        except:
            audio, orig_sr = torchaudio.load(wav_path)
            audio = audio[0].numpy()
            
        # Resample if needed
        if orig_sr != self.args.target_sr:
            audio = resample(audio, orig_sr, self.args.target_sr)
            
        audio = torch.from_numpy(audio).float()
        
        # Get speaker ID from the provided map
        spk_id = self.speaker_to_id[speaker_id]
        
        phone_ids = self.phone_id_cache[wav_path]
        return audio, spk_id, wav_path, phone_ids
        
    def get_num_speakers(self):
        return self.num_speakers
        
    def get_speaker_files(self, speaker_id: str) -> List[str]:
        return [wav_path for wav_path, spk in self.wav_files if spk == speaker_id]

def collate_fn(batch):
    """Custom collate function to handle variable length audio."""
    audios, spk_ids, wav_paths, phone_id_lists = zip(*batch)
    
    # Pad audio and phoneme sequences
    audios_padded = pad_sequence(list(audios), batch_first=True, padding_value=0)
    phone_ids_padded = pad_sequence([torch.tensor(p) for p in phone_id_lists], batch_first=True, padding_value=0)
    
    # Create attention mask for the audio
    attention_mask = (audios_padded != 0).long()
    
    return audios_padded, torch.tensor(spk_ids), wav_paths, phone_ids_padded, attention_mask

def create_dataloader(args: VCTKArgs,
                     batch_size: int,
                     shuffle: bool,
                     num_workers: int = 0):
    """Create a VCTK dataloader."""
    dataset = VCTKDataset(args)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader, dataset 