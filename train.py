#!/usr/bin/env python3

import argparse
import os

# Mitigate threading issues with BLAS libraries (for resource-limited environments)
# This must be done BEFORE importing torch, numpy, etc.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from resemblyzer import VoiceEncoder

from config import Config
from data.vctk import create_dataloader
from models.rcop import RCOP
from utils.logging import setup_logger, log_config, log_model_summary
from utils.phonemes import get_num_phones, text_to_phones, phones_to_ids

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_models(config, num_speakers, num_phones, device):
    """Load pre-trained models and initialize RCOP."""
    
    # Load WavLM-Large (frozen)
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm_model.eval()
    wavlm_model.requires_grad_(False)
    wavlm_model.to(device)
    
    # Load Resemblyzer (frozen)
    voice_encoder = VoiceEncoder()
    voice_encoder.eval()
    
    # Initialize RCOP model
    rcop_model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_speakers
    )
    rcop_model.to(device)
    
    return wavlm_processor, wavlm_model, voice_encoder, rcop_model

def extract_features(audio, wavlm_processor, wavlm_model, voice_encoder, device):
    """Extract SSL features and speaker embeddings."""
    
    # Ensure audio is the right shape and on CPU for processing
    if audio.dim() == 1:
        audio_np = audio.numpy()
    else:
        audio_np = audio.squeeze().numpy()
    
    # Extract SSL features with WavLM
    inputs = wavlm_processor(audio_np, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        ssl_outputs = wavlm_model(**inputs)
        ssl_features = ssl_outputs.last_hidden_state.squeeze(0)  # (T, d_ssl)
    
    # Extract speaker embedding with Resemblyzer
    with torch.no_grad():
        # Resemblyzer expects numpy array
        spk_embed = voice_encoder.embed_utterance(audio_np)
        spk_embed = torch.from_numpy(spk_embed).float().to(device)  # (d_spk,)
    
    return ssl_features, spk_embed

def train_epoch(model, dataloader, optimizer, criterion_ce, device, epoch, total_epochs, logger, phoneme_cache, wavlm_processor, wavlm_model, voice_encoder):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ph_loss = 0
    total_sp_loss = 0
    
    # Lambda schedule
    lambd = (epoch + 1) / total_epochs if total_epochs > 0 else 1.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (audio, spk_id, wav_path) in enumerate(pbar):
        optimizer.zero_grad()
        
        try:
            # Extract features
            ssl_features, spk_embed = extract_features(
                audio, wavlm_processor, 
                wavlm_model, 
                voice_encoder, 
                device
            )
            
            # Forward pass
            ph_logits, sp_logits = model(ssl_features, spk_embed, lambd)
            
            # ----------------------------------------------------------------
            # Derive phoneme targets from the VCTK transcription that matches
            # the current wav.  VCTK keeps per-utterance text files in
            #   <root>/txt/<speaker>/<utt_id>.txt
            # Example wav:  wav48_silence_trimmed/p225/p225_001_mic1.flac
            # Corresponding txt:            txt/p225/p225_001.txt
            # We map the wav path to its txt path, load the transcript, convert
            # to IPA phones and finally to IDs.  The phone sequence is then
            # stretched/repeated to the frame count T so that each SSL frame
            # receives a target label.  If anything goes wrong we gracefully
            # fall back to a "UNK" label so training can continue.
            # ----------------------------------------------------------------
            try:
                if wav_path in phoneme_cache:
                    phone_ids = phoneme_cache[wav_path]
                else:
                    # 1. Build expected .txt path robustly
                    wav_p = Path(wav_path)
                    data_root = dataloader.dataset.data_root

                    speaker_id = wav_p.parent.name
                    utt_id = wav_p.stem.replace("_mic1", "")
                    
                    txt_path = data_root / "txt" / speaker_id / f"{utt_id}.txt"

                    # 2. Read transcript
                    with open(txt_path, "r") as f_txt:
                        transcript = f_txt.read().strip()

                    # 3. Text ➔ phones ➔ IDs
                    phone_seq   = text_to_phones(transcript)
                    phone_ids   = phones_to_ids(phone_seq)
                    
                    # Cache the result
                    phoneme_cache[wav_path] = phone_ids

                if len(phone_ids) == 0:
                    raise ValueError("Empty phone sequence from transcript")

                # 4. Expand / repeat to match number of frames T
                import math
                T          = ph_logits.size(0)
                repeats    = math.ceil(T / len(phone_ids))
                expanded   = (phone_ids * repeats)[:T]
                ph_targets = torch.tensor(expanded, dtype=torch.long, device=device)

            except FileNotFoundError:
                # This is a recoverable error for a single missing transcript.
                logger.warning(f"Transcript not found for {wav_path}; using UNK label")
                unk_id     = get_num_phones() - 1
                ph_targets = torch.full((ph_logits.size(0),), unk_id, dtype=torch.long, device=device)

            except Exception as e:
                # Any other exception here is likely a critical setup error (e.g., espeak missing).
                # This is not recoverable. Log it and re-raise to stop the training run.
                logger.error(f"A critical error occurred during phoneme generation for {wav_path}: {e}")
                logger.error("This may indicate a problem with your environment, such as the 'espeak-ng' library not being installed or accessible. Please run setup.sh or install it manually.")
                raise e
            
            sp_targets = spk_id.to(device)
            
            # Compute losses
            ph_loss = criterion_ce(ph_logits, ph_targets)
            sp_loss = criterion_ce(sp_logits, sp_targets.unsqueeze(0))
            
            # Total loss
            total_batch_loss = ph_loss + sp_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            # Update running losses
            total_loss += total_batch_loss.item()
            total_ph_loss += ph_loss.item()
            total_sp_loss += sp_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Ph': f'{ph_loss.item():.4f}',
                'Sp': f'{sp_loss.item():.4f}',
                'λ': f'{lambd:.3f}'
            })
            
        except Exception as e:
            logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader)
    avg_ph_loss = total_ph_loss / len(dataloader)
    avg_sp_loss = total_sp_loss / len(dataloader)
    
    logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Ph={avg_ph_loss:.4f}, Sp={avg_sp_loss:.4f}, λ={lambd:.3f}")
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, save_path, num_speakers=None, num_phones=None):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # Add metadata for proper model loading
    if num_speakers is not None:
        checkpoint['num_speakers'] = num_speakers
    if num_phones is not None:
        checkpoint['num_phones'] = num_phones
        
    torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser(description="Train RC-OP model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to VCTK dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--subset", type=int, default=500, help="Subset size for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("train", args.log_dir)
    
    # Create config
    config = Config()
    config.device = args.device
    config.epochs = args.epochs
    config.subset = args.subset
    config.batch_size = args.batch_size
    config.lr = args.lr
    
    log_config(logger, config)
    
    # Set seed
    set_seed(config.seed)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataloader
    logger.info("Loading dataset...")
    dataloader, dataset = create_dataloader(
        data_root=args.data_root,
        target_sr=config.target_sr,
        subset=config.subset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    num_speakers = dataset.get_num_speakers()
    num_phones = get_num_phones()
    
    logger.info(f"Dataset: {len(dataset)} samples, {num_speakers} speakers, {num_phones} phonemes")
    
    # Load models
    logger.info("Loading models...")
    wavlm_processor, wavlm_model, voice_encoder, rcop_model = load_models(
        config, num_speakers, num_phones, device
    )
    
    log_model_summary(logger, rcop_model)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(rcop_model.parameters(), lr=config.lr)
    criterion_ce = nn.CrossEntropyLoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        rcop_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    phoneme_cache = {}  # Cache for phoneme conversions
    for epoch in range(start_epoch, config.epochs):
        avg_loss = train_epoch(
            rcop_model, dataloader, optimizer, criterion_ce, 
            device, epoch, config.epochs, logger, phoneme_cache,
            wavlm_processor, wavlm_model, voice_encoder
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"rcop_epoch{epoch+1}.pt")
        save_checkpoint(rcop_model, optimizer, epoch, avg_loss, checkpoint_path, num_speakers, num_phones)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 