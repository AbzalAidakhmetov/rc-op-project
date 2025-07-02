#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

# --- Environment Setup ---
# This must be done BEFORE importing torch, numpy, etc. to mitigate threading issues.
from utils.environment import setup_environment
setup_environment()
# -------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from resemblyzer import VoiceEncoder
import torchaudio.transforms as T

from config import Config
from data.vctk import create_dataloader
from models.rcop import RCOP
from utils.logging import setup_logger, log_config, log_model_summary
from utils.phonemes import get_num_phones, text_to_phones, phones_to_ids
from utils.environment import set_seed
from utils.features import extract_features
from utils.checkpoint import save_checkpoint

def load_pretrained_models(device):
    """Load pre-trained models that do not depend on data stats."""
    
    # Load WavLM-Large (frozen)
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm_model.eval()
    wavlm_model.requires_grad_(False)
    wavlm_model.to(device)
    
    # Load Resemblyzer (frozen)
    voice_encoder = VoiceEncoder()
    voice_encoder.eval()
    
    return wavlm_processor, wavlm_model, voice_encoder

def train_epoch(model, dataloader, optimizer, criterion_ce, criterion_l1, device, epoch, total_epochs, logger, wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_ph_loss = 0
    total_sp_loss = 0
    total_recon_loss = 0
    
    # Lambda schedule
    lambd = (epoch + 1) / total_epochs if total_epochs > 0 else 1.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (audio, spk_id, wav_path, phone_ids) in enumerate(pbar):
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
            ph_logits, sp_logits, pred_mels = model(ssl_features, spk_embed, lambd)
            
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
                if len(phone_ids) == 0:
                    raise ValueError("Empty phone sequence from transcript")

                # 4. Expand / repeat to match number of frames T
                import math
                T          = ph_logits.size(0)
                repeats    = math.ceil(T / len(phone_ids))
                expanded   = (phone_ids * repeats)[:T]
                ph_targets = torch.tensor(expanded, dtype=torch.long, device=device)

            except Exception as e:
                logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
                continue
            
            sp_targets = spk_id.to(device)
            
            # --- Reconstruction Target ---
            gt_mels = mel_spectrogram(audio.to(device))
            
            # Match dimensions (pad or trim predicted mels)
            T_gt = gt_mels.size(-1)
            T_pred = pred_mels.size(0)
            
            # WavLM features may have slightly different T, so we align
            if T_pred > T_gt:
                pred_mels_aligned = pred_mels[:T_gt, :]
            else:
                pred_mels_aligned = torch.nn.functional.pad(pred_mels, (0, 0, 0, T_gt - T_pred))
            
            # The mel spectrogram transform returns (n_channels, n_mels, time)
            # We need (time, n_mels) to match the model output
            gt_mels_aligned = gt_mels.squeeze(0).transpose(0, 1)

            # Compute losses
            ph_loss = criterion_ce(ph_logits, ph_targets)
            sp_loss = criterion_ce(sp_logits, sp_targets.unsqueeze(0))
            recon_loss = criterion_l1(pred_mels_aligned, gt_mels_aligned)
            
            # Total loss
            total_batch_loss = ph_loss + sp_loss + recon_loss
            
            # Backward pass
            total_batch_loss.backward()
            optimizer.step()
            
            # Update running losses
            total_loss += total_batch_loss.item()
            total_ph_loss += ph_loss.item()
            total_sp_loss += sp_loss.item()
            total_recon_loss += recon_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Ph': f'{ph_loss.item():.4f}',
                'Sp': f'{sp_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'λ': f'{lambd:.3f}'
            })
            
        except Exception as e:
            logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
            continue
    
    avg_loss = total_loss / len(dataloader)
    avg_ph_loss = total_ph_loss / len(dataloader)
    avg_sp_loss = total_sp_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    
    logger.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Ph={avg_ph_loss:.4f}, Sp={avg_sp_loss:.4f}, Recon={avg_recon_loss:.4f}, λ={lambd:.3f}")
    
    return avg_loss

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
    parser.add_argument("--max_duration", type=int, default=15, help="Maximum audio duration in seconds to filter dataset.")
    
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
    
    # --- Load models FIRST to avoid thread exhaustion ---
    logger.info("Loading pre-trained models (WavLM, Resemblyzer)...")
    wavlm_processor, wavlm_model, voice_encoder = load_pretrained_models(device)
    
    # --- Mel Spectrogram transform ---
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.target_sr,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_mels=80,
        f_min=0,
        f_max=8000
    ).to(device)

    # --- Create dataloader SECOND ---
    # This will now run the phonemizing step after heavy models are loaded
    logger.info("Loading dataset...")
    dataloader, dataset = create_dataloader(
        data_root=args.data_root,
        target_sr=config.target_sr,
        subset=config.subset,
        batch_size=config.batch_size,
        shuffle=True,
        max_duration_s=args.max_duration
    )
    
    num_speakers = dataset.get_num_speakers()
    num_phones = get_num_phones()
    
    logger.info(f"Dataset: {len(dataset)} samples, {num_speakers} speakers, {num_phones} phonemes")
    
    # --- Initialize RCOP model THIRD, now that we have num_speakers ---
    logger.info("Initializing RCOP model...")
    rcop_model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_speakers
    )
    rcop_model.to(device)

    log_model_summary(logger, rcop_model)
    
    # Setup optimizer and losses
    optimizer = optim.Adam(rcop_model.parameters(), lr=config.lr)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss()
    
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
    for epoch in range(start_epoch, config.epochs):
        avg_loss = train_epoch(
            rcop_model, dataloader, optimizer, criterion_ce, criterion_l1,
            device, epoch, config.epochs, logger,
            wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram
        )
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"rcop_epoch{epoch+1}.pt")
        save_checkpoint(rcop_model, optimizer, epoch, avg_loss, checkpoint_path, num_speakers, num_phones)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 