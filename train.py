#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import random

# --- Environment Setup ---
# This must be done BEFORE importing torch, numpy, etc. to mitigate threading issues.
from utils.environment import setup_environment
setup_environment()
# -------------------------

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from resemblyzer import VoiceEncoder
import torchaudio.transforms as T
import wandb

from config import Config
from data.vctk import create_dataloader, VCTKArgs, get_vctk_files
from models.rcop import RCOP
from utils.logging import setup_logger, log_config, log_model_summary
from utils.phonemes import get_num_phones, text_to_phones, phones_to_ids
from utils.environment import set_seed
from utils.features import extract_features
from utils.checkpoint import save_checkpoint, load_model_from_checkpoint

MODEL_CACHE_DIR = "./models"

def load_pretrained_models(device):
    """Load pre-trained models that do not depend on data stats."""
    
    # Load WavLM-Large (frozen)
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model.eval()
    wavlm_model.requires_grad_(False)
    wavlm_model.to(device)
    
    # Load Resemblyzer (frozen)
    voice_encoder = VoiceEncoder()
    voice_encoder.eval()
    
    return wavlm_processor, wavlm_model, voice_encoder

def train_epoch(model, dataloader, optimizer, criterion_ph, criterion_sp, criterion_l1, device, epoch, total_epochs, logger, config, wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram):
    """Train for one epoch."""
    model.train()
    total_loss, total_ph_loss, total_sp_loss, total_recon_loss = 0, 0, 0, 0
    num_samples = 0
    
    lambd = (epoch + 1) / total_epochs if total_epochs > 0 else 1.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (audios_padded, spk_ids, wav_paths, phone_ids_padded, attention_mask) in enumerate(pbar):
        optimizer.zero_grad()
        
        batch_loss = torch.tensor(0.0, device=device)
        processed_in_batch = 0
        
        # Process each item in the batch
        for i in range(audios_padded.size(0)):
            audio = audios_padded[i]
            spk_id = spk_ids[i]
            phone_ids_list = phone_ids_padded[i].tolist()
            mask = attention_mask[i]
            
            # Remove padding before feature extraction
            unpadded_audio = audio[mask == 1]
            if unpadded_audio.nelement() == 0:
                continue
        
            try:
                # Extract features
                ssl_features, spk_embed = extract_features(
                    unpadded_audio, wavlm_processor, wavlm_model, voice_encoder, device
                )
                
                # Forward pass
                ph_logits, sp_logits, pred_mels = model(ssl_features, spk_embed, lambd)
            
                # --- Phoneme targets for CTC Loss ---
                # The padding value from collate_fn is 0, which corresponds to our BLANK token.
                # text_to_phones does not produce BLANK, so we can safely filter padding by checking for 0.
                phone_ids = [p for p in phone_ids_list if p != 0]
                if not phone_ids:
                    logger.warning(f"Skipping sample {i} in batch {batch_idx} due to empty phone sequence.")
                    continue
                
                ph_targets = torch.tensor(phone_ids, dtype=torch.long)
                ph_target_lengths = torch.tensor([len(phone_ids)], dtype=torch.long)

                # --- Input for CTC Loss ---
                # Model output needs to be log probabilities.
                ph_log_probs = nn.functional.log_softmax(ph_logits, dim=-1)
                
                # CTC Loss expects input of shape (T, N, C), where T=time, N=batch, C=classes.
                # In this loop, our batch size N is 1.
                T = ph_log_probs.size(0)
                ph_log_probs_for_ctc = ph_log_probs.unsqueeze(1)
                input_lengths = torch.tensor([T], dtype=torch.long)

                # --- Speaker targets ---
                sp_targets = spk_id.to(device)
                
                # --- Reconstruction Target ---
                gt_mels = mel_spectrogram(unpadded_audio.to(device))
                
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
                ph_loss = criterion_ph(ph_log_probs_for_ctc, ph_targets, input_lengths, ph_target_lengths)
                sp_loss = criterion_sp(sp_logits, sp_targets.unsqueeze(0))
                recon_loss = criterion_l1(pred_mels_aligned, gt_mels_aligned)
                
                # Total loss (with weights from config)
                item_loss = (
                    config.lambda_ph * ph_loss + 
                    config.lambda_sp * sp_loss + 
                    config.lambda_recon * recon_loss
                )
                
                batch_loss += item_loss
                processed_in_batch += 1
            
                # Update running losses for logging
                total_loss += item_loss.item()
                total_ph_loss += ph_loss.item()
                total_sp_loss += sp_loss.item()
                total_recon_loss += recon_loss.item()
                
            except Exception as e:
                logger.warning(f"Skipping sample {i} in batch {batch_idx} due to error: {e}")
                continue

        num_samples += processed_in_batch

        # Backward pass on the average batch loss
        if processed_in_batch > 0:
            avg_batch_loss = batch_loss / processed_in_batch
            avg_batch_loss.backward()
            optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{avg_batch_loss.item():.4f}',
                'Ph': f'{ph_loss.item():.4f}',
                'Sp': f'{sp_loss.item():.4f}',
                'Recon': f'{recon_loss.item():.4f}',
                'λ': f'{lambd:.3f}'
            })
            
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_ph_loss = total_ph_loss / num_samples if num_samples > 0 else 0
    avg_sp_loss = total_sp_loss / num_samples if num_samples > 0 else 0
    avg_recon_loss = total_recon_loss / num_samples if num_samples > 0 else 0
    
    logger.info(f"Epoch {epoch+1} TRAIN: Loss={avg_loss:.4f}, Ph={avg_ph_loss:.4f}, Sp={avg_sp_loss:.4f}, Recon={avg_recon_loss:.4f}, λ={lambd:.3f}")
    
    return avg_loss, avg_ph_loss, avg_sp_loss, avg_recon_loss, lambd

def validate_epoch(model, dataloader, criterion_ph, criterion_l1, device, logger, config, wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram):
    """Validate for one epoch."""
    model.eval()
    total_val_loss, total_val_ph_loss, total_val_recon_loss = 0, 0, 0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch_idx, (audios_padded, spk_ids, wav_paths, phone_ids_padded, attention_mask) in enumerate(pbar):
            # Process each item in the batch
            for i in range(audios_padded.size(0)):
                audio = audios_padded[i]
                phone_ids_list = phone_ids_padded[i].tolist()
                mask = attention_mask[i]
                
                unpadded_audio = audio[mask == 1]
                if unpadded_audio.nelement() == 0:
                    continue

                try:
                    ssl_features, spk_embed = extract_features(
                        unpadded_audio, wavlm_processor, wavlm_model, voice_encoder, device
                    )
                    
                    # Forward pass for validation (no GRL needed)
                    ph_logits, _, pred_mels = model(ssl_features, spk_embed, lambd=0.0)
                    
                    phone_ids = [p for p in phone_ids_list if p != 0]
                    if not phone_ids:
                        continue

                    ph_targets = torch.tensor(phone_ids, dtype=torch.long)
                    ph_target_lengths = torch.tensor([len(phone_ids)], dtype=torch.long)

                    ph_log_probs = nn.functional.log_softmax(ph_logits, dim=-1)
                    T = ph_log_probs.size(0)
                    ph_log_probs_for_ctc = ph_log_probs.unsqueeze(1)
                    input_lengths = torch.tensor([T], dtype=torch.long)

                    gt_mels = mel_spectrogram(unpadded_audio.to(device))
                    T_gt = gt_mels.size(-1)
                    T_pred = pred_mels.size(0)
                    
                    if T_pred > T_gt:
                        pred_mels_aligned = pred_mels[:T_gt, :]
                    else:
                        pred_mels_aligned = torch.nn.functional.pad(pred_mels, (0, 0, 0, T_gt - T_pred))
                    
                    gt_mels_aligned = gt_mels.squeeze(0).transpose(0, 1)
                
                    ph_loss = criterion_ph(ph_log_probs_for_ctc, ph_targets, input_lengths, ph_target_lengths)
                    recon_loss = criterion_l1(pred_mels_aligned, gt_mels_aligned)
                    
                    # Val loss excludes speaker classification, as speakers may be unseen
                    item_loss = config.lambda_ph * ph_loss + config.lambda_recon * recon_loss
                    
                    total_val_loss += item_loss.item()
                    total_val_ph_loss += ph_loss.item()
                    total_val_recon_loss += recon_loss.item()
                    num_samples += 1

                except Exception as e:
                    logger.warning(f"Skipping val sample {i} in batch {batch_idx} due to error: {e}")
                    continue
    
    avg_val_loss = total_val_loss / num_samples if num_samples > 0 else 0
    avg_ph_loss = total_val_ph_loss / num_samples if num_samples > 0 else 0
    avg_recon_loss = total_val_recon_loss / num_samples if num_samples > 0 else 0
    
    logger.info(f"Epoch VAL: Loss={avg_val_loss:.4f}, Ph={avg_ph_loss:.4f}, Recon={avg_recon_loss:.4f}")
    
    return avg_val_loss, avg_ph_loss, avg_recon_loss

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
    parser.add_argument("--save_interval", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Ratio of speakers for validation set.")
    parser.add_argument("--val_subset", type=int, default=100, help="Subset size for validation set, for faster validation.")
    parser.add_argument("--lambda_ph", type=float, default=None, help="Weight for phoneme loss")
    parser.add_argument("--lambda_sp", type=float, default=None, help="Weight for speaker loss")
    parser.add_argument("--lambda_recon", type=float, default=None, help="Weight for reconstruction loss")
    parser.add_argument("--wandb_project", type=str, default="rc-op-project", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="A specific name for the W&B run")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    
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
    
    # Allow overriding loss weights from CLI
    if args.lambda_ph is not None: config.lambda_ph = args.lambda_ph
    if args.lambda_sp is not None: config.lambda_sp = args.lambda_sp
    if args.lambda_recon is not None: config.lambda_recon = args.lambda_recon
    
    log_config(logger, config)
    
    # --- W&B Setup ---
    if not args.no_wandb:
        try:
            wandb.init(
                name=args.wandb_run_name,
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={**config.__dict__, **vars(args)}
            )
            if wandb.run:
                config.run_name = wandb.run.name
                logger.info(f"Weights & Biases integration enabled. Run name: {wandb.run.name}")
        except ImportError:
            logger.warning("wandb package not found, disabling W&B. Run `pip install wandb`.")
            args.no_wandb = True

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

    # --- Dataset Loading and Splitting ---
    logger.info("Scanning dataset once to get all files...")
    all_files = get_vctk_files(Path(args.data_root), args.max_duration)
    
    all_speakers = sorted(list(set(f[1] for f in all_files)))
    random.shuffle(all_speakers)
    
    val_split_idx = int(len(all_speakers) * args.val_split_ratio)
    val_speakers = all_speakers[:val_split_idx]
    train_speakers = all_speakers[val_split_idx:]
    
    # Filter the file list based on the speaker split
    train_files = [f for f in all_files if f[1] in train_speakers]
    val_files = [f for f in all_files if f[1] in val_speakers]
    
    speaker_to_id = {spk: i for i, spk in enumerate(all_speakers)}
    
    logger.info(f"Total speakers: {len(all_speakers)}")
    logger.info(f"Training speakers: {len(train_speakers)} ({len(train_files)} files)")
    logger.info(f"Validation speakers: {len(val_speakers)} ({len(val_files)} files)")

    # --- Create Dataloaders ---
    logger.info("Creating dataloaders...")
    train_args = VCTKArgs(
        target_sr=config.target_sr,
        subset=config.subset,
        max_duration_s=args.max_duration,
        speaker_to_id=speaker_to_id,
        file_list=train_files
    )
    train_dataloader, train_dataset = create_dataloader(
        args=train_args,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_args = VCTKArgs(
        target_sr=config.target_sr,
        subset=args.val_subset,
        max_duration_s=args.max_duration,
        speaker_to_id=speaker_to_id,
        file_list=val_files
    )
    val_dataloader, val_dataset = create_dataloader(
        args=val_args,
        batch_size=config.batch_size,
        shuffle=False
    )

    num_total_speakers = len(all_speakers)
    num_phones = get_num_phones()
    
    logger.info(f"Dataset: {len(train_dataset)} train samples, {len(val_dataset)} val samples, {num_total_speakers} total speakers")
    
    # --- Initialize RCOP model THIRD, now that we have num_speakers ---
    logger.info("Initializing RCOP model...")
    rcop_model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_total_speakers
    )
    rcop_model.to(device)
    
    if not args.no_wandb:
        wandb.watch(rcop_model, log="all", log_freq=250)

    log_model_summary(logger, rcop_model)
    
    # Setup optimizer and losses
    optimizer = optim.Adam(rcop_model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    criterion_ph = nn.CTCLoss(blank=0, zero_infinity=True)  # BLANK token is at index 0
    criterion_sp = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        rcop_model, _ = load_model_from_checkpoint(args.resume, 0, 0, device)
        rcop_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Resumed scheduler state.")
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        avg_loss, avg_ph_loss, avg_sp_loss, avg_recon_loss, lambd = train_epoch(
            rcop_model, train_dataloader, optimizer, criterion_ph, criterion_sp, criterion_l1,
            device, epoch, config.epochs, logger, config,
            wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram
        )
        
        avg_val_loss, avg_val_ph_loss, avg_val_recon_loss = validate_epoch(
            rcop_model, val_dataloader, criterion_ph, criterion_l1, device, logger, config,
            wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram
        )

        # Log metrics to W&B
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_ph_loss": avg_ph_loss,
                "train_sp_loss": avg_sp_loss,
                "train_recon_loss": avg_recon_loss,
                "val_loss": avg_val_loss,
                "val_ph_loss": avg_val_ph_loss,
                "val_recon_loss": avg_val_recon_loss,
                "lambda": lambd,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Step the LR scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Save checkpoint periodically and at the end
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == config.epochs:
            checkpoint_path = os.path.join(args.save_dir, f"rcop_epoch{epoch+1}.pt")
            save_checkpoint(rcop_model, optimizer, scheduler, epoch, avg_loss, checkpoint_path, num_total_speakers, num_phones, best_val_loss)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.save_dir, "rcop_best.pt")
            save_checkpoint(rcop_model, optimizer, scheduler, epoch, avg_loss, best_checkpoint_path, num_total_speakers, num_phones, best_val_loss)
            logger.info(f"Saved new best model to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    logger.info("Training completed!")

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 