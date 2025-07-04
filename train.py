#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import random
import traceback

# --- Environment Setup ---
# This must be done BEFORE importing torch, numpy, etc. to mitigate threading issues.
from utils.environment import setup_environment
setup_environment()
# -------------------------

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from transformers.modeling_outputs import BaseModelOutput
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
    """Train for one epoch with batch processing."""
    model.train()
    running_loss, running_ph_loss, running_sp_loss, running_recon_loss = 0, 0, 0, 0
    num_samples = 0
    
    lambd = (epoch + 1) / total_epochs if total_epochs > 0 else 1.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, (audios_padded, spk_ids, wav_paths, phone_ids_padded, attention_mask) in enumerate(pbar):
        optimizer.zero_grad()
        
        try:
            # --- Batched Feature Extraction ---
            # 1. WavLM SSL features (batched)
            inputs = wavlm_processor(
                audios_padded.cpu().numpy(),
                sampling_rate=config.target_sr,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            inputs['attention_mask'] = inputs['attention_mask'].bool()
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                ssl_outputs: BaseModelOutput = wavlm_model(**inputs)
                ssl_features = ssl_outputs.last_hidden_state  # (N, T_ssl, D)
                # The attention mask from the processor is for the raw audio, so we calculate
                # the downsampled feature lengths for CTC loss below.

            # 2. Resemblyzer speaker embeddings (looped)
            spk_embeds = []
            for i in range(audios_padded.size(0)):
                unpadded_audio_np = audios_padded[i, :attention_mask[i].sum()].cpu().numpy()
                if unpadded_audio_np.size == 0: continue
                spk_embed = voice_encoder.embed_utterance(unpadded_audio_np)
                spk_embeds.append(spk_embed)
            if not spk_embeds: continue # Skip batch if all audios are empty
            spk_embeds = torch.from_numpy(np.array(spk_embeds)).float().to(device) # (N, d_spk)
            # Ensure we extracted an embedding for every sample in the batch. If not, skip to avoid
            # tensor dimension mismatches later on.
            if spk_embeds.size(0) != audios_padded.size(0):
                continue

            # --- Forward pass ---
            ph_logits, sp_logits, pred_mels = model(ssl_features, spk_embeds, lambd)

            # --- Loss Calculation ---
            # 1. Phoneme CTC Loss
            ph_log_probs = F.log_softmax(ph_logits, dim=-1).permute(1, 0, 2) # (T_ssl, N, n_phones)
            
            # Correctly calculate the input lengths for CTC loss after WavLM's downsampling
            raw_audio_lengths = inputs['attention_mask'].sum(dim=1)
            input_lengths = wavlm_model._get_feat_extract_output_lengths(raw_audio_lengths)

            ph_target_lengths = (phone_ids_padded != 0).sum(dim=1)
            # --- Move CTC targets and lengths to the correct device ---
            phone_ids_padded = phone_ids_padded.to(device)
            ph_target_lengths = ph_target_lengths.to(device)
            input_lengths = input_lengths.to(device)

            ph_loss = criterion_ph(ph_log_probs, phone_ids_padded, input_lengths, ph_target_lengths)

            # 2. Speaker Classification Loss
            sp_loss = criterion_sp(sp_logits, spk_ids.to(device))

            # 3. Masked Reconstruction Loss
            gt_mels = mel_spectrogram(audios_padded.to(device)) # (N, n_mels, T_mel)
            gt_mels = gt_mels.permute(0, 2, 1) # (N, T_mel, n_mels)

            T_gt = gt_mels.size(1)
            T_pred = pred_mels.size(1)
            if T_pred > T_gt:
                pred_mels_aligned = pred_mels[:, :T_gt, :]
            else:
                pred_mels_aligned = F.pad(pred_mels, (0, 0, 0, T_gt - T_pred), 'constant', 0)

            mel_lengths = torch.floor(attention_mask.sum(1) / mel_spectrogram.hop_length).long().to(device)
            mel_mask = torch.arange(T_gt, device=device)[None, :] < mel_lengths[:, None]
            mel_mask = mel_mask.unsqueeze(2).expand_as(gt_mels)
            
            recon_loss_unreduced = criterion_l1(pred_mels_aligned, gt_mels)
            recon_loss = (recon_loss_unreduced * mel_mask).sum() / mel_mask.sum()

            # --- Total Loss & Backward Pass ---
            batch_loss = (config.lambda_ph * ph_loss + 
                          config.lambda_sp * sp_loss + 
                          config.lambda_recon * recon_loss)
            batch_loss.backward()
            optimizer.step()

            # --- Logging ---
            running_loss += batch_loss.item() * audios_padded.size(0)
            running_ph_loss += ph_loss.item() * audios_padded.size(0)
            running_sp_loss += sp_loss.item() * audios_padded.size(0)
            running_recon_loss += recon_loss.item() * audios_padded.size(0)
            num_samples += audios_padded.size(0)
            
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}', 'Ph': f'{ph_loss.item():.4f}',
                'Sp': f'{sp_loss.item():.4f}', 'Recon': f'{recon_loss.item():.4f}', 'λ': f'{lambd:.3f}'
            })

        except Exception as e:
            logger.error(f"Skipping batch {batch_idx} due to error: {e}\n{traceback.format_exc()}")
            continue

    avg_loss = running_loss / num_samples if num_samples > 0 else 0
    avg_ph_loss = running_ph_loss / num_samples if num_samples > 0 else 0
    avg_sp_loss = running_sp_loss / num_samples if num_samples > 0 else 0
    avg_recon_loss = running_recon_loss / num_samples if num_samples > 0 else 0
    
    logger.info(f"Epoch {epoch+1} TRAIN: Loss={avg_loss:.4f}, Ph={avg_ph_loss:.4f}, Sp={avg_sp_loss:.4f}, Recon={avg_recon_loss:.4f}, λ={lambd:.3f}")
    
    return avg_loss, avg_ph_loss, avg_sp_loss, avg_recon_loss, lambd

def validate_epoch(model, dataloader, criterion_ph, criterion_l1, device, logger, config, wavlm_processor, wavlm_model, voice_encoder, mel_spectrogram):
    """Validate for one epoch with batch processing."""
    model.eval()
    running_val_loss, running_val_ph_loss, running_val_recon_loss = 0, 0, 0
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch_idx, (audios_padded, spk_ids, wav_paths, phone_ids_padded, attention_mask) in enumerate(pbar):
            try:
                # --- Batched Feature Extraction ---
                inputs = wavlm_processor(
                    audios_padded.cpu().numpy(), sampling_rate=config.target_sr,
                    return_tensors="pt", padding=True, return_attention_mask=True
                )
                inputs['attention_mask'] = inputs['attention_mask'].bool()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                ssl_outputs: BaseModelOutput = wavlm_model(**inputs)
                ssl_features = ssl_outputs.last_hidden_state
                # The attention mask from the processor is for the raw audio, so we calculate
                # the downsampled feature lengths for CTC loss below.

                spk_embeds = []
                for i in range(audios_padded.size(0)):
                    unpadded_audio_np = audios_padded[i, :attention_mask[i].sum()].cpu().numpy()
                    if unpadded_audio_np.size == 0: continue
                    spk_embed = voice_encoder.embed_utterance(unpadded_audio_np)
                    spk_embeds.append(spk_embed)
                if not spk_embeds: continue
                spk_embeds = torch.from_numpy(np.array(spk_embeds)).float().to(device)
                # Skip batch if we failed to get embeddings for every sample
                if spk_embeds.size(0) != audios_padded.size(0):
                    continue

                # --- Forward pass (no GRL) ---
                ph_logits, _, pred_mels = model(ssl_features, spk_embeds, lambd=0.0)

                # --- Loss Calculation ---
                # 1. Phoneme CTC Loss
                ph_log_probs = F.log_softmax(ph_logits, dim=-1).permute(1, 0, 2)

                # Correctly calculate the input lengths for CTC loss after WavLM's downsampling
                raw_audio_lengths = inputs['attention_mask'].sum(dim=1)
                input_lengths = wavlm_model._get_feat_extract_output_lengths(raw_audio_lengths)

                ph_target_lengths = (phone_ids_padded != 0).sum(dim=1)
                # --- Move CTC targets and lengths to the correct device ---
                phone_ids_padded = phone_ids_padded.to(device)
                ph_target_lengths = ph_target_lengths.to(device)
                input_lengths = input_lengths.to(device)

                ph_loss = criterion_ph(ph_log_probs, phone_ids_padded, input_lengths, ph_target_lengths)

                # 2. Masked Reconstruction Loss
                gt_mels = mel_spectrogram(audios_padded.to(device)).permute(0, 2, 1)
                T_gt = gt_mels.size(1)
                T_pred = pred_mels.size(1)
                if T_pred > T_gt:
                    pred_mels_aligned = pred_mels[:, :T_gt, :]
                else:
                    pred_mels_aligned = F.pad(pred_mels, (0, 0, 0, T_gt - T_pred), 'constant', 0)

                mel_lengths = torch.floor(attention_mask.sum(1) / mel_spectrogram.hop_length).long().to(device)
                mel_mask = torch.arange(T_gt, device=device)[None, :] < mel_lengths[:, None]
                mel_mask = mel_mask.unsqueeze(2).expand_as(gt_mels)

                recon_loss_unreduced = criterion_l1(pred_mels_aligned, gt_mels)
                recon_loss = (recon_loss_unreduced * mel_mask).sum() / mel_mask.sum()
                
                # Val loss excludes speaker classification
                item_loss = config.lambda_ph * ph_loss + config.lambda_recon * recon_loss
                
                running_val_loss += item_loss.item() * audios_padded.size(0)
                running_val_ph_loss += ph_loss.item() * audios_padded.size(0)
                running_val_recon_loss += recon_loss.item() * audios_padded.size(0)
                num_samples += audios_padded.size(0)
                
                pbar.set_postfix({'Val Loss': f'{item_loss.item():.4f}'})

            except Exception as e:
                logger.warning(f"Skipping val batch {batch_idx} due to error: {e}")
                continue
    
    avg_val_loss = running_val_loss / num_samples if num_samples > 0 else 0
    avg_ph_loss = running_val_ph_loss / num_samples if num_samples > 0 else 0
    avg_recon_loss = running_val_recon_loss / num_samples if num_samples > 0 else 0
    
    logger.info(f"Epoch VAL: Loss={avg_val_loss:.4f}, Ph={avg_ph_loss:.4f}, Recon={avg_recon_loss:.4f}")
    
    return avg_val_loss, avg_ph_loss, avg_recon_loss

def main():
    parser = argparse.ArgumentParser(description="Train RC-OP model")
    parser.add_argument("--data_root", type=str, required=True, help="Path to VCTK dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--subset", type=int, default=500, help="Subset size for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion_ph = nn.CTCLoss(blank=0, zero_infinity=True)  # BLANK token is at index 0
    criterion_sp = nn.CrossEntropyLoss()
    criterion_l1 = nn.L1Loss(reduction='none') # Use 'none' for masked loss calculation
    
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