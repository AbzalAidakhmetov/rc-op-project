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
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoModel, SpeechT5HifiGan
from transformers.modeling_outputs import BaseModelOutput
from speechbrain.inference import EncoderClassifier
import torchaudio.transforms as T
import wandb

from config import Config
from data.vctk import create_dataloader, VCTKArgs, get_vctk_files
from models.rcop import RCOP
from utils.logging import setup_logger, log_config, log_model_summary
from utils.phonemes import get_num_phones, text_to_phones, phones_to_ids
from utils.environment import set_seed
from utils.checkpoint import save_checkpoint, load_model_from_checkpoint
from utils.loss import MultiResolutionSTFTLoss

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
    
    # Load SpeechBrain ECAPA-TDNN for speaker embeddings (frozen)
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(MODEL_CACHE_DIR, "spkrec-ecapa-voxceleb"),
        run_opts={"device": device}
    )
    if not speaker_model:
        # This is a fatal error, so we raise an exception to stop execution.
        raise RuntimeError("Failed to load the SpeechBrain speaker model, which is essential for training.")
    speaker_model.eval()
    speaker_model.requires_grad_(False)
    
    # Load HiFi-GAN vocoder (frozen)
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", cache_dir=MODEL_CACHE_DIR
    )
    vocoder.eval()
    vocoder.requires_grad_(False)
    vocoder.to(device)
    
    return wavlm_processor, wavlm_model, speaker_model, vocoder

def train_epoch(model, dataloader, optimizer, criterion_ph, criterion_l1, criterion_spectral, device, epoch, total_epochs, logger, config, wavlm_processor, wavlm_model, speaker_model, vocoder, mel_spectrogram, bypass_projection=False):
    """Train for one epoch with batch processing."""
    model.train()
    running_loss, running_ph_loss, running_sp_loss, running_recon_loss, running_spectral_loss = 0, 0, 0, 0, 0
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

            # 2. SpeechBrain speaker embeddings (batched)
            with torch.no_grad():
                # SpeechBrain model expects audio on the correct device
                audios_padded_device = audios_padded.to(device)
                spk_embeds = speaker_model.encode_batch(audios_padded_device).squeeze(1)


            # --- Ground truth mel-spectrogram for loss calculation ---
            gt_mels_linear = mel_spectrogram(audios_padded.to(device)) # (N, n_mels, T_mel)
            # Clamp and convert to log-scale for a more stable L1 loss
            gt_mels = torch.log(torch.clamp(gt_mels_linear, min=1e-5))
            gt_mels = gt_mels.permute(0, 2, 1) # (N, T_mel, n_mels)

            # --- Forward pass ---
            ph_logits, sp_logits, pred_mels = model(ssl_features, spk_embeds, audios_padded.to(device), lambd, bypass_projection=bypass_projection)

            # --- Align Time Dimensions ---
            # Predicted mels are already at the target 62.5 Hz frame-rate thanks to the
            # learned up-sampler integrated in RCOP. (Shape match check kept for safety.)
            if pred_mels.size(1) != gt_mels.size(1):
                # Small mismatch (usually 1 frame) due to rounding – fall back to
                # a quick linear interpolation so that shapes align for the loss.
                pred_mels = F.interpolate(
                    pred_mels.permute(0, 2, 1),  # (N, C, T)
                    size=gt_mels.size(1),
                    mode="linear",
                    align_corners=False
                ).permute(0, 2, 1)

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

            # 2. Speaker Classification Loss (Adversarial)
            # The goal is to make the speaker classifier CONFUSED, not accurate.
            # We train it to predict a uniform distribution over all speakers
            # using KL-Divergence loss, providing a stable adversarial signal.
            sp_ids_device = spk_ids.to(device)
            sp_loss = F.cross_entropy(sp_logits, sp_ids_device)

            # 3. Masked Reconstruction Loss
            mel_lengths = torch.floor(attention_mask.sum(1) / mel_spectrogram.hop_length).long().to(device)
            mel_mask = torch.arange(gt_mels.size(1), device=device)[None, :] < mel_lengths[:, None]
            mel_mask = mel_mask.unsqueeze(2).expand_as(gt_mels)
            
            # The model already predicts log-mels, so we compare them directly
            # to the log-transformed ground truth mels.
            recon_loss_unreduced = criterion_l1(pred_mels, gt_mels)
            recon_loss = (recon_loss_unreduced * mel_mask).sum() / mel_mask.sum()

            # 4. High-quality Spectral Loss
            # Pass the predicted mel-spectrogram through the frozen HiFi-GAN vocoder.
            # We intentionally keep the vocoder parameters frozen, but we DO allow
            # gradients to flow back to `pred_mels` so that the spectral loss
            # provides a meaningful training signal.
            pred_audio = vocoder(pred_mels).squeeze(1)
            
            # Use the original padded audio as the ground truth
            gt_audio = audios_padded.to(device)
            spectral_loss = criterion_spectral(pred_audio, gt_audio)

            # --- Total Loss & Backward Pass ---
            batch_loss = (config.lambda_ph * ph_loss +
                          config.lambda_sp * sp_loss +
                          config.lambda_recon * recon_loss +
                          config.lambda_spectral * spectral_loss)
            batch_loss.backward()
            optimizer.step()

            # --- Logging ---
            running_loss += batch_loss.item() * audios_padded.size(0)
            running_ph_loss += ph_loss.item() * audios_padded.size(0)
            running_sp_loss += sp_loss.item() * audios_padded.size(0)
            running_recon_loss += recon_loss.item() * audios_padded.size(0)
            running_spectral_loss += spectral_loss.item() * audios_padded.size(0)
            num_samples += audios_padded.size(0)
            
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}', 'Ph': f'{ph_loss.item():.4f}',
                'Sp': f'{sp_loss.item():.4f}', 'Recon': f'{recon_loss.item():.4f}',
                'Spect': f'{spectral_loss.item():.4f}', 'λ': f'{lambd:.3f}'
            })

        except Exception as e:
            logger.error(f"Skipping batch {batch_idx} due to error: {e}\n{traceback.format_exc()}")
            continue

    avg_loss = running_loss / num_samples if num_samples > 0 else 0
    avg_ph_loss = running_ph_loss / num_samples if num_samples > 0 else 0
    avg_sp_loss = running_sp_loss / num_samples if num_samples > 0 else 0
    avg_recon_loss = running_recon_loss / num_samples if num_samples > 0 else 0
    avg_spectral_loss = running_spectral_loss / num_samples if num_samples > 0 else 0
    
    logger.info(f"Epoch {epoch+1} TRAIN: Loss={avg_loss:.4f}, Ph={avg_ph_loss:.4f}, Sp={avg_sp_loss:.4f}, Recon={avg_recon_loss:.4f}, Spect={avg_spectral_loss:.4f}, λ={lambd:.3f}")
    
    return avg_loss, avg_ph_loss, avg_sp_loss, avg_recon_loss, avg_spectral_loss, lambd

def validate_epoch(model, dataloader, criterion_ph, criterion_l1, device, logger, config, wavlm_processor, wavlm_model, speaker_model, mel_spectrogram, bypass_projection=False):
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

                # Batched speaker embeddings
                # SpeechBrain model expects audio on the correct device
                audios_padded_device = audios_padded.to(device)
                spk_embeds = speaker_model.encode_batch(audios_padded_device).squeeze(1)

                # --- Ground truth mel-spectrogram ---
                gt_mels_linear = mel_spectrogram(audios_padded.to(device))
                gt_mels = torch.log(torch.clamp(gt_mels_linear, min=1e-5))
                gt_mels = gt_mels.permute(0, 2, 1) # (N, T_mel, n_mels)

                # --- Forward pass (no GRL) ---
                ph_logits, _, pred_mels = model(
                    ssl_features, spk_embeds, audios_padded.to(device), lambd=0.0, bypass_projection=bypass_projection
                )

                # --- Align Time Dimensions ---
                if pred_mels.size(1) != gt_mels.size(1):
                    pred_mels = F.interpolate(
                        pred_mels.permute(0, 2, 1),
                        size=gt_mels.size(1),
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)

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
                mel_lengths = torch.floor(attention_mask.sum(1) / mel_spectrogram.hop_length).long().to(device)
                mel_mask = torch.arange(gt_mels.size(1), device=device)[None, :] < mel_lengths[:, None]
                mel_mask = mel_mask.unsqueeze(2).expand_as(gt_mels)

                recon_loss_unreduced = criterion_l1(pred_mels, gt_mels)
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
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--max_duration", type=int, default=15, help="Maximum audio duration in seconds to filter dataset.")
    parser.add_argument("--save_interval", type=int, default=5, help="Save a checkpoint every N epochs.")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Ratio of speakers for validation set.")
    parser.add_argument("--val_subset", type=int, default=100, help="Subset size for validation set, for faster validation.")
    parser.add_argument("--skip_validation", action="store_true", help="Skip validation entirely (useful for overfitting tests)")
    parser.add_argument("--finetune_layers", type=int, default=None, help="Number of final vocoder head layers to finetune. Overrides config.")
    parser.add_argument("--lambda_ph", type=float, default=None, help="Weight for phoneme loss")
    parser.add_argument("--lambda_sp", type=float, default=None, help="Weight for speaker loss")
    parser.add_argument("--lambda_recon", type=float, default=None, help="Weight for reconstruction loss")
    parser.add_argument("--lambda_spectral", type=float, default=None, help="Weight for spectral loss")
    parser.add_argument("--wandb_project", type=str, default="rc-op-project", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="A specific name for the W&B run")
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--overfit_on_file", type=str, default=None, help="Path to a single audio file to overfit on, bypassing normal dataset loading.")
    parser.add_argument("--bypass_projection", action="store_true", help="Bypass orthogonal projection for debugging reconstruction.")
    
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
    
    # Allow overriding from CLI
    if args.finetune_layers is not None: config.finetune_layers = args.finetune_layers
    if args.lambda_ph is not None: config.lambda_ph = args.lambda_ph
    if args.lambda_sp is not None: config.lambda_sp = args.lambda_sp
    if args.lambda_recon is not None: config.lambda_recon = args.lambda_recon
    if args.lambda_spectral is not None: config.lambda_spectral = args.lambda_spectral
    
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
    logger.info("Loading pre-trained models (WavLM, SpeechBrain, HiFi-GAN)...")
    wavlm_processor, wavlm_model, speaker_model, vocoder = load_pretrained_models(device)
    
    # --- Mel Spectrogram transform ---
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=config.target_sr,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        f_min=config.f_min,
        f_max=config.f_max
    ).to(device)

    # --- Dataset Loading and Splitting ---
    if args.overfit_on_file:
        logger.info(f"--- OVERFITTING ON A SINGLE FILE: {args.overfit_on_file} ---")
        if not os.path.exists(args.overfit_on_file):
            raise FileNotFoundError(f"File specified for overfitting not found: {args.overfit_on_file}")
        
        overfit_wav_path = Path(args.overfit_on_file)
        speaker_id = overfit_wav_path.parent.name
        utt_id = overfit_wav_path.stem.replace("_mic1", "")
        txt_root = Path(args.data_root) / "txt"
        txt_path = txt_root / speaker_id / f"{utt_id}.txt"

        if not txt_path.exists():
            raise FileNotFoundError(f"Could not find transcript for the overfit file: {txt_path}")
            
        # When overfitting, create a minimal file list and speaker map for training.
        train_files = [(str(overfit_wav_path), speaker_id, txt_path)]
        val_files = []
        train_speaker_to_id = {speaker_id: 0}
        val_speaker_to_id = {}

        args.skip_validation = True # Overfitting implies no validation
        config.subset = 1 # Force subset to 1
        logger.info(f"Single speaker for overfitting: {speaker_id}")
    else:
        logger.info("Scanning dataset once to get all files...")
        all_files = get_vctk_files(Path(args.data_root), args.max_duration)
        all_speakers = sorted(list(set(f[1] for f in all_files)))
        random.shuffle(all_speakers)

        if args.skip_validation:
            # Use all speakers for training when skipping validation
            train_speakers = all_speakers
            val_speakers = []
        else:
            val_split_idx = int(len(all_speakers) * args.val_split_ratio)
            val_speakers = all_speakers[:val_split_idx]
            train_speakers = all_speakers[val_split_idx:]

        # Filter the file list based on the speaker split
        train_files = [f for f in all_files if f[1] in train_speakers]
        val_files = [f for f in all_files if f[1] in val_speakers]

        # --- Create Speaker Mappings ---
        # The model should only know about the speakers it is trained on.
        # Create a mapping for training speakers with IDs from 0 to N_train-1.
        train_speaker_to_id = {spk: i for i, spk in enumerate(train_speakers)}
        
        # Create a separate mapping for validation speakers. This is good practice,
        # though the IDs are not used in the validation loss calculation itself.
        val_speaker_to_id = {spk: i for i, spk in enumerate(val_speakers)}

        logger.info(f"Total speakers: {len(all_speakers)}")
        logger.info(f"Training speakers: {len(train_speakers)} ({len(train_files)} files)")
        logger.info(f"Validation speakers: {len(val_speakers)} ({len(val_files)} files)")

    # --- Create Dataloaders ---
    logger.info("Creating dataloaders...")
    train_args = VCTKArgs(
        target_sr=config.target_sr,
        subset=config.subset,
        max_duration_s=args.max_duration,
        speaker_to_id=train_speaker_to_id,
        file_list=train_files
    )
    train_dataloader, train_dataset = create_dataloader(
        args=train_args,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    if not args.skip_validation and len(val_files) > 0:
        val_args = VCTKArgs(
            target_sr=config.target_sr,
            subset=args.val_subset,
            max_duration_s=args.max_duration,
            speaker_to_id=val_speaker_to_id, # Use the validation speaker map
            file_list=val_files
        )
        val_dataloader, val_dataset = create_dataloader(
            args=val_args,
            batch_size=config.batch_size,
            shuffle=False
        )
    else:
        val_dataloader, val_dataset = None, None

    num_train_speakers = len(train_dataset.speaker_to_id) if not args.overfit_on_file else 1
    num_phones = get_num_phones()
    
    val_size_log = len(val_dataset) if val_dataset is not None else 0
    logger.info(f"Dataset: {len(train_dataset)} train samples, {val_size_log} val samples, {num_train_speakers} training speakers")
    
    # --- Initialize RCOP model THIRD, now that we have num_speakers ---
    logger.info("Initializing RCOP model...")
    rcop_model = RCOP(
        d_spk=config.d_spk,
        d_ssl=config.d_ssl,
        n_phones=num_phones,
        n_spk=num_train_speakers,
        hop_length=config.hop_length,
    )
    rcop_model.to(device)
    
    if not args.no_wandb:
        wandb.watch(rcop_model, log="all", log_freq=250)

    log_model_summary(logger, rcop_model)
    
    # Setup optimizer and losses
    optimizer = optim.Adam(rcop_model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion_ph = nn.CTCLoss(blank=0, zero_infinity=True)  # BLANK token is at index 0
    criterion_l1 = nn.L1Loss(reduction='none') # Use 'none' for masked loss calculation
    criterion_spectral = MultiResolutionSTFTLoss(
        fft_sizes=config.stft_loss_fft_sizes,
        hop_sizes=config.stft_loss_hop_sizes,
        win_lengths=config.stft_loss_win_sizes,
    ).to(device)
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        avg_loss, avg_ph_loss, avg_sp_loss, avg_recon_loss, avg_spectral_loss, lambd = train_epoch(
            rcop_model, train_dataloader, optimizer, criterion_ph, criterion_l1, criterion_spectral,
            device, epoch, config.epochs, logger, config,
            wavlm_processor, wavlm_model, speaker_model, vocoder, mel_spectrogram,
            bypass_projection=args.bypass_projection
        )
        
        if not args.skip_validation and val_dataloader is not None:
            avg_val_loss, avg_val_ph_loss, avg_val_recon_loss = validate_epoch(
                rcop_model, val_dataloader, criterion_ph, criterion_l1, device, logger, config,
                wavlm_processor, wavlm_model, speaker_model, mel_spectrogram,
                bypass_projection=args.bypass_projection
            )
        else:
            avg_val_loss = avg_val_ph_loss = avg_val_recon_loss = 0.0

        # Log metrics to W&B
        if not args.no_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_ph_loss": avg_ph_loss,
                "train_sp_loss": avg_sp_loss,
                "train_recon_loss": avg_recon_loss,
                "train_spectral_loss": avg_spectral_loss,
                "val_loss": avg_val_loss,
                "val_ph_loss": avg_val_ph_loss,
                "val_recon_loss": avg_val_recon_loss,
                "lambda": lambd,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # Step the LR scheduler based on validation loss
        if not args.skip_validation:
            scheduler.step(avg_val_loss)
        else:
            scheduler.step(avg_loss)
        
        # Save checkpoint periodically and at the end
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == config.epochs:
            checkpoint_path = os.path.join(args.save_dir, f"rcop_epoch{epoch+1}.pt")
            save_checkpoint(rcop_model, optimizer, scheduler, epoch, avg_loss, checkpoint_path, num_train_speakers, num_phones, best_val_loss=best_val_loss)
            logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        # Save best model based on validation loss
        if not args.skip_validation and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(args.save_dir, "rcop_best.pt")
            save_checkpoint(rcop_model, optimizer, scheduler, epoch, avg_loss, best_checkpoint_path, num_train_speakers, num_phones, best_val_loss=best_val_loss)
            logger.info(f"Saved new best model to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    logger.info("Training completed!")

    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 