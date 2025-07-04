#!/usr/bin/env python3

import argparse
import os

# --- Environment Setup ---
# This must be done BEFORE importing torch, numpy, etc. to mitigate threading issues.
from utils.environment import setup_environment
setup_environment()
# -------------------------

import torch
torch.set_num_threads(1)
import numpy as np
from torch.utils.data import DataLoader
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from resemblyzer import VoiceEncoder
from tqdm import tqdm

from config import Config
from data.vctk import create_dataloader, VCTKArgs
from models.rcop import RCOP
from utils.logging import setup_logger
from utils.phonemes import get_num_phones
from utils.environment import set_seed
from utils.features import extract_features
from utils.checkpoint import load_model_from_checkpoint

MODEL_CACHE_DIR = "./models"

def evaluate_speaker_classification(model, dataloader, wavlm_processor, wavlm_model, voice_encoder, device, logger):
    """Evaluate speaker classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, spk_id, wav_path, _, attention_mask in tqdm(dataloader, desc="Evaluating speaker classification"):
            try:
                # Features are extracted one-by-one if batch size > 1
                ssl_features_list, spk_embed_list, spk_id_list = [], [], []
                for i in range(audio.size(0)):
                    unpadded_audio = audio[i, :attention_mask[i].sum()]
                    if unpadded_audio.numel() == 0: continue
                    
                    ssl_feats, spk_embed = extract_features(
                        unpadded_audio, wavlm_processor, wavlm_model, voice_encoder, device
                    )
                    ssl_features_list.append(ssl_feats)
                    spk_embed_list.append(spk_embed)
                    spk_id_list.append(spk_id[i])

                if not ssl_features_list: continue

                # Process each item from the batch individually
                for ssl_features, spk_embed, actual_id in zip(ssl_features_list, spk_embed_list, spk_id_list):
                    # Forward pass needs batch dimension
                    ph_logits, sp_logits, _ = model(ssl_features.unsqueeze(0), spk_embed.unsqueeze(0), lambd=0.0)
                    
                    predicted = sp_logits.argmax(dim=1)
                    actual = actual_id.to(device).unsqueeze(0)
                    
                    correct += (predicted == actual).sum().item()
                    total += actual.numel()

            except Exception as e:
                logger.warning(f"Skipping sample due to error: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Speaker classification accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_speaker_disentanglement(model, dataloader, wavlm_processor, wavlm_model, voice_encoder, device, logger):
    """Evaluate how well speaker information is removed from content features."""
    model.eval()
    
    speaker_embeddings = {}
    content_features = {}
    
    with torch.no_grad():
        for audio, spk_id, wav_path, _, attention_mask in tqdm(dataloader, desc="Extracting features for disentanglement"):
            # This evaluation assumes a batch size of 1 for simplicity.
            if audio.size(0) != 1:
                logger.warning("Disentanglement evaluation should be run with batch size 1. Skipping batch.")
                continue

            try:
                unpadded_audio = audio[0, :attention_mask[0].sum()]
                if unpadded_audio.numel() == 0: continue

                # Extract features
                ssl_features, spk_embed = extract_features(
                    unpadded_audio, wavlm_processor, wavlm_model, voice_encoder, device
                )
                
                # Get projected features (content without speaker info)
                axis = model.W_proj(spk_embed)
                proj_feats = model.project_orthogonally(ssl_features, axis)
                
                # Store mean content features per speaker
                spk_id_str = str(spk_id.item())
                if spk_id_str not in content_features:
                    content_features[spk_id_str] = []
                    speaker_embeddings[spk_id_str] = []
                
                content_features[spk_id_str].append(proj_feats.mean(0).cpu())
                speaker_embeddings[spk_id_str].append(spk_embed.cpu())
                
            except Exception as e:
                logger.warning(f"Skipping sample due to error: {e}")
                continue
    
    # Calculate average similarity within speakers for content vs speaker embeddings
    content_similarities = []
    speaker_similarities = []
    
    for spk_id in content_features:
        if len(content_features[spk_id]) > 1:
            # Content features should be similar within speaker
            feats = torch.stack(content_features[spk_id])
            mean_feat = feats.mean(0)
            similarities = torch.cosine_similarity(feats, mean_feat.unsqueeze(0))
            content_similarities.extend(similarities.tolist())
            
            # Speaker embeddings should also be similar within speaker
            embeds = torch.stack(speaker_embeddings[spk_id])
            mean_embed = embeds.mean(0)
            similarities = torch.cosine_similarity(embeds, mean_embed.unsqueeze(0))
            speaker_similarities.extend(similarities.tolist())
    
    if not content_similarities or not speaker_similarities:
        logger.warning("Could not compute similarity scores. This typically requires more than one sample per speaker in the evaluation set.")
        return 0, 0
    
    avg_content_sim = np.mean(content_similarities) if content_similarities else 0
    avg_speaker_sim = np.mean(speaker_similarities) if speaker_similarities else 0
    
    logger.info(f"Average content similarity within speakers: {avg_content_sim:.4f}")
    logger.info(f"Average speaker embedding similarity within speakers: {avg_speaker_sim:.4f}")
    
    return avg_content_sim, avg_speaker_sim

def evaluate_model(checkpoint_path, data_root, device, logger, subset=100, max_duration_s=20):
    """Comprehensive model evaluation."""
    logger.info(f"Evaluating model from {checkpoint_path}")
    
    # Create dataloader
    config = Config()
    vctk_args = VCTKArgs(
        data_root=data_root,
        target_sr=config.target_sr,
        subset=subset,
        max_duration_s=max_duration_s,
    )
    dataloader, dataset = create_dataloader(
        args=vctk_args,
        batch_size=1, # Use batch size 1 for evaluation for simplicity
        shuffle=False,
    )
    
    num_speakers_in_dataset = dataset.get_num_speakers()
    num_phones = get_num_phones()
    
    logger.info(f"Evaluation dataset: {len(dataset)} samples, {num_speakers_in_dataset} speakers")
    
    # Load models
    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model = WavLMModel.from_pretrained(
        "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
    )
    wavlm_model.eval()
    wavlm_model.to(device)
    
    voice_encoder = VoiceEncoder()
    
    # The number of speakers is determined by the CHECKPOINT.
    # Pass a dummy value for num_speakers; it will be overwritten by the checkpoint metadata.
    rcop_model, num_speakers_from_ckpt = load_model_from_checkpoint(checkpoint_path, 0, num_phones, device)
    logger.info(f"Loaded model trained on {num_speakers_from_ckpt} speakers.")
    
    # Evaluate speaker classification
    speaker_acc = evaluate_speaker_classification(
        rcop_model, dataloader, wavlm_processor, wavlm_model, voice_encoder, device, logger
    )
    
    # Evaluate speaker disentanglement
    content_sim, speaker_sim = evaluate_speaker_disentanglement(
        rcop_model, dataloader, wavlm_processor, wavlm_model, voice_encoder, device, logger
    )
    
    # Summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Speaker classification accuracy: {speaker_acc:.4f}")
    logger.info(f"Content feature consistency: {content_sim:.4f}")
    logger.info(f"Speaker embedding consistency: {speaker_sim:.4f}")
    logger.info("=" * 50)
    
    return {
        'speaker_accuracy': speaker_acc,
        'content_similarity': content_sim,
        'speaker_similarity': speaker_sim
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate RC-OP model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to VCTK dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--subset", type=int, default=100, help="Subset size for evaluation")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--max_duration", type=int, default=15, help="Maximum audio duration in seconds to filter dataset.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logger("evaluate", args.log_dir)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(Config().seed)
    logger.info(f"Using random seed: {Config().seed}")
    
    # Evaluate model
    results = evaluate_model(args.ckpt, args.data_root, device, logger, args.subset, args.max_duration)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 