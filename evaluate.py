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
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoModel
from speechbrain.inference import EncoderClassifier
from tqdm import tqdm

from config import Config
from data.vctk import create_dataloader, VCTKArgs
from models.rcop import RCOP
from utils.logging import setup_logger
from utils.phonemes import get_num_phones
from utils.environment import set_seed
from utils.checkpoint import load_model_from_checkpoint

MODEL_CACHE_DIR = "./models"

def evaluate_phoneme_accuracy(model, dataloader, device, logger, wavlm_processor, wavlm_model, speaker_model):
    """
    Placeholder for evaluating phoneme recognition accuracy (e.g., PER).
    This is a non-trivial task requiring alignment between predicted phoneme sequences
    and ground truth, typically using a library like `jiwer`.
    """
    logger.warning("Phoneme Error Rate (PER) evaluation is not yet implemented.")
    logger.warning("This requires aligning predicted phoneme sequences with ground truth, which is a complex task.")
    return -1.0 # Return a value that clearly indicates it's not implemented

def evaluate_speaker_classification(model, dataloader, wavlm_processor, wavlm_model, speaker_model, device, logger):
    """Evaluate speaker classification accuracy in a batched manner."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audios_padded, spk_ids, wav_paths, _, attention_mask in tqdm(dataloader, desc="Evaluating speaker classification"):
            try:
                # --- Batched Feature Extraction ---
                # 1. WavLM SSL features
                inputs = wavlm_processor(
                    audios_padded.cpu().numpy(), sampling_rate=16000,
                    return_tensors="pt", padding=True, return_attention_mask=True
                )
                inputs['attention_mask'] = inputs['attention_mask'].bool()
                inputs = {k: v.to(device) for k, v in inputs.items()}
                ssl_outputs = wavlm_model(**inputs)
                ssl_features = ssl_outputs.last_hidden_state

                # 2. SpeechBrain speaker embeddings
                audios_padded_device = audios_padded.to(device)
                spk_embeds = speaker_model.encode_batch(audios_padded_device).squeeze(1)

                # --- Forward pass for speaker classification ---
                _, sp_logits, _ = model(ssl_features, spk_embeds, audios_padded_device, lambd=0.0)
                
                predicted = sp_logits.argmax(dim=1)
                actual = spk_ids.to(device)
                
                correct += (predicted == actual).sum().item()
                total += actual.numel()

            except Exception as e:
                logger.warning(f"Skipping batch due to error: {e}")
                continue
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Speaker classification accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def evaluate_reconstruction_quality(model, dataloader, device, logger):
    """
    Placeholder for evaluating reconstruction quality using objective metrics.
    A common metric is Mel Cepstral Distortion (MCD) between synthesized and ground truth audio.
    """
    logger.warning("Reconstruction quality evaluation (e.g., MCD) is not yet implemented.")
    logger.warning("This would require synthesizing audio for each validation sample and comparing with the original.")
    return -1.0 # Return a value that clearly indicates it's not implemented

def evaluate_model(checkpoint_path, data_root, device, logger, subset=100, max_duration_s=20, batch_size=16):
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
        batch_size=batch_size, # Use batch size from args for evaluation
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
    
    speaker_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(MODEL_CACHE_DIR, "spkrec-ecapa-voxceleb"),
        run_opts={"device": device}
    )
    if not speaker_model:
        logger.error("Fatal: SpeechBrain speaker model failed to load.")
        raise RuntimeError("Could not load SpeechBrain speaker model.")
    speaker_model.eval()
    
    # The number of speakers is determined by the CHECKPOINT.
    # Pass a dummy value for num_speakers; it will be overwritten by the checkpoint metadata.
    rcop_model, num_speakers_from_ckpt = load_model_from_checkpoint(checkpoint_path, 0, num_phones, device)
    logger.info(f"Loaded model trained on {num_speakers_from_ckpt} speakers.")
    
    # NOTE: The speaker classification evaluation is commented out because it's not a
    # meaningful metric when evaluating on unseen speakers. The model's speaker
    # classifier was only trained on the training speakers, so it cannot correctly
    # identify speakers from the validation/test set. The primary goal is voice
    # conversion quality (evaluated with PER and MCD), not auxiliary classifier accuracy.
    # speaker_acc = evaluate_speaker_classification(
    #     rcop_model, dataloader, wavlm_processor, wavlm_model, speaker_model, device, logger
    # )
    
    # --- Placeholder for more comprehensive evaluation ---
    logger.info("-" * 20)
    logger.info("NOTE: A complete evaluation requires implementing PER and MCD metrics,")
    logger.info("which are essential for assessing voice conversion quality.")
    evaluate_phoneme_accuracy(rcop_model, dataloader, device, logger, wavlm_processor, wavlm_model, speaker_model)
    evaluate_reconstruction_quality(rcop_model, dataloader, device, logger)
    logger.info("-" * 20)
    
    # Summary
    logger.info("=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    # logger.info(f"Speaker classification accuracy: {speaker_acc:.4f} (auxiliary metric)")
    logger.info("NOTE: Speaker classification accuracy is not reported for unseen speakers.")
    logger.info("-" * 50)
    
    return {
        # 'speaker_accuracy': speaker_acc,
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate RC-OP model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Path to VCTK dataset")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--subset", type=int, default=100, help="Subset size for evaluation")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--max_duration", type=int, default=15, help="Maximum audio duration in seconds to filter dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    
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
    results = evaluate_model(
        args.ckpt, 
        args.data_root, 
        device, 
        logger, 
        subset=args.subset, 
        max_duration_s=args.max_duration, 
        batch_size=args.batch_size
    )
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 