#!/usr/bin/env python3

import torch
import os
from config import Config
from models.rcop import RCOP
from utils.phonemes import get_num_phones
from utils.logging import setup_logger

# This must be done BEFORE importing torch
from utils.environment import setup_environment
setup_environment()

MODEL_CACHE_DIR = "./models"

def test_model_loading(logger):
    """Attempt to load all pre-trained models."""
    logger.info("--- Testing Model Loading ---")
    device = "cpu" # No need for GPU for this test
    all_ok = True
    
    try:
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor, SpeechT5HifiGan
        from speechbrain.inference import EncoderClassifier
        logger.info("✓ Successfully imported HuggingFace and SpeechBrain libraries.")
    except ImportError as e:
        logger.error(f"✗ Failed to import required libraries: {e}")
        logger.error("Please run `pip install -r requirements.txt` in your conda environment.")
        return False, None

    # 1. WavLM
    try:
        Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR)
        WavLMModel.from_pretrained("microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR)
        logger.info("✓ WavLM model loaded successfully.")
    except Exception as e:
        logger.error(f"✗ Failed to load WavLM model: {e}")
        all_ok = False
        
    # 2. SpeechBrain Speaker Encoder
    try:
        EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(MODEL_CACHE_DIR, "spkrec-ecapa-voxceleb"),
            run_opts={"device": device}
        )
        logger.info("✓ SpeechBrain speaker model loaded successfully.")
    except Exception as e:
        logger.error(f"✗ Failed to load SpeechBrain speaker model: {e}")
        all_ok = False

    # 3. HiFi-GAN Vocoder
    vocoder = None
    try:
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir=MODEL_CACHE_DIR)
        logger.info("✓ HiFi-GAN vocoder loaded successfully.")
    except Exception as e:
        logger.error(f"✗ Failed to load HiFi-GAN vocoder: {e}")
        all_ok = False
        
    if all_ok:
        logger.info("✅ All pre-trained models are available and loadable.")
    else:
        logger.error("🔥 One or more models failed to load. Please check the errors above.")
        
    return all_ok, vocoder

def test_forward_pass(logger, vocoder):
    """Perform a dummy forward pass to check for runtime errors."""
    if vocoder is None:
        logger.warning("Skipping forward pass test because vocoder failed to load.")
        return False
        
    logger.info("--- Testing Forward Pass ---")
    device = "cpu"
    all_ok = True
    
    try:
        # Dummy config and inputs
        cfg = Config()
        num_phones = get_num_phones()
        num_speakers = 10 # Dummy value for initialization
        
        # Initialize model
        model = RCOP(
            d_spk=cfg.d_spk,
            d_ssl=cfg.d_ssl,
            n_phones=num_phones,
            n_spk=num_speakers
        ).to(device)
        model.eval()
        
        # Create dummy input tensors
        batch_size = 2
        seq_len_ssl = 150
        ssl_features = torch.randn(batch_size, seq_len_ssl, cfg.d_ssl).to(device)
        spk_embed = torch.randn(batch_size, cfg.d_spk).to(device)
        
        # Forward pass
        with torch.no_grad():
            ph_logits, sp_logits, pred_mels = model(ssl_features, spk_embed, lambd=0.0)
        
        # Check output shapes
        assert ph_logits.shape == (batch_size, seq_len_ssl, num_phones)
        assert sp_logits.shape == (batch_size, num_speakers)
        assert pred_mels.shape[0] == batch_size
        assert pred_mels.shape[1] == seq_len_ssl
        assert pred_mels.shape[2] == 80 # Mel dimension
        
        logger.info("✓ RCOP model forward pass completed successfully.")
        logger.info(f"  - Predicted mels shape: {pred_mels.shape}")
        
        # Test vocoder pass
        vocoder.to(device)
        with torch.no_grad():
            # Vocoder expects (B, T, n_mels)
            wav = vocoder(pred_mels)
        
        assert wav.shape[0] == batch_size
        assert wav.ndim == 2
        logger.info(f"✓ HiFi-GAN vocoder pass completed successfully. Output audio shape: {wav.shape}")
        
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}", exc_info=True)
        all_ok = False
        
    if all_ok:
        logger.info("✅ Forward pass test completed successfully.")
    else:
        logger.error("🔥 The model forward pass encountered a runtime error.")
        
    return all_ok

def main():
    """Run all setup verification tests."""
    os.makedirs("logs", exist_ok=True)
    logger = setup_logger("test_setup", "logs")
    
    # Import here to provide a clear error message if libraries are missing
    try:
        from transformers import SpeechT5HifiGan
        from speechbrain.inference import EncoderClassifier
    except ImportError as e:
        logger.error(f"✗ Failed to import required libraries: {e}")
        logger.error("Please run `pip install -r requirements.txt` in your conda environment first.")
        return

    logger.info("========================================")
    logger.info("  RC-OP Project Setup Verification  ")
    logger.info("========================================")
    
    loading_ok, vocoder_instance = test_model_loading(logger)
    forward_pass_ok = False
    
    if loading_ok:
        forward_pass_ok = test_forward_pass(logger, vocoder_instance)
        
    logger.info("----------------------------------------")
    if loading_ok and forward_pass_ok:
        logger.info("✅ SUCCESS: Your environment is set up correctly!")
        logger.info("You are ready to train the model.")
    else:
        logger.error("🔥 FAILURE: Setup verification failed.")
        logger.error("Please review the error messages above to diagnose the issue.")
        logger.error("Common issues include failed downloads or library mismatches.")
    logger.info("========================================")

if __name__ == "__main__":
    main() 