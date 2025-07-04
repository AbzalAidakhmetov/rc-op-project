#!/usr/bin/env python3
"""
Test script to verify RC-OP setup is working correctly.
Run this after setup.sh completes successfully.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✓ TorchAudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"✗ TorchAudio import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False
    
    try:
        import resemblyzer
        print(f"✓ Resemblyzer")
    except ImportError as e:
        print(f"✗ Resemblyzer import failed: {e}")
        return False
    
    try:
        import soundfile
        print(f"✓ SoundFile {soundfile.__version__}")
    except ImportError as e:
        print(f"✗ SoundFile import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that WavLM model can be loaded."""
    print("\nTesting model loading...")
    
    MODEL_CACHE_DIR = "./models"
    
    try:
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        
        print("Loading WavLM feature extractor...")
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
        )
        print("✓ WavLM feature extractor loaded successfully")
        
        print("Loading WavLM model...")
        model = WavLMModel.from_pretrained(
            "microsoft/wavlm-large", cache_dir=MODEL_CACHE_DIR
        )
        print("✓ WavLM model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        print("This is normal if models haven't been downloaded yet.")
        print("Models will be downloaded automatically when first used.")
        return False

def test_data_directory():
    """Test that data directory structure is correct."""
    print("\nTesting data directory...")
    
    # Path from default setup.sh
    default_path = "./data/VCTK-Corpus-0.92"
    # Path if user just unzips in ./data
    alt_path = "./data"

    data_dir = None
    if os.path.exists(os.path.join(default_path, "wav48_silence_trimmed")):
        data_dir = default_path
    elif os.path.exists(os.path.join(alt_path, "wav48_silence_trimmed")):
        data_dir = alt_path

    if data_dir:
        print(f"✓ VCTK data directory found: {data_dir}")
        return True
    else:
        print(f"⚠ VCTK wav data not found in {default_path} or {alt_path}")
        print("This is normal if you haven't downloaded the VCTK dataset yet.")
        return False

def main():
    print("RC-OP Setup Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test model loading
    if not test_model_loading():
        success = False
    
    # Test data directory
    if not test_data_directory():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! RC-OP is ready to use.")
        print("\nNext steps:")
        print("1. Set VCTK_ROOT environment variable:")
        print("   export VCTK_ROOT=./data/VCTK-Corpus-0.92")
        print("2. Start training:")
        print("   python train.py --data_root $VCTK_ROOT --epochs 20")
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        print("You may need to:")
        print("- Run setup.sh again")
        print("- Download the VCTK dataset manually")
        print("- Check your internet connection for model downloads")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 