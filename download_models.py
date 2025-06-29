import os
import sys

def download_model_with_retry(model_name, model_class, processor_class=None):
    """Download model with retry logic and proper error handling."""
    print(f"Downloading {model_name}...")
    
    try:
        # First try to download the processor if specified
        if processor_class:
            print(f"Downloading processor for {model_name}...")
            processor = processor_class.from_pretrained(model_name, cache_dir="./models")
            print(f"✓ Processor downloaded successfully")
        
        # Download the main model
        model = model_class.from_pretrained(model_name, cache_dir="./models")
        print(f"✓ {model_name} downloaded successfully")
        return model
        
    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        return None

def try_alternative_approaches():
    """Try alternative approaches for downloading WavLM."""
    print("Trying alternative approaches...")
    
    # Try 1: Use AutoProcessor instead of Wav2Vec2Processor
    try:
        from transformers import AutoProcessor, WavLMModel
        print("Trying with AutoProcessor...")
        processor = AutoProcessor.from_pretrained("microsoft/wavlm-large", cache_dir="./models")
        model = WavLMModel.from_pretrained("microsoft/wavlm-large", cache_dir="./models")
        print("✓ Success with AutoProcessor!")
        return True
    except Exception as e:
        print(f"AutoProcessor approach failed: {e}")
    
    # Try 2: Use AutoModel instead of WavLMModel
    try:
        from transformers import AutoProcessor, AutoModel
        print("Trying with AutoModel...")
        processor = AutoProcessor.from_pretrained("microsoft/wavlm-large", cache_dir="./models")
        model = AutoModel.from_pretrained("microsoft/wavlm-large", cache_dir="./models")
        print("✓ Success with AutoModel!")
        return True
    except Exception as e:
        print(f"AutoModel approach failed: {e}")
    
    return False

def main():
    # Create models directory
    os.makedirs("./models", exist_ok=True)
    
    # Check if transformers is available
    try:
        from transformers import WavLMModel, Wav2Vec2Processor
    except ImportError:
        print("Transformers not available yet. Models will be downloaded automatically when first used.")
        return 0
    
    # Try the standard approach first
    wavlm_model = download_model_with_retry(
        "microsoft/wavlm-large", 
        WavLMModel, 
        Wav2Vec2Processor
    )
    
    if wavlm_model is not None:
        print("All models downloaded and cached successfully!")
        return 0
    
    # If standard approach failed, try alternatives
    if try_alternative_approaches():
        print("Models downloaded successfully using alternative approach!")
        return 0
    
    # If all approaches failed
    print("All download approaches failed.")
    print("This might be due to:")
    print("- Network connectivity issues")
    print("- Model availability on HuggingFace")
    print("- Transformers version compatibility")
    print("The models will be downloaded automatically when first used in training/inference.")
    return 1

if __name__ == "__main__":
    sys.exit(main()) 