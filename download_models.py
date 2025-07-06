import os
import sys

def download_transformer_model(model_name, model_class, processor_class=None):
    """Download a HuggingFace Transformer model."""
    print(f"Downloading Transformer model: {model_name}...")
    try:
        if processor_class:
            processor_class.from_pretrained(model_name, cache_dir="./models")
        model_class.from_pretrained(model_name, cache_dir="./models")
        print(f"✓ {model_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def download_speechbrain_model(model_name, model_class):
    """Download a SpeechBrain model."""
    print(f"Downloading SpeechBrain model: {model_name}...")
    try:
        # Use last part of name for a clean directory
        savedir = os.path.join("./models", model_name.split('/')[-1])
        model_class.from_hparams(
            source=model_name, 
            savedir=savedir,
            run_opts={"device": "cpu"} # Download on CPU to avoid CUDA init
        )
        print(f"✓ {model_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error downloading {model_name}: {e}")
        return False

def main():
    os.makedirs("./models", exist_ok=True)
    
    try:
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        from speechbrain.inference import EncoderClassifier
    except ImportError as e:
        print(f"Required library not found: {e}. Please run `pip install -r requirements.txt` first.")
        return 1
        
    # Download WavLM
    wavlm_ok = download_transformer_model(
        "microsoft/wavlm-large", 
        WavLMModel, 
        Wav2Vec2FeatureExtractor
    )

    # Download SpeechBrain ECAPA-TDNN
    speaker_model_ok = download_speechbrain_model(
        "speechbrain/spkrec-ecapa-voxceleb",
        EncoderClassifier
    )
    
    if wavlm_ok and speaker_model_ok:
        print("\nAll models downloaded and cached successfully!")
        return 0
    else:
        print("\nOne or more model downloads failed. Please check the errors above.")
        print("This might be due to network issues. The models will attempt to download again when first used.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 