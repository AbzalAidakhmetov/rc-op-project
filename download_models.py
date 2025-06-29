from transformers import WavLMModel, Wav2Vec2Processor

print("Downloading Wav2Vec2Processor...")
Wav2Vec2Processor.from_pretrained("microsoft/wavlm-large")
print("Downloading WavLMModel...")
WavLMModel.from_pretrained("microsoft/wavlm-large")
print("All models downloaded and cached.") 