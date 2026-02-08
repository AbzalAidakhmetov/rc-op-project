#!/usr/bin/env python3
"""
Inference script for second.py checkpoints (FlowMatchingResNet).
Loads a trained checkpoint and converts audio files.

Usage:
    # Basic conversion (auto-picks VCTK speakers if no source/target given)
    python inference.py --checkpoint outputs_source/checkpoint_0002000.pt

    # Specify source and target audio
    python inference.py --checkpoint outputs_svd_20k/model_final.pt \
        --source path/to/source.wav --target path/to/target_speaker.wav

    # More ODE steps for better quality
    python inference.py --checkpoint outputs_svd_20k/model_final.pt --ode-steps 64

    # Batch: convert source to multiple target speakers
    python inference.py --checkpoint outputs_svd_20k/model_final.pt \
        --source path/to/source.wav \
        --target speaker1.wav speaker2.wav speaker3.wav
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Import everything from second.py (model, feature extraction, inference logic)
from second import (
    FlowMatchingResNet,
    FeatureExtractor,
    FlowMode,
    convert_voice,
    load_audio,
    WAVLM_SR,
    VOCOS_SR,
    VCTK_ROOT,
    ODE_STEPS,
    GUIDANCE_SCALE,
)


def load_checkpoint(path: str, device: str):
    """
    Load a second.py checkpoint and return model + metadata.
    Works with both model_final.pt and checkpoint_NNNNNNN.pt files.
    """
    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")

    # Extract saved config (with sensible defaults for older checkpoints)
    mode = ckpt.get("mode", "svd")
    svd_proj = ckpt.get("svd_proj", None)
    step = ckpt.get("step", "?")
    wavlm_layer = ckpt.get("wavlm_layer", -1)
    use_instance_norm = ckpt.get("instance_norm", False)

    # Build and load model
    model = FlowMatchingResNet()
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Step           : {step}")
    print(f"  Mode           : {mode}")
    print(f"  WavLM layer    : {wavlm_layer} ({'last' if wavlm_layer == -1 else wavlm_layer})")
    print(f"  Instance norm  : {use_instance_norm}")
    print(f"  SVD projection : {'yes' if svd_proj is not None else 'no'}")
    print(f"  Parameters     : {n_params:,}")

    return model, svd_proj, FlowMode(mode), wavlm_layer, use_instance_norm


def find_default_audio(data_dir: Path, speaker_idx: int = 0, file_idx: int = 0):
    """Pick a default audio file from VCTK."""
    speakers = sorted(
        d.name for d in data_dir.iterdir()
        if d.is_dir() and d.name.startswith("p")
    )
    if not speakers:
        raise FileNotFoundError(f"No speaker directories found in {data_dir}")
    spk = speakers[min(speaker_idx, len(speakers) - 1)]
    spk_dir = data_dir / spk
    files = sorted(spk_dir.glob("*_mic1.flac"))
    if not files:
        files = sorted(spk_dir.glob("*.flac"))
    if not files:
        files = sorted(spk_dir.glob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No audio files in {spk_dir}")
    return str(files[min(file_idx, len(files) - 1)]), spk


def main():
    parser = argparse.ArgumentParser(
        description="Voice Conversion Inference (for second.py checkpoints)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--source", type=str, default=None,
                        help="Source audio file (default: auto-pick from VCTK)")
    parser.add_argument("--target", type=str, nargs="+", default=None,
                        help="Target speaker reference(s) -- one or more files")
    parser.add_argument("--output-dir", type=str, default="outputs_inference",
                        help="Output directory (default: outputs_inference)")
    parser.add_argument("--data-dir", type=str, default=str(VCTK_ROOT),
                        help="VCTK data dir (used for auto-picking source/target)")
    parser.add_argument("--ode-steps", type=int, default=ODE_STEPS,
                        help=f"ODE solver steps (default: {ODE_STEPS})")
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE,
                        help=f"CFG guidance scale (default: {GUIDANCE_SCALE})")
    parser.add_argument("--max-duration", type=float, default=10.0,
                        help="Max source audio duration in seconds (default: 10)")
    # Allow overriding checkpoint settings
    parser.add_argument("--wavlm-layer", type=int, default=None,
                        help="Override WavLM layer from checkpoint")
    parser.add_argument("--instance-norm", action="store_true", default=None,
                        help="Override: force instance norm on")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ---- Load model ----
    model, svd_proj, mode, wavlm_layer, use_instance_norm = load_checkpoint(
        args.checkpoint, device
    )

    # Allow CLI overrides
    if args.wavlm_layer is not None:
        wavlm_layer = args.wavlm_layer
        print(f"  (override) WavLM layer: {wavlm_layer}")
    if args.instance_norm is not None:
        use_instance_norm = args.instance_norm
        print(f"  (override) Instance norm: {use_instance_norm}")

    # ---- Load feature extractor ----
    print("\nLoading frozen models (WavLM, Vocos, ECAPA) ...")
    feat_ext = FeatureExtractor(device=device).to(device)

    # ---- Resolve source audio ----
    data_dir = Path(args.data_dir)
    if args.source:
        source_path = args.source
    else:
        source_path, src_spk = find_default_audio(data_dir, speaker_idx=0)
        print(f"\nAuto-selected source speaker: {src_spk}")

    # ---- Resolve target audio(s) ----
    if args.target:
        target_paths = args.target
    else:
        # Pick a different speaker than source
        tp, tgt_spk = find_default_audio(data_dir, speaker_idx=4)
        target_paths = [tp]
        print(f"Auto-selected target speaker: {tgt_spk}")

    # ---- Load source audio ----
    print(f"\nSource: {source_path}")
    src_16k = load_audio(source_path, WAVLM_SR, max_sec=args.max_duration)
    src_24k = load_audio(source_path, VOCOS_SR, max_sec=args.max_duration)
    src_dur = len(src_24k) / VOCOS_SR
    print(f"  Duration: {src_dur:.2f}s")

    # Save original for easy comparison
    src_name = Path(source_path).stem
    orig_path = output_dir / f"{src_name}_original.wav"
    sf.write(str(orig_path), src_24k.numpy(), VOCOS_SR)
    print(f"  Saved original: {orig_path}")

    # ---- Convert to each target ----
    print(f"\nODE steps: {args.ode_steps}  |  Guidance: {args.guidance_scale}  |  Mode: {mode.value}")
    print("-" * 60)

    for target_path in target_paths:
        tgt_name = Path(target_path).stem
        print(f"\nTarget ref: {target_path}")

        tgt_16k = load_audio(target_path, WAVLM_SR, max_sec=5.0)

        # Save target speaker reference so you can hear their real voice
        tgt_24k = load_audio(target_path, VOCOS_SR, max_sec=5.0)
        tgt_ref_path = output_dir / f"{tgt_name}_target_reference.wav"
        sf.write(str(tgt_ref_path), tgt_24k.numpy(), VOCOS_SR)
        print(f"  Target voice : {tgt_ref_path} ({len(tgt_24k) / VOCOS_SR:.2f}s)")

        # Conversion
        conv_np, conv_dur = convert_voice(
            model, feat_ext, src_16k, src_24k, tgt_16k,
            svd_proj, mode, device,
            ode_steps=args.ode_steps, guidance=args.guidance_scale,
            wavlm_layer=wavlm_layer, use_instance_norm=use_instance_norm,
        )
        conv_path = output_dir / f"{src_name}_to_{tgt_name}.wav"
        sf.write(str(conv_path), conv_np, VOCOS_SR)
        print(f"  Converted    : {conv_path} ({conv_dur:.2f}s)")

        # Reconstruction (same speaker as source)
        recon_np, recon_dur = convert_voice(
            model, feat_ext, src_16k, src_24k, src_16k,
            svd_proj, mode, device,
            ode_steps=args.ode_steps, guidance=args.guidance_scale,
            wavlm_layer=wavlm_layer, use_instance_norm=use_instance_norm,
        )
        recon_path = output_dir / f"{src_name}_reconstructed.wav"
        sf.write(str(recon_path), recon_np, VOCOS_SR)
        print(f"  Reconstructed: {recon_path} ({recon_dur:.2f}s)")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("DONE! All outputs in:", output_dir)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
