#!/usr/bin/env python3
"""
Evaluate voice conversion quality across experiments.

Computes:
  1. Speaker similarity: cosine distance between converted audio and target
     speaker reference (using ECAPA-TDNN embeddings). Higher = more like target.
  2. Source leakage: cosine distance between converted audio and source speaker.
     Lower = less source speaker leaking through.
  3. WER (Word Error Rate): ASR content preservation check using Whisper.
     Lower = better content preservation.

Usage:
    # Evaluate all experiments at step 30000 (speaker sim only, default)
    python evaluate.py --step 30000

    # Evaluate WER across all experiments at their latest step
    python evaluate.py --wer

    # Evaluate WER for specific experiments and step
    python evaluate.py --wer --step 30000 --exp-dirs outputs_30k_svd_k2_inorm

    # Evaluate a specific experiment
    python evaluate.py --exp-dirs outputs_30k_source outputs_30k_svd_k2

    # Evaluate final inference outputs
    python evaluate.py --inference-dir outputs_inference
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier


VOCOS_SR = 24000
ECAPA_SR = 16000
WHISPER_SR = 16000


class SpeakerSimilarity:
    """Compute speaker similarity using ECAPA-TDNN embeddings."""

    def __init__(self, device="cuda"):
        print("Loading ECAPA-TDNN speaker encoder ...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa_voxceleb",
            run_opts={"device": device},
        )
        self.encoder.eval()
        self.device = device

    @torch.no_grad()
    def embed(self, audio_path: str) -> torch.Tensor:
        """Extract speaker embedding from audio file."""
        wav_np, sr = sf.read(audio_path)
        wav = torch.from_numpy(wav_np.astype(np.float32))
        if wav.dim() > 1:
            wav = wav.mean(-1)
        wav = torchaudio.functional.resample(wav, sr, ECAPA_SR)
        emb = self.encoder.encode_batch(wav.unsqueeze(0).to(self.device))
        return emb.squeeze()  # (192,)

    def cosine_sim(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(
            emb_a.unsqueeze(0), emb_b.unsqueeze(0)
        ).item()


class ASRScorer:
    """Compute WER using Whisper ASR from HuggingFace transformers."""

    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cuda"):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        print(f"Loading Whisper ASR model: {model_name} ...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to text."""
        wav_np, sr = sf.read(audio_path)
        wav = torch.from_numpy(wav_np.astype(np.float32))
        if wav.dim() > 1:
            wav = wav.mean(-1)
        # Resample to 16kHz for Whisper
        if sr != WHISPER_SR:
            wav = torchaudio.functional.resample(wav, sr, WHISPER_SR)

        inputs = self.processor(
            wav.numpy(), sampling_rate=WHISPER_SR, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        # Force English, no timestamps
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="en", task="transcribe"
        )
        generated_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=256
        )
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip().lower()


def evaluate_wer_val_samples(
    exp_dir: Path, step: int, asr: ASRScorer
) -> list[dict] | None:
    """
    Evaluate WER for validation samples at a given step.
    Transcribes source reference and converted audio, computes WER.
    """
    from jiwer import wer as compute_wer

    val_dir = exp_dir / "val_samples"
    ref_dir = val_dir / "references"

    if not ref_dir.exists():
        print(f"  No references/ folder in {val_dir}")
        return None

    # Find conversion files for this step
    pattern = f"step{step:07d}_*_to_*.wav"
    conv_files = sorted(val_dir.glob(pattern))

    if not conv_files:
        # Try to find the latest step available
        all_wavs = sorted(val_dir.glob("step*.wav"))
        if not all_wavs:
            print(f"  No val samples in {val_dir}")
            return None
        available_steps = sorted(set(
            int(w.name.split("_")[0].replace("step", ""))
            for w in all_wavs
        ))
        step = available_steps[-1]  # Use latest
        pattern = f"step{step:07d}_*_to_*.wav"
        conv_files = sorted(val_dir.glob(pattern))
        print(f"  Using latest available step: {step}")

    results = []
    for conv_path in conv_files:
        # Parse filename: step0030000_p237_to_p307.wav
        parts = conv_path.stem.split("_")
        to_idx = parts.index("to")
        src_spk = "_".join(parts[1:to_idx])

        # Find matching source reference
        src_refs = sorted(ref_dir.glob(f"source_{src_spk}_*.wav"))
        if not src_refs:
            continue

        # Transcribe source (ground truth content) and converted audio
        ref_text = asr.transcribe(str(src_refs[0]))
        hyp_text = asr.transcribe(str(conv_path))

        if not ref_text:
            print(f"    Warning: empty transcription for source {src_refs[0].name}")
            continue

        pair_wer = compute_wer(ref_text, hyp_text) if hyp_text else 1.0

        tgt_spk = "_".join(parts[to_idx + 1:])
        results.append({
            "pair": f"{src_spk}->{tgt_spk}",
            "ref_text": ref_text,
            "hyp_text": hyp_text,
            "wer": pair_wer,
        })

    return results


def print_wer_results(name: str, results: list[dict]):
    """Print WER results for an experiment."""
    if not results:
        return

    avg_wer = np.mean([r["wer"] for r in results])

    print(f"\n  {'Pair':<20s}  {'WER':>6s}  {'Reference':^30s}  {'Hypothesis':^30s}")
    print(f"  {'-' * 20}  {'-' * 6}  {'-' * 30}  {'-' * 30}")
    for r in results:
        ref_short = r["ref_text"][:28] + ".." if len(r["ref_text"]) > 30 else r["ref_text"]
        hyp_short = r["hyp_text"][:28] + ".." if len(r["hyp_text"]) > 30 else r["hyp_text"]
        print(f"  {r['pair']:<20s}  {r['wer']:>5.1%}  {ref_short:<30s}  {hyp_short:<30s}")
    print(f"  {'-' * 20}  {'-' * 6}")
    print(f"  {'AVERAGE WER':<20s}  {avg_wer:>5.1%}")

    if avg_wer < 0.10:
        print(f"  --> Excellent content preservation (WER < 10%)")
    elif avg_wer < 0.30:
        print(f"  --> Good content preservation (WER < 30%)")
    else:
        print(f"  --> WARNING: Significant content destruction (WER > 30%)")
        print(f"      Instance norm or speaker stripping may be too aggressive.")

    return avg_wer


def evaluate_val_samples(exp_dir: Path, step: int, scorer: SpeakerSimilarity):
    """Evaluate validation samples from a training run at a given step."""
    val_dir = exp_dir / "val_samples"
    ref_dir = val_dir / "references"

    if not ref_dir.exists():
        print(f"  No references/ folder in {val_dir}")
        return None

    # Find all conversion files for this step
    pattern = f"step{step:07d}_*_to_*.wav"
    conv_files = sorted(val_dir.glob(pattern))

    if not conv_files:
        print(f"  No files matching step {step} in {val_dir}")
        return None

    results = []
    for conv_path in conv_files:
        # Parse filename: step0030000_p237_to_p307.wav
        parts = conv_path.stem.split("_")
        # Find "to" index to split source and target speaker names
        to_idx = parts.index("to")
        src_spk = "_".join(parts[1:to_idx])
        tgt_spk = "_".join(parts[to_idx + 1:])

        # Find matching reference files
        src_refs = sorted(ref_dir.glob(f"source_{src_spk}_*.wav"))
        tgt_refs = sorted(ref_dir.glob(f"target_{tgt_spk}_*.wav"))

        if not src_refs or not tgt_refs:
            continue

        emb_conv = scorer.embed(str(conv_path))
        emb_src = scorer.embed(str(src_refs[0]))
        emb_tgt = scorer.embed(str(tgt_refs[0]))

        sim_to_target = scorer.cosine_sim(emb_conv, emb_tgt)
        sim_to_source = scorer.cosine_sim(emb_conv, emb_src)

        results.append({
            "pair": f"{src_spk}->{tgt_spk}",
            "target_sim": sim_to_target,
            "source_sim": sim_to_source,
        })

    return results


def evaluate_inference_dir(inf_dir: Path, scorer: SpeakerSimilarity):
    """Evaluate inference output directory."""
    results = []

    # Find converted files (pattern: *_to_*.wav)
    conv_files = sorted(inf_dir.glob("*_to_*.wav"))

    for conv_path in conv_files:
        # Parse: p225_003_mic1_to_p229_010_mic1.wav
        stem = conv_path.stem
        to_idx = stem.index("_to_")
        src_stem = stem[:to_idx]
        tgt_stem = stem[to_idx + 4:]

        # Look for reference files
        orig = inf_dir / f"{src_stem}_original.wav"
        tgt_ref = inf_dir / f"{tgt_stem}_target_reference.wav"
        recon = inf_dir / f"{src_stem}_reconstructed.wav"

        if not orig.exists() or not tgt_ref.exists():
            continue

        emb_conv = scorer.embed(str(conv_path))
        emb_src = scorer.embed(str(orig))
        emb_tgt = scorer.embed(str(tgt_ref))

        r = {
            "pair": f"{src_stem} -> {tgt_stem}",
            "target_sim": scorer.cosine_sim(emb_conv, emb_tgt),
            "source_sim": scorer.cosine_sim(emb_conv, emb_src),
        }

        if recon.exists():
            emb_recon = scorer.embed(str(recon))
            r["recon_sim"] = scorer.cosine_sim(emb_recon, emb_src)

        results.append(r)

    return results


def print_results(name: str, results: list[dict]):
    if not results:
        return

    avg_tgt = np.mean([r["target_sim"] for r in results])
    avg_src = np.mean([r["source_sim"] for r in results])

    print(f"\n  {'Pair':<25s}  {'Target Sim':>10s}  {'Source Sim':>10s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}")
    for r in results:
        print(f"  {r['pair']:<25s}  {r['target_sim']:>10.4f}  {r['source_sim']:>10.4f}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}")
    print(f"  {'AVERAGE':<25s}  {avg_tgt:>10.4f}  {avg_src:>10.4f}")

    # Interpretation
    print(f"\n  Target similarity: {avg_tgt:.4f}  (higher = sounds more like target speaker)")
    print(f"  Source leakage:    {avg_src:.4f}  (lower  = less source speaker leaking through)")
    delta = avg_tgt - avg_src
    if delta > 0.05:
        print(f"  Verdict: converted voice is CLOSER to target than source (delta={delta:+.4f})")
    elif delta > -0.05:
        print(f"  Verdict: ambiguous -- similar distance to both (delta={delta:+.4f})")
    else:
        print(f"  Verdict: converted voice still sounds like SOURCE (delta={delta:+.4f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate voice conversion experiments")
    parser.add_argument("--step", type=int, default=30000,
                        help="Step number to evaluate val samples at (default: 30000)")
    parser.add_argument("--exp-dirs", type=str, nargs="*", default=None,
                        help="Experiment directories to evaluate (default: auto-find outputs_*)")
    parser.add_argument("--inference-dir", type=str, default=None,
                        help="Evaluate an inference output directory instead")
    parser.add_argument("--wer", action="store_true",
                        help="Evaluate WER (content preservation) using Whisper ASR")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-base",
                        help="Whisper model to use for ASR (default: openai/whisper-base)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- WER evaluation mode ---
    if args.wer:
        asr = ASRScorer(model_name=args.whisper_model, device=device)

        if args.exp_dirs:
            exp_dirs = [Path(d) for d in args.exp_dirs]
        else:
            # Auto-find all output directories with val samples
            exp_dirs = sorted(
                d for d in Path(".").glob("outputs_*")
                if (d / "val_samples").exists()
            )

        if not exp_dirs:
            print("No experiment directories found.")
            return

        print(f"\n{'=' * 70}")
        print(f"WER EVALUATION (Content Preservation)")
        print(f"ASR model: {args.whisper_model}")
        print(f"{'=' * 70}")

        wer_summary = []
        for exp_dir in exp_dirs:
            name = exp_dir.name
            print(f"\n--- {name} ---")
            results = evaluate_wer_val_samples(exp_dir, args.step, asr)
            if results:
                avg_wer = print_wer_results(name, results)
                wer_summary.append((name, avg_wer))

        # Final WER comparison
        if len(wer_summary) > 1:
            print(f"\n{'=' * 70}")
            print("WER COMPARISON SUMMARY")
            print(f"{'=' * 70}")
            print(f"  {'Experiment':<40s}  {'Avg WER':>8s}")
            print(f"  {'-' * 40}  {'-' * 8}")
            wer_summary.sort(key=lambda x: x[1])
            for name, w in wer_summary:
                marker = " <-- BEST" if w == wer_summary[0][1] else ""
                print(f"  {name:<40s}  {w:>7.1%}{marker}")
            print()
            if wer_summary[0][1] < 0.10:
                print("  Best model has < 10% WER -- excellent content preservation!")
            elif wer_summary[0][1] < 0.30:
                print("  Best model has < 30% WER -- decent content preservation.")
            else:
                print("  WARNING: All models have > 30% WER -- phonemes being destroyed.")
                print("  Consider relaxing instance norm or reducing SVD components.")
        return

    # --- Speaker similarity evaluation mode ---
    scorer = SpeakerSimilarity(device=device)

    # Inference mode
    if args.inference_dir:
        inf_dir = Path(args.inference_dir)
        print(f"\n{'=' * 60}")
        print(f"Evaluating inference: {inf_dir}")
        print(f"{'=' * 60}")
        results = evaluate_inference_dir(inf_dir, scorer)
        print_results(str(inf_dir), results)
        return

    # Training experiment mode
    if args.exp_dirs:
        exp_dirs = [Path(d) for d in args.exp_dirs]
    else:
        # Auto-find all outputs_30k_* directories
        exp_dirs = sorted(Path(".").glob("outputs_30k_*"))

    if not exp_dirs:
        print("No experiment directories found. Use --exp-dirs or --inference-dir.")
        return

    print(f"\n{'=' * 60}")
    print(f"Evaluating {len(exp_dirs)} experiments at step {args.step}")
    print(f"{'=' * 60}")

    # Collect summary for final comparison table
    summary = []

    for exp_dir in exp_dirs:
        name = exp_dir.name
        print(f"\n--- {name} ---")
        results = evaluate_val_samples(exp_dir, args.step, scorer)
        if results:
            print_results(name, results)
            avg_tgt = np.mean([r["target_sim"] for r in results])
            avg_src = np.mean([r["source_sim"] for r in results])
            summary.append((name, avg_tgt, avg_src))

    # Final comparison table
    if len(summary) > 1:
        print(f"\n{'=' * 60}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Experiment':<35s}  {'Tgt Sim':>7s}  {'Src Sim':>7s}  {'Delta':>7s}")
        print(f"  {'-' * 35}  {'-' * 7}  {'-' * 7}  {'-' * 7}")

        # Sort by target similarity (higher is better)
        summary.sort(key=lambda x: x[1], reverse=True)
        for name, tgt, src in summary:
            delta = tgt - src
            print(f"  {name:<35s}  {tgt:>7.4f}  {src:>7.4f}  {delta:>+7.4f}")

        print(f"\n  Best target similarity: {summary[0][0]}")
        print(f"  Lowest source leakage:  {min(summary, key=lambda x: x[2])[0]}")


if __name__ == "__main__":
    main()
