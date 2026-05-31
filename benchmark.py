#!/usr/bin/env python3
"""NeuralKNN-VC benchmark.

For NUM_PAIRS held-out (source-speaker, target-speaker) pairs, run BOTH backends
(kNN-VC teacher and the pool-free neural converter) and report a markdown table:

    backend | ECAPA target-sim | ECAPA source-leak | delta | RTF

  - ECAPA target-sim : cosine(converted, target reference)  -- higher is better.
  - ECAPA source-leak: cosine(converted, source utterance)  -- lower is better.
  - delta            : target-sim - source-leak             -- higher is better.
  - RTF              : real-time factor of the CONVERSION region only, measured
                       apples-to-apples across backends. Reference/pool encoding
                       (WavLM over target refs, ECAPA) and source feature
                       extraction are amortizable and are precomputed OUTSIDE the
                       timed block for BOTH backends, so the timed region is
                       kNN-match+vocode (knn) vs converter+vocode (neural).
                       wall_time / audio_duration; <1 is faster than real time.

The OLD from-scratch-mel system reported target-sim ~0.38 -- that is the baseline
to beat. State-of-the-art any-to-any zero-shot VC sits around 0.6-0.85.

Optional WER via faster-whisper (extra 'eval'); skipped cleanly if not installed.

Examples
--------
    .venv/bin/python benchmark.py \
        --converter outputs/converter.pt \
        --data-dir data/librispeech/LibriSpeech/dev-clean \
        --num-pairs 5 --output-dir outputs
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import from_args
from utils import load_audio, save_wav
from backbone.knnvc import KNNVC
from models.speaker import SpeakerEncoder

OLD_BASELINE_TARGET_SIM = 0.38  # old from-scratch-mel system, reported.


def parse_args():
    p = argparse.ArgumentParser(description="NeuralKNN-VC benchmark (knn vs neural)")
    p.add_argument("--converter", type=str, default=None,
                   help="Converter checkpoint (.pt). If omitted, only knn is benchmarked.")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Root with <speaker>/.../*.flac|*.wav.")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--num-pairs", type=int, default=5)
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-utts", type=int, default=8,
                   help="Max target utterances used to build the kNN pool / ref.")
    return p.parse_args()


def _glob_audio(d: Path):
    files = []
    for ext in ("*.flac", "*.wav"):
        files.extend(sorted(d.rglob(ext)))
    return files


def collect_speakers(data_dir: str):
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")
    spk_files = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = _glob_audio(d)
        if files:
            spk_files[d.name] = files
    if len(spk_files) < 2:
        raise RuntimeError(f"Need >=2 speakers with audio under {data_dir}.")
    return spk_files


def make_pairs(spk_files, num_pairs, pool_utts, rng):
    """Build (src_path, [tgt_ref_paths]) pairs across distinct speakers."""
    speakers = sorted(spk_files.keys())
    pairs = []
    used = set()
    attempts = 0
    while len(pairs) < num_pairs and attempts < num_pairs * 50:
        attempts += 1
        src_spk = rng.choice(speakers)
        tgt_spk = rng.choice([s for s in speakers if s != src_spk])
        key = (src_spk, tgt_spk)
        if key in used:
            continue
        used.add(key)
        src_path = rng.choice(spk_files[src_spk])
        tgt_refs = spk_files[tgt_spk][:pool_utts]
        if not tgt_refs:
            continue
        pairs.append({
            "src_spk": src_spk, "tgt_spk": tgt_spk,
            "src_path": src_path, "tgt_refs": tgt_refs,
        })
    return pairs


@torch.no_grad()
def ecapa_emb(spk_enc, wav, device):
    w = wav.to(device)
    e = spk_enc.encode(w)
    if e.dim() == 1:
        e = e.unsqueeze(0)
    return e  # (1, 192)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).item()


def try_load_whisper(device):
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return None
    try:
        compute = "float16" if device == "cuda" else "int8"
        return WhisperModel("base.en", device=device, compute_type=compute)
    except Exception:
        try:
            return WhisperModel("base.en", device="cpu", compute_type="int8")
        except Exception:
            return None


def transcribe(model, wav, sr):
    import numpy as np
    audio = wav.detach().cpu().float().reshape(-1).numpy().astype(np.float32)
    segs, _ = model.transcribe(audio, language="en", beam_size=1)
    return " ".join(s.text for s in segs).strip()


def wer(ref: str, hyp: str) -> float:
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r:
        return 0.0 if not h else 1.0
    # Levenshtein over words.
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / len(r)


def main():
    args = parse_args()
    cfg = from_args(args)
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable -> CPU.")
        device = "cpu"
    if not args.data_dir:
        args.data_dir = cfg.data_dir
    topk = args.topk if args.topk is not None else cfg.knn_topk
    sr = cfg.wavlm_sr
    rng = random.Random(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "benchmark_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("Loading kNN-VC backbone + ECAPA ...")
    knnvc = KNNVC(device=device)
    spk_enc = SpeakerEncoder(device=device)

    backends = ["knn"]
    converter = None
    if args.converter:
        from models.converter import NeuralConverter
        ckpt = torch.load(args.converter, map_location="cpu")
        ccfg = ckpt.get("config", {})
        converter = NeuralConverter(
            feat_dim=ccfg.get("wavlm_dim", cfg.wavlm_dim),
            hidden_dim=ccfg.get("hidden_dim", cfg.hidden_dim),
            spk_dim=ccfg.get("ecapa_dim", cfg.ecapa_dim),
            num_blocks=ccfg.get("num_res_blocks", cfg.num_res_blocks),
        )
        converter.load_state_dict(ckpt["model"])
        converter = converter.to(device).eval()
        backends.append("neural")
    else:
        print("No --converter given: benchmarking kNN backend only.")

    whisper = try_load_whisper(device)
    use_wer = whisper is not None
    print(f"WER (faster-whisper): {'enabled' if use_wer else 'skipped (not installed)'}")

    spk_files = collect_speakers(args.data_dir)
    pairs = make_pairs(spk_files, args.num_pairs, args.pool_utts, rng)
    print(f"Benchmarking {len(pairs)} pairs x {len(backends)} backends.\n")

    # accumulators per backend
    acc = {b: {"tsim": [], "leak": [], "delta": [], "rtf": [], "wer": []} for b in backends}

    # RTF measures only the apples-to-apples CONVERSION region. Reference/pool
    # encoding (WavLM over target refs, ECAPA over the ref) and SOURCE feature
    # extraction are amortizable / shared and are precomputed OUTSIDE the timed
    # block for BOTH backends, so the two RTFs are directly comparable:
    #   knn    : timed = kNN match + vocode
    #   neural : timed = converter forward + vocode
    @torch.no_grad()
    def run_knn(precomp):
        converted = knnvc.match_features(precomp["source_feats"], precomp["pool"], topk=topk)
        return knnvc.vocode(converted)

    @torch.no_grad()
    def run_neural(precomp):
        converted = converter(precomp["source_feats"].unsqueeze(0), precomp["spk_emb"])[0]
        return knnvc.vocode(converted)

    runners = {"knn": run_knn, "neural": run_neural}

    for idx, pair in enumerate(pairs):
        # Reference embeddings (target ref + source) computed once per pair.
        src_wav = load_audio(pair["src_path"], target_sr=sr)
        tgt_wav = load_audio(pair["tgt_refs"][0], target_sr=sr)
        src_emb = ecapa_emb(spk_enc, src_wav, device)
        tgt_emb = ecapa_emb(spk_enc, tgt_wav, device)

        src_text = transcribe(whisper, src_wav, sr) if use_wer else None

        # ---- precompute (untimed) reference/pool + source features ----
        with torch.no_grad():
            source_feats = knnvc.get_features(pair["src_path"]).to(device)  # (T,1024)
            precomp = {"source_feats": source_feats}
            if "knn" in backends:
                precomp["pool"] = knnvc.build_pool(pair["tgt_refs"])        # (Np,1024)
            if "neural" in backends:
                ref_wav = load_audio(pair["tgt_refs"][0], target_sr=sr).to(device)
                spk_emb = spk_enc.encode(ref_wav)
                if spk_emb.dim() == 1:
                    spk_emb = spk_emb.unsqueeze(0)
                precomp["spk_emb"] = spk_emb

        for b in backends:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            wav = runners[b](precomp)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            wav = wav.detach().cpu().float().reshape(-1)
            dur = wav.shape[0] / sr
            rtf = elapsed / max(dur, 1e-9)

            conv_emb = ecapa_emb(spk_enc, wav, device)
            tsim = cos(conv_emb, tgt_emb)
            leak = cos(conv_emb, src_emb)

            acc[b]["tsim"].append(tsim)
            acc[b]["leak"].append(leak)
            acc[b]["delta"].append(tsim - leak)
            acc[b]["rtf"].append(rtf)

            if use_wer:
                hyp = transcribe(whisper, wav, sr)
                acc[b]["wer"].append(wer(src_text, hyp))

            save_wav(str(wav_dir / f"pair{idx}_{pair['src_spk']}_to_{pair['tgt_spk']}_{b}.wav"),
                     wav, sr=sr)
            print(f"[{idx}] {pair['src_spk']}->{pair['tgt_spk']} {b:6s} "
                  f"tsim={tsim:.3f} leak={leak:.3f} delta={tsim-leak:+.3f} rtf={rtf:.3f}")

    # ---- Build markdown ----
    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    lines = []
    lines.append("# NeuralKNN-VC Benchmark\n")
    lines.append(f"- Data dir: `{args.data_dir}`")
    lines.append(f"- Pairs: {len(pairs)}  |  topk: {topk}  |  device: {device}")
    lines.append(f"- Old from-scratch-mel baseline target-sim (to beat): "
                 f"**{OLD_BASELINE_TARGET_SIM:.2f}**")
    lines.append("")
    header = "| backend | ECAPA target-sim | ECAPA source-leak | delta | RTF (convert-only) |"
    sep = "|---|---|---|---|---|"
    if use_wer:
        header = header[:-1] + " WER |"
        sep = sep[:-1] + "---|"
    lines.append(header)
    lines.append(sep)
    for b in backends:
        a = acc[b]
        row = (f"| {b} | {mean(a['tsim']):.3f} | {mean(a['leak']):.3f} | "
               f"{mean(a['delta']):+.3f} | {mean(a['rtf']):.3f} |")
        if use_wer:
            row = row[:-1] + f" {mean(a['wer']) * 100:.1f}% |"
        lines.append(row)
    lines.append("")
    lines.append("Higher target-sim and delta are better; lower source-leak and RTF are better.")
    lines.append(f"Reference: old system target-sim {OLD_BASELINE_TARGET_SIM:.2f}; "
                 f"SOTA any-to-any zero-shot VC ~0.6-0.85.")
    md = "\n".join(lines) + "\n"

    md_path = out_dir / "benchmark.md"
    md_path.write_text(md)
    print("\n" + md)
    print(f"Saved: {md_path.resolve()}")
    print(f"Converted wavs: {wav_dir.resolve()}")


if __name__ == "__main__":
    main()
