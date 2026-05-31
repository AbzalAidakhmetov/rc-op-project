#!/usr/bin/env python3
"""NeuralKNN-VC inference CLI.

Two backends, both vocoded by the SAME pretrained prematched HiFi-GAN so their
outputs stay in-distribution for the vocoder:

  --backend knn     (default)  pure kNN-VC: SOTA-quality, NO training.
      source --WavLM L6--> query feats
      target ref(s) --WavLM L6--> pool
      per-frame cosine-kNN (topk) average into pool -> converted feats
      converted feats --prematched HiFi-GAN--> wav

  --backend neural             the novel POOL-FREE converter (distilled from kNN-VC):
      source --WavLM L6--> source feats
      target ref --ECAPA--> 192-d speaker embedding (NO pool needed)
      NeuralConverter(source_feats, spk_emb) -> converted feats (single forward pass)
      converted feats --prematched HiFi-GAN--> wav

Everything runs at 16 kHz.

Examples
--------
    # instant SOTA-quality demo, no training:
    .venv/bin/python infer.py --backend knn \
        --source a.wav --target b.wav --output-dir outputs

    # multiple reference utterances build a richer target pool (knn backend):
    .venv/bin/python infer.py --backend knn \
        --source a.wav --target ref1.wav ref2.wav ref3.wav

    # pool-free neural converter (needs a trained checkpoint):
    .venv/bin/python infer.py --backend neural \
        --source a.wav --target b.wav --converter outputs/converter.pt

    # auto-pick two speakers from a data dir if no --source/--target given:
    .venv/bin/python infer.py --backend knn --data-dir data/librispeech/LibriSpeech/dev-clean
"""

import argparse
import time
from pathlib import Path

import torch

from config import from_args
from utils import load_audio, save_wav
from backbone.knnvc import KNNVC


def parse_args():
    p = argparse.ArgumentParser(description="NeuralKNN-VC inference")
    p.add_argument("--backend", choices=["knn", "neural"], default="knn",
                   help="knn = pure kNN-VC (no training); neural = pool-free converter")
    p.add_argument("--source", type=str, default=None,
                   help="Source (content) wav.")
    p.add_argument("--target", type=str, nargs="+", default=None,
                   help="Target reference wav(s). knn uses all as the pool; "
                        "neural uses the first for the ECAPA embedding.")
    p.add_argument("--converter", type=str, default=None,
                   help="Converter checkpoint (.pt) for --backend neural.")
    p.add_argument("--topk", type=int, default=None,
                   help="kNN top-k (knn backend). Default from config (4).")
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--data-dir", type=str, default=None,
                   help="If --source/--target omitted, auto-pick 2 speakers from here.")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _glob_audio(d: Path):
    files = []
    for ext in ("*.flac", "*.wav"):
        files.extend(sorted(d.rglob(ext)))
    return files


def autopick_pair(data_dir: str):
    """Pick (source_path, [target_path]) from the first two speaker dirs."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")
    spk_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    spk_dirs = [d for d in spk_dirs if _glob_audio(d)]
    if len(spk_dirs) < 2:
        raise RuntimeError(
            f"Need >=2 speaker dirs with audio under {data_dir}, found {len(spk_dirs)}."
        )
    src_files = _glob_audio(spk_dirs[0])
    tgt_files = _glob_audio(spk_dirs[1])
    print(f"Auto-picked source speaker '{spk_dirs[0].name}', "
          f"target speaker '{spk_dirs[1].name}'")
    return str(src_files[0]), [str(tgt_files[0])]


def main():
    args = parse_args()
    cfg = from_args(args)
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable -> falling back to CPU.")
        device = "cpu"
    topk = args.topk if args.topk is not None else cfg.knn_topk

    # Resolve source / target.
    if args.source and args.target:
        source_path = args.source
        ref_paths = list(args.target)
    elif args.data_dir:
        source_path, ref_paths = autopick_pair(args.data_dir)
    else:
        raise SystemExit(
            "Provide --source and --target, or --data-dir to auto-pick a pair."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"NeuralKNN-VC inference  |  backend={args.backend}  device={device}")
    print(f"  source : {source_path}")
    print(f"  target : {ref_paths}")
    print("=" * 64)

    print("Loading kNN-VC backbone (WavLM-Large + prematched HiFi-GAN) ...")
    knnvc = KNNVC(device=device)

    sr = cfg.wavlm_sr
    src_name = Path(source_path).stem
    tgt_name = Path(ref_paths[0]).stem

    t0 = time.time()
    if args.backend == "knn":
        wav = knnvc.convert(source_path, ref_paths, topk=topk)
    else:
        if not args.converter:
            raise SystemExit("--backend neural requires --converter CKPT.")
        from models.converter import NeuralConverter
        from models.speaker import SpeakerEncoder

        print(f"Loading converter checkpoint: {args.converter}")
        ckpt = torch.load(args.converter, map_location="cpu")
        ccfg = ckpt.get("config", {})
        feat_dim = ccfg.get("wavlm_dim", cfg.wavlm_dim)
        hidden_dim = ccfg.get("hidden_dim", cfg.hidden_dim)
        spk_dim = ccfg.get("ecapa_dim", cfg.ecapa_dim)
        num_blocks = ccfg.get("num_res_blocks", cfg.num_res_blocks)
        converter = NeuralConverter(
            feat_dim=feat_dim, hidden_dim=hidden_dim,
            spk_dim=spk_dim, num_blocks=num_blocks,
        )
        converter.load_state_dict(ckpt["model"])
        converter = converter.to(device).eval()

        spk_enc = SpeakerEncoder(device=device)

        with torch.no_grad():
            source_feats = knnvc.get_features(source_path).to(device)  # (T, 1024)
            ref_wav = load_audio(ref_paths[0], target_sr=sr).to(device)
            spk_emb = spk_enc.encode(ref_wav)                          # (1, 192)
            if spk_emb.dim() == 1:
                spk_emb = spk_emb.unsqueeze(0)
            converted = converter(source_feats.unsqueeze(0), spk_emb)[0]  # (T, 1024)
            wav = knnvc.vocode(converted)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    wav = wav.detach().cpu().float().reshape(-1)
    duration = wav.shape[0] / sr
    rtf = elapsed / max(duration, 1e-9)

    out_path = out_dir / f"{src_name}_to_{tgt_name}_{args.backend}.wav"
    save_wav(str(out_path), wav, sr=sr)

    print("-" * 64)
    print(f"Saved : {out_path.resolve()}")
    print(f"Duration : {duration:.2f}s   |   wall time : {elapsed:.2f}s   |   RTF : {rtf:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
