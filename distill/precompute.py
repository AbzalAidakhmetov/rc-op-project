#!/usr/bin/env python3
"""Precompute the distillation feature cache (one pass over the corpus).

The on-the-fly teacher in ``distill.dataset_gen`` re-runs ~9 WavLM-Large forwards
per batch item EVERY step, re-extracting the same utterances' features over and
over -> ~5-8 s/step, i.e. days for a real run. Since the WavLM-Large layer-6
features and the ECAPA embeddings of each utterance never change, we extract them
ONCE here and cache them. Training then becomes a cheap GPU cosine-kNN over cached
features + the tiny converter forward/backward (~50x faster).

Cache layout (under ``--cache-dir``):
    index.json            {sr, layer, dim, speakers: {spk: [utt_key, ...]}}
    feats/<utt_key>.pt     float16 (T, 1024) WavLM-Large layer-6 features
    ecapa.pt               {utt_key: float32 (192,)} ECAPA speaker embeddings

``utt_key`` = ``f"{speaker}__{path.stem}"``.

Usage:
    .venv/bin/python -m distill.precompute \
        --data-dir data/librispeech/LibriSpeech/dev-clean \
        --cache-dir cache/dev-clean
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from backbone.knnvc import KNNVC
from models.speaker import SpeakerEncoder
from data.dataset import AudioFolderDataset
from utils import load_audio


def parse_args():
    p = argparse.ArgumentParser(description="Precompute WavLM-L6 + ECAPA cache for distillation.")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--max-files-per-speaker", type=int, default=None)
    p.add_argument("--max-utt-sec", type=float, default=20.0,
                   help="Cap per-utterance length fed to WavLM (bounds 12GB OOM on long clips).")
    p.add_argument("--topk", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    cache_dir = Path(args.cache_dir)
    feats_dir = cache_dir / "feats"
    feats_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frozen teacher (WavLM-Large) + ECAPA on {device} ...")
    knnvc = KNNVC(device=device, topk=args.topk, prematched=True)
    spk_enc = SpeakerEncoder(device=device)
    sr = knnvc.sr

    dataset = AudioFolderDataset(
        root_dir=args.data_dir, sr=sr,
        max_speakers=args.max_speakers,
        max_files_per_speaker=args.max_files_per_speaker,
    )

    cap = args.max_utt_sec if args.max_utt_sec and args.max_utt_sec > 0 else None
    index = {"sr": sr, "layer": int(knnvc.layer), "dim": int(knnvc.dim) if hasattr(knnvc, "dim") else 1024,
             "speakers": {}}
    ecapa = {}

    total = sum(len(v) for v in dataset.speaker_files.values())
    pbar = tqdm(total=total, desc="Caching features")
    n_ok = 0
    for spk in dataset.speakers:
        keys = []
        for path in dataset.speaker_files[spk]:
            utt_key = f"{spk}__{Path(path).stem}"
            out_path = feats_dir / f"{utt_key}.pt"
            try:
                wav = load_audio(path, target_sr=sr, max_sec=cap)
                if wav.shape[0] < int(0.4 * sr):
                    pbar.update(1)
                    continue
                feats = knnvc.get_features(wav).float().half().cpu()      # (T,1024) fp16
                emb = spk_enc.encode(wav.to(device)).float().squeeze(0).cpu()  # (192,)
                torch.save(feats, out_path)
                ecapa[utt_key] = emb
                keys.append(utt_key)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {utt_key}: {e}")
            pbar.update(1)
        if keys:
            index["speakers"][spk] = keys
    pbar.close()

    torch.save(ecapa, cache_dir / "ecapa.pt")
    with open(cache_dir / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\nCached {n_ok} utterances from {len(index['speakers'])} speakers -> {cache_dir}")


if __name__ == "__main__":
    main()
