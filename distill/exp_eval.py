#!/usr/bin/env python3
"""Evaluate an experimental converter (proto/film, optional adv) zero-shot.

Mirrors benchmark.py's pair sampling + ECAPA cosine so numbers are directly
comparable to run1/run2. Reports kNN baseline vs the neural converter on the
same pairs. For the proto arch, target prototypes are built from the reference
audio's WavLM frames (a few seconds) -- no full pool at inference.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from config import from_args
from backbone.knnvc import KNNVC
from models.speaker import SpeakerEncoder
from models.converter import NeuralConverter
from models.proto_converter import PrototypeConverter
from benchmark import collect_speakers, make_pairs, ecapa_emb, cos
from utils import load_audio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--converter", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--num-pairs", type=int, default=30)
    p.add_argument("--n-proto", type=int, default=64)
    p.add_argument("--ref-sec", type=float, default=8.0, help="seconds of reference for protos/ECAPA")
    p.add_argument("--pool-utts", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tag", type=str, default="exp")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = from_args(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    sr = cfg.wavlm_sr
    rng = random.Random(args.seed)

    ck = torch.load(args.converter, map_location=device)
    arch = ck.get("arch", "film")
    n_proto = args.n_proto  # CLI controls eval-time prototype count (probe train/eval mismatch)
    if arch == "proto":
        model = PrototypeConverter(feat_dim=cfg.wavlm_dim, hidden_dim=cfg.hidden_dim,
                                   spk_dim=cfg.ecapa_dim, num_blocks=cfg.num_res_blocks,
                                   in_instance_norm=ck.get("in_instance_norm", False)).to(device)
    else:
        model = NeuralConverter(feat_dim=cfg.wavlm_dim, hidden_dim=cfg.hidden_dim,
                                spk_dim=cfg.ecapa_dim, num_blocks=cfg.num_res_blocks).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"[eval] arch={arch} n_proto={n_proto} step={ck.get('step')}")

    knnvc = KNNVC(device=device); spk_enc = SpeakerEncoder(device=device)
    spk_files = collect_speakers(args.data_dir)
    pairs = make_pairs(spk_files, args.num_pairs, args.pool_utts, rng)

    agg = {"knn": {"t": [], "s": []}, "neural": {"t": [], "s": []}}
    for i, pr in enumerate(pairs):
        src_wav = load_audio(pr["src_path"], target_sr=sr)
        src_feats = knnvc.get_features(src_wav).float()                       # (T,1024)
        ref_wavs = [load_audio(p, target_sr=sr, max_sec=args.ref_sec) for p in pr["tgt_refs"]]
        pool = knnvc.build_pool(ref_wavs).float()                            # (Np,1024)
        ref_for_emb = load_audio(pr["tgt_refs"][0], target_sr=sr, max_sec=args.ref_sec).to(device)
        spk = spk_enc.encode(ref_for_emb).float()
        if spk.dim() == 1: spk = spk.unsqueeze(0)

        # speaker embeddings of references (target) and source for sim/leak
        tgt_emb = ecapa_emb(spk_enc, load_audio(pr["tgt_refs"][0], target_sr=sr), device)
        src_emb = ecapa_emb(spk_enc, src_wav, device)

        # kNN baseline
        knn_feats = knnvc.match_features(src_feats, pool, topk=cfg.knn_topk)
        knn_wav = knnvc.vocode(knn_feats)
        ke = ecapa_emb(spk_enc, knn_wav, device)
        agg["knn"]["t"].append(cos(ke, tgt_emb)); agg["knn"]["s"].append(cos(ke, src_emb))

        # neural
        if arch == "proto":
            idx = torch.randint(0, pool.shape[0], (n_proto,), device=device)
            protos = pool[idx].unsqueeze(0)
            conv = model(src_feats.unsqueeze(0).to(device), protos, spk)[0]
        else:
            conv = model(src_feats.unsqueeze(0).to(device), spk)[0]
        nwav = knnvc.vocode(conv)
        ne = ecapa_emb(spk_enc, nwav, device)
        agg["neural"]["t"].append(cos(ne, tgt_emb)); agg["neural"]["s"].append(cos(ne, src_emb))
        if i % 5 == 0:
            print(f"  [{i}] {pr['src_spk']}->{pr['tgt_spk']} "
                  f"knn t={agg['knn']['t'][-1]:.3f} neural t={agg['neural']['t'][-1]:.3f}", flush=True)

    print(f"\n=== {args.tag} | {len(pairs)} pairs | arch={arch} ===")
    print("| backend | tgt-sim | src-leak | delta |")
    print("|---|---|---|---|")
    for bk in ("knn", "neural"):
        t = sum(agg[bk]["t"]) / len(agg[bk]["t"]); s = sum(agg[bk]["s"]) / len(agg[bk]["s"])
        print(f"| {bk} | {t:.3f} | {s:.3f} | {t - s:+.3f} |")


if __name__ == "__main__":
    main()
