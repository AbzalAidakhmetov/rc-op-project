#!/usr/bin/env python3
"""Experimental trainer toward SOTA: prototype conditioning + optional adversarial loss.

Two hypotheses under test vs the pool-free L1 baseline (zero-shot tgt-sim ~0.39):
  H1 (prototypes): cross-attend source to M target prototype frames (restores the
     identity information the single 192-d ECAPA vector cannot carry).  --arch proto
  H2 (adversarial): add a speaker-conditioned frame discriminator + feature matching
     so the converter outputs land on the REAL WavLM-feature manifold instead of the
     over-smoothed kNN mean (fixes high source-leak).  --adv

Uses the precomputed cache (distill.precompute) for fast iteration.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import from_args
from distill.cached_gen import CachedDistillSampler
from models.converter import NeuralConverter
from models.proto_converter import PrototypeConverter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--arch", choices=["film", "proto"], default="proto")
    p.add_argument("--in-instancenorm", action="store_true",
                   help="instance-normalise source features (strip source-speaker stats)")
    p.add_argument("--adv", action="store_true", help="enable adversarial + feature-matching loss")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d-lr", type=float, default=2e-4)
    p.add_argument("--crop-sec", type=float, default=3.0)
    p.add_argument("--pool-utts", type=int, default=8)
    p.add_argument("--n-proto", type=int, default=64)
    p.add_argument("--n-real", type=int, default=64)
    p.add_argument("--lambda-adv", type=float, default=0.1)
    p.add_argument("--lambda-fm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--adv-start", type=int, default=1000, help="step to switch adversarial on")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


class FrameDiscriminator(nn.Module):
    """Speaker-conditioned per-frame discriminator over WavLM features."""

    def __init__(self, feat_dim=1024, spk_dim=192, hidden=512):
        super().__init__()
        self.spk = nn.Linear(spk_dim, hidden)
        self.l1 = nn.Linear(feat_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, feats, spk):
        # feats: (B, N, D) ; spk: (B, spk_dim)
        h = self.act(self.l1(feats))
        h = h + self.spk(spk).unsqueeze(1)
        h = self.act(self.l2(h))
        logits = self.head(h).squeeze(-1)   # (B, N)
        return logits, h                    # h = penultimate features for FM


def masked_l1_cos(pred, target, mask):
    denom = mask.sum().clamp_min(1.0)
    l1 = ((pred - target).abs().mean(-1) * mask).sum() / denom
    cos = F.cosine_similarity(pred, target, dim=-1)
    cos_loss = ((1 - cos) * mask).sum() / denom
    return l1, cos_loss, (cos * mask).sum() / denom


def main():
    args = parse_args()
    config = from_args(args)
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[exp] arch={args.arch} adv={args.adv} steps={args.steps} cache={args.cache_dir}")
    sampler = CachedDistillSampler(
        cache_dir=args.cache_dir, device=device, topk=config.knn_topk,
        crop_frames=int(round(args.crop_sec * 50)), pool_utts=args.pool_utts,
    )

    if args.arch == "proto":
        model = PrototypeConverter(feat_dim=config.wavlm_dim, hidden_dim=config.hidden_dim,
                                   spk_dim=config.ecapa_dim, num_blocks=config.num_res_blocks,
                                   in_instance_norm=args.in_instancenorm).to(device)
    else:
        model = NeuralConverter(feat_dim=config.wavlm_dim, hidden_dim=config.hidden_dim,
                                spk_dim=config.ecapa_dim, num_blocks=config.num_res_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[exp] generator params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = args.warmup_steps
    lr_lambda = lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min((s - warm) / max(1, args.steps - warm), 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    disc = dopt = None
    if args.adv:
        disc = FrameDiscriminator(feat_dim=config.wavlm_dim, spk_dim=config.ecapa_dim).to(device)
        dopt = torch.optim.AdamW(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9))

    start = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"]); start = int(ck.get("step", 0))
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        if "sched" in ck: sched.load_state_dict(ck["sched"])
        if disc is not None and ck.get("disc"): disc.load_state_dict(ck["disc"])
        print(f"[exp] resumed at {start}")

    def save(step):
        d = {"model": model.state_dict(), "step": step, "arch": args.arch,
             "n_proto": args.n_proto, "in_instance_norm": args.in_instancenorm,
             "config": vars(config),
             "opt": opt.state_dict(), "sched": sched.state_dict()}
        if disc is not None: d["disc"] = disc.state_dict()
        torch.save(d, out_dir / "converter.pt")

    run = []
    for step in range(start, args.steps):
        b = sampler.sample_batch(args.batch_size,
                                 n_proto=args.n_proto if args.arch == "proto" else 0,
                                 n_real=args.n_real if args.adv else 0)
        src, tgt, spk, lengths = b["source_feats"], b["target_feats"], b["spk_emb"], b["lengths"]
        B, T, _ = src.shape
        ar = torch.arange(T, device=device).unsqueeze(0)
        mask = (ar < lengths.unsqueeze(1)).float()                # (B,T)
        adv_on = args.adv and step >= args.adv_start

        # ---- generator ----
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            if args.arch == "proto":
                pred = model(src, b["protos"], spk, lengths=lengths)
            else:
                pred = model(src, spk, lengths=lengths)
            l1, cos_loss, cos_val = masked_l1_cos(pred.float(), tgt.float(), mask)
            g_loss = l1 + cos_loss
            g_adv = torch.tensor(0.0, device=device); g_fm = torch.tensor(0.0, device=device)
            if adv_on:
                fake_logits, fake_h = disc(pred.float(), spk)
                g_adv = -(fake_logits * mask).sum() / mask.sum().clamp_min(1.0)
                with torch.no_grad():
                    _, real_h = disc(b["real_tgt"].float(), spk)
                fake_mean = (fake_h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
                g_fm = F.l1_loss(fake_mean, real_h.mean(1))
                g_loss = g_loss + args.lambda_adv * g_adv + args.lambda_fm * g_fm
        if not torch.isfinite(g_loss):
            print(f"  [warn] non-finite g_loss step {step}; skip"); continue
        scaler.scale(g_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt); scaler.update(); sched.step()

        # ---- discriminator ----
        d_loss = torch.tensor(0.0, device=device)
        if adv_on:
            dopt.zero_grad(set_to_none=True)
            real = b["real_tgt"].float()
            rl, _ = disc(real, spk)
            fl, _ = disc(pred.detach().float(), spk)
            real_term = F.relu(1.0 - rl).mean()
            fake_term = (F.relu(1.0 + fl) * mask).sum() / mask.sum().clamp_min(1.0)
            d_loss = real_term + fake_term
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
            dopt.step()

        run.append(l1.item()); run = run[-200:]
        if step % args.log_every == 0:
            print(f"step {step:6d}/{args.steps} | L1 {l1.item():.4f} (avg {sum(run)/len(run):.4f}) "
                  f"cos {cos_val.item():.4f} | adv {g_adv.item():.3f} fm {g_fm.item():.3f} d {d_loss.item():.3f} "
                  f"| lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if args.save_every and step > 0 and step % args.save_every == 0:
            save(step)
    save(args.steps)
    print(f"[exp] saved {out_dir/'converter.pt'}")


if __name__ == "__main__":
    main()
