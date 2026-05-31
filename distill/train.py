#!/usr/bin/env python3
"""Train the pool-free NEURAL CONVERTER by distilling the kNN-VC teacher.

The kNN-VC backbone (``backbone.knnvc.KNNVC``) is the TEACHER: for a source
utterance and a random target speaker it produces kNN-matched target features
(averaged target-speaker neighbours) entirely in WavLM-Large space. We train the
parametric ``models.converter.NeuralConverter`` -- conditioned only on the
target speaker's ECAPA embedding (NO pool) -- to regress those features:

    converted = converter(source_feats, spk_emb)
    loss      = L1(converted, knn_target_feats)  [+ (1 - cosine) if enabled]

At inference the converter replaces the kNN lookup; its output is vocoded by the
SAME pretrained prematched HiFi-GAN, so it stays in-distribution for the vocoder.

12 GB-friendly: AMP autocast + GradScaler, batch_size 8, crop 3 s by default.

Example:
    .venv/bin/python -m distill.train --steps 50000 --data-dir \
        data/librispeech/LibriSpeech/dev-clean
    # tiny smoke run:
    .venv/bin/python -m distill.train --steps 50 --batch-size 2 \
        --max-speakers 4 --max-files-per-speaker 8 --pool-utts 4 --no-wandb
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import from_args
from backbone.knnvc import KNNVC
from models.speaker import SpeakerEncoder
from models.converter import NeuralConverter
from data.dataset import AudioFolderDataset
from distill.dataset_gen import make_distill_batch
from utils import load_audio, save_wav

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def parse_args():
    p = argparse.ArgumentParser(
        description="Distil the kNN-VC teacher into a pool-free neural converter."
    )
    p.add_argument("--steps", type=int, default=None, help="Training steps.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--crop-sec", type=float, default=None,
                   help="Source crop length in seconds (bounds memory).")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-speakers", type=int, default=None)
    p.add_argument("--max-files-per-speaker", type=int, default=None)
    p.add_argument("--pool-utts", type=int, default=8,
                   help="Target utterances concatenated into the kNN pool.")
    p.add_argument("--pool-sec", type=float, default=10.0,
                   help="Cap (sec) per target pool ref fed to WavLM (bounds OOM). "
                        "Use <=0 for whole clip.")
    p.add_argument("--ref-sec", type=float, default=10.0,
                   help="Cap (sec) on the ECAPA reference clip. Use <=0 for whole clip.")
    p.add_argument("--topk", type=int, default=None,
                   help="Neighbours averaged by the teacher (default Config.knn_topk).")
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--no-cosine-loss", dest="use_cosine_loss", action="store_false",
                   default=None, help="Disable the cosine term (L1 only).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--demo-every", type=int, default=1000,
                   help="Run a quick demo conversion every N steps (0 = off).")
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def masked_distill_loss(pred, target, lengths, use_cosine: bool):
    """L1 (+ optional 1-cosine) over valid (unpadded) frames only.

    pred / target: (B, T, D).  lengths: (B,) valid time steps per item.
    """
    B, T, _ = pred.shape
    # (B, T) boolean mask of real (non-pad) frames.
    ar = torch.arange(T, device=pred.device).unsqueeze(0)        # (1, T)
    mask = (ar < lengths.unsqueeze(1)).float()                   # (B, T)
    denom = mask.sum().clamp_min(1.0)

    # ---- L1 ----
    l1 = (pred - target).abs().mean(dim=-1)                      # (B, T)
    l1 = (l1 * mask).sum() / denom

    loss = l1
    cos_val = None
    if use_cosine:
        # 1 - cosine similarity per frame.
        cos = F.cosine_similarity(pred, target, dim=-1)          # (B, T)
        cos_loss = ((1.0 - cos) * mask).sum() / denom
        loss = loss + cos_loss
        cos_val = (cos * mask).sum() / denom
    return loss, l1.detach(), (cos_val.detach() if cos_val is not None else None)


@torch.no_grad()
def run_demo(converter, knnvc, spk_enc, dataset, device, out_path, crop_sec):
    """Convert a held-out source toward a different speaker and vocode it.

    Proves the converter output is vocoder-in-distribution end to end.
    """
    converter.eval()
    speakers = dataset.speakers
    src_spk = speakers[0]
    tgt_spk = speakers[-1] if len(speakers) > 1 else speakers[0]
    src_path = dataset.speaker_files[src_spk][0]
    tgt_path = dataset.speaker_files[tgt_spk][0]

    src_wav = load_audio(src_path, target_sr=knnvc.sr, max_sec=max(crop_sec, 4.0))
    src_feats = knnvc.get_features(src_wav).float()                   # (T, 1024)
    ref_wav = load_audio(tgt_path, target_sr=knnvc.sr).to(device)
    spk_emb = spk_enc.encode(ref_wav).float()                        # (1, 192)

    converted = converter(src_feats.unsqueeze(0).to(device), spk_emb)[0]  # (T, 1024)
    wav = knnvc.vocode(converted)
    save_wav(out_path, wav, sr=knnvc.sr)
    converter.train()
    return out_path, wav.shape[-1] / knnvc.sr


def main():
    args = parse_args()
    config = from_args(args)
    device = config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu"
    if not torch.cuda.is_available() and device == "cuda":
        print("[warn] CUDA unavailable; falling back to CPU.")
        device = "cpu"
        config.device = "cpu"

    topk = args.topk if args.topk is not None else config.knn_topk
    use_wandb = HAS_WANDB and not args.no_wandb

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    demo_dir = out_dir / "distill_demos"
    demo_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NeuralKNN-VC :: distilling kNN-VC teacher -> pool-free converter")
    print("=" * 70)
    print(f"  device       : {device}")
    print(f"  data_dir     : {config.data_dir}")
    print(f"  steps        : {config.steps}")
    print(f"  batch_size   : {config.batch_size}")
    print(f"  lr           : {config.lr}")
    print(f"  crop_sec     : {config.crop_sec}")
    print(f"  topk / pool  : {topk} / {args.pool_utts}")
    print(f"  cosine loss  : {config.use_cosine_loss}")
    print(f"  output_dir   : {out_dir}")
    print()

    # ---- frozen teacher + speaker encoder ----
    print("Loading frozen kNN-VC teacher (WavLM-Large + prematched HiFi-GAN) ...")
    knnvc = KNNVC(
        device=device,
        topk=topk,
        prematched=True,
    )
    print("Loading frozen ECAPA speaker encoder ...")
    spk_enc = SpeakerEncoder(device=device)

    # ---- dataset (paths only; features come from the teacher on the fly) ----
    dataset = AudioFolderDataset(
        root_dir=config.data_dir,
        crop_sec=config.crop_sec,
        sr=config.wavlm_sr,
        max_speakers=config.max_speakers,
        max_files_per_speaker=config.max_files_per_speaker,
    )

    # ---- trainable converter ----
    converter = NeuralConverter(
        feat_dim=config.wavlm_dim,
        hidden_dim=config.hidden_dim,
        spk_dim=config.ecapa_dim,
        num_blocks=config.num_res_blocks,
    ).to(device)
    converter.train()
    n_params = sum(p.numel() for p in converter.parameters())
    print(f"NeuralConverter: {n_params:,} trainable parameters\n")

    # ---- optimiser: AdamW + linear warmup -> cosine decay ----
    optimizer = torch.optim.AdamW(converter.parameters(), lr=config.lr, weight_decay=0.01)
    warmup = config.warmup_steps

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, config.steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if use_wandb:
        wandb.init(project="neural-knn-vc", config=vars(config))

    # ---- training loop ----
    running = []
    for step in range(config.steps):
        batch = make_distill_batch(
            knnvc, spk_enc, dataset, device,
            batch_size=config.batch_size,
            topk=topk,
            pool_utts=args.pool_utts,
            crop_sec=config.crop_sec,
            pool_sec=(args.pool_sec if args.pool_sec and args.pool_sec > 0 else None),
            ref_sec=(args.ref_sec if args.ref_sec and args.ref_sec > 0 else None),
        )
        source_feats = batch["source_feats"]   # (B, T, 1024)
        spk_emb = batch["spk_emb"]             # (B, 192)
        target_feats = batch["target_feats"]   # (B, T, 1024)
        lengths = batch["lengths"]             # (B,)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            converted = converter(source_feats, spk_emb, lengths=lengths)
            loss, l1, cos = masked_distill_loss(
                converted.float(), target_feats.float(), lengths, config.use_cosine_loss
            )

        scaler.scale(loss).backward()
        if config.grad_clip and config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(converter.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        lv = loss.item()
        running.append(lv)
        if len(running) > 200:
            running.pop(0)

        if step % args.log_every == 0 or step == config.steps - 1:
            avg = sum(running) / len(running)
            lr_now = scheduler.get_last_lr()[0]
            cos_s = f" cos={cos.item():.4f}" if cos is not None else ""
            print(f"step {step:6d}/{config.steps} | loss {lv:.4f} "
                  f"(avg {avg:.4f}) | L1 {l1.item():.4f}{cos_s} | lr {lr_now:.2e}")
            if use_wandb:
                log = {"loss": lv, "avg_loss": avg, "l1": l1.item(), "lr": lr_now}
                if cos is not None:
                    log["cosine"] = cos.item()
                wandb.log(log, step=step)

        if args.demo_every and step > 0 and step % args.demo_every == 0:
            try:
                path, dur = run_demo(
                    converter, knnvc, spk_enc, dataset, device,
                    demo_dir / f"demo_step{step:06d}.wav", config.crop_sec,
                )
                print(f"  [demo] {path} ({dur:.2f}s)")
            except Exception as e:  # noqa: BLE001 - demo must never kill training
                print(f"  [demo skipped] {e}")

        if args.save_every and step > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / "converter.pt"
            torch.save({"model": converter.state_dict(), "config": vars(config)}, ckpt_path)

    # ---- final save ----
    ckpt_path = out_dir / "converter.pt"
    torch.save({"model": converter.state_dict(), "config": vars(config)}, ckpt_path)
    print(f"\nSaved converter checkpoint: {ckpt_path}")

    if args.demo_every:
        try:
            path, dur = run_demo(
                converter, knnvc, spk_enc, dataset, device,
                demo_dir / "demo_final.wav", config.crop_sec,
            )
            print(f"Final demo: {path} ({dur:.2f}s)")
        except Exception as e:  # noqa: BLE001
            print(f"Final demo skipped: {e}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
