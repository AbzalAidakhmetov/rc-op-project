#!/usr/bin/env python3
"""
Training script for Voice Conversion with Rectified Flow Matching.

Supports two methods:
- baseline: Standard CFM from Gaussian noise
- sg_flow: Rectified Flow from orthogonally projected content subspace

Training Phases:
- Phase A (First 2k steps): Train ONLY the Decoder (freeze Flow)
- Phase B (Remaining steps): Train both Flow and Decoder

Usage:
    python train.py --mode sg_flow --data_dir ./preprocessed
"""

import argparse
import os
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import Config
from data.dataset import create_dataloader
from models.flow_matching import create_flow_model
from models.decoder import WavLMToMelDecoder
from models.system import VoiceConversionSystem
from models.projection import OrthogonalProjection
from utils.logging import setup_logger


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(
    model, batch, optimizer, device, grad_clip, scaler,
    train_flow=True, train_decoder=True, mode="sg_flow", use_amp=True
):
    """Single training step with AMP support."""
    model.train()
    optimizer.zero_grad()
    
    target_wavlm = batch["target_wavlm"].to(device)
    target_spk = batch["target_spk"].to(device)
    target_mel = batch["target_mel"].to(device)
    target_mask = batch["target_mask"].to(device)
    
    with autocast(enabled=use_amp):
        outputs = model.compute_loss(
            x1=target_wavlm,
            target_spk=target_spk,
            target_mel=target_mel,
            mask=target_mask,
            mode=mode,
            train_flow=train_flow,
            train_decoder=train_decoder,
        )
        loss = outputs["loss"]
    
    # Backward with gradient scaling for AMP
    scaler.scale(loss).backward()
    
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    
    return {
        "loss": loss.item(),
        "flow_loss": outputs["flow_loss"].item() if torch.is_tensor(outputs["flow_loss"]) else outputs["flow_loss"],
        "decoder_loss": outputs["decoder_loss"].item() if torch.is_tensor(outputs["decoder_loss"]) else outputs["decoder_loss"],
    }


@torch.no_grad()
def validate(model, val_loader, device, mode="sg_flow", use_amp=True):
    """Validation loop."""
    model.eval()
    total_loss, total_flow, total_dec, n = 0, 0, 0, 0
    
    for batch in val_loader:
        target_wavlm = batch["target_wavlm"].to(device)
        target_spk = batch["target_spk"].to(device)
        target_mel = batch["target_mel"].to(device)
        target_mask = batch["target_mask"].to(device)
        
        with autocast(enabled=use_amp):
            out = model.compute_loss(target_wavlm, target_spk, target_mel, target_mask, mode)
        
        total_loss += out["loss"].item()
        total_flow += out["flow_loss"].item() if torch.is_tensor(out["flow_loss"]) else out["flow_loss"]
        total_dec += out["decoder_loss"].item() if torch.is_tensor(out["decoder_loss"]) else out["decoder_loss"]
        n += 1
    
    return {"val_loss": total_loss/n, "val_flow": total_flow/n, "val_dec": total_dec/n}


def freeze_module(module):
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def main():
    parser = argparse.ArgumentParser(description="Train Voice Conversion Flow Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Preprocessed data directory")
    parser.add_argument("--projection_path", type=str, default=None, help="SVD projection matrix path")
    
    # Mode selection (as per prompt)
    parser.add_argument("--mode", type=str, default="sg_flow", 
                       choices=["baseline", "sg_flow"],
                       help="Training mode: baseline (CFM) or sg_flow")
    
    # Training phases
    parser.add_argument("--num_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--phase_a_steps", type=int, default=2000, 
                       help="Phase A steps (decoder only training)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="LR warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dim")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Mixed precision
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    
    # Logging
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--log_interval", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_interval", type=int, default=2000, help="Save every N steps")
    parser.add_argument("--eval_interval", type=int, default=1000, help="Evaluate every N steps")
    
    # W&B
    parser.add_argument("--wandb_project", type=str, default="vc-flow-matching", help="W&B project")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    logger = setup_logger("train", args.log_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    use_amp = not args.no_amp and torch.cuda.is_available()
    logger.info(f"Device: {device}, Mode: {args.mode}, AMP: {use_amp}")
    
    config = Config()
    
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project, 
            name=f"{args.mode}_{args.num_steps}steps",
            config=vars(args)
        )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, train_ds = create_dataloader(
        args.data_dir, "train", args.batch_size, args.num_workers
    )
    val_loader, val_ds = create_dataloader(
        args.data_dir, "val", args.batch_size, args.num_workers
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Handle projection path for sg_flow
    projection = None
    if args.mode == "sg_flow":
        if args.projection_path is None:
            default_path = Path(args.data_dir) / "projection_matrix.pt"
            if default_path.exists():
                args.projection_path = str(default_path)
            else:
                raise ValueError("--projection_path required for sg_flow mode")
        projection = OrthogonalProjection(projection_path=args.projection_path)
        logger.info(f"Loaded projection matrix from {args.projection_path}")
    
    # Create model
    logger.info("Creating model...")
    flow_model = create_flow_model(
        method=args.mode,
        d_input=config.WAVLM_DIM,
        d_model=args.d_model,
        d_spk=config.d_spk,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        projection_path=args.projection_path,
    )
    
    decoder = WavLMToMelDecoder(
        d_wavlm=config.WAVLM_DIM,
        d_spk=config.d_spk,
        d_hidden=config.decoder_hidden_dim,
        n_mels=config.n_mels,
        num_layers=config.decoder_num_layers,
    )
    
    model = VoiceConversionSystem(flow_model, decoder, projection).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.num_steps)
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=use_amp)
    
    # Training loop
    logger.info("="*60)
    logger.info("TRAINING PHASES:")
    logger.info(f"  Phase A (steps 1-{args.phase_a_steps}): Decoder ONLY (Flow frozen)")
    logger.info(f"  Phase B (steps {args.phase_a_steps+1}-{args.num_steps}): Flow + Decoder")
    logger.info("="*60)
    
    step = 0
    best_val = float("inf")
    train_iter = iter(train_loader)
    pbar = tqdm(total=args.num_steps, desc="Training")
    running = {"loss": 0, "flow": 0, "dec": 0}
    
    # Start in Phase A: freeze flow, train decoder only
    freeze_module(model.flow_model)
    current_phase = "A"
    logger.info("Starting Phase A: Training Decoder only (Flow frozen)")
    
    while step < args.num_steps:
        # Phase transition
        if step == args.phase_a_steps and current_phase == "A":
            current_phase = "B"
            unfreeze_module(model.flow_model)
            logger.info("="*60)
            logger.info(f"PHASE TRANSITION at step {step}")
            logger.info("Starting Phase B: Training Flow + Decoder")
            logger.info("="*60)
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Train step
        train_flow = (current_phase == "B")
        train_decoder = True
        
        metrics = train_step(
            model, batch, optimizer, device, args.grad_clip, scaler,
            train_flow=train_flow, train_decoder=train_decoder,
            mode=args.mode, use_amp=use_amp
        )
        scheduler.step()
        
        running["loss"] += metrics["loss"]
        running["flow"] += metrics["flow_loss"]
        running["dec"] += metrics["decoder_loss"]
        step += 1
        pbar.update(1)
        
        # Logging
        if step % args.log_interval == 0:
            avg = {k: v/args.log_interval for k, v in running.items()}
            lr = optimizer.param_groups[0]["lr"]
            phase_str = f"[Phase {current_phase}]"
            pbar.set_postfix(
                phase=current_phase,
                loss=f"{avg['loss']:.4f}",
                flow=f"{avg['flow']:.4f}",
                dec=f"{avg['dec']:.4f}"
            )
            
            if not args.no_wandb:
                wandb.log({
                    "train/loss": avg["loss"],
                    "train/flow_loss": avg["flow"],
                    "train/decoder_loss": avg["dec"],
                    "train/lr": lr,
                    "train/phase": 1 if current_phase == "A" else 2,
                }, step=step)
            
            running = {"loss": 0, "flow": 0, "dec": 0}
        
        # Validation
        if step % args.eval_interval == 0:
            val = validate(model, val_loader, device, args.mode, use_amp)
            logger.info(f"Step {step} [Phase {current_phase}]: val_loss={val['val_loss']:.4f}, "
                       f"val_flow={val['val_flow']:.4f}, val_dec={val['val_dec']:.4f}")
            
            if not args.no_wandb:
                wandb.log(val, step=step)
            
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val,
                    "mode": args.mode,
                    "config": vars(args),
                }, f"{args.save_dir}/{args.mode}_best.pt")
                logger.info(f"Saved best model (val_loss={best_val:.4f})")
        
        # Save checkpoint
        if step % args.save_interval == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "mode": args.mode,
                "config": vars(args),
            }, f"{args.save_dir}/{args.mode}_step{step}.pt")
            logger.info(f"Saved checkpoint at step {step}")
    
    pbar.close()
    
    # Save final model
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "mode": args.mode,
        "config": vars(args),
    }, f"{args.save_dir}/{args.mode}_final.pt")
    logger.info(f"Saved final model to {args.save_dir}/{args.mode}_final.pt")
    logger.info("Training complete!")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
