#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
set -x # Print each command before executing it.

# --- Common Hyperparameters ---
# These settings are shared across all experimental runs.
# We are using a subset of 300 speakers for 500 epochs.
# BATCH SIZE REDUCED from 16 to 8 and MAX DURATION to 12s to prevent CUDA OOM errors.
COMMON_ARGS="--data_root ./data \
--subset 300 \
--epochs 500 \
--batch_size 8 \
--max_duration 12 \
--val_split_ratio 0.1 \
--save_interval 100 \
--wandb_project rc-op-final-project"

echo "=========================================================="
echo "Sweep 1: Finding the Optimal Learning Rate"
echo "=========================================================="

# Run 1.1: Slower Learning Rate
python train.py $COMMON_ARGS \
  --lr 5e-5 \
  --save_dir checkpoints/sweep_lr/lr_5e-5 \
  --wandb_run_name "sweep-lr-5e-5"

# Run 1.2: Default Learning Rate (Baseline)
python train.py $COMMON_ARGS \
  --lr 1e-4 \
  --save_dir checkpoints/sweep_lr/lr_1e-4 \
  --wandb_run_name "sweep-lr-1e-4"

# Run 1.3: Faster Learning Rate
python train.py $COMMON_ARGS \
  --lr 2e-4 \
  --save_dir checkpoints/sweep_lr/lr_2e-4 \
  --wandb_run_name "sweep-lr-2e-4"


echo "=========================================================="
echo "Sweep 2: The Content vs. Style Trade-off"
echo "=========================================================="

# Run 2.1: Emphasize Content Clarity
python train.py $COMMON_ARGS \
  --lambda_ph 2.0 --lambda_sp 0.5 \
  --save_dir checkpoints/sweep_content_style/ph2.0_sp0.5 \
  --wandb_run_name "sweep-ph2.0-sp0.5"

# Run 2.2: Balanced (Default)
python train.py $COMMON_ARGS \
  --lambda_ph 1.0 --lambda_sp 1.0 \
  --save_dir checkpoints/sweep_content_style/ph1.0_sp1.0 \
  --wandb_run_name "sweep-ph1.0-sp1.0"

# Run 2.3: Emphasize Speaker Disentanglement
python train.py $COMMON_ARGS \
  --lambda_ph 0.5 --lambda_sp 2.0 \
  --save_dir checkpoints/sweep_content_style/ph0.5_sp2.0 \
  --wandb_run_name "sweep-ph0.5-sp2.0"


echo "=========================================================="
echo "Sweep 3: The Reconstruction Quality Trade-off"
echo "=========================================================="

# Run 3.1: Fewer Trainable Layers (More Conservative)
python train.py $COMMON_ARGS \
  --finetune_layers 4 \
  --save_dir checkpoints/sweep_finetune/layers_4 \
  --wandb_run_name "sweep-finetune-4"

# Run 3.2: Default Number of Layers
python train.py $COMMON_ARGS \
  --finetune_layers 8 \
  --save_dir checkpoints/sweep_finetune/layers_8 \
  --wandb_run_name "sweep-finetune-8"

# Run 3.3: More Trainable Layers (More Capacity)
python train.py $COMMON_ARGS \
  --finetune_layers 12 \
  --save_dir checkpoints/sweep_finetune/layers_12 \
  --wandb_run_name "sweep-finetune-12"

echo "=========================================================="
echo "All sweeps completed!"
echo "==========================================================" 