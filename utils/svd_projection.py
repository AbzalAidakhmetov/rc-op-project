#!/usr/bin/env python3
"""
SVD-based projection for separating speaker and content subspaces in WavLM features.

Usage:
    python utils/svd_projection.py --data_dir ./preprocessed --output projection_matrix.pt
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
from tqdm import tqdm


def load_random_frames(data_dir: str, num_samples: int = 10000, seed: int = 42) -> torch.Tensor:
    """Load random WavLM frames from preprocessed data for SVD computation."""
    data_dir = Path(data_dir)
    metadata_path = data_dir / "metadata.json"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    utterances = list(metadata["utterances"].values())
    random.seed(seed)
    random.shuffle(utterances)
    
    all_frames = []
    frames_per_file = max(1, num_samples // len(utterances) + 1)
    
    print(f"Loading random frames from {len(utterances)} utterances...")
    for utt_info in tqdm(utterances, desc="Loading frames"):
        if len(all_frames) >= num_samples:
            break
            
        wavlm_path = data_dir / utt_info["wavlm_path"]
        try:
            features = torch.load(wavlm_path)
            T = features.shape[0]
            if T > frames_per_file:
                indices = random.sample(range(T), frames_per_file)
                sampled = features[indices]
            else:
                sampled = features
            all_frames.append(sampled)
        except Exception as e:
            print(f"Error loading {wavlm_path}: {e}")
            continue
    
    all_frames = torch.cat(all_frames, dim=0)
    if all_frames.shape[0] > num_samples:
        indices = random.sample(range(all_frames.shape[0]), num_samples)
        all_frames = all_frames[indices]
    
    print(f"Loaded {all_frames.shape[0]} frames with dimension {all_frames.shape[1]}")
    return all_frames


def compute_speaker_subspace(
    wavlm_features: torch.Tensor,
    k: int = 64,
    center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute SVD on WavLM features to find speaker subspace.
    
    Args:
        wavlm_features: (N, D) matrix of WavLM frames
        k: Number of top singular vectors for speaker subspace
        center: Whether to center the data before SVD
        
    Returns:
        V_k: (D, k) top-k right singular vectors (speaker basis)
        P_speaker: (D, D) projection onto speaker subspace
        P_content: (D, D) projection onto content subspace
        mean: (D,) mean vector
    """
    N, D = wavlm_features.shape
    print(f"Computing SVD on {N} samples with dimension {D}...")
    
    if center:
        mean = wavlm_features.mean(dim=0)
        centered = wavlm_features - mean
    else:
        mean = torch.zeros(D)
        centered = wavlm_features
    
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    
    V_k = Vh[:k].T
    P_speaker = V_k @ V_k.T
    P_content = torch.eye(D) - P_speaker
    
    total_var = (S ** 2).sum()
    speaker_var = (S[:k] ** 2).sum()
    
    print(f"SVD complete:")
    print(f"  Speaker subspace (top {k}): {100 * speaker_var / total_var:.1f}% variance")
    print(f"  Content subspace: {100 * (1 - speaker_var / total_var):.1f}% variance")
    
    return V_k, P_speaker, P_content, mean


def project_to_content(
    features: torch.Tensor,
    P_content: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project WavLM features onto the content subspace."""
    original_shape = features.shape
    D = original_shape[-1]
    features_flat = features.reshape(-1, D)
    
    if mean is not None:
        features_flat = features_flat - mean
    
    projected = features_flat @ P_content.T
    
    if mean is not None:
        projected = projected + mean
    
    return projected.reshape(original_shape)


def save_projection_matrix(
    output_path: str,
    V_k: torch.Tensor,
    P_speaker: torch.Tensor,
    P_content: torch.Tensor,
    mean: torch.Tensor,
    k: int,
):
    """Save projection matrices to disk."""
    torch.save({
        "V_k": V_k,
        "P_speaker": P_speaker,
        "P_content": P_content,
        "mean": mean,
        "k": k,
    }, output_path)
    print(f"Saved projection matrices to {output_path}")


def load_projection_matrix(path: str) -> dict:
    """Load projection matrices from disk."""
    return torch.load(path)


def main():
    parser = argparse.ArgumentParser(description="Compute SVD projection matrices")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="projection_matrix.pt")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    frames = load_random_frames(args.data_dir, args.num_samples, args.seed)
    V_k, P_speaker, P_content, mean = compute_speaker_subspace(frames, args.rank)
    save_projection_matrix(args.output, V_k, P_speaker, P_content, mean, args.rank)


if __name__ == "__main__":
    main()
