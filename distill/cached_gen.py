"""Fast distillation batches from a precomputed feature cache.

Drop-in replacement for ``make_distill_batch`` that does ZERO WavLM forwards at
train time. Per batch item it: picks a source utterance (random crop) and a
random target speaker, builds the target pool by concatenating that speaker's
cached features, runs a GPU cosine-kNN (the SAME ``batched_cosine_knn`` the
teacher uses) to get the supervision target, and grabs a cached ECAPA embedding
for conditioning. ~50x faster than the on-the-fly teacher.

Numerically identical supervision to ``make_distill_batch`` (same kNN matcher),
modulo float16 storage of the cached features (kNN is cosine -> robust).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch

from utils import batched_cosine_knn


class CachedDistillSampler:
    def __init__(
        self,
        cache_dir: str,
        device: str = "cuda",
        topk: int = 4,
        crop_frames: int = 150,          # ~3 s at 50 Hz
        pool_utts: int = 8,
        pool_frames: Optional[int] = 6000,
        allowed_speakers: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        cache_dir = Path(cache_dir)
        with open(cache_dir / "index.json") as f:
            self.index = json.load(f)
        self.device = device
        self.topk = topk
        self.crop_frames = crop_frames
        self.pool_utts = pool_utts
        self.pool_frames = pool_frames
        self.rng = random.Random(seed)

        all_spk = list(self.index["speakers"].keys())
        self.speakers = [s for s in all_spk if (allowed_speakers is None or s in allowed_speakers)]
        if len(self.speakers) < 1:
            raise ValueError("CachedDistillSampler: no speakers available after filtering.")

        # Load every cached feature tensor for the allowed speakers into CPU RAM
        # (fp16). dev-clean ~= 1.9 GB; trivially fits. Moved to GPU per use.
        ecapa = torch.load(cache_dir / "ecapa.pt", map_location="cpu")
        self.feats: Dict[str, torch.Tensor] = {}
        self.ecapa: Dict[str, torch.Tensor] = {}
        self.spk_keys: Dict[str, List[str]] = {}
        feats_dir = cache_dir / "feats"
        for spk in self.speakers:
            keys = [k for k in self.index["speakers"][spk] if k in ecapa]
            keys = [k for k in keys if (feats_dir / f"{k}.pt").exists()]
            if not keys:
                continue
            self.spk_keys[spk] = keys
            for k in keys:
                self.feats[k] = torch.load(feats_dir / f"{k}.pt", map_location="cpu")  # fp16 (T,1024)
                self.ecapa[k] = ecapa[k].float()
        self.speakers = [s for s in self.speakers if s in self.spk_keys]
        n_utt = sum(len(v) for v in self.spk_keys.values())
        print(f"CachedDistillSampler: {len(self.speakers)} speakers, {n_utt} utterances in RAM.")

    def _random_crop(self, feats: torch.Tensor) -> torch.Tensor:
        T = feats.shape[0]
        if T <= self.crop_frames:
            return feats
        start = self.rng.randint(0, T - self.crop_frames)
        return feats[start:start + self.crop_frames]

    def _build_pool(self, tgt_spk: str) -> torch.Tensor:
        keys = self.spk_keys[tgt_spk]
        n = min(self.pool_utts, len(keys))
        chosen = self.rng.sample(keys, n)
        pool = torch.cat([self.feats[k] for k in chosen], dim=0).float().to(self.device)
        if self.pool_frames is not None and pool.shape[0] > self.pool_frames:
            idx = torch.randperm(pool.shape[0], device=pool.device)[: self.pool_frames]
            pool = pool[idx]
        ref_key = self.rng.choice(chosen)
        return pool, ref_key

    @torch.no_grad()
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        src_list, tgt_list, emb_list, lengths = [], [], [], []
        for _ in range(batch_size):
            src_spk = self.rng.choice(self.speakers)
            cands = [s for s in self.speakers if s != src_spk] or [src_spk]
            tgt_spk = self.rng.choice(cands)

            src_key = self.rng.choice(self.spk_keys[src_spk])
            src_feats = self._random_crop(self.feats[src_key]).float().to(self.device)  # (Tq,1024)

            pool, ref_key = self._build_pool(tgt_spk)
            tgt_feats = batched_cosine_knn(src_feats, pool, topk=self.topk)             # (Tq,1024)
            spk_emb = self.ecapa[ref_key].to(self.device)                              # (192,)

            src_list.append(src_feats)
            tgt_list.append(tgt_feats)
            emb_list.append(spk_emb)
            lengths.append(src_feats.shape[0])

        max_t = max(lengths)
        dim = src_list[0].shape[-1]
        B = batch_size
        source_feats = torch.zeros(B, max_t, dim, device=self.device)
        target_feats = torch.zeros(B, max_t, dim, device=self.device)
        for i, (sf, tf) in enumerate(zip(src_list, tgt_list)):
            t = sf.shape[0]
            source_feats[i, :t] = sf
            target_feats[i, :t] = tf
        return {
            "source_feats": source_feats,
            "target_feats": target_feats,
            "spk_emb": torch.stack(emb_list, dim=0),
            "lengths": torch.tensor(lengths, dtype=torch.long, device=self.device),
        }
