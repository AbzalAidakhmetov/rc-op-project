"""Generate converter training pairs from the kNN-VC teacher.

The pool-free neural converter is DISTILLED from the kNN-VC backbone. For every
training example we:

  1. pick a SOURCE utterance (provides the linguistic content),
  2. pick a random TARGET speaker (!= source speaker),
  3. build the target speaker's matching POOL from a few of their utterances,
  4. extract the source's WavLM-Large layer-6 features (the converter input),
  5. run the teacher's kNN match (averaged target-speaker neighbour features) to
     get the SUPERVISION target -- this is exactly what kNN-VC would feed the
     vocoder, but produced WITHOUT a network,
  6. compute the target speaker's ECAPA embedding from a reference clip (the
     ONLY target-side information the converter sees at inference -- no pool).

The converter is then trained to map (source_feats, spk_emb) -> kNN_target_feats
so at inference it replaces the pool lookup with a single forward pass.

All teacher computation is frozen / no-grad. Returned tensors are padded to the
batch's max time length with a per-item ``lengths`` tensor for masked losses.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List

import torch

from utils import load_audio


def _pick_source_and_target(dataset, rng: random.Random):
    """Pick (source_path, source_spk, target_spk, target_paths).

    SOURCE supplies content; TARGET (a different speaker) supplies identity.
    ``target_paths`` is the full list of the target speaker's utterances.
    """
    speakers = dataset.speakers
    src_spk = rng.choice(speakers)
    # If there is only one speaker, fall back to same-speaker conversion so the
    # pipeline still runs (degenerate but valid for tiny smoke datasets).
    candidates = [s for s in speakers if s != src_spk] or [src_spk]
    tgt_spk = rng.choice(candidates)

    src_path = rng.choice(dataset.speaker_files[src_spk])
    tgt_paths = list(dataset.speaker_files[tgt_spk])
    return src_path, src_spk, tgt_spk, tgt_paths


@torch.no_grad()
def make_distill_batch(
    knnvc,
    spk_enc,
    dataset,
    device,
    batch_size: int,
    topk: int = 4,
    pool_utts: int = 8,
    crop_sec: float | None = None,
    pool_sec: float | None = 10.0,
    ref_sec: float | None = 10.0,
    rng: random.Random | None = None,
    to_device: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build one distillation batch from the frozen kNN-VC teacher.

    Args:
        knnvc:      ``backbone.knnvc.KNNVC`` teacher (frozen).
        spk_enc:    ``models.speaker.SpeakerEncoder`` (frozen ECAPA).
        dataset:    ``data.dataset.AudioFolderDataset`` (exposes ``.speakers`` /
                    ``.speaker_files``).
        device:     torch device for the returned tensors.
        batch_size: number of (source, target) pairs.
        topk:       neighbours averaged by the teacher's kNN match.
        pool_utts:  max target utterances concatenated into the matching pool.
        crop_sec:   if set, truncate the SOURCE utterance to this many seconds
                    (bounds time length / memory).
        pool_sec:   cap (seconds) on EACH target pool reference fed to WavLM
                    (default 10 s). Uncapped full clips are the main 12 GB OOM
                    exposure (WavLM-Large activations on long refs); a few
                    seconds per ref is plenty for kNN-VC quality. None = whole clip.
        ref_sec:    cap (seconds) on the ECAPA reference clip (default 10 s).
                    None = whole clip.
        rng:        optional ``random.Random`` for reproducibility.
        to_device:  put tensors on ``device`` (else leave on CPU).

    Returns:
        dict of tensors:
          'source_feats': (B, T, 1024) WavLM-Large layer-6 features (converter input)
          'spk_emb':      (B, 192)     target-speaker ECAPA embedding
          'target_feats': (B, T, 1024) kNN-matched supervision (teacher output)
          'lengths':      (B,)         valid (unpadded) time length per item
    """
    rng = rng or random
    sr = knnvc.sr if hasattr(knnvc, "sr") else 16000

    source_feats_list: List[torch.Tensor] = []
    target_feats_list: List[torch.Tensor] = []
    spk_emb_list: List[torch.Tensor] = []
    lengths: List[int] = []

    for _ in range(batch_size):
        src_path, _src_spk, _tgt_spk, tgt_paths = _pick_source_and_target(dataset, rng)

        # ---- source content features (converter input) ----
        src_wav = load_audio(src_path, target_sr=sr, max_sec=crop_sec)
        # Guard against ultra-short crops that would yield <2 frames.
        if src_wav.shape[0] < int(0.4 * sr):
            src_wav = load_audio(src_path, target_sr=sr, max_sec=None)
        source_feats = knnvc.get_features(src_wav)              # (Tq, 1024)

        # ---- target speaker matching pool ----
        # Cap EACH pool ref length so WavLM-Large activation memory stays bounded
        # (uncapped long refs are the main 12 GB OOM exposure).
        n_pool = min(pool_utts, len(tgt_paths))
        pool_paths = rng.sample(tgt_paths, n_pool) if n_pool > 0 else tgt_paths[:1]
        pool_wavs = [load_audio(p, target_sr=sr, max_sec=pool_sec) for p in pool_paths]
        pool = knnvc.build_pool(pool_wavs)                     # (Np, 1024)

        # ---- teacher supervision: kNN-averaged target features ----
        target_feats = knnvc.match_features(source_feats, pool, topk=topk)  # (Tq, 1024)

        # ---- target ECAPA embedding (the only target-side info at inference) ----
        ref_path = rng.choice(pool_paths)
        ref_wav = load_audio(ref_path, target_sr=sr, max_sec=ref_sec).to(device)
        spk_emb = spk_enc.encode(ref_wav)                      # (1, 192)

        source_feats_list.append(source_feats.float().cpu())
        target_feats_list.append(target_feats.float().cpu())
        spk_emb_list.append(spk_emb.float().squeeze(0).cpu())
        lengths.append(source_feats.shape[0])

    # ---- pad time dimension to the batch max ----
    max_t = max(lengths)
    dim = source_feats_list[0].shape[-1]
    B = batch_size

    source_feats = torch.zeros(B, max_t, dim, dtype=torch.float32)
    target_feats = torch.zeros(B, max_t, dim, dtype=torch.float32)
    for i, (sf_i, tf_i) in enumerate(zip(source_feats_list, target_feats_list)):
        t = sf_i.shape[0]
        source_feats[i, :t] = sf_i
        target_feats[i, :t] = tf_i

    spk_emb = torch.stack(spk_emb_list, dim=0)                 # (B, 192)
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    batch = {
        "source_feats": source_feats,
        "spk_emb": spk_emb,
        "target_feats": target_feats,
        "lengths": lengths_t,
    }
    if to_device:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def distill_batch_stream(
    knnvc,
    spk_enc,
    dataset,
    device,
    batch_size: int,
    topk: int = 4,
    pool_utts: int = 8,
    crop_sec: float | None = None,
    pool_sec: float | None = 10.0,
    ref_sec: float | None = 10.0,
    seed: int | None = None,
) -> Iterator[Dict[str, torch.Tensor]]:
    """Infinite streaming generator of distillation batches.

    Lets ``distill/train.py`` pull fresh teacher-labelled batches on the fly
    without materialising a finite dataset. Yields the same dict as
    :func:`make_distill_batch`.
    """
    rng = random.Random(seed)
    while True:
        yield make_distill_batch(
            knnvc,
            spk_enc,
            dataset,
            device,
            batch_size=batch_size,
            topk=topk,
            pool_utts=pool_utts,
            crop_sec=crop_sec,
            pool_sec=pool_sec,
            ref_sec=ref_sec,
            rng=rng,
        )
