"""Distillation package for NeuralKNN-VC.

Turns the non-parametric kNN-VC teacher (``backbone.knnvc.KNNVC``) into a
parametric, pool-free neural converter (``models.converter.NeuralConverter``).

  * ``dataset_gen`` -- generate (source_feats, spk_emb) -> kNN_target_feats
    supervision pairs on the fly from the frozen teacher.
  * ``train``       -- CLI training loop that regresses the converter onto the
    teacher's kNN-matched features.
"""

from distill.dataset_gen import make_distill_batch, distill_batch_stream

__all__ = ["make_distill_batch", "distill_batch_stream"]
