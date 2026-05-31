# NeuralKNN-VC: a pool-free neural converter distilled from kNN-VC

## 1. Motivation

The previous incarnation of this project trained a ~15M-parameter 1D-ResNet
**from scratch** to predict 100-band mel-spectrograms from **WavLM-base
last-layer** features, then vocoded with a generic **Vocos** mel vocoder. Despite
a reasonable architecture (FiLM-conditioned dilated ResNet, rectified
flow-matching), the result was muddy and robotic, with a reported **target
speaker-similarity of ~0.38** -- well below the SOTA range of ~0.6-0.85 for
zero-shot any-to-any VC. The fundamental problems were (a) training a generative
mel predictor from scratch on limited data, (b) a domain mismatch between the
16 kHz WavLM features and the 24 kHz Vocos vocoder, and (c) using last-layer
WavLM features, which entangle speaker and content.

NeuralKNN-VC abandons the from-scratch-mel approach entirely and adopts a
**quality-first, feature-space** design.

## 2. Quality backbone: kNN-VC

We re-implement **kNN-VC** (Baas, van Niekerk, Kamper, *"Voice Conversion With
Just Nearest Neighbors"*, Interspeech 2023). The insight is that **self-
supervised speech features are nearly speaker-agnostic at the right layer**:
frames with the same phonetic content land near each other in feature space
regardless of speaker. Conversion is therefore a *retrieval* problem, not a
generation problem.

Pipeline (all at 16 kHz, no mel, no Vocos):

1. Encode the **source** with WavLM-Large, taking **layer 6** (1-indexed; the
   paper's "speaker-information layer" sweet spot): query features `(Tq, 1024)`.
2. Encode one or more **target reference** utterances the same way and
   concatenate them into a **matching pool** `(Np, 1024)`.
3. For each query frame, find its **cosine top-`k` (k=4)** neighbours in the pool
   and **average** those neighbour feature vectors. This swaps the source
   speaker's realisation of each sound for the target speaker's, while preserving
   content and prosody.
4. Vocode the converted features with a **pretrained prematched HiFi-GAN**
   (trained directly on WavLM-Large layer-6 features), yielding 16 kHz audio.

Crucially, **nothing here is trained by us** -- WavLM-Large and the prematched
HiFi-GAN are pretrained and loaded from the `bshall/knn-vc` `torch.hub` repo.
This alone gives SOTA-comparable conversion and is our reference ceiling.

**Cost.** kNN-VC's weakness is inference: it must hold the target's feature
**pool** in memory and run a nearest-neighbour search per frame. The pool needs
a non-trivial chunk of target audio, and search cost scales with pool size.

## 3. Novelty: pool-free neural converter (distilled from kNN-VC)

We **distil** the non-parametric kNN lookup into a small parametric network that
needs **no pool at inference** -- only a single 192-dim **ECAPA-TDNN** speaker
embedding from a short reference clip.

**Model.**
`NeuralConverter(source_feats: (B,T,1024), spk_emb: (B,192)) -> (B,T,1024)`,
reusing the proven FiLM 1D-ResNet from the old `main.py`, adapted to
feature->feature:

- input `Conv1d(1024 -> 512)`;
- a global FiLM conditioning vector `g = MLP(spk_emb)` (192->512->512) -- note
  there is **no flow timestep**, unlike the old velocity field;
- 8 dilated `ResidualBlock1D`s (dilations `[1,2,4,8,1,2,4,8]`), each
  FiLM-modulated by `g`;
- output `GroupNorm + GELU + Conv1d(512 -> 1024)`;
- a **residual skip**: `output = source_feats + delta`. The network only has to
  learn the *correction* that moves source frames toward the target speaker's
  manifold, which is an easier and more stable target than reconstructing
  features outright.

**Teacher signal.** For each training example we pick a source utterance and a
random target speaker (`!=` source). We build the target's pool from up to
`pool_utts` of its utterances, compute `source_feats = WavLM(source)`, and obtain
the supervision target `target_feats = kNN-match(source_feats, pool, topk)` --
i.e. the **averaged-neighbour features, before vocoding**. The converter is
conditioned on the target's ECAPA embedding and regressed onto `target_feats`.

**Loss.** Length-masked `L1(converted, target_feats)` plus an optional
`(1 - cosine)` term (`use_cosine_loss=True` by default), which directly rewards
directional agreement -- the same cosine geometry the kNN matcher uses.
Optimisation is AdamW with linear warmup + cosine decay, gradient clipping, and
AMP autocast + GradScaler to fit comfortably in 12 GB.

**Why it stays in-distribution.** Both the teacher target and the converter
output live in WavLM-Large layer-6 space, and **both backends vocode with the
same prematched HiFi-GAN**. The converter is therefore producing exactly the kind
of features the vocoder expects, avoiding the domain mismatch that plagued the
old mel/Vocos system.

## 4. Evaluation protocol

`benchmark.py` runs **both** backends (kNN and neural) over `--num-pairs`
held-out `(source, target)` speaker pairs and reports a markdown table with:

- **ECAPA target-sim** -- cosine similarity between the converted audio's ECAPA
  embedding and the *target* speaker's embedding (higher = better identity
  transfer). The old system scored ~0.38; this is the baseline to beat.
- **ECAPA source-leak** -- cosine similarity to the *source* speaker (lower =
  less source identity leaking through).
- **delta** = target-sim - source-leak (a single conversion-quality summary).
- **RTF** -- real-time factor on GPU, with `torch.cuda.synchronize()` around the
  full `convert + vocode` so the kNN search / forward pass is timed honestly.
- **WER** (optional) -- intelligibility via `faster-whisper`, guarded by
  `try/except` so the benchmark runs without the `eval` extra installed.

We expect the kNN backend to set the quality ceiling, and the distilled neural
converter to approach it while being **pool-free** and a **single forward pass**
at inference -- trading a small similarity gap for a large reduction in
inference-time memory and dependence on having a sizeable target reference.

## 5. Summary of the change

| | Old (from-scratch mel) | NeuralKNN-VC |
|---|---|---|
| Feature space | WavLM-base **last** layer, 768-d | WavLM-**Large** **layer 6**, 1024-d |
| Target | 100-band mel (generated) | WavLM features (retrieved/distilled) |
| Vocoder | generic Vocos @ 24 kHz | prematched HiFi-GAN @ 16 kHz |
| Trained component | whole mel predictor | only the pool-free converter (distilled) |
| Target-sim | ~0.38 | kNN ceiling + distilled converter |
| Inference need | source + target ref | kNN: target pool; **neural: 1 ECAPA emb** |

The kNN-VC backbone gives quality essentially for free; the pool-free neural
converter is the contribution -- a parametric, speaker-conditioned, single-pass
approximation of kNN-VC that removes the pool requirement at inference.
