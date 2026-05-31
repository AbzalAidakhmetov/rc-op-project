# NeuralKNN-VC

Quality-first, any-to-any zero-shot **voice conversion** at 16 kHz, built on two pieces:

1. **kNN-VC backbone (the quality teacher).** A faithful re-implementation of
   *kNN-VC* (Baas, van Niekerk, Kamper, "Voice Conversion With Just Nearest
   Neighbors", Interspeech 2023). It works entirely in **WavLM-Large layer-6
   feature space** and is vocoded by a **pretrained prematched HiFi-GAN** -- so
   it needs essentially **no training** and is SOTA-comparable out of the box.
2. **Pool-free neural converter (the research novelty).** kNN-VC needs the
   target speaker's *feature pool* at inference time (a chunk of reference audio
   it searches with nearest-neighbours). We **distil** that non-parametric
   lookup into a small, **speaker-conditioned** network that needs **no pool at
   inference** -- only a short reference, reduced to a single 192-dim ECAPA
   speaker embedding. One forward pass replaces the search.

Both backends emit features that the **same** pretrained prematched HiFi-GAN
vocodes, so the neural converter's outputs stay in-distribution for the vocoder.

> **Why this rewrite?** The previous system trained a tiny 15M ResNet *from
> scratch* to predict 100-band mel from WavLM-base *last-layer* features, then
> vocoded with generic Vocos. Result: muddy/robotic audio, reported target
> speaker-similarity ~**0.38** -- far below SOTA (~0.6-0.85). NeuralKNN-VC
> replaces the whole stack with the kNN-VC feature-space design.

---

## How it works

```
            source.wav  --[WavLM-Large L6]-->  query feats (Tq, 1024)
target ref.wav(s)        --[WavLM-Large L6]-->  matching pool (Np, 1024)

  kNN backend:   for each query frame, cosine top-4 into the pool, AVERAGE the
                 4 neighbour vectors  ->  converted feats (Tq, 1024)
  neural backend: converted = source_feats + NeuralConverter(source_feats, ecapa(ref))

            converted feats  --[prematched HiFi-GAN]-->  converted.wav (16 kHz)
```

- WavLM-Large hidden size is **1024**; **layer 6** (1-indexed, as in the paper)
  is the voice-conversion sweet spot.
- WavLM runs at 50 Hz on 16 kHz audio; HiFi-GAN reconstructs 16 kHz.
  **Everything is 16 kHz** -- no mel, no Vocos, no resampling mismatch.
- The pretrained WavLM-Large and prematched HiFi-GAN come from the
  [`bshall/knn-vc`](https://github.com/bshall/knn-vc) `torch.hub` repo.

### The novel pool-free converter

`NeuralConverter(source_feats: (B,T,1024), spk_emb: (B,192)) -> (B,T,1024)`

- Input projection `Conv1d(1024 -> 512)`.
- Global FiLM conditioning vector `g = MLP(spk_emb)` (no flow timestep).
- 8 dilated residual blocks (`ResidualBlock1D`, dilations `[1,2,4,8,1,2,4,8]`),
  each FiLM-modulated by `g`. Architecture ported from the proven FiLM 1D-ResNet
  in the old `main.py`, adapted to **feature -> feature**.
- Output projection `Conv1d(512 -> 1024)`, with a **residual skip**:
  `output = source_feats + delta`, so the net learns a *correction* toward the
  target speaker rather than rebuilding features from scratch.

**Distillation.** The kNN-VC backbone is the **teacher**. For a source utterance
and a randomly chosen target speaker, we build the target's pool, run the kNN
match **in feature space** (the averaged-neighbour features, *before* vocoding)
to get the supervision target, condition the converter on the target's ECAPA
embedding, and regress `converted -> kNN_target` with L1 (+ optional cosine)
loss. At inference the converter replaces the lookup; features are vocoded by the
same prematched HiFi-GAN.

---

## Install

Python 3.11. A virtualenv lives at `.venv` (use `.venv/bin/python`).

```bash
# from the repo root (the .venv is uv-managed; use uv pip)
VIRTUAL_ENV=.venv uv pip install -e .          # core deps
VIRTUAL_ENV=.venv uv pip install -e '.[eval]'  # + faster-whisper for optional WER
# (if your .venv has pip, `.venv/bin/python -m pip install -e .` also works)
```

GPU: defaults are tuned for a single **RTX 3060 (12 GB)** -- AMP training,
`batch_size=8`, `crop_sec=3.0`. WavLM-Large + HiFi-GAN are fetched from
`torch.hub` on first use (cached under the default hub dir; ECAPA caches under
`models/ecapa_voxceleb`).

---

## Quick start

```bash
# 1. Get data (LibriSpeech dev-clean, ~337 MB, 16 kHz, idempotent)
bash scripts/download_data.sh
#    -> data/librispeech/LibriSpeech/dev-clean/<spk>/<chapter>/*.flac

# 2. Instant SOTA-quality demo with NO TRAINING (pure kNN-VC backbone).
#    Auto-picks two speakers from the data dir if you omit --source/--target.
.venv/bin/python infer.py --backend knn \
    --data-dir data/librispeech/LibriSpeech/dev-clean
#    or with explicit files:
.venv/bin/python infer.py --backend knn \
    --source path/to/source.wav --target path/to/target_ref.wav

# 3. Distil the pool-free neural converter (12 GB-friendly defaults).
.venv/bin/python -m distill.train \
    --data-dir data/librispeech/LibriSpeech/dev-clean \
    --output-dir outputs --steps 50000

# 4. Pool-free neural conversion (single forward pass, no pool needed).
.venv/bin/python infer.py --backend neural \
    --converter outputs/converter.pt \
    --source path/to/source.wav --target path/to/target_ref.wav

# 5. Benchmark both backends: ECAPA target-sim / source-leak / RTF (+ optional WER).
.venv/bin/python benchmark.py \
    --converter outputs/converter.pt \
    --data-dir data/librispeech/LibriSpeech/dev-clean \
    --num-pairs 10
#    -> writes outputs/benchmark.md
```

### One-shot smoke test

Tiny end-to-end run (small caps, a few steps) that exercises every stage and
finishes fast on a 3060:

```bash
bash scripts/smoke_test.sh
```

It runs: pure-kNN conversion -> tiny distillation -> neural conversion ->
benchmark. Override knobs via env vars, e.g. `STEPS=50 MAX_SPK=6 bash scripts/smoke_test.sh`.

---

## Repository layout

```
config.py                 # Config dataclass + from_args()
utils.py                  # load_audio / peak_norm / save_wav / batched_cosine_knn
data/dataset.py           # AudioFolderDataset, collate_fn, build_dataloader
models/speaker.py         # SpeakerEncoder (frozen ECAPA-TDNN, 192-dim)
models/converter.py       # NeuralConverter (pool-free, FiLM 1D-ResNet) + blocks
backbone/knnvc.py         # KNNVC: WavLM-Large + prematched HiFi-GAN teacher/vocoder
distill/dataset_gen.py    # make_distill_batch (teacher supervision)
distill/train.py          # CLI: train the converter (AMP, warmup+cosine)
infer.py                  # CLI: --backend knn | neural
benchmark.py              # CLI: ECAPA sim/leak + RTF table -> benchmark.md
scripts/download_data.sh  # LibriSpeech dev-clean (VCTK alt documented inside)
scripts/smoke_test.sh     # tiny end-to-end pipeline check
```

> The legacy from-scratch-mel + Vocos system (`main.py`, `inference.py`,
> `evaluate.py`, `data/preprocess.py`) has been **removed** from the repo.

## Data

- **Default:** LibriSpeech `dev-clean` (40 speakers, ~337 MB, 16 kHz FLAC).
  `scripts/download_data.sh` fetches and extracts it.
- **VCTK alternative:** `bash download_vctk.sh` (~11 GB, 48 kHz). The dataset
  loader globs `p###/*_mic1.flac`; everything is resampled to 16 kHz on load.
  Point any command at `--data-dir data/vctk/wav48_silence_trimmed`.

`AudioFolderDataset` groups audio by top-level speaker directory and recursively
globs `*.flac` / `*.wav`, so LibriSpeech (`spk/chapter/*.flac`), VCTK
(`p###/*_mic1.flac`) and flat `spk/*.wav` layouts all work.

## License / credits

kNN-VC models and method: [bshall/knn-vc](https://github.com/bshall/knn-vc)
(Baas, van Niekerk, Kamper, Interspeech 2023). Speaker embeddings:
`speechbrain/spkrec-ecapa-voxceleb`.
