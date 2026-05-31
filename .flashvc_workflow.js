export const meta = {
  name: 'neuralknnvc-build',
  description: 'Rebuild the broken flow-VC repo into NeuralKNN-VC: a kNN-VC quality backbone (WavLM-Large L6 + pretrained prematched HiFi-GAN) for SOTA-comparable audio, PLUS a novel pool-free speaker-conditioned feed-forward converter distilled from kNN. Then smoke-test end-to-end on the RTX 3060.',
  phases: [
    { title: 'Implement', detail: '7 parallel agents author disjoint modules against one shared SPEC' },
    { title: 'Integrate', detail: 'wire repo, remove orphans, fetch tiny data, run kNN-VC (no-train) audio, distill+train converter, infer both backends, benchmark; fix bugs' },
    { title: 'Review', detail: '3 parallel reviewers: kNN-VC API + distillation correctness, repo hygiene, 12GB/quality sanity' },
    { title: 'Fix', detail: 'apply blocking findings and re-verify' },
  ],
}

// ============================================================================
// SHARED INTERFACE SPEC  (contract every implement agent codes against)
// ============================================================================
const SPEC = `
NeuralKNN-VC INTERFACE SPEC  (all modules MUST match these signatures exactly)
Working dir: /workspace/rc-op-project  (git repo). Python 3.11 venv at .venv (use .venv/bin/python).
torch 2.10+cu128, CUDA on an RTX 3060 (12 GB). Internet is available.

WHY THIS PROJECT EXISTS
  The previous code trained a tiny 15M ResNet FROM SCRATCH to predict 100-band mel from WavLM-base
  LAST-layer features, vocoded with generic Vocos. Result: poor, muddy/robotic audio (reported target
  speaker-similarity ~0.38, far below SOTA ~0.6-0.85). We are REPLACING that with a quality-first design.

QUALITY BACKBONE = kNN-VC (Baas, van Niekerk, Kamper, "Voice Conversion With Just Nearest Neighbors",
  Interspeech 2023). It is SOTA-comparable for any-to-any zero-shot VC and needs essentially NO training
  (its HiFi-GAN vocoder is pretrained). Pipeline (NO mel, NO Vocos -- all in WavLM-Large feature space,
  16 kHz audio in and out):
    source wav --WavLM-Large layer 6--> query features (Tq, 1024)
    target ref wav(s) --WavLM-Large layer 6--> matching pool (Np, 1024)
    for each query frame: cosine-kNN (topk=4) into the pool, AVERAGE the k neighbour feature vectors
       -> converted features (Tq, 1024)
    converted features --PREMATCHED HiFi-GAN--> converted waveform (16 kHz)
  The pretrained models come from the bshall/knn-vc torch.hub repo. YOU MUST VERIFY THE REAL API by
  fetching https://raw.githubusercontent.com/bshall/knn-vc/master/hubconf.py and the matcher/vocoder
  source in that repo. Known entrypoints (verify, do not assume blindly):
    torch.hub.load('bshall/knn-vc', 'wavlm_large', trust_repo=True, device=...)        # WavLM-Large
    torch.hub.load('bshall/knn-vc', 'hifigan_wavlm', prematched=True, trust_repo=True, device=...)
    (also a bundled 'knn_vc' convenience object with .get_features/.get_matching_set/.match)
  WavLM-Large hidden size = 1024; layer 6 (1-indexed as in the paper) is the VC sweet spot. WavLM runs
  at 50 Hz on 16 kHz audio; HiFi-GAN reconstructs 16 kHz. Keep everything at 16 kHz.

NOVEL CONTRIBUTION = POOL-FREE NEURAL CONVERTER (distilled from kNN-VC).
  kNN-VC needs the target speaker's feature POOL at inference (a chunk of reference audio) and does a
  non-parametric lookup. We distil this into a parametric, speaker-conditioned, single-forward-pass
  network that needs NO pool at inference -- only a short reference (reduced to one ECAPA speaker
  embedding). This is the project's research novelty.
    NeuralConverter(source_feats:(B,T,1024), spk_emb:(B,192)) -> converted_feats:(B,T,1024)
    Architecture: reuse the FiLM-conditioned 1D ResNet from the OLD main.py (ResidualBlock1D), adapted
    to feature->feature: input proj Conv1d 1024->hidden(512), global FiLM cond = MLP(spk_emb) [no flow
    timestep], 8 dilated residual blocks, output proj Conv1d hidden->1024, plus a residual skip so it
    learns a DELTA over the source feats (output = source_feats + net(...)).
  Distillation: the kNN-VC backbone is the TEACHER. For a source utterance and a randomly chosen target
  speaker, build the target's pool, run kNN MATCH IN FEATURE SPACE (the averaged-neighbour features,
  BEFORE vocoding) to get the supervision target; condition the converter on the target's ECAPA embedding;
  regress converted_feats -> kNN_target_feats with L1 (+ optional cosine) loss. At inference the converter
  replaces the kNN lookup; features are vocoded by the SAME pretrained prematched HiFi-GAN (so the
  converter outputs are in-distribution for the vocoder).

REFERENCE CODE (READ IT): /workspace/rc-op-project/main.py has the proven ResidualBlock1D, SinusoidalPosEmb,
  FiLM ResNet (FlowMatchingResNet), ECAPA SpeakerEncoder, VCTKDataset, load_audio, upsample_to_length,
  collate, the torchaudio.list_audio_backends shim, and ECAPA-based eval ideas (also see legacy evaluate.py
  for ECAPA cosine-sim + Whisper WER). main.py / inference.py / evaluate.py / data/preprocess.py /
  data/dataset.py are LEGACY and will be DELETED by the integrator -- do NOT import from them.

TARGET LAYOUT
  config.py
  utils.py
  data/__init__.py        data/dataset.py
  backbone/__init__.py    backbone/knnvc.py
  models/__init__.py      models/converter.py    models/speaker.py
  distill/__init__.py     distill/dataset_gen.py distill/train.py
  infer.py   benchmark.py
  scripts/download_data.sh   scripts/smoke_test.sh
  README.md   WRITEUP.md   pyproject.toml

CONTRACTS (exact names/signatures):

config.py
  - @dataclass Config:
      wavlm_sr=16000, wavlm_layer=6, wavlm_dim=1024, ecapa_dim=192
      knn_topk=4, knn_pool_seconds=None   # None = use whole reference
      # converter net
      hidden_dim=512, num_res_blocks=8
      # distillation training
      batch_size=8, lr=2e-4, steps=50000, warmup_steps=500, grad_clip=1.0, crop_sec=3.0,
      use_cosine_loss=True
      # data caps (smoke knobs; None=all)
      data_dir='data/librispeech/LibriSpeech/dev-clean'
      max_speakers=None, max_files_per_speaker=None
      output_dir='outputs', device='cuda'
  - def from_args(args)->Config (override any field whose name matches an args attribute).

utils.py  (torch, torch.nn.functional as F, torchaudio, soundfile as sf, numpy as np)
  - load_audio(path, target_sr=16000, max_sec=None)->1D float tensor (mono, peak-norm 0.95, resampled, trunc).
  - peak_norm(wav). save_wav(path, wav, sr=16000).
  - batched_cosine_knn(query:(Tq,D), pool:(Np,D), topk=4)->converted:(Tq,D): cosine sim, top-k, mean of
    neighbour vectors. Chunk over Tq to bound memory. This is the core kNN matcher used by backbone+distill.

models/speaker.py
  - MUST start (before importing speechbrain) with the torchaudio shim:
      import torchaudio
      if not hasattr(torchaudio,'list_audio_backends'): torchaudio.list_audio_backends=lambda:['soundfile']
  - class SpeakerEncoder(nn.Module)(device='cuda'): frozen ECAPA (speechbrain EncoderClassifier.from_hparams
      source='speechbrain/spkrec-ecapa-voxceleb', savedir='models/ecapa_voxceleb', run_opts={'device':device}).
      .encode(audio_16k:(B,T) or (T,))->(B,192). No grad. (Port from main.py SpeakerEncoder.)

backbone/knnvc.py   (the kNN-VC quality teacher + vocoder; NO transformers/vocos)
  - class KNNVC: loads WavLM-Large + prematched HiFi-GAN from the bshall/knn-vc torch.hub (verify API
      against the real repo; pass trust_repo=True; cache under ./models or default hub dir). Methods:
      .get_features(wav_or_path)->(T,1024)            # WavLM-Large layer 6 features for one utterance
      .build_pool(list_of_wavs_or_paths)->(Np,1024)   # concat features of reference utterances
      .match_features(query:(Tq,1024), pool:(Np,1024), topk=4)->(Tq,1024)  # kNN-average (uses utils.batched_cosine_knn)
      .vocode(feats:(T,1024))->wav_16k (1D tensor)     # prematched HiFi-GAN
      .convert(source_wav, ref_wavs, topk=4)->wav_16k  # full kNN-VC: features->match->vocode
    Everything frozen / no-grad. Works on CUDA. Document the exact hub calls you settled on.

models/converter.py   (the NOVEL pool-free converter; reuse FiLM ResNet from main.py)
  - class SinusoidalPosEmb / ResidualBlock1D: port from main.py (ResidualBlock1D(channels,dilation,cond_dim),
    FiLM from a global cond vector).
  - class NeuralConverter(nn.Module)(feat_dim=1024, hidden_dim=512, spk_dim=192, num_blocks=8):
      forward(source_feats:(B,T,1024), spk_emb:(B,192)) -> converted_feats:(B,T,1024).
      input Conv1d(1024->hidden,k3,pad1); global cond g=spk_mlp(spk_emb) (MLP 192->hidden->hidden);
      num_blocks dilated ResidualBlock1D([1,2,4,8,1,2,4,8][:n]) each FiLM-modulated by g; out GroupNorm+GELU+
      Conv1d(hidden->1024,k1). RESIDUAL: return source_feats + delta (so it learns a correction). Shapes
      are (B,T,C) externally; transpose to channel-first internally for Conv1d.

data/dataset.py  (torch, soundfile, torchaudio, numpy, random; uses utils.load_audio)
  - class AudioFolderDataset(Dataset)(root_dir, crop_sec=3.0, sr=16000, max_speakers=None,
      max_files_per_speaker=None): group audio by TOP-LEVEL speaker dir; recursively glob *.flac/*.wav
      (covers LibriSpeech/LibriTTS spk/chapter/*.flac, VCTK p###/*_mic1.flac, flat spk/*.wav). Apply caps.
      Expose .speakers (sorted) and .speaker_files (dict spk->list[Path]). __getitem__ returns
      {'audio_16k','speaker','path'} with a random crop_sec crop (skip <0.5s). 16 kHz only (kNN-VC is 16k).
  - collate_fn(batch)->{'audio_16k','lengths','speakers','paths'} (pad).
  - def build_dataloader(config, shuffle=True)->(DataLoader, AudioFolderDataset).

distill/dataset_gen.py   (generate converter training pairs from the kNN-VC teacher)
  - @torch.no_grad() def make_distill_batch(knnvc:KNNVC, spk_enc:SpeakerEncoder, dataset, device, batch_size,
      topk=4, pool_utts=8) -> dict of tensors {'source_feats','spk_emb','target_feats','lengths'}:
      For each item: pick a SOURCE utterance (content) and a random TARGET speaker != source. Build the
      target pool from up to pool_utts of the target speaker's utterances (knnvc.build_pool). Compute
      source_feats=knnvc.get_features(source). target_feats=knnvc.match_features(source_feats, pool, topk)
      (the kNN supervision -- averaged target-speaker neighbour features). spk_emb=spk_enc.encode(a target
      reference clip). Pad to max T; return CPU or device tensors + valid lengths. This is the teacher signal.
  - Optionally an iterable/streaming generator so distill/train.py can pull batches on the fly.

distill/train.py   (CLI: train the pool-free converter)
  - argparse exposing Config distill fields (--steps --batch-size --lr --crop-sec --data-dir --output-dir
    --max-speakers --max-files-per-speaker --pool-utts --topk --no-wandb). Build Config.from_args, KNNVC
    teacher (frozen), SpeakerEncoder (frozen), dataset, NeuralConverter (trainable). Loop: get a distill
    batch (make_distill_batch), predict converted=converter(source_feats, spk_emb), loss = L1(converted,
    target_feats) [+ (1 - cosine) if use_cosine_loss], masked to valid lengths. AdamW, warmup+cosine,
    grad clip, AMP autocast+GradScaler (12GB). Periodic loss print + optional quick demo conversion
    (vocode a held-out sample). Save output_dir/converter.pt = {'model':state_dict, 'config':vars(config)}.

infer.py   (CLI)
  - args: --backend knn|neural (default knn), --source WAV, --target WAV (or multiple refs), --converter CKPT
    (for neural), --topk, --output-dir, [--data-dir auto-pick 2 speakers if no source/target].
    knn: KNNVC.convert(source, [refs]) -> wav. neural: source_feats=knnvc.get_features(source); spk_emb=
    SpeakerEncoder.encode(load_audio(ref)); converted=converter(source_feats[None], spk_emb)[0];
    wav=knnvc.vocode(converted). Write <src>_to_<tgt>_<backend>.wav (16k) + print path/duration.

benchmark.py   (CLI)
  - args: --converter CKPT, --data-dir, --output-dir, --num-pairs, [--topk]. For NUM_PAIRS held-out
    (src,tgt) speaker pairs, run BOTH backends (knn and neural). Report a markdown table:
      backend | ECAPA target-sim | ECAPA source-leak | delta | RTF (GPU, synchronize around convert+vocode).
    ECAPA cosine via SpeakerEncoder. Optional WER via faster-whisper in try/except (skip if absent). Also
    note the OLD system's reported target-sim 0.38 as the baseline to beat. Save output_dir/benchmark.md, print it.

scripts/download_data.sh
  - Idempotent. Download LibriSpeech dev-clean (~337MB, http://www.openslr.org/resources/12/dev-clean.tar.gz)
    to data/librispeech/ extracted as data/librispeech/LibriSpeech/dev-clean/<spk>/<chapter>/*.flac. Print
    speaker/file counts. Document the VCTK alternative in a comment.
  - scripts/smoke_test.sh: tiny end-to-end (small caps + few steps) calling: a pure-kNN conversion (infer
    --backend knn) to prove SOTA-quality audio with NO training, then distill/train (tiny), infer --backend
    neural, benchmark. Parameterised to finish fast on a 3060.

pyproject.toml
  - name 'neural-knn-vc'. Keep torch/torchaudio/numpy/soundfile/librosa/speechbrain/accelerate/tqdm/wandb/
    matplotlib/ipykernel. WavLM-Large + HiFi-GAN come via torch.hub (no extra hard dep). transformers/vocos
    may be dropped from required deps (not used) -- but leaving them is harmless; your call. Add optional
    extra 'eval'=['faster-whisper']. [tool.hatch...packages]=['data','models','backbone','distill'].

README.md + WRITEUP.md
  - README: ACCURATE to the real files/commands (the old README described nonexistent files -- primary fix).
    Quick start: setup -> scripts/download_data.sh -> infer --backend knn (instant SOTA-quality demo, no
    training) -> distill/train.py -> infer --backend neural -> benchmark.py. Explain the kNN-VC backbone and
    the novel pool-free converter, and the 12GB-friendly defaults. WRITEUP: short method note (kNN-VC
    backbone; pool-free neural converter distilled from kNN; speaker conditioning via ECAPA; eval protocol:
    ECAPA sim/leak + RTF, contrasting with the old from-scratch-mel approach and its 0.38 similarity).

GLOBAL RULES
  - Complete, runnable code. No TODOs/placeholders/pseudocode.
  - Match signatures EXACTLY for cross-module imports.
  - Features/mel are (B,T,C); Conv1d work is channel-first. Everything 16 kHz.
  - Frozen teacher/vocoder/speaker-encoder: no grad, @torch.no_grad on feature extraction + kNN + vocode.
  - Default device 'cuda'. Create parent dirs as needed (mkdir -p in scripts).
`;

// ============================================================================
// PHASE 1 -- IMPLEMENT (7 parallel module-authors, disjoint file ownership)
// ============================================================================
phase('Implement')

const MODULES = [
  { key: 'config',    files: ['config.py', 'utils.py'] },
  { key: 'data',      files: ['data/__init__.py', 'data/dataset.py'] },
  { key: 'backbone',  files: ['backbone/__init__.py', 'backbone/knnvc.py'] },
  { key: 'models',    files: ['models/__init__.py', 'models/converter.py', 'models/speaker.py'] },
  { key: 'distill',   files: ['distill/__init__.py', 'distill/dataset_gen.py', 'distill/train.py'] },
  { key: 'usecli',    files: ['infer.py', 'benchmark.py'] },
  { key: 'docs',      files: ['README.md', 'WRITEUP.md', 'pyproject.toml', 'scripts/download_data.sh', 'scripts/smoke_test.sh'] },
]

const implResults = await parallel(MODULES.map(m => () =>
  agent(
    `You are implementing part of NeuralKNN-VC, rebuilding a broken flow-matching voice-conversion repo ` +
    `into a kNN-VC quality backbone plus a novel pool-free neural converter. Read ` +
    `/workspace/rc-op-project/main.py (proven reference for the ResNet/ECAPA/dataset/utils) and any legacy ` +
    `file you need. ${m.key==='backbone' ? 'CRITICAL: fetch and read the REAL bshall/knn-vc torch.hub repo ' +
    '(hubconf.py + matcher/vocoder source on github raw) so the WavLM-Large + prematched HiFi-GAN calls are ' +
    'correct -- do not guess the API. ' : ''}Implement ONLY these files, completely and runnably:\n  ${m.files.join('\n  ')}\n\n` +
    `Follow this shared interface SPEC EXACTLY (other agents implement the rest in parallel against the same ` +
    `contract, so signatures must match to the letter):\n${SPEC}\n\n` +
    `Write each assigned file with the Write tool (mkdir -p parents first via Bash if needed). Do NOT create ` +
    `or modify files outside your list. Return a <=5-line summary of what you wrote and any contract ` +
    `assumptions other modules should know (especially, for backbone, the exact torch.hub calls you used).`,
    { label: `impl:${m.key}`, phase: 'Implement' }
  )
))

// ============================================================================
// PHASE 2 -- INTEGRATE (single serial agent)
// ============================================================================
phase('Integrate')

const integrationReport = await agent(
  `You are the INTEGRATOR for NeuralKNN-VC at /workspace/rc-op-project (git repo). Seven agents just wrote ` +
  `the modules per this SPEC:\n${SPEC}\n\nTheir summaries:\n` +
  `${implResults.filter(Boolean).map((r,i)=>`[${MODULES[i].key}] ${r}`).join('\n')}\n\n` +
  `Make the whole thing run end-to-end on the RTX 3060, then SMOKE-TEST. Steps:\n` +
  `1. Inspect new files; fix cross-module import/signature mismatches, missing __init__.py, dangling refs. ` +
  `   Ensure the torchaudio.list_audio_backends shim is applied in models/speaker.py BEFORE speechbrain import.\n` +
  `2. Use the venv (.venv/bin/python, .venv/bin/pip). Run '.venv/bin/pip install -e .'. Then an import smoke: ` +
  `   python -c 'import config,utils; import models.converter,models.speaker; import backbone.knnvc; ` +
  `   import data.dataset; import distill.dataset_gen,distill.train'  -- fix every error. Install any missing ` +
  `   runtime deps the kNN-VC hub code needs.\n` +
  `3. VALIDATE THE QUALITY BACKBONE FIRST (the whole point -- SOTA audio with NO training): fetch the tiny ` +
  `   dataset via scripts/download_data.sh (LibriSpeech dev-clean ~337MB; if unreachable, grab a few speakers ` +
  `   another way). Then run 'python infer.py --backend knn ...' on one real (source,target) pair. Confirm it ` +
  `   loads WavLM-Large + prematched HiFi-GAN from torch.hub and writes a real converted 16kHz wav. This must ` +
  `   work -- it is the headline. Fix any hub/API issues (re-read the real bshall/knn-vc repo if needed).\n` +
  `4. SMOKE-TEST THE NOVEL PATH with TINY settings (point = it runs, not final quality): ` +
  `   --max-speakers 8 --max-files-per-speaker 6, --crop-sec 2.0, --batch-size 4, --steps ~150, small pool-utts. ` +
  `   Sequence: a) python distill/train.py ... -> converter.pt ; b) python infer.py --backend neural ` +
  `   --converter converter.pt ... -> wav ; c) python benchmark.py --converter converter.pt --num-pairs 2 ` +
  `   -> benchmark.md. Reduce batch/steps if OOM/slow. Fix every bug (this is the real test of the code).\n` +
  `5. After the new code runs, remove LEGACY files with git rm: main.py, inference.py, evaluate.py, ` +
  `   data/preprocess.py, and the old data/dataset.py IF replaced. Do NOT touch .venv, data/, .git. Also ` +
  `   git rm the now-unused download_vctk.sh/setup.sh ONLY if README no longer references them (otherwise ` +
  `   update README). Delete the stale .flashvc_workflow.js? NO -- leave it.\n` +
  `6. Do NOT git commit.\n\n` +
  `Return a structured report: (a) backend=knn quality-demo result (did it produce real audio? file path), ` +
  `(b) each smoke stage PASS/FAIL with exact commands, (c) the benchmark.md table contents (ECAPA sim + RTF), ` +
  `(d) bugs fixed, (e) remaining known issues, (f) final file tree (excluding .venv/data/.git).`,
  { label: 'integrate+smoke-test', phase: 'Integrate' }
)

// ============================================================================
// PHASE 3 -- REVIEW (3 parallel reviewers)
// ============================================================================
phase('Review')

const FINDINGS_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { findings: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      severity: { type: 'string', enum: ['blocking','major','minor'] },
      file: { type: 'string' }, issue: { type: 'string' }, fix: { type: 'string' },
    }, required: ['severity','file','issue','fix'],
  }}}, required: ['findings'],
}

const REVIEW_LENSES = [
  { key: 'method', prompt:
    `Review METHOD CORRECTNESS of NeuralKNN-VC. Read backbone/knnvc.py, utils.batched_cosine_knn, ` +
    `distill/dataset_gen.py, distill/train.py, models/converter.py, infer.py. Verify: (1) WavLM-Large ` +
    `LAYER 6 features (1024-d) are used, not last layer / not base; (2) kNN matching is cosine top-k then ` +
    `MEAN of neighbour vectors, producing target-speaker features; (3) the prematched HiFi-GAN is fed the ` +
    `matched/converted FEATURES (in-distribution), and the torch.hub calls match the real bshall/knn-vc API; ` +
    `(4) distillation target = kNN match_features output, converter conditioned on the TARGET speaker ECAPA ` +
    `embedding, loss masked to valid lengths, residual (delta) parameterisation correct; (5) neural infer ` +
    `path = converter -> SAME vocoder, pool-free. Report concrete correctness bugs.` },
  { key: 'hygiene', prompt:
    `Review REPO HYGIENE of NeuralKNN-VC. Verify new files match the SPEC and import each other consistently ` +
    `(no dangling imports; no references to deleted legacy main.py/inference.py/evaluate.py/data/preprocess.py). ` +
    `README.md must match the REAL files/commands (old README described nonexistent files -- must be fixed); ` +
    `pyproject package list correct; __init__.py present where imported as packages; CLI flags referenced in ` +
    `README/smoke_test.sh actually exist. Report mismatches.` },
  { key: 'mem-quality', prompt:
    `Review NeuralKNN-VC for 12GB-VRAM SAFETY and QUALITY-SANITY. Read distill/train.py, distill/dataset_gen.py, ` +
    `backbone/knnvc.py, benchmark.py. Check: AMP used in training; teacher/vocoder/WavLM/ECAPA all no-grad and ` +
    `not accumulating graphs; batched_cosine_knn chunks over time to avoid a giant (Tq x Np) matrix blowing ` +
    `12GB; pool features stored sensibly; RTF measured with torch.cuda.synchronize around the right region; ` +
    `defaults (batch/crop) sane for a 3060. Report real OOM risks, quality footguns (e.g. wrong sample rate, ` +
    `feature normalisation mismatch with the pretrained vocoder), or misleading benchmark numbers.` },
]

const reviews = await parallel(REVIEW_LENSES.map(l => () =>
  agent(
    `${l.prompt}\n\nContext: the integrator already ran a smoke test and reported:\n${integrationReport}\n\n` +
    `Be concrete and file-specific. Only report real bugs/risks -- no style nitpicks. Return findings via the schema.`,
    { label: `review:${l.key}`, phase: 'Review', schema: FINDINGS_SCHEMA }
  )
))

const allFindings = reviews.filter(Boolean).flatMap(r => r.findings || [])
const blocking = allFindings.filter(f => f.severity === 'blocking' || f.severity === 'major')

// ============================================================================
// PHASE 4 -- FIX
// ============================================================================
phase('Fix')

let fixReport = 'No blocking/major findings -- nothing to fix.'
if (blocking.length > 0) {
  fixReport = await agent(
    `You are finalising NeuralKNN-VC at /workspace/rc-op-project. Reviewers found these BLOCKING/MAJOR issues:\n` +
    `${blocking.map((f,i)=>`${i+1}. [${f.severity}] ${f.file}: ${f.issue}\n   -> fix: ${f.fix}`).join('\n')}\n\n` +
    `Apply the fixes. Then re-run the import smoke and, if a core path changed (knnvc/converter/distill/infer), ` +
    `re-run the smallest end-to-end check you can (.venv/bin/python, tiny settings, the dataset already downloaded) ` +
    `to confirm it still runs and 'infer --backend knn' still produces audio. Do NOT git commit. Return what you ` +
    `changed and the result of your re-verification (commands + pass/fail).`,
    { label: 'apply-fixes', phase: 'Fix' }
  )
}

// ============================================================================
return {
  project: 'NeuralKNN-VC -- kNN-VC quality backbone + novel pool-free neural converter',
  filesImplemented: MODULES.flatMap(m => m.files),
  integration: integrationReport,
  reviewFindings: allFindings,
  blockingFixed: blocking.length,
  fixReport,
}
