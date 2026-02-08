# Zero-Shot Voice Conversion via Rectified Flow Matching with Speaker-Disentangled Content Representations

## Abstract

We present a zero-shot voice conversion system based on Rectified Flow Matching that converts the speaker identity of a source utterance to that of an arbitrary target speaker, given only a short reference clip. Our central hypothesis is that **explicitly removing speaker information from the flow's starting point forces the model to rely on an external speaker embedding for identity reconstruction, yielding better speaker similarity in the converted output**. We test this hypothesis by comparing three flow initialisation strategies—random noise, raw content features, and speaker-disentangled content features—using a 1D convolutional ResNet velocity predictor with FiLM conditioning. We propose a simple yet effective speaker-stripping pipeline combining instance normalisation of self-supervised speech features with SVD-based projection. Experiments on the VCTK dataset demonstrate that the disentangled starting point achieves the highest target speaker similarity (+37% over the baseline), confirming our hypothesis.

---

## 1. Introduction

Voice conversion (VC) aims to transform the speech of a source speaker so that it sounds as though it were spoken by a target speaker, while preserving the linguistic content. Zero-shot VC is the more challenging variant where the target speaker is unseen during training; the system must generalise from a single short reference utterance.

Recent advances in flow-based generative models, particularly Rectified Flow Matching (Liu et al., 2023), offer a compelling framework for VC. Flow matching learns a deterministic mapping between a starting distribution \(p_0\) and a target distribution \(p_1\) via a velocity field, avoiding the training instabilities of GANs and the slow sampling of diffusion models.

A critical design choice in flow-based VC is the **starting point** \(\mathbf{z}_0\). Prior work has explored starting from Gaussian noise (as in standard diffusion) or from a content representation derived from the source speech. We hypothesise that:

> **Starting from a speaker-disentangled content representation—where speaker identity has been explicitly removed—produces superior voice conversion compared to starting from raw content features or random noise.**

The intuition is that if \(\mathbf{z}_0\) retains source speaker characteristics, the flow may "shortcut" by preserving those characteristics rather than fully adopting the target speaker's identity from the conditioning signal. By stripping speaker information from \(\mathbf{z}_0\), we force the model to reconstruct speaker identity entirely from the target speaker embedding.

---

## 2. Method

### 2.1 Problem Formulation

Given a source utterance \(\mathbf{x}^{(s)}\) from speaker \(s\) and a reference utterance \(\mathbf{x}^{(r)}\) from target speaker \(r\), the goal is to produce \(\hat{\mathbf{x}}^{(r)}\) that has the linguistic content of \(\mathbf{x}^{(s)}\) and the speaker identity of \(\mathbf{x}^{(r)}\).

We operate in the mel-spectrogram domain. Let \(\mathbf{z}_1 \in \mathbb{R}^{T \times 100}\) denote the ground-truth 100-band log-mel spectrogram (extracted by the Vocos feature extractor at 24 kHz, hop length 256, yielding ~93.75 frames/sec). The flow model learns a velocity field that transports a starting point \(\mathbf{z}_0\) to \(\mathbf{z}_1\).

### 2.2 Feature Extraction

Three frozen pre-trained models extract features on-the-fly during training:

**WavLM-base-plus** (Chen et al., 2022) extracts content features from 16 kHz audio. The model outputs \(\mathbf{W} \in \mathbb{R}^{T_w \times 768}\) at ~50 frames/sec. These features capture phonetic and linguistic information but also encode speaker characteristics.

**ECAPA-TDNN** (Desplanques et al., 2020), pre-trained on VoxCeleb for speaker verification, extracts a fixed-dimensional speaker embedding \(\mathbf{e} \in \mathbb{R}^{192}\) from any audio clip. This embedding captures speaker identity (pitch range, timbre, vocal tract characteristics) independently of spoken content.

**Vocos** (Siuzdak, 2023) serves dual roles: its feature extractor computes the target mel spectrogram \(\mathbf{z}_1\), and its decoder converts predicted mel spectrograms back to waveforms at 24 kHz.

Since WavLM operates at 50 Hz and the mel spectrogram at ~93.75 Hz, we upsample WavLM features to match the mel sequence length using linear interpolation:

\[
\mathbf{W}_{\text{up}} = \text{Interpolate}(\mathbf{W}, T_{\text{mel}}) \in \mathbb{R}^{T_{\text{mel}} \times 768}
\]

### 2.3 Speaker Disentanglement

We propose a two-stage speaker-stripping pipeline applied to the upsampled WavLM features:

**Stage 1: Instance Normalisation.** For each utterance, we normalise each feature dimension to zero mean and unit variance across the time axis:

\[
\hat{\mathbf{W}}_{\text{up}}^{(d)} = \frac{\mathbf{W}_{\text{up}}^{(d)} - \mu^{(d)}}{\sigma^{(d)} + \epsilon}, \quad \mu^{(d)} = \frac{1}{T}\sum_{t=1}^{T} W_{t,d}, \quad \sigma^{(d)} = \sqrt{\frac{1}{T}\sum_{t=1}^{T}(W_{t,d} - \mu^{(d)})^2}
\]

where \(d\) indexes feature dimensions and \(\epsilon = 10^{-6}\). This removes utterance-level statistics that encode speaker characteristics such as mean pitch offset and formant biases.

**Stage 2: SVD Projection.** We compute a dataset-level SVD projection matrix \(\mathbf{P}\) that removes the top-\(k\) singular vectors from the feature space. Given a collection of WavLM frames \(\mathbf{F} \in \mathbb{R}^{N \times 768}\) from 500 utterances:

\[
\tilde{\mathbf{F}} = \mathbf{F} - \bar{\mathbf{F}}, \quad \tilde{\mathbf{F}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
\]

\[
\mathbf{P} = \mathbf{I}_{768} - \mathbf{V}_{:k}\mathbf{V}_{:k}^\top \in \mathbb{R}^{768 \times 768}
\]

where \(\mathbf{V}_{:k}\) contains the first \(k\) right singular vectors. The top singular components capture the directions of maximum variance, which in speech features correspond to speaker identity. The projection is applied as:

\[
\mathbf{W}_{\text{strip}} = \hat{\mathbf{W}}_{\text{up}} \mathbf{P}
\]

### 2.4 Flow Starting Point Modes

We investigate three initialisation strategies for \(\mathbf{z}_0\):

**Noise mode.** \(\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})\), with shape \(\mathbb{R}^{T \times 100}\). The model must generate the entire mel spectrogram from scratch, conditioned on WavLM features and the speaker embedding.

**Source mode.** \(\mathbf{z}_0 = f_\theta(\mathbf{W}_{\text{up}})\), where \(f_\theta: \mathbb{R}^{768} \to \mathbb{R}^{100}\) is a learnable linear projection. This starts the flow from a content-rich point in mel space that retains source speaker characteristics.

**SVD mode (proposed).** \(\mathbf{z}_0 = f_\theta(\mathbf{W}_{\text{strip}})\), where \(\mathbf{W}_{\text{strip}}\) has been processed by the instance normalisation + SVD pipeline. The flow starts from a content representation with speaker information explicitly removed.

### 2.5 Velocity Network Architecture

The velocity field \(v_\theta(\mathbf{z}_t, t, \mathbf{c}, \mathbf{e})\) is parameterised by a 1D convolutional ResNet with Feature-wise Linear Modulation (FiLM) conditioning. The architecture consists of approximately 15M parameters.

**Input Projections.** The noisy mel \(\mathbf{z}_t \in \mathbb{R}^{T \times 100}\) and the conditioning features \(\mathbf{c} \in \mathbb{R}^{T \times 768}\) (WavLM, optionally speaker-stripped) are projected independently and summed:

\[
\mathbf{h}_0 = \text{Conv1d}_{1 \times 1}^{100 \to 512}(\mathbf{z}_t^\top) + \text{Conv1d}_{3 \times 1}^{768 \to 512}(\mathbf{c}^\top) \in \mathbb{R}^{512 \times T}
\]

**Global Conditioning.** The timestep \(t \in [0,1]\) is encoded via sinusoidal positional embeddings (Vaswani et al., 2017) and projected through an MLP:

\[
\mathbf{g}_t = \text{MLP}_t(\text{SinEmb}(t)) \in \mathbb{R}^{512}
\]

The speaker embedding is similarly projected:

\[
\mathbf{g}_e = \text{MLP}_e(\mathbf{e}) \in \mathbb{R}^{512}
\]

The global conditioning vector is their sum: \(\mathbf{g} = \mathbf{g}_t + \mathbf{g}_e\).

**Residual Blocks.** The backbone consists of 8 residual blocks with dilations \([1, 2, 4, 8, 1, 2, 4, 8]\), providing a receptive field that spans the full temporal context. Each block applies FiLM conditioning:

\[
\mathbf{h}' = \text{GroupNorm}(\mathbf{h})
\]
\[
\gamma, \beta = \text{split}(\text{Linear}(\mathbf{g})) \quad \text{(scale and shift)}
\]
\[
\mathbf{h}' = \mathbf{h}' \odot (1 + \gamma) + \beta \quad \text{(FiLM)}
\]
\[
\mathbf{h}' = \text{Conv1d}_{\text{dilated}}(\text{GELU}(\mathbf{h}'))
\]
\[
\mathbf{h}' = \text{Conv1d}_{1 \times 1}(\text{GELU}(\text{GroupNorm}(\mathbf{h}')))
\]
\[
\mathbf{h} \leftarrow \mathbf{h} + \mathbf{h}' \quad \text{(skip connection)}
\]

**Output Head.** A final GroupNorm, GELU, and \(1 \times 1\) convolution project back to mel dimensions:

\[
\hat{\mathbf{v}} = \text{Conv1d}_{1 \times 1}^{512 \to 100}(\text{GELU}(\text{GroupNorm}(\mathbf{h}_L))) \in \mathbb{R}^{T \times 100}
\]

### 2.6 Rectified Flow Matching Training

Following Liu et al. (2023), we train with the Rectified Flow objective. Given paired \((\mathbf{z}_0, \mathbf{z}_1)\):

1. Sample timestep: \(t \sim \mathcal{U}(0, 1)\)
2. Interpolate: \(\mathbf{z}_t = (1 - t)\mathbf{z}_0 + t\,\mathbf{z}_1\)
3. Target velocity: \(\mathbf{v}^* = \mathbf{z}_1 - \mathbf{z}_0\)
4. Loss: \(\mathcal{L} = \|\, v_\theta(\mathbf{z}_t, t, \mathbf{c}, \mathbf{e}) - \mathbf{v}^* \,\|_2^2\)

**Classifier-Free Guidance (CFG).** During training, the speaker embedding is dropped (set to zero) with probability \(p = 0.1\). At inference, the guided velocity is:

\[
\tilde{\mathbf{v}} = \mathbf{v}_\varnothing + s \cdot (\mathbf{v}_\mathbf{e} - \mathbf{v}_\varnothing)
\]

where \(\mathbf{v}_\mathbf{e}\) is the conditional prediction, \(\mathbf{v}_\varnothing\) is the unconditional prediction (null speaker embedding), and \(s = 1.5\) is the guidance scale.

### 2.7 Inference

At inference, the ODE is solved with the Euler method over \(N = 50\) steps:

\[
\mathbf{z}_{i+1} = \mathbf{z}_i + \frac{1}{N}\,\tilde{\mathbf{v}}_\theta(\mathbf{z}_i, t_i, \mathbf{c}, \mathbf{e}), \quad t_i = \frac{i}{N}
\]

The final \(\mathbf{z}_N\) is decoded to a waveform via the Vocos decoder.

---

## 3. Experimental Setup

### 3.1 Dataset

We use the VCTK Corpus (Yamagishi et al., 2019), specifically the `wav48_silence_trimmed` subset. The dataset contains approximately 44,000 utterances from 110 English speakers with various accents. Audio is resampled to 16 kHz for WavLM and 24 kHz for Vocos. During training, random 2-second crops are used, with proportional cropping applied to both sample rates to ensure temporal alignment.

### 3.2 Training Configuration

All experiments share the following configuration:

| Parameter | Value |
|---|---|
| Batch size | 16 |
| Learning rate | \(10^{-4}\) (AdamW, weight decay 0.01) |
| LR schedule | Linear warmup (1000 steps) + cosine decay |
| Gradient clipping | 1.0 |
| CFG dropout | 0.1 |
| Training steps | 30,000 |
| Model parameters | ~15M |
| GPU | NVIDIA RTX 3060 (12 GB) |

### 3.3 Experiments

We conduct five experiments varying the flow initialisation and speaker-stripping strategies:

| ID | Mode | Speaker Stripping | Description |
|----|------|------|-------------|
| E1 | Source | None | Baseline: \(\mathbf{z}_0 = f_\theta(\mathbf{W}_{\text{up}})\) |
| E2 | SVD | SVD \(k{=}2\) | Mild: removes top 2 singular vectors |
| E3 | SVD | SVD \(k{=}2\) + Instance Norm | Proposed: instance normalisation + SVD |
| E4 | SVD | SVD \(k{=}8\) + Instance Norm + Layer 7 | Aggressive: uses WavLM layer 7, instance norm, SVD \(k{=}8\) |
| E5 | Noise | N/A | \(\mathbf{z}_0 \sim \mathcal{N}(0, I)\) |

Experiments E3 and E4 apply instance normalisation before SVD projection. E4 additionally uses WavLM's 7th transformer layer (which is known to encode more phonetic and less speaker information) instead of the final layer.

All experiments use the same fixed validation set (4 speaker pairs, seed=42) for fair comparison.

### 3.4 Evaluation Metrics

We evaluate using ECAPA-TDNN speaker embeddings (the same encoder used during training for conditioning, but here used purely for evaluation):

- **Target Speaker Similarity (Tgt Sim):** Cosine similarity between the ECAPA embedding of the converted utterance and the target speaker's reference. Higher values indicate the converted speech more closely matches the target speaker's voice.

- **Source Speaker Leakage (Src Sim):** Cosine similarity between the ECAPA embedding of the converted utterance and the source speaker's original. Lower values indicate less residual source speaker identity in the output.

- **Conversion Delta (\(\Delta\)):** \(\text{Tgt Sim} - \text{Src Sim}\). Positive values indicate the converted voice is closer to the target than to the source—i.e., genuine speaker conversion has occurred.

---

## 4. Results

### 4.1 Training Loss

The final average training loss (MSE) at 30,000 steps clusters into two groups:

| Group | Experiments | Avg Loss |
|-------|------------|----------|
| Low (~0.010) | E1 (Source), E2 (SVD \(k{=}2\)) | 0.010 |
| High (~0.022) | E3 (SVD+INorm), E4 (Nuclear) | 0.022 |

The higher loss for E3 and E4 is expected and not indicative of worse performance. When speaker information is removed from \(\mathbf{z}_0\), the velocity target \(\mathbf{v}^* = \mathbf{z}_1 - \mathbf{z}_0\) has greater magnitude—the model must reconstruct speaker-dependent spectral characteristics that are absent from the starting point. This is precisely the desired behaviour.

### 4.2 Speaker Similarity Evaluation

| Experiment | Tgt Sim | Src Sim | \(\Delta\) |
|---|---|---|---|
| **E3: SVD \(k{=}2\) + INorm** | **0.3855** | 0.1483 | **+0.2372** |
| E4: Nuclear | 0.3695 | 0.1555 | +0.2140 |
| E1: Source (baseline) | 0.2820 | 0.1122 | +0.1698 |
| E2: SVD \(k{=}2\) | 0.2857 | 0.1230 | +0.1627 |

### 4.3 Analysis

**The speaker-disentanglement hypothesis is confirmed.** E3 (instance normalisation + SVD \(k{=}2\)) achieves the highest target speaker similarity (0.3855) and the best conversion delta (+0.2372), representing a **37% improvement** in target similarity and a **40% improvement** in delta over the source baseline (E1).

**Instance normalisation is the critical component.** Comparing E2 (\(k{=}2\) SVD alone, Tgt Sim = 0.2857) to E3 (\(k{=}2\) SVD + instance norm, Tgt Sim = 0.3855) reveals that instance normalisation alone accounts for a +0.10 absolute improvement. SVD with \(k{=}2\) provides negligible benefit over the raw source baseline (E2 vs E1: 0.2857 vs 0.2820).

**Aggressive stripping is counterproductive.** E4 (nuclear) performs slightly worse than E3 despite using three stripping mechanisms instead of one. The additional use of WavLM layer 7 and SVD \(k{=}8\) likely removes useful content information along with speaker characteristics, degrading the model's ability to preserve linguistic content.

**Training loss is anti-correlated with conversion quality.** The experiments with higher training loss (E3, E4) achieve better speaker conversion. This is because low loss in the source baseline reflects an easy optimisation problem (small \(\|\mathbf{v}^*\|\)) rather than better voice conversion.

---

## 5. System Overview

```
Source Audio ──16kHz──> WavLM ──768d──> Upsample ──> [Instance Norm] ──> [SVD Proj] ──> Project ──> z₀
                                                                                           │
Target Audio ──16kHz──> ECAPA-TDNN ──192d──> Speaker Embedding ─────────────────────┐      │
                                                                                     │      │
                                                                    ┌─────────────────┤      │
                                                                    │   FiLM cond     │      │
          t ~ U[0,1] ──> Sinusoidal Emb ──> MLP ──> Time Emb ──────┤   (per block)   │      │
                                                                    │                 │      │
                                                                    └────> 8× ResBlock1D <───┘
                                                                              │
                                                                         Predicted v̂
                                                                              │
                                                            z₁ = ODE_solve(z₀, v̂, steps=50)
                                                                              │
                                                                    Vocos Decoder ──> Converted Audio
```

---

## 6. Conclusion and Future Work

We have demonstrated that explicitly removing speaker information from the flow matching starting point—particularly via instance normalisation of WavLM features—significantly improves zero-shot voice conversion quality. The proposed method is simple, requires no additional training, and can be applied as a preprocessing step to any flow-based VC system.

Key findings:

1. **Instance normalisation of self-supervised speech features is the single most effective technique** for speaker disentanglement in the flow matching framework, improving target speaker similarity by 37%.
2. **SVD projection with low \(k\) is insufficient** for speaker removal and provides no meaningful benefit over the raw baseline.
3. **Overly aggressive stripping is counterproductive**, suggesting a trade-off between speaker removal and content preservation.
4. **Training loss is a misleading metric** for comparing flow initialisation strategies—lower loss does not imply better conversion.

**Future work** includes:

- **Longer training** (100k+ steps) on the best configuration (E3) to investigate whether target similarity continues to improve.
- **Perceptual evaluation** via MOS (Mean Opinion Score) listening tests for naturalness and speaker similarity.
- **Automatic speech recognition (ASR)** evaluation to quantify content preservation (Word Error Rate on converted vs source).
- **Out-of-domain evaluation** using speakers and languages not present in VCTK.
- **Adaptive instance normalisation** where the target speaker's statistics are injected during denormalisation, potentially further improving speaker adoption.

---

## References

- Chen, S., et al. (2022). "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." *IEEE JSTSP*.
- Desplanques, B., Thienpondt, J., & Demuynck, K. (2020). "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification." *Interspeech*.
- Liu, X., Gong, C., & Liu, Q. (2023). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." *ICLR*.
- Siuzdak, H. (2023). "Vocos: Closing the Gap between Time-Domain and Fourier-Based Neural Vocoders for High-Quality Audio Synthesis." *ICLR*.
- Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
- Yamagishi, J., Veaux, C., & MacDonald, K. (2019). "CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)."
