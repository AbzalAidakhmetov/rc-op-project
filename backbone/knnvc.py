"""kNN-VC quality backbone (the distillation teacher).

Wraps the pretrained models from the official ``bshall/knn-vc`` torch.hub repo:

    https://github.com/bshall/knn-vc   (hubconf.py + matcher.py)

The pipeline runs entirely in WavLM-Large feature space at 16 kHz -- NO mel,
NO Vocos:

    source wav --WavLM-Large layer 6--> query feats (Tq, 1024)
    target ref wav(s) --WavLM-Large layer 6--> matching pool (Np, 1024)
    per query frame: cosine top-k (k=4) into the pool, AVERAGE the k neighbours
        -> converted feats (Tq, 1024)
    converted feats --prematched HiFi-GAN--> waveform (16 kHz)

Everything is frozen / no-grad.

EXACT torch.hub calls used (verified against the real repo's hubconf.py):

    wavlm  = torch.hub.load('bshall/knn-vc', 'wavlm_large',
                            trust_repo=True, device=<device>)
    hifigan = torch.hub.load('bshall/knn-vc', 'hifigan_wavlm',
                             prematched=True, trust_repo=True, device=<device>)

The hub's ``hifigan_wavlm`` entrypoint returns a ``(generator, config)`` tuple;
we keep both (the config holds ``sampling_rate`` == 16000).

WavLM feature extraction follows the repo's ``KNeighborsVC.get_features``:

    feats = wavlm.extract_features(wav_16khz, output_layer=6,
                                   ret_layer_results=False)[0]   # (1, T, 1024)
    feats = feats.squeeze(0)                                     # (T, 1024)

with ``wav_16khz`` shaped ``(channels, T)`` (i.e. ``(1, T)``).

HiFi-GAN vocoding follows ``KNeighborsVC.vocode``:

    y = hifigan(feats[None])     # feats (T, 1024) -> input (1, T, 1024)
    wav = y.squeeze(1)           # (1, T_wav) -> we return (T_wav,)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import torch
import torchaudio

# Reuse the project's shared audio loader + kNN matcher so the backbone and the
# distillation pipeline use the EXACT same matcher implementation.
from utils import load_audio, batched_cosine_knn

# WavLM layer 6 (1-indexed, as in the kNN-VC paper) is the VC sweet spot.
WAVLM_LAYER = 6
WAVLM_SR = 16000

WavOrPath = Union[str, Path, torch.Tensor]


class KNNVC:
    """Pretrained kNN-VC teacher: WavLM-Large + prematched HiFi-GAN.

    All models are frozen and run under ``torch.no_grad``.

    Args:
        device: torch device string (default ``'cuda'``).
        prematched: use the HiFi-GAN trained on prematched data (recommended; the
            converter / kNN outputs are in-distribution for it).
        topk: default number of neighbours for kNN averaging.
        progress: show download progress bars when fetching checkpoints.
        cache_dir: optional dir to use as the torch.hub cache (checkpoints land
            under ``<cache_dir>/checkpoints``). Defaults to the standard hub dir.
    """

    def __init__(
        self,
        device: str = "cuda",
        prematched: bool = True,
        topk: int = 4,
        progress: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.topk = int(topk)
        self.sr = WAVLM_SR
        self.layer = WAVLM_LAYER

        if cache_dir is not None:
            cache_dir = str(Path(cache_dir).absolute())
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(cache_dir)

        # --- WavLM-Large (feature encoder) -------------------------------
        self.wavlm = torch.hub.load(
            "bshall/knn-vc",
            "wavlm_large",
            trust_repo=True,
            progress=progress,
            device=str(self.device),
        )
        self.wavlm.eval()
        for p in self.wavlm.parameters():
            p.requires_grad = False

        # --- prematched HiFi-GAN (vocoder) -------------------------------
        # The hub entrypoint returns (generator, config).
        hifigan, hifigan_cfg = torch.hub.load(
            "bshall/knn-vc",
            "hifigan_wavlm",
            prematched=prematched,
            trust_repo=True,
            progress=progress,
            device=str(self.device),
        )
        self.hifigan = hifigan.eval()
        self.hifigan_cfg = hifigan_cfg
        for p in self.hifigan.parameters():
            p.requires_grad = False

        # HiFi-GAN config reports the (16 kHz) sampling rate it reconstructs.
        cfg_sr = getattr(hifigan_cfg, "sampling_rate", None)
        if cfg_sr is not None:
            self.sr = int(cfg_sr)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _to_wav_tensor(self, wav_or_path: WavOrPath) -> torch.Tensor:
        """Return a mono 1-D 16 kHz float tensor for `wav_or_path`."""
        if isinstance(wav_or_path, (str, Path)):
            wav = load_audio(str(wav_or_path), target_sr=self.sr)
        else:
            wav = wav_or_path
            if not torch.is_tensor(wav):
                wav = torch.as_tensor(wav)
            wav = wav.float()
            if wav.dim() > 1:
                # collapse any channel dims to mono 1-D
                wav = wav.reshape(wav.shape[0] if wav.shape[0] < wav.shape[-1] else -1, -1)
                wav = wav.mean(0) if wav.dim() == 2 else wav.reshape(-1)
        return wav.reshape(-1)

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_features(self, wav_or_path: WavOrPath) -> torch.Tensor:
        """WavLM-Large layer-6 features for one utterance.

        Returns: ``(T, 1024)`` on ``self.device``.
        """
        wav = self._to_wav_tensor(wav_or_path)            # (T_audio,)
        x = wav.unsqueeze(0).to(self.device)              # (1, T_audio) = (channels, T)
        feats = self.wavlm.extract_features(
            x, output_layer=self.layer, ret_layer_results=False
        )[0]                                              # (1, T, 1024)
        return feats.squeeze(0)                           # (T, 1024)

    @torch.no_grad()
    def build_pool(self, wavs_or_paths: List[WavOrPath]) -> torch.Tensor:
        """Concatenated WavLM features of all reference utterances.

        Returns: ``(Np, 1024)`` on ``self.device``.
        """
        if isinstance(wavs_or_paths, (str, Path, torch.Tensor)):
            wavs_or_paths = [wavs_or_paths]
        feats = [self.get_features(w) for w in wavs_or_paths]
        return torch.cat(feats, dim=0)

    # ------------------------------------------------------------------
    # matching
    # ------------------------------------------------------------------
    @torch.no_grad()
    def match_features(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        """kNN-average matching IN FEATURE SPACE (before vocoding).

        For each query frame, find the `topk` cosine-nearest pool vectors and
        average them. This is the supervision target for the neural converter.

        Args:
            query: ``(Tq, 1024)``
            pool:  ``(Np, 1024)``
            topk:  number of neighbours (defaults to ``self.topk``).
        Returns:
            ``(Tq, 1024)`` converted features on ``self.device``.
        """
        k = self.topk if topk is None else int(topk)
        query = query.to(self.device)
        pool = pool.to(self.device)
        return batched_cosine_knn(query, pool, topk=k)

    # ------------------------------------------------------------------
    # vocoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def vocode(self, feats: torch.Tensor) -> torch.Tensor:
        """Prematched HiFi-GAN vocoder.

        Args:
            feats: ``(T, 1024)`` (or ``(1, T, 1024)``) WavLM-space features.
        Returns:
            1-D waveform tensor (16 kHz) on CPU.
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)                    # (1, T, 1024)
        feats = feats.to(self.device)
        y = self.hifigan(feats)                           # (1, 1, T_wav)
        wav = y.squeeze(1).squeeze(0)                     # (T_wav,)
        return wav.detach().cpu()

    # ------------------------------------------------------------------
    # full pipeline
    # ------------------------------------------------------------------
    @torch.no_grad()
    def convert(
        self,
        source_wav: WavOrPath,
        ref_wavs: Union[WavOrPath, List[WavOrPath]],
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        """Full kNN-VC conversion: features -> kNN match -> vocode.

        Args:
            source_wav: content/source utterance (path or 1-D 16 kHz tensor).
            ref_wavs: one or more target-speaker reference utterances.
            topk: neighbours for kNN averaging (defaults to ``self.topk``).
        Returns:
            converted 1-D waveform (16 kHz) on CPU.
        """
        if isinstance(ref_wavs, (str, Path, torch.Tensor)):
            ref_wavs = [ref_wavs]
        query = self.get_features(source_wav)             # (Tq, 1024)
        pool = self.build_pool(ref_wavs)                  # (Np, 1024)
        converted = self.match_features(query, pool, topk=topk)  # (Tq, 1024)
        return self.vocode(converted)                     # (T_wav,)
