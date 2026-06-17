"""Speech Emotion Recognition backends emitting dimensional VAD + a categorical label.

``HeuristicSER`` is the offline default: it reads the (already-computed) word-level
prosody features -- dims 0..2 are treated as VAD -- and maps them to the nearest
emotion centroid. ``Wav2Vec2SER`` wraps a Wagner-et-al.-style wav2vec2 dimensional
model for real audio (lazy import; install ``emoji-asr[asr]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..emoji_set import EMOTION_VAD, va_distance


@dataclass
class SERResult:
    valence: float
    arousal: float
    dominance: float
    emotion: str

    @property
    def vad(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)


def _nearest_emotion(vad: Tuple[float, float, float]) -> str:
    best, best_d = "neutral", float("inf")
    for emo, centroid in EMOTION_VAD.items():
        d = va_distance(vad, centroid)
        if d < best_d:
            best_d, best = d, emo
    return best


class SERBackend:
    def predict_from_prosody(self, prosody: np.ndarray) -> SERResult:
        raise NotImplementedError

    def predict_from_audio(self, audio_path: str) -> SERResult:  # pragma: no cover
        raise NotImplementedError


class HeuristicSER(SERBackend):
    """Pool word-level VAD (dims 0..2 of the prosody matrix) into an utterance VAD."""

    def predict_from_prosody(self, prosody: np.ndarray) -> SERResult:
        prosody = np.asarray(prosody, dtype=np.float32)
        vad = prosody[:, :3].mean(axis=0)
        v, a, d = float(vad[0]), float(vad[1]), float(vad[2])
        return SERResult(v, a, d, _nearest_emotion((v, a, d)))


class Wav2Vec2SER(SERBackend):  # pragma: no cover - requires model + audio
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"):
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        self.fe = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.eval()

    def predict_from_audio(self, audio_path: str) -> SERResult:
        import soundfile as sf
        import torch
        wav, sr = sf.read(audio_path)
        inputs = self.fe(wav, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze(0).tolist()
        # Model outputs arousal, dominance, valence in [0, 1]; rescale to [-1, 1].
        a, d, v = (2 * x - 1 for x in logits[:3])
        return SERResult(v, a, d, _nearest_emotion((v, a, d)))


def build_ser(backend: str = "heuristic", **kw) -> SERBackend:
    if backend == "heuristic":
        return HeuristicSER()
    if backend == "wav2vec2":
        return Wav2Vec2SER(model_name=kw.get("wav2vec2_model"))
    raise ValueError(f"unknown SER backend: {backend}")
