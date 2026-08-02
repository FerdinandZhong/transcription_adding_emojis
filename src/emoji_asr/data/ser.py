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


# MELD is annotated with exactly these seven categories. The SER->emotion label (used for
# the divergent flag and annotator conditioning) is restricted to them: the EMOTION_VAD
# table also holds 'love'/'sarcasm', which are not MELD classes and otherwise capture a
# spurious ~44% plurality of clips under nearest-centroid.
MELD_EMOTIONS = ("neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise")


def _nearest_emotion(vad: Tuple[float, float, float], allowed=None) -> str:
    best, best_d = "neutral", float("inf")
    for emo, centroid in EMOTION_VAD.items():
        if allowed is not None and emo not in allowed:
            continue
        d = va_distance(vad, centroid)
        if d < best_d:
            best_d, best = d, emo
    return best


class SERBackend:
    def predict_from_prosody(self, prosody: np.ndarray) -> SERResult:
        """Pool word-level VAD (dims 0..2 of the prosody matrix) into an utterance VAD.

        Generic and backend-independent: with a real audio extractor those dims already
        hold per-word audio VAD, so pooling them yields the utterance emotion used to
        condition the annotator. Every backend inherits this.
        """
        prosody = np.asarray(prosody, dtype=np.float32)
        vad = prosody[:, :3].mean(axis=0)
        v, a, d = float(vad[0]), float(vad[1]), float(vad[2])
        allowed = getattr(self, "allowed_emotions", None)
        return SERResult(v, a, d, _nearest_emotion((v, a, d), allowed))

    def predict_from_audio(self, audio_path: str) -> SERResult:  # pragma: no cover
        raise NotImplementedError


class HeuristicSER(SERBackend):
    """Offline backend: reads the (emotion-prior) prosody VAD via the base pooling."""


def _build_audeering_emotion_model(model_name: str):
    """Load the audeering dimensional SER model with its *correct* custom head.

    The checkpoint is not a plain ``Wav2Vec2ForSequenceClassification``: it uses a bespoke
    regression head (``EmotionModel`` + ``RegressionHead`` from the model card). Loading it
    via ``AutoModelForAudioClassification`` silently drops the trained head and initializes a
    random one, yielding saturated near-constant VAD. Defined lazily so torch/transformers
    import only when the wav2vec2 backend is actually used.
    """
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model, Wav2Vec2PreTrainedModel,
    )

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            x = self.dropout(features)
            x = torch.tanh(self.dense(x))
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0]
            pooled = torch.mean(hidden, dim=1)
            return self.classifier(pooled)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).eval()
    return processor, model, torch


class Wav2Vec2SER(SERBackend):  # pragma: no cover - requires model + audio
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                 allowed_emotions=MELD_EMOTIONS):
        self.processor, self.model, self._torch = _build_audeering_emotion_model(model_name)
        # Restrict the derived categorical label to MELD's classes (see MELD_EMOTIONS).
        self.allowed_emotions = allowed_emotions

    def predict_from_audio(self, audio_path: str) -> SERResult:
        import soundfile as sf
        wav, sr = sf.read(audio_path)
        if getattr(wav, "ndim", 1) > 1:
            wav = wav.mean(axis=1)
        return self.predict_from_samples(np.asarray(wav, dtype=np.float32), sr)

    def predict_from_samples(self, samples: np.ndarray, sr: int) -> SERResult:
        """Run the dimensional model on an in-memory waveform (used for per-word VAD)."""
        torch = self._torch
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return SERResult(0.0, 0.0, 0.0, "neutral")
        y = self.processor(samples, sampling_rate=sr).input_values[0]
        y = torch.tensor(np.asarray(y, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(y).squeeze(0).tolist()
        # Model outputs [arousal, dominance, valence] in [0, 1]; rescale to [-1, 1] to match
        # the EMOTION_VAD centroid space used by _nearest_emotion.
        a, d, v = (2 * x - 1 for x in out[:3])
        return SERResult(v, a, d, _nearest_emotion((v, a, d), self.allowed_emotions))


def build_ser(backend: str = "heuristic", **kw) -> SERBackend:
    if backend == "heuristic":
        return HeuristicSER()
    if backend == "wav2vec2":
        return Wav2Vec2SER(model_name=kw.get("wav2vec2_model"))
    raise ValueError(f"unknown SER backend: {backend}")
