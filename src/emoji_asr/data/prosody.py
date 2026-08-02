"""Real audio -> word-level prosody features.

This module closes the gap between the paper's described pipeline (``word-level VAD
features aligned to the word grid``) and the offline emotion-prior fallback in
``datasets._prosody_from_emotion``. Given an utterance's audio and its words it returns a
``[n_words, prosody_dim]`` matrix of *genuine acoustic* features:

* dims 0..2   : valence, arousal, dominance from a wav2vec2 dimensional SER model, run
                per word segment (not a single utterance value broadcast to every word);
* dims 3..D-1 : framewise acoustic descriptors (log-energy, zero-crossing rate, spectral
                centroid/bandwidth/rolloff/flatness/crest, and band log-energies) pooled
                within each word segment.

Word segments come from a :class:`WordAligner`. The default is a character-duration
weighted uniform split (a documented approximation of forced alignment); callers with
Whisper/forced-alignment timestamps can pass them directly via ``word_spans``.

Design goals: (a) produce real acoustic content, not a relabelled gold-emotion prior;
(b) degrade gracefully and stay unit-testable on a synthetic waveform with a stub SER,
so no ~10 GB download or model fetch is needed to exercise the code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np

_EPS = 1e-8
_TARGET_SR = 16000


class _SERLike(Protocol):
    def predict_from_samples(self, samples: np.ndarray, sr: int): ...


@dataclass
class WordSpan:
    start: float  # seconds
    end: float    # seconds


class WordAligner:
    """Character-duration weighted uniform alignment.

    Splits ``duration`` seconds across ``words`` proportionally to token length, which
    approximates the pacing of natural speech far better than an equal split. This is an
    explicit stand-in for forced alignment; pass real timestamps via ``word_spans`` on
    the extractor when a Whisper/CTC aligner is available.
    """

    def align(self, words: Sequence[str], duration: float) -> List[WordSpan]:
        n = len(words)
        if n == 0 or duration <= 0:
            return [WordSpan(0.0, max(duration, 0.0)) for _ in range(n)]
        weights = np.array([max(len(w), 1) for w in words], dtype=np.float64)
        frac = weights / weights.sum()
        bounds = np.concatenate([[0.0], np.cumsum(frac)]) * duration
        return [WordSpan(float(bounds[i]), float(bounds[i + 1])) for i in range(n)]


def _resample(x: np.ndarray, sr: int, target_sr: int = _TARGET_SR) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return x, sr
    try:  # prefer librosa when the [asr] extra is installed
        import librosa

        return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr), target_sr
    except Exception:
        # Dependency-free linear-interpolation fallback (adequate for feature extraction).
        if x.size < 2:
            return x, target_sr
        new_n = max(int(round(x.size * target_sr / sr)), 1)
        xp = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
        xq = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        return np.interp(xq, xp, x).astype(np.float32), target_sr


def _load_audio(audio_path: str, target_sr: int = _TARGET_SR) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:  # mix down to mono
        wav = wav.mean(axis=1)
    return _resample(np.asarray(wav, dtype=np.float32), sr, target_sr)


def _acoustic_features(seg: np.ndarray, sr: int, n_feats: int) -> np.ndarray:
    """Return exactly ``n_feats`` numpy-only acoustic descriptors for one word segment."""
    seg = np.asarray(seg, dtype=np.float32)
    if seg.size < 8:
        return np.zeros(n_feats, dtype=np.float32)

    rms = float(np.sqrt(np.mean(seg ** 2) + _EPS))
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
    mean_abs = float(np.mean(np.abs(seg)))
    crest = float(np.max(np.abs(seg)) / (rms + _EPS))

    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / sr)
    power = mag ** 2
    psum = float(power.sum() + _EPS)
    centroid = float((freqs * power).sum() / psum)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / psum))
    cumpow = np.cumsum(power)
    rolloff = float(freqs[np.searchsorted(cumpow, 0.85 * cumpow[-1])]) if cumpow[-1] > 0 else 0.0
    gmean = float(np.exp(np.mean(np.log(mag + _EPS))))
    flatness = gmean / (float(np.mean(mag)) + _EPS)

    base = [
        np.log(rms + _EPS), zcr, mean_abs, crest,
        centroid / sr, bandwidth / sr, rolloff / sr, flatness,
    ]
    # Fill remaining dims with band log-energies over a linear frequency partition.
    n_bands = max(n_feats - len(base), 0)
    if n_bands > 0:
        edges = np.linspace(0, power.size, n_bands + 1).astype(int)
        bands = [np.log(power[edges[b]:edges[b + 1]].sum() + _EPS) for b in range(n_bands)]
        base.extend(bands)

    feats = np.asarray(base[:n_feats], dtype=np.float32)
    if feats.size < n_feats:  # pad if base already shorter than requested
        feats = np.concatenate([feats, np.zeros(n_feats - feats.size, dtype=np.float32)])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


class WordProsodyExtractor:
    """Audio + words -> ``[n_words, prosody_dim]`` real word-level prosody matrix."""

    def __init__(self, ser: Optional[_SERLike] = None, prosody_dim: int = 32,
                 target_sr: int = _TARGET_SR, aligner: Optional[WordAligner] = None,
                 normalize_acoustic: bool = True):
        if prosody_dim < 3:
            raise ValueError("prosody_dim must be >= 3 (v,a,d occupy the first 3 dims)")
        self.ser = ser
        self.prosody_dim = prosody_dim
        self.target_sr = target_sr
        self.aligner = aligner or WordAligner()
        self.normalize_acoustic = normalize_acoustic

    def extract(self, audio_path: str, words: Sequence[str],
                word_spans: Optional[Sequence[WordSpan]] = None) -> np.ndarray:
        wav, sr = _load_audio(audio_path, self.target_sr)
        duration = wav.size / float(sr)
        spans = list(word_spans) if word_spans is not None else self.aligner.align(words, duration)
        if len(spans) != len(words):
            raise ValueError(f"got {len(spans)} spans for {len(words)} words")

        n_ac = self.prosody_dim - 3
        rows = []
        for span in spans:
            i0, i1 = int(span.start * sr), int(span.end * sr)
            seg = wav[i0:max(i1, i0 + 1)]
            vad = self._vad(seg, sr)
            ac = _acoustic_features(seg, sr, n_ac) if n_ac > 0 else np.zeros(0, dtype=np.float32)
            rows.append(np.concatenate([vad, ac]).astype(np.float32))

        mat = np.stack(rows) if rows else np.zeros((0, self.prosody_dim), dtype=np.float32)
        if self.normalize_acoustic and mat.shape[0] > 1 and n_ac > 0:
            block = mat[:, 3:]
            std = block.std(axis=0)
            std[std < _EPS] = 1.0
            mat[:, 3:] = (block - block.mean(axis=0)) / std
        return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    def _vad(self, seg: np.ndarray, sr: int) -> np.ndarray:
        if self.ser is None or not hasattr(self.ser, "predict_from_samples"):
            return np.zeros(3, dtype=np.float32)
        try:
            r = self.ser.predict_from_samples(seg, sr)
            return np.asarray([r.valence, r.arousal, r.dominance], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)
