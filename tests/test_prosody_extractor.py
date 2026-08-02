"""Offline dry-run for the audio -> word-level prosody path.

Uses a synthetic waveform + a stub SER so the whole WordProsodyExtractor code path runs
without downloading MELD.Raw (~10 GB) or the wav2vec2 model. This guards the wiring that
the real re-run depends on.
"""

import numpy as np
import pytest

from emoji_asr.data.prosody import WordAligner, WordProsodyExtractor, _acoustic_features


class _StubSER:
    """Deterministic VAD from a segment's energy/zcr, standing in for Wav2Vec2SER."""

    def predict_from_samples(self, samples, sr):
        from emoji_asr.data.ser import SERResult
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            return SERResult(0.0, 0.0, 0.0, "neutral")
        v = float(np.tanh(samples.mean() * 5))
        a = float(np.tanh(np.sqrt(np.mean(samples ** 2)) * 3))
        d = float(np.tanh(np.abs(samples).mean() * 2))
        return SERResult(v, a, d, "neutral")


def _write_sine(path, sr=16000, seconds=1.2, freq=220.0):
    sf = pytest.importorskip("soundfile")
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wav = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), wav, sr)
    return str(path)


def test_aligner_char_weighted_covers_duration():
    spans = WordAligner().align(["a", "bbbb", "cc"], duration=3.0)
    assert len(spans) == 3
    assert spans[0].start == 0.0
    assert spans[-1].end == pytest.approx(3.0)
    # longer token gets a longer span
    assert (spans[1].end - spans[1].start) > (spans[0].end - spans[0].start)


def test_acoustic_features_shape_and_finite():
    seg = np.random.default_rng(0).normal(0, 0.1, 4000).astype(np.float32)
    feats = _acoustic_features(seg, sr=16000, n_feats=29)
    assert feats.shape == (29,)
    assert np.isfinite(feats).all()


def test_extractor_produces_word_level_real_prosody(tmp_path):
    path = _write_sine(tmp_path / "utt.wav")
    words = ["hello", "there", "friend"]
    ext = WordProsodyExtractor(ser=_StubSER(), prosody_dim=32)
    mat = ext.extract(path, words)
    assert mat.shape == (3, 32)          # one row per word, 32-dim prosody
    assert np.isfinite(mat).all()
    # VAD dims are populated (not all zero) and acoustic dims are standardized per-file.
    assert np.abs(mat[:, :3]).sum() > 0
    assert mat[:, 3:].std() > 0


def test_extractor_respects_external_spans(tmp_path):
    from emoji_asr.data.prosody import WordSpan
    path = _write_sine(tmp_path / "utt.wav")
    spans = [WordSpan(0.0, 0.6), WordSpan(0.6, 1.2)]
    mat = WordProsodyExtractor(ser=_StubSER(), prosody_dim=16).extract(path, ["a", "b"], word_spans=spans)
    assert mat.shape == (2, 16)


def test_silver_labeler_preserves_real_vad(tmp_path):
    # overwrite_vad=False must keep the extractor's per-word VAD intact.
    from emoji_asr.data.silver_labels import SilverLabeler
    path = _write_sine(tmp_path / "utt.wav")
    words = ["keep", "these", "values"]
    prosody = WordProsodyExtractor(ser=_StubSER(), prosody_dim=32).extract(path, words)
    before = prosody[:, :3].copy()
    ex = SilverLabeler().label("u1", words, prosody, overwrite_vad=False)
    assert np.allclose(ex.prosody[:, :3], before)
