"""ASR backends producing words + word-level timestamps.

``PassthroughASR`` is the offline default: it accepts a reference transcript and emits
uniform timestamps, so the rest of the pipeline runs without audio. ``WhisperASR``
wraps openai-whisper for real audio (lazy import; install ``emoji-asr[asr]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ASRWord:
    word: str
    start: float
    end: float


@dataclass
class ASRResult:
    words: List[ASRWord]

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


class ASRBackend:
    def transcribe(self, audio_path: Optional[str] = None,
                   reference_text: Optional[str] = None) -> ASRResult:
        raise NotImplementedError


class PassthroughASR(ASRBackend):
    """Use the reference transcript and synthesize uniform word timestamps."""

    def __init__(self, words_per_second: float = 3.0):
        self.wps = words_per_second

    def transcribe(self, audio_path: Optional[str] = None,
                   reference_text: Optional[str] = None) -> ASRResult:
        if reference_text is None:
            raise ValueError("PassthroughASR requires reference_text")
        toks = reference_text.split()
        dur = 1.0 / self.wps
        words = [ASRWord(w, i * dur, (i + 1) * dur) for i, w in enumerate(toks)]
        return ASRResult(words)


class WhisperASR(ASRBackend):  # pragma: no cover - requires audio + model download
    def __init__(self, model_name: str = "base"):
        import whisper  # type: ignore
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: Optional[str] = None,
                   reference_text: Optional[str] = None) -> ASRResult:
        if audio_path is None:
            raise ValueError("WhisperASR requires audio_path")
        result = self.model.transcribe(audio_path, word_timestamps=True)
        words: List[ASRWord] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(ASRWord(w["word"].strip(), float(w["start"]),
                                     float(w["end"])))
        return ASRResult(words)


def build_asr(backend: str = "passthrough", **kw) -> ASRBackend:
    if backend == "passthrough":
        return PassthroughASR()
    if backend == "whisper":
        return WhisperASR(model_name=kw.get("whisper_model", "base"))
    raise ValueError(f"unknown ASR backend: {backend}")
