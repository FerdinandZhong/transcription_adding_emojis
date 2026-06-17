"""Silver-label pipeline: raw utterance -> (ASR) -> SER -> emoji annotator -> Example.

This is the realistic data-construction path for real corpora (MELD/IEMOCAP/MOSEI) and
the implementation of dataset idea (c): the emoji annotator is conditioned on *both* the
transcript and the detected speech emotion. It also runs on synthetic raw text (using
precomputed prosody) so the whole pipeline is testable offline.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..emoji_set import EmojiSet
from .asr import ASRBackend, PassthroughASR
from .llm import EmojiAnnotator, OfflineEmojiAnnotator
from .schema import Example
from .ser import HeuristicSER, SERBackend


class SilverLabeler:
    def __init__(self, asr: Optional[ASRBackend] = None,
                 ser: Optional[SERBackend] = None,
                 annotator: Optional[EmojiAnnotator] = None,
                 emoji_set: Optional[EmojiSet] = None):
        self.emoji_set = emoji_set or EmojiSet()
        self.asr = asr or PassthroughASR()
        self.ser = ser or HeuristicSER()
        self.annotator = annotator or OfflineEmojiAnnotator(emoji_set=self.emoji_set)

    def label(self, uid: str, words: List[str], prosody: np.ndarray,
              audio_path: Optional[str] = None, emotion: str = "neutral",
              divergent: bool = False) -> Example:
        """Label one utterance whose words and word-level prosody are already known.

        ``prosody`` feeds the SER backend (heuristic reads its VAD dims); for real audio
        provide ``audio_path`` and use ``Wav2Vec2SER`` instead.
        """
        if audio_path is not None and hasattr(self.ser, "predict_from_audio"):
            try:
                ser_out = self.ser.predict_from_audio(audio_path)
            except NotImplementedError:
                ser_out = self.ser.predict_from_prosody(prosody)
        else:
            ser_out = self.ser.predict_from_prosody(prosody)
        # Keep the word-level prosody stream aligned with the detected speech state.
        # This lets audio-backed SER influence both labeling and model input features.
        prosody = np.asarray(prosody, dtype=np.float32).copy()
        if prosody.ndim != 2 or prosody.shape[0] != len(words):
            raise ValueError(f"{uid}: invalid prosody shape {prosody.shape}, expected [T, D]")
        if prosody.shape[1] >= 3:
            prosody[:, 0] = ser_out.valence
            prosody[:, 1] = ser_out.arousal
            prosody[:, 2] = ser_out.dominance
        ann = self.annotator.annotate(words, ser=ser_out)
        return Example(uid=uid, words=words, prosody=prosody,
                       insertion=ann.insertion, emoji_ids=ann.emoji_ids,
                       emotion=emotion, divergent=divergent,
                       audio_path=audio_path,
                       meta={"ser_emotion": ser_out.emotion})

    def relabel(self, examples: List[Example]) -> List[Example]:
        """Produce silver labels for examples that already carry words + prosody.

        Useful to (a) build a text-only vs fusion silver set for comparison, and (b)
        measure silver-label noise against gold on synthetic data.
        """
        out = []
        for ex in examples:
            out.append(self.label(ex.uid, ex.words, ex.prosody,
                                  audio_path=ex.audio_path, emotion=ex.emotion,
                                  divergent=ex.divergent))
        return out


def silver_agreement(gold: List[Example], silver: List[Example],
                     emoji_set: EmojiSet) -> dict:
    """How well silver labels match gold (insertion + emoji emotion).

    Reported per group (all / congruent / divergent) so we can show that text-only
    silver labels degrade on prosody-divergent utterances.
    """
    def _acc(pairs):
        ins_correct = ins_total = emo_correct = emo_total = 0
        for g, s in pairs:
            for gi, si in zip(g.insertion, s.insertion):
                ins_total += 1
                ins_correct += int(gi == si)
            for ge, se in zip(g.emoji_ids, s.emoji_ids):
                if ge > 0:
                    emo_total += 1
                    emo_correct += int(emoji_set.emotion_of(se) == emoji_set.emotion_of(ge))
        return {
            "insertion_acc": ins_correct / max(ins_total, 1),
            "emoji_emotion_acc": emo_correct / max(emo_total, 1),
            "n": len(pairs),
        }

    pairs = list(zip(gold, silver))
    return {
        "all": _acc(pairs),
        "congruent": _acc([(g, s) for g, s in pairs if not g.divergent]),
        "divergent": _acc([(g, s) for g, s in pairs if g.divergent]),
    }
