"""Annotator-as-predictor baseline (zero training).

Runs an :class:`EmojiAnnotator` (offline rule-based or OpenAI LLM) directly as the
prediction model. With ``condition_on_speech=False`` this is the text-only LLM baseline
(idea a); with ``True`` it is the fusion LLM baseline (idea c). Predictions are emitted
in the same per-word format as the neural decoder so the same metrics apply.
"""

from __future__ import annotations

from typing import List, Optional

from ..emoji_set import EmojiSet
from ..data.llm import EmojiAnnotator, OfflineEmojiAnnotator
from ..data.ser import HeuristicSER, SERBackend


def annotator_predict(examples, emoji_set: Optional[EmojiSet] = None,
                      annotator: Optional[EmojiAnnotator] = None,
                      ser: Optional[SERBackend] = None,
                      condition_on_speech: bool = True,
                      topk: int = 3) -> List[List[dict]]:
    emoji_set = emoji_set or EmojiSet()
    annotator = annotator or OfflineEmojiAnnotator(
        emoji_set=emoji_set, condition_on_speech=condition_on_speech)
    ser = ser or HeuristicSER()
    preds: List[List[dict]] = []
    for ex in examples:
        ser_out = ser.predict_from_prosody(ex.prosody) if condition_on_speech else None
        ann = annotator.annotate(ex.words, ser=ser_out)
        seq = []
        for j in range(ex.num_words):
            inserted = ann.insertion[j]
            emoji_id = ann.emoji_ids[j]
            seq.append({
                "insertion": int(inserted),
                "emoji_id": emoji_id,
                "topk": [emoji_id] + [0] * (topk - 1),
                "ins_prob": 1.0 if inserted else 0.0,
            })
        preds.append(seq)
    return preds
