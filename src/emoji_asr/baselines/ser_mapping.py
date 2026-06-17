"""Speejis-style baseline: fixed VA->emoji mapping, no learned text placement.

For each utterance we run SER, and if the voice is clearly emotional we place the
VA-nearest emoji at the end of the transcript. This mirrors Speejis (prosody-only,
fixed lookup) and serves as the "no learned placement / no text" reference point.
"""

from __future__ import annotations

from typing import List, Optional

from ..emoji_set import EmojiSet
from ..data.ser import HeuristicSER, SERBackend


def ser_mapping_predict(examples, emoji_set: Optional[EmojiSet] = None,
                        ser: Optional[SERBackend] = None,
                        arousal_threshold: float = 0.25,
                        topk: int = 3) -> List[List[dict]]:
    emoji_set = emoji_set or EmojiSet()
    ser = ser or HeuristicSER()
    preds: List[List[dict]] = []
    for ex in examples:
        ser_out = ser.predict_from_prosody(ex.prosody)
        strong = ser_out.emotion != "neutral" and abs(ser_out.arousal) >= arousal_threshold
        emoji_id = emoji_set.nearest_by_va(ser_out.valence, ser_out.arousal,
                                           ser_out.dominance) if strong else 0
        seq = []
        n = ex.num_words
        for j in range(n):
            insert_here = strong and j == n - 1
            seq.append({
                "insertion": int(insert_here),
                "emoji_id": emoji_id if insert_here else 0,
                "topk": [emoji_id] + [0] * (topk - 1) if insert_here else [0] * topk,
                "ins_prob": 1.0 if insert_here else 0.0,
            })
        preds.append(seq)
    return preds
