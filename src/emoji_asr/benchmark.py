"""Human-validated benchmark construction.

Workflow:

1. :func:`sample_benchmark` -- draw a test set from the silver pool, *oversampling*
   prosody-divergent and sarcasm utterances (the cases that isolate prosody's value).
2. :func:`export_for_annotation` -- write a JSONL annotation task (transcript, audio
   path, emoji palette, and a hidden silver suggestion) for human raters.
3. :func:`import_annotations` -- read corrected labels back into :class:`Example`s.
4. :func:`inter_annotator_agreement` -- Cohen's kappa on insertion (per token) and on
   emoji *emotion* class (at agreed insertion points).

The validated examples + emoji set + VA mapping form the released benchmark.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .emoji_set import EmojiSet
from .data.schema import Example


def sample_benchmark(examples: List[Example], n: int,
                     divergent_target: float = 0.4, sarcasm_target: float = 0.15,
                     seed: int = 13) -> List[Example]:
    """Stratified sample that oversamples divergent and sarcasm utterances."""
    rng = np.random.default_rng(seed)
    divergent = [e for e in examples if e.divergent]
    sarcasm = [e for e in examples if e.emotion == "sarcasm" and not e.divergent]
    other = [e for e in examples if not e.divergent and e.emotion != "sarcasm"]

    n_div = min(len(divergent), int(round(n * divergent_target)))
    n_sar = min(len(sarcasm), int(round(n * sarcasm_target)))
    n_oth = max(0, n - n_div - n_sar)

    def take(pool, k):
        if k <= 0 or not pool:
            return []
        idx = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return [pool[i] for i in idx]

    chosen = take(divergent, n_div) + take(sarcasm, n_sar) + take(other, n_oth)
    rng.shuffle(chosen)
    return chosen


def export_for_annotation(examples: List[Example], path: str, emoji_set: EmojiSet,
                          include_silver_hint: bool = True) -> None:
    """Write a JSONL annotation task; gold labels are NOT exposed to annotators."""
    palette = emoji_set.id_to_char[1:]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            item = {
                "uid": ex.uid,
                "transcript": " ".join(ex.words),
                "tokens": ex.words,
                "audio_path": ex.audio_path,
                "palette": palette,
                "instructions": ("Place at most one emoji after a token to convey the "
                                 "speaker's emotion as heard in the audio."),
            }
            if include_silver_hint:
                item["silver_suggestion"] = {
                    "position": next((j for j, v in enumerate(ex.insertion) if v == 1), -1),
                    "emoji": next((emoji_set.char(e) for e in ex.emoji_ids if e > 0), ""),
                }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def import_annotations(path: str, examples: List[Example],
                       emoji_set: EmojiSet) -> List[Example]:
    """Read corrected annotations (one JSON object per line) into validated examples.

    Each line: ``{"uid", "position": int, "emoji": str}``. Examples without a matching
    annotation are dropped.
    """
    by_uid = {ex.uid: ex for ex in examples}
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ex = by_uid.get(rec["uid"])
            if ex is None:
                continue
            n = ex.num_words
            insertion = [0] * n
            emoji_ids = [0] * n
            pos = int(rec.get("position", -1))
            ch = (rec.get("emoji") or "").strip()
            if 0 <= pos < n and ch in emoji_set.char_to_id:
                insertion[pos] = 1
                emoji_ids[pos] = emoji_set.id(ch)
            out.append(Example(uid=ex.uid, words=ex.words, prosody=ex.prosody,
                               insertion=insertion, emoji_ids=emoji_ids,
                               emotion=ex.emotion, divergent=ex.divergent,
                               audio_path=ex.audio_path, meta={"validated": True}))
    return out


def inter_annotator_agreement(annotations: Dict[str, List[Example]],
                              emoji_set: EmojiSet) -> Dict[str, float]:
    """Average pairwise Cohen's kappa across annotators.

    ``annotations`` maps annotator id -> aligned list of Examples (same order/uids).
    Returns kappa for the insertion decision (per token) and for the emoji emotion
    class at positions where both annotators inserted an emoji.
    """
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    ann_ids = list(annotations.keys())
    if len(ann_ids) < 2:
        return {"insertion_kappa": float("nan"), "emoji_emotion_kappa": float("nan"),
                "n_annotators": len(ann_ids)}

    ins_kappas, emo_kappas = [], []
    for a, b in combinations(ann_ids, 2):
        exa, exb = annotations[a], annotations[b]
        ins_a, ins_b, emo_a, emo_b = [], [], [], []
        for ea, eb in zip(exa, exb):
            for ia, ib in zip(ea.insertion, eb.insertion):
                ins_a.append(ia)
                ins_b.append(ib)
            for ja, jb in zip(ea.emoji_ids, eb.emoji_ids):
                if ja > 0 and jb > 0:
                    emo_a.append(emoji_set.emotion_of(ja))
                    emo_b.append(emoji_set.emotion_of(jb))
        if ins_a:
            ins_kappas.append(cohen_kappa_score(ins_a, ins_b))
        if emo_a and len(set(emo_a + emo_b)) > 1:
            emo_kappas.append(cohen_kappa_score(emo_a, emo_b))
    return {
        "insertion_kappa": float(np.mean(ins_kappas)) if ins_kappas else float("nan"),
        "emoji_emotion_kappa": float(np.mean(emo_kappas)) if emo_kappas else float("nan"),
        "n_annotators": len(ann_ids),
    }
