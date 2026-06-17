"""Serialization utilities for processed emoji-ASR datasets."""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from .schema import Example


def example_to_record(ex: Example) -> Dict:
    """Convert an :class:`Example` to a JSON-serializable dict."""
    return {
        "uid": ex.uid,
        "words": ex.words,
        "prosody": np.asarray(ex.prosody, dtype=np.float32).tolist(),
        "insertion": ex.insertion,
        "emoji_ids": ex.emoji_ids,
        "emotion": ex.emotion,
        "divergent": ex.divergent,
        "audio_path": ex.audio_path,
        "meta": ex.meta,
    }


def record_to_example(rec: Dict) -> Example:
    """Convert a JSON record back to :class:`Example`."""
    return Example(
        uid=rec["uid"],
        words=list(rec["words"]),
        prosody=np.asarray(rec["prosody"], dtype=np.float32),
        insertion=list(rec["insertion"]),
        emoji_ids=list(rec["emoji_ids"]),
        emotion=rec.get("emotion", "neutral"),
        divergent=bool(rec.get("divergent", False)),
        audio_path=rec.get("audio_path"),
        meta=dict(rec.get("meta", {})),
    )


def save_jsonl(examples: List[Example], path: str) -> None:
    """Write examples to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(example_to_record(ex), ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Example]:
    """Read examples from a JSONL file."""
    out: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(record_to_example(json.loads(line)))
    return out


def split_stats(examples: List[Example]) -> Dict:
    """Lightweight split summary for sanity checks."""
    n_examples = len(examples)
    n_words = sum(ex.num_words for ex in examples)
    n_insert = sum(sum(ex.insertion) for ex in examples)
    n_divergent = sum(1 for ex in examples if ex.divergent)
    emo_counts: Dict[str, int] = {}
    for ex in examples:
        emo_counts[ex.emotion] = emo_counts.get(ex.emotion, 0) + 1
    return {
        "n_examples": n_examples,
        "n_words": n_words,
        "n_insertions": n_insert,
        "insertion_rate": (n_insert / n_words) if n_words else 0.0,
        "n_divergent": n_divergent,
        "divergent_rate": (n_divergent / n_examples) if n_examples else 0.0,
        "emotion_counts": emo_counts,
    }
