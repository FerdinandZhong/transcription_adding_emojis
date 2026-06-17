"""Speejis-style human UX study tooling.

Real human studies need people; this module provides the *protocol plumbing*:

* :func:`export_ab_survey` -- write an A/B task pairing a plain transcript with the
  emoji-augmented transcript, randomizing left/right, for expressiveness + preference
  ratings.
* :func:`aggregate_ratings` -- summarize collected ratings (mean expressiveness,
  preference rate, win rate) with simple confidence via bootstrap.
* :func:`simulate_ratings` -- a deterministic mock rater (prefers correct, expressive
  emoji) so the pipeline and tests run end-to-end offline.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from .emoji_set import EmojiSet
from .data.schema import Example


def render_with_preds(ex: Example, pred_seq: List[dict], emoji_set: EmojiSet) -> str:
    out = []
    for j, w in enumerate(ex.words):
        out.append(w)
        if j < len(pred_seq) and pred_seq[j]["insertion"] and pred_seq[j]["emoji_id"] > 0:
            out.append(emoji_set.char(pred_seq[j]["emoji_id"]))
    return " ".join(out)


def export_ab_survey(examples: List[Example], preds: List[List[dict]],
                     emoji_set: EmojiSet, path: str, seed: int = 13) -> None:
    rng = np.random.default_rng(seed)
    with open(path, "w", encoding="utf-8") as f:
        for ex, pred in zip(examples, preds):
            plain = " ".join(ex.words)
            augmented = render_with_preds(ex, pred, emoji_set)
            flip = bool(rng.integers(0, 2))
            item = {
                "uid": ex.uid,
                "option_A": augmented if not flip else plain,
                "option_B": plain if not flip else augmented,
                "augmented_side": "A" if not flip else "B",
                "questions": [
                    "Which version better conveys the speaker's emotion? (A/B)",
                    "Rate expressiveness of each version (1-7).",
                ],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def aggregate_ratings(ratings: List[dict], n_boot: int = 1000,
                      seed: int = 13) -> Dict[str, float]:
    """Aggregate ratings of the form
    ``{"prefers_augmented": bool, "expr_augmented": int, "expr_plain": int}``.
    """
    if not ratings:
        return {"n": 0}
    pref = np.array([1.0 if r["prefers_augmented"] else 0.0 for r in ratings])
    ea = np.array([r["expr_augmented"] for r in ratings], dtype=float)
    ep = np.array([r["expr_plain"] for r in ratings], dtype=float)
    rng = np.random.default_rng(seed)
    boot = [pref[rng.integers(0, len(pref), len(pref))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": len(ratings),
        "preference_augmented": float(pref.mean()),
        "preference_ci95": [float(lo), float(hi)],
        "expressiveness_augmented": float(ea.mean()),
        "expressiveness_plain": float(ep.mean()),
        "expressiveness_gain": float(ea.mean() - ep.mean()),
    }


def simulate_ratings(examples: List[Example], preds: List[List[dict]],
                     emoji_set: EmojiSet, seed: int = 13) -> List[dict]:
    """Mock raters: prefer the augmented version when its emoji matches the gold
    utterance emotion; add bounded noise. For demos / tests only."""
    rng = np.random.default_rng(seed)
    ratings = []
    for ex, pred in zip(examples, preds):
        pred_emoji = next((p["emoji_id"] for p in pred
                           if p["insertion"] and p["emoji_id"] > 0), 0)
        correct = pred_emoji > 0 and emoji_set.emotion_of(pred_emoji) == ex.emotion
        base = 6.0 if correct else (3.5 if pred_emoji > 0 else 3.0)
        expr_aug = float(np.clip(base + rng.normal(0, 0.5), 1, 7))
        expr_plain = float(np.clip(3.0 + rng.normal(0, 0.5), 1, 7))
        ratings.append({
            "uid": ex.uid,
            "prefers_augmented": bool(expr_aug > expr_plain),
            "expr_augmented": round(expr_aug),
            "expr_plain": round(expr_plain),
        })
    return ratings
