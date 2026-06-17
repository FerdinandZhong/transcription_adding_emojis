"""Metrics for emoji insertion.

Two prediction targets are scored:

* **Placement** (insertion points): precision / recall / F1 with the positive class
  being "an emoji follows this word" -- the same metric family as punctuation
  restoration, for continuity with the prior paper.
* **Emoji selection** at gold-insertion positions:
  - ``top1`` / ``topk`` exact-match accuracy,
  - ``macro_f1`` over emoji classes,
  - ``semantics_preservation``: predicted emoji shares the gold emoji's *emotion*
    class (multiple emoji may validly express the same utterance), a proxy for the
    downstream-classifier metric of Qiu et al. (2024),
  - ``emotion_fidelity``: predicted emoji's emotion matches the gold *utterance*
    emotion.

``evaluate_model`` runs a model over examples and reports metrics for ``all``,
``congruent`` and ``divergent`` groups, so the multimodal gain on prosody-divergent
utterances is explicit.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..emoji_set import EmojiSet


def placement_prf(gold_ins: List[int], pred_ins: List[int]) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for g, p in zip(gold_ins, pred_ins):
        if g == 1 and p == 1:
            tp += 1
        elif g == 0 and p == 1:
            fp += 1
        elif g == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def emoji_metrics(gold_ids: List[int], pred_top1: List[int], pred_topk: List[List[int]],
                  gold_utt_emotion: List[str], emoji_set: EmojiSet) -> Dict[str, float]:
    """Score emoji selection over gold-insertion positions only."""
    if not gold_ids:
        return {"n": 0, "top1": 0.0, "topk": 0.0, "macro_f1": 0.0,
                "semantics_preservation": 0.0, "emotion_fidelity": 0.0}
    n = len(gold_ids)
    top1 = sum(int(g == p) for g, p in zip(gold_ids, pred_top1)) / n
    topk = sum(int(g in ks) for g, ks in zip(gold_ids, pred_topk)) / n
    sem = sum(int(emoji_set.emotion_of(p) == emoji_set.emotion_of(g))
              for g, p in zip(gold_ids, pred_top1)) / n
    fid = sum(int(emoji_set.emotion_of(p) == e)
              for p, e in zip(pred_top1, gold_utt_emotion)) / n
    try:
        from sklearn.metrics import f1_score
        labels = list(range(1, emoji_set.num_emoji + 1))
        macro_f1 = float(f1_score(gold_ids, pred_top1, labels=labels,
                                  average="macro", zero_division=0))
    except Exception:  # pragma: no cover
        macro_f1 = 0.0
    return {"n": n, "top1": top1, "topk": topk, "macro_f1": macro_f1,
            "semantics_preservation": sem, "emotion_fidelity": fid}


def evaluate_predictions(examples, preds: List[List[dict]], emoji_set: EmojiSet,
                         group: Optional[str] = None) -> Dict[str, Dict]:
    """Aggregate metrics from decoded predictions aligned to ``examples``.

    ``preds[i]`` is a list of per-word dicts (see ``decode_predictions``) for
    ``examples[i]``. Returns placement + emoji metrics for the requested group
    (``None`` = all, ``"congruent"``, ``"divergent"``).
    """
    gold_ins: List[int] = []
    pred_ins: List[int] = []
    gold_ids: List[int] = []
    pred_top1: List[int] = []
    pred_topk: List[List[int]] = []
    gold_utt_emotion: List[str] = []

    for ex, pred in zip(examples, preds):
        if group == "congruent" and ex.divergent:
            continue
        if group == "divergent" and not ex.divergent:
            continue
        for j, word_pred in enumerate(pred):
            gi = ex.insertion[j]
            gold_ins.append(gi)
            pred_ins.append(word_pred["insertion"])
            if gi == 1:
                gold_ids.append(ex.emoji_ids[j])
                pred_top1.append(word_pred["topk"][0])
                pred_topk.append(word_pred["topk"])
                gold_utt_emotion.append(ex.emotion)

    return {
        "placement": placement_prf(gold_ins, pred_ins),
        "emoji": emoji_metrics(gold_ids, pred_top1, pred_topk, gold_utt_emotion, emoji_set),
        "n_examples": sum(1 for ex in examples
                          if group is None
                          or (group == "congruent" and not ex.divergent)
                          or (group == "divergent" and ex.divergent)),
    }


def evaluate_model(model, examples, vocab, emoji_set: EmojiSet, device="cpu",
                   batch_size: int = 32, topk: int = 3,
                   threshold: float = 0.5) -> Dict[str, Dict]:
    """Run ``model`` over ``examples`` and report metrics per group."""
    import torch
    from ..data.schema import EmojiDataset, collate
    from ..models.fusion_model import decode_predictions

    ds = EmojiDataset(examples, vocab)
    model.eval()
    all_preds: List[List[dict]] = []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            items = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
            batch = collate(items)
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(batch)
            all_preds.extend(decode_predictions(out, batch["mask"],
                                                threshold=threshold, topk=topk))
    return {
        "all": evaluate_predictions(examples, all_preds, emoji_set, group=None),
        "congruent": evaluate_predictions(examples, all_preds, emoji_set, group="congruent"),
        "divergent": evaluate_predictions(examples, all_preds, emoji_set, group="divergent"),
    }
