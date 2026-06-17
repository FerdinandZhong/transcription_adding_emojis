"""Adapters from real ERC corpora to :class:`Example` via the silver-label pipeline.

Expected layouts (summarized):

* MELD: ``train_sent_emo.csv`` etc. with columns ``Utterance``, ``Emotion``,
  ``Sentiment``, ``Dialogue_ID``, ``Utterance_ID``; audio under a split-specific
  directory containing files named ``dia<Dialogue_ID>_utt<Utterance_ID>.wav``.
* IEMOCAP: per-utterance text + categorical/VAD labels (parse from the release).
* CMU-MOSEI: aligned transcripts + sentiment/emotion intensities.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..emoji_set import EMOTION_LEXICON, EmojiSet, emotion_to_va
from .schema import Example
from .silver_labels import SilverLabeler

# MELD emotion strings -> our taxonomy.
_MELD_EMO = {
    "neutral": "neutral", "joy": "joy", "sadness": "sadness", "anger": "anger",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
}


def _has_text_emotion(words: List[str]) -> bool:
    lex = {w for ws in EMOTION_LEXICON.values() for w in ws}
    return any(w.lower() in lex for w in words)


def _resolve_meld_audio_path(audio_dir: Optional[str], dialogue_id: int,
                             utt_id: int) -> Optional[str]:
    if not audio_dir:
        return None
    cand = os.path.join(audio_dir, f"dia{dialogue_id}_utt{utt_id}.wav")
    return cand if os.path.exists(cand) else None


def _prosody_from_emotion(emotion: str, n_words: int, prosody_dim: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Fallback word-level prosody when audio SER is unavailable: VAD prior + noise."""
    v, a, d = emotion_to_va(emotion)
    rows = []
    for _ in range(n_words):
        vad = np.array([v, a, d], dtype=np.float32) + rng.normal(0, 0.05, 3).astype(np.float32)
        if prosody_dim > 3:
            emb = rng.normal(0, 0.3, prosody_dim - 3).astype(np.float32)
            rows.append(np.concatenate([vad, emb]))
        else:
            rows.append(vad[:prosody_dim])
    return np.stack(rows).astype(np.float32)


def load_meld_split(csv_path: str, audio_dir: Optional[str] = None,
                    labeler: Optional[SilverLabeler] = None,
                    emoji_set: Optional[EmojiSet] = None,
                    prosody_dim: int = 32, seed: int = 13,
                    max_rows: Optional[int] = None,
                    mark_divergent: bool = True,
                    start_row: int = 0) -> List[Example]:
    """Load one MELD split CSV into silver-labeled examples.

    If ``audio_dir`` and an audio-capable SER backend are provided, prosody comes from
    real speech; otherwise a VAD prior derived from the gold emotion label is used.
    """
    import pandas as pd

    emoji_set = emoji_set or EmojiSet()
    labeler = labeler or SilverLabeler(emoji_set=emoji_set)
    rng = np.random.default_rng(seed + max(0, start_row))
    df = pd.read_csv(csv_path)
    examples: List[Example] = []
    n_kept = 0
    for i, row in enumerate(df.iterrows()):
        if i < start_row:
            continue
        if max_rows is not None and n_kept >= max_rows:
            break
        _, row = row
        words = str(row["Utterance"]).split()
        if not words:
            continue
        emotion = _MELD_EMO.get(str(row["Emotion"]).lower(), "neutral")
        did = int(row.get("Dialogue_ID", 0))
        uid_ = int(row.get("Utterance_ID", 0))
        uid = f"meld-{did}-{uid_}"
        audio_path = _resolve_meld_audio_path(audio_dir, did, uid_)
        prosody = _prosody_from_emotion(emotion, len(words), prosody_dim, rng)
        ex = labeler.label(uid, words, prosody, audio_path=audio_path, emotion=emotion)
        if mark_divergent:
            speech_emotion = str(ex.meta.get("ser_emotion", emotion))
            ex.divergent = speech_emotion != "neutral" and (not _has_text_emotion(words))
        ex.meta.update({
            "dataset": "meld",
            "dialogue_id": did,
            "utterance_id": uid_,
            "split_csv": os.path.basename(csv_path),
        })
        examples.append(ex)
        n_kept += 1
    return examples


def load_meld_splits(split_csv_paths: Dict[str, str],
                     split_audio_dirs: Optional[Dict[str, Optional[str]]] = None,
                     labeler: Optional[SilverLabeler] = None,
                     emoji_set: Optional[EmojiSet] = None,
                     prosody_dim: int = 32, seed: int = 13,
                     max_rows_per_split: Optional[int] = None,
                     mark_divergent: bool = True) -> Dict[str, List[Example]]:
    """Load multiple MELD splits with consistent settings."""
    out: Dict[str, List[Example]] = {}
    for split, csv_path in split_csv_paths.items():
        audio_dir = (split_audio_dirs or {}).get(split)
        out[split] = load_meld_split(
            csv_path=csv_path,
            audio_dir=audio_dir,
            labeler=labeler,
            emoji_set=emoji_set,
            prosody_dim=prosody_dim,
            seed=seed,
            max_rows=max_rows_per_split,
            mark_divergent=mark_divergent,
        )
    return out


def load_dataset(source: str, **kw) -> List[Example]:
    """Dispatch to a corpus adapter. ``synthetic`` is handled in ``synthetic.py``."""
    if source == "meld":
        return load_meld_split(**kw)
    raise NotImplementedError(
        f"Adapter for '{source}' is a documented stub; implement parsing for the "
        f"release layout or use source=synthetic for offline runs."
    )
