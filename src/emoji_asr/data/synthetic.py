"""Synthetic affective-speech corpus.

The generator fabricates word-level examples with controllable *text/prosody
congruence* so the codebase runs end-to-end without external datasets, GPUs, or API
keys, while still exercising the scientific claim:

* **text-congruent** examples carry the emotion in a surface word -> a text-only model
  can solve them;
* **prosody-divergent** examples have emotion-neutral text but emotional prosody ->
  only a model that reads the acoustic stream can solve them.

Prosody features are ``[T, prosody_dim]``: dims 0..2 are a (noisy) VAD vector for the
utterance emotion; the rest are an emotion-correlated random embedding. Emotional
utterances receive an end-of-utterance emoji; neutral utterances receive none, so the
*insertion* decision is also non-trivial.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..emoji_set import EMOTION_LEXICON, EMOTIONS, EmojiSet, emotion_to_va
from .schema import Example

_FILLERS = [
    "i", "you", "we", "they", "the", "a", "this", "that", "it", "he", "she",
    "went", "saw", "had", "today", "yesterday", "now", "here", "there", "then",
    "meeting", "lunch", "work", "home", "phone", "car", "dog", "movie", "game",
    "and", "but", "so", "because", "when", "after", "before", "really", "just",
    "think", "know", "said", "told", "asked", "came", "left", "got", "made",
]


class SyntheticCorpus:
    def __init__(self, emoji_set: Optional[EmojiSet] = None,
                 prosody_dim: int = 32, seed: int = 13):
        self.emoji_set = emoji_set or EmojiSet()
        self.prosody_dim = prosody_dim
        self.rng = np.random.default_rng(seed)
        # Deterministic per-emotion embedding centroids for the non-VAD prosody dims.
        emb_dim = max(0, prosody_dim - 3)
        self._centroids = {
            emo: self.rng.normal(0, 1, size=emb_dim).astype(np.float32)
            for emo in EMOTIONS
        }

    def _prosody(self, emotion: str, n_words: int) -> np.ndarray:
        v, a, d = emotion_to_va(emotion)
        rows = []
        for _ in range(n_words):
            vad = np.array([v, a, d], dtype=np.float32)
            vad = vad + self.rng.normal(0, 0.08, size=3).astype(np.float32)
            if self.prosody_dim > 3:
                emb = self._centroids[emotion] + self.rng.normal(
                    0, 0.5, size=self.prosody_dim - 3
                ).astype(np.float32)
                row = np.concatenate([vad, emb])
            else:
                row = vad[: self.prosody_dim]
            rows.append(row)
        return np.stack(rows).astype(np.float32)

    def _sentence(self, emotion: str, congruent: bool) -> List[str]:
        n = int(self.rng.integers(4, 9))
        words = list(self.rng.choice(_FILLERS, size=n, replace=True))
        if congruent and emotion != "neutral":
            lex = EMOTION_LEXICON.get(emotion, [])
            if lex:
                pos = int(self.rng.integers(1, n))
                words[pos] = str(self.rng.choice(lex))
        return [str(w) for w in words]

    def _example(self, uid: str, emotion: str, congruent: bool) -> Example:
        words = self._sentence(emotion, congruent)
        n = len(words)
        prosody = self._prosody(emotion, n)
        insertion = [0] * n
        emoji_ids = [0] * n
        if emotion != "neutral":
            ids = self.emoji_set.ids_for_emotion(emotion)
            chosen = int(self.rng.choice(ids)) if ids else 0
            if chosen > 0:
                insertion[-1] = 1
                emoji_ids[-1] = chosen
        divergent = (emotion != "neutral") and (not congruent)
        return Example(
            uid=uid, words=words, prosody=prosody, insertion=insertion,
            emoji_ids=emoji_ids, emotion=emotion, divergent=divergent,
        )

    def generate(self, n: int, split: str = "train",
                 divergent_fraction: float = 0.0,
                 neutral_fraction: float = 0.25) -> List[Example]:
        """Generate ``n`` examples.

        ``divergent_fraction`` is the share of *emotional* examples whose text is made
        neutral (prosody-divergent). For train/dev keep it 0 (or small); for the test
        split set it high to stress-test prosody usage.
        """
        examples: List[Example] = []
        emo_pool = [e for e in EMOTIONS if e != "neutral"]
        for i in range(n):
            if self.rng.random() < neutral_fraction:
                emotion, congruent = "neutral", True
            else:
                emotion = str(self.rng.choice(emo_pool))
                congruent = self.rng.random() >= divergent_fraction
            examples.append(self._example(f"{split}-{i:05d}", emotion, congruent))
        return examples


def make_splits(n_train: int, n_dev: int, n_test: int,
                divergent_test_fraction: float = 0.5,
                prosody_dim: int = 32, seed: int = 13,
                emoji_set: Optional[EmojiSet] = None):
    """Build train/dev/test synthetic splits.

    Train/dev are text-congruent (the easy regime); the test split mixes congruent and
    prosody-divergent examples so we can report the multimodal gain separately.
    """
    corpus = SyntheticCorpus(emoji_set=emoji_set, prosody_dim=prosody_dim, seed=seed)
    train = corpus.generate(n_train, "train", divergent_fraction=0.0)
    dev = corpus.generate(n_dev, "dev", divergent_fraction=0.0)
    test = corpus.generate(n_test, "test", divergent_fraction=divergent_test_fraction)
    return train, dev, test
