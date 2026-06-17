"""Core data structures for word-level emoji insertion.

An :class:`Example` is one utterance aligned at the *word* level:

* ``words``            -- list of transcript word strings
* ``prosody``          -- word-level acoustic feature matrix ``[T, prosody_dim]``
* ``insertion``        -- per-word 0/1: should an emoji follow this word?
* ``emoji_ids``        -- per-word emoji label id (0 = NO_EMOJI), see ``EmojiSet``
* ``emotion``          -- utterance emotion (gold, for analysis / fidelity)
* ``divergent``        -- True if text is emotion-neutral but the voice is emotional
                          (the prosody-divergent subset)

The two supervised targets mirror the plan's two heads: an *insertion-point* head
(binary, from ``insertion``) and an *emoji-selection* head (which emoji, from
``emoji_ids`` restricted to positions where ``insertion == 1``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:  # torch is required for training/eval but schema stays importable without it
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False
    Dataset = object  # type: ignore


PAD = "<pad>"
UNK = "<unk>"


@dataclass
class Example:
    uid: str
    words: List[str]
    prosody: np.ndarray                  # [T, prosody_dim], float32
    insertion: List[int]                 # [T], 0/1
    emoji_ids: List[int]                 # [T], 0..K
    emotion: str = "neutral"
    divergent: bool = False
    audio_path: Optional[str] = None
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        t = len(self.words)
        assert len(self.insertion) == t, f"{self.uid}: insertion length mismatch"
        assert len(self.emoji_ids) == t, f"{self.uid}: emoji_ids length mismatch"
        self.prosody = np.asarray(self.prosody, dtype=np.float32)
        assert self.prosody.shape[0] == t, f"{self.uid}: prosody rows mismatch"

    @property
    def num_words(self) -> int:
        return len(self.words)

    def rendered(self, emoji_set) -> str:
        """Transcript with gold emoji inserted, for display / semantics scoring."""
        out = []
        for w, e in zip(self.words, self.emoji_ids):
            out.append(w)
            if e > 0:
                out.append(emoji_set.char(e))
        return " ".join(out)


class Vocab:
    """Minimal word-level vocabulary for the lite text encoder."""

    def __init__(self, tokens: Optional[List[str]] = None):
        self.itos: List[str] = [PAD, UNK]
        self.stoi: Dict[str, int] = {PAD: 0, UNK: 1}
        if tokens:
            for t in tokens:
                self.add(t)

    def add(self, token: str) -> int:
        token = token.lower()
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def encode(self, words: List[str]) -> List[int]:
        return [self.stoi.get(w.lower(), 1) for w in words]

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, examples: List[Example], min_freq: int = 1) -> "Vocab":
        from collections import Counter
        c = Counter()
        for ex in examples:
            for w in ex.words:
                c[w.lower()] += 1
        v = cls()
        for tok, freq in c.most_common():
            if freq >= min_freq:
                v.add(tok)
        return v


class EmojiDataset(Dataset):
    """Wraps a list of :class:`Example` with a word :class:`Vocab`."""

    def __init__(self, examples: List[Example], vocab: Vocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return {
            "uid": ex.uid,
            "word_ids": self.vocab.encode(ex.words),
            "words": ex.words,
            "prosody": ex.prosody,
            "insertion": ex.insertion,
            "emoji_ids": ex.emoji_ids,
            "length": ex.num_words,
            "emotion": ex.emotion,
            "divergent": ex.divergent,
        }


def collate(batch: List[dict]):
    """Pad a batch to ``[B, T]`` tensors plus a length mask.

    Returns a dict of tensors; ``mask`` is 1 for real tokens, 0 for padding.
    """
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError("torch is required for collate()")
    b = len(batch)
    lengths = [item["length"] for item in batch]
    t = max(lengths)
    prosody_dim = batch[0]["prosody"].shape[1]

    word_ids = torch.zeros(b, t, dtype=torch.long)
    prosody = torch.zeros(b, t, prosody_dim, dtype=torch.float32)
    insertion = torch.zeros(b, t, dtype=torch.float32)
    emoji_ids = torch.zeros(b, t, dtype=torch.long)
    mask = torch.zeros(b, t, dtype=torch.bool)

    for i, item in enumerate(batch):
        n = item["length"]
        word_ids[i, :n] = torch.tensor(item["word_ids"], dtype=torch.long)
        prosody[i, :n] = torch.tensor(np.asarray(item["prosody"]), dtype=torch.float32)
        insertion[i, :n] = torch.tensor(item["insertion"], dtype=torch.float32)
        emoji_ids[i, :n] = torch.tensor(item["emoji_ids"], dtype=torch.long)
        mask[i, :n] = True

    return {
        "uid": [item["uid"] for item in batch],
        "words": [item["words"] for item in batch],
        "word_ids": word_ids,
        "prosody": prosody,
        "insertion": insertion,
        "emoji_ids": emoji_ids,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "emotion": [item["emotion"] for item in batch],
        "divergent": [item["divergent"] for item in batch],
    }
