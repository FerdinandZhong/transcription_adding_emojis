"""Load train/dev/test splits from the experiment config."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..emoji_set import EmojiSet
from .io import load_jsonl
from .schema import Example
from .synthetic import make_splits


def load_train_dev_test(cfg: dict, emoji_set: Optional[EmojiSet] = None) -> Tuple[
    List[Example], List[Example], List[Example]
]:
    """Return ``(train, dev, test)`` according to ``cfg["data"]["source"]``."""
    data = cfg.get("data", {})
    source = data.get("source", "synthetic")
    seed = cfg.get("seed", 13)
    prosody_dim = cfg.get("model", {}).get("prosody_dim", 32)
    es = emoji_set or EmojiSet()

    if source == "synthetic":
        sc = data["synthetic"]
        return make_splits(
            sc["n_train"],
            sc["n_dev"],
            sc["n_test"],
            divergent_test_fraction=sc.get("divergent_test_fraction", 0.5),
            prosody_dim=prosody_dim,
            seed=seed,
            emoji_set=es,
        )

    if source == "jsonl":
        paths = data["jsonl"]
        return (
            load_jsonl(paths["train"]),
            load_jsonl(paths["dev"]),
            load_jsonl(paths["test"]),
        )

    raise ValueError(
        f"unknown data source {source!r}; use 'synthetic' or 'jsonl'"
    )
