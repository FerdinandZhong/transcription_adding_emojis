"""Data layer: schema, synthetic generator, real-corpus adapters, and the
silver-label pipeline (ASR + SER + LLM emoji insertion)."""

from .schema import Example, EmojiDataset, Vocab, collate
from .io import load_jsonl, save_jsonl, split_stats
from .loader import load_train_dev_test

__all__ = [
    "Example",
    "EmojiDataset",
    "Vocab",
    "collate",
    "load_jsonl",
    "load_train_dev_test",
    "save_jsonl",
    "split_stats",
]
