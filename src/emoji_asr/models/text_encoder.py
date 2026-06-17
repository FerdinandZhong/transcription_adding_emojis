"""Word-level text encoders.

* :class:`LiteTextEncoder` is a small from-scratch Transformer over a word vocabulary.
  It needs no network or pretrained weights, so it is the default for offline runs and
  CI. It consumes ``batch["word_ids"]``.
* :class:`HFTextEncoder` wraps a HuggingFace encoder (default ModernBERT; any
  BERT/RoBERTa/DeBERTa works) and pools subword states back to the word grid using
  ``word_ids``. Use it for real experiments (``model.text_encoder.backend=hf``). It
  consumes ``batch["words"]``. ModernBERT requires ``transformers>=4.48``.

Both return per-word hidden states ``[B, T, H]`` aligned to the word/prosody axis.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LiteTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_size))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch) -> torch.Tensor:
        word_ids = batch["word_ids"]            # [B, T]
        mask = batch["mask"]                    # [B, T] bool, True = real
        t = word_ids.size(1)
        x = self.embed(word_ids) + self.pos[:, :t, :]
        x = self.dropout(x)
        # TransformerEncoder expects src_key_padding_mask True where padding.
        h = self.encoder(x, src_key_padding_mask=~mask)
        return h


class HFTextEncoder(nn.Module):  # pragma: no cover - requires model download
    def __init__(self, hf_model: str = "answerdotai/ModernBERT-base", dropout: float = 0.1,
                 freeze: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModel.from_pretrained(hf_model)
        self.hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, batch) -> torch.Tensor:
        words: List[List[str]] = batch["words"]
        device = next(self.model.parameters()).device
        t = batch["mask"].size(1)
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt",
                             padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state          # [B, S, H]
        b = out.size(0)
        pooled = torch.zeros(b, t, self.hidden_size, device=device)
        for i in range(b):
            word_ids = self.tokenizer(
                words[i], is_split_into_words=True, truncation=True
            ).word_ids()
            sums = torch.zeros(t, self.hidden_size, device=device)
            counts = torch.zeros(t, 1, device=device)
            for s_idx, w_idx in enumerate(word_ids):
                if w_idx is None or w_idx >= t:
                    continue
                sums[w_idx] += out[i, s_idx]
                counts[w_idx] += 1
            pooled[i] = sums / counts.clamp(min=1.0)
        return self.dropout(pooled)


def build_text_encoder(cfg: dict, vocab_size: int) -> nn.Module:
    backend = cfg.get("backend", "lite")
    if backend == "lite":
        return LiteTextEncoder(
            vocab_size=vocab_size,
            hidden_size=cfg.get("hidden_size", 128),
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 4),
            max_len=cfg.get("max_len", 64),
            dropout=cfg.get("dropout", 0.1),
        )
    if backend == "hf":
        return HFTextEncoder(hf_model=cfg.get("hf_model", "answerdotai/ModernBERT-base"),
                             dropout=cfg.get("dropout", 0.1),
                             freeze=cfg.get("freeze", False))
    raise ValueError(f"unknown text encoder backend: {backend}")
