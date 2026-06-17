"""Multimodal emoji-insertion model.

Architecture (extends BERT punctuation token-classification with a prosody stream):

    words  -> TextEncoder ---\
                              >-- Fusion --> [insertion head] (where?)
    prosody -> ProsodyEncoder/                \\-> [emoji head]  (which emoji?)

* **Insertion head**: per-word binary logit (should an emoji follow this word?).
* **Emoji head**: per-word distribution over the K real emoji; supervised only at
  positions where the gold insertion is 1.

The ``fusion`` mode selects how the streams combine:

* ``cross_attention`` -- text queries the prosody sequence (default),
* ``concat``          -- concatenate + project,
* ``none``            -- text only (the ablation / text-only baseline).

Setting ``use_prosody=False`` at the config level (or zeroing the prosody stream)
recovers the text-only model exactly, giving a clean with/without-prosody ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import build_text_encoder


@dataclass
class ModelConfig:
    num_emoji: int                     # K (real emoji, excluding NO_EMOJI)
    vocab_size: int = 2
    prosody_dim: int = 32
    hidden_size: int = 128
    fusion: str = "cross_attention"    # cross_attention | concat | none
    use_prosody: bool = True
    dropout: float = 0.1
    text_encoder: Dict = field(default_factory=lambda: {"backend": "lite"})

    def __post_init__(self):
        # Keep the lite encoder's hidden size in sync.
        self.text_encoder = dict(self.text_encoder)
        self.text_encoder.setdefault("hidden_size", self.hidden_size)
        self.text_encoder.setdefault("dropout", self.dropout)


class ProsodyEncoder(nn.Module):
    def __init__(self, prosody_dim: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prosody_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, prosody: torch.Tensor) -> torch.Tensor:
        return self.net(prosody)


class EmojiInsertionModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = build_text_encoder(cfg.text_encoder, cfg.vocab_size)
        h = getattr(self.text_encoder, "hidden_size", cfg.hidden_size)
        self.hidden_size = h
        self.prosody_encoder = ProsodyEncoder(cfg.prosody_dim, h, cfg.dropout)

        self.fusion_mode = cfg.fusion if cfg.use_prosody else "none"
        if self.fusion_mode == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(h, num_heads=4, dropout=cfg.dropout,
                                                    batch_first=True)
            self.fuse_norm = nn.LayerNorm(h)
        elif self.fusion_mode == "concat":
            self.fuse_proj = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(),
                                           nn.Dropout(cfg.dropout))

        # Insertion is a boundary decision, so the head also sees positional features
        # (normalized position, distance-from-end, is-last) derived from the mask.
        self._pos_feat_dim = 3
        self.insertion_head = nn.Linear(h + self._pos_feat_dim, 1)
        self.emoji_head = nn.Linear(h, cfg.num_emoji)

    def _fuse(self, text_h: torch.Tensor, prosody: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
        if self.fusion_mode == "none":
            return text_h
        pros_h = self.prosody_encoder(prosody)
        if self.fusion_mode == "cross_attention":
            key_padding = ~mask                       # True where padding
            attn_out, _ = self.cross_attn(text_h, pros_h, pros_h,
                                          key_padding_mask=key_padding,
                                          need_weights=False)
            return self.fuse_norm(text_h + attn_out)
        if self.fusion_mode == "concat":
            return self.fuse_proj(torch.cat([text_h, pros_h], dim=-1))
        raise ValueError(self.fusion_mode)

    @staticmethod
    def _position_features(mask: torch.Tensor) -> torch.Tensor:
        """Per-token [norm_pos, dist_from_end, is_last] from a [B, T] bool mask."""
        b, t = mask.shape
        lengths = mask.sum(dim=1).clamp(min=1)                       # [B]
        idx = torch.arange(t, device=mask.device).unsqueeze(0).expand(b, t).float()
        denom = (lengths - 1).clamp(min=1).unsqueeze(1).float()
        norm_pos = (idx / denom).clamp(0, 1)
        last_idx = (lengths - 1).unsqueeze(1).float()
        dist_end = ((last_idx - idx) / denom).clamp(0, 1)
        is_last = (idx == last_idx).float()
        feats = torch.stack([norm_pos, dist_end, is_last], dim=-1)   # [B, T, 3]
        return feats * mask.unsqueeze(-1).float()

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        mask = batch["mask"]
        text_h = self.text_encoder(batch)
        fused = self._fuse(text_h, batch["prosody"], mask)
        pos_feats = self._position_features(mask)
        ins_in = torch.cat([fused, pos_feats], dim=-1)
        insertion_logits = self.insertion_head(ins_in).squeeze(-1)  # [B, T]
        emoji_logits = self.emoji_head(fused)                       # [B, T, K]
        return {"insertion_logits": insertion_logits,
                "emoji_logits": emoji_logits, "fused": fused}

    def compute_loss(self, batch, out: Dict[str, torch.Tensor],
                     insertion_weight: float = 1.0,
                     emoji_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        mask = batch["mask"]
        ins_logits = out["insertion_logits"]
        ins_target = batch["insertion"]
        # Insertion points are rare (~1 per utterance); reweight the positive class.
        n_pos = (ins_target * mask).sum()
        n_neg = mask.sum() - n_pos
        pos_weight = (n_neg / n_pos.clamp(min=1)).clamp(1.0, 20.0)
        ins_loss = F.binary_cross_entropy_with_logits(
            ins_logits, ins_target, reduction="none", pos_weight=pos_weight)
        ins_loss = (ins_loss * mask).sum() / mask.sum().clamp(min=1)

        # Emoji loss only where gold insertion == 1.
        emoji_logits = out["emoji_logits"]
        emoji_target = (batch["emoji_ids"] - 1).clamp(min=0)        # 0..K-1
        sel = (batch["insertion"] > 0.5) & mask
        if sel.any():
            emoji_loss = F.cross_entropy(emoji_logits[sel], emoji_target[sel])
        else:
            emoji_loss = torch.zeros((), device=ins_logits.device)

        total = insertion_weight * ins_loss + emoji_weight * emoji_loss
        return {"loss": total, "insertion_loss": ins_loss.detach(),
                "emoji_loss": emoji_loss.detach()}


@torch.no_grad()
def decode_predictions(out: Dict[str, torch.Tensor], mask: torch.Tensor,
                       threshold: float = 0.5, topk: int = 1):
    """Convert logits to per-word predictions.

    Returns lists (per example) of dicts: insertion (0/1), emoji_id (0..K, 0 if not
    inserted), and top-k emoji ids (1..K) for the emoji head.
    """
    ins_prob = torch.sigmoid(out["insertion_logits"])
    emoji_logits = out["emoji_logits"]
    k = min(topk, emoji_logits.size(-1))
    topk_ids = emoji_logits.topk(k, dim=-1).indices            # [B, T, k], 0..K-1

    results: List[List[dict]] = []
    b, t = ins_prob.shape
    for i in range(b):
        seq = []
        for j in range(t):
            if not bool(mask[i, j]):
                continue
            inserted = int(ins_prob[i, j] >= threshold)
            top = [int(x) + 1 for x in topk_ids[i, j].tolist()]   # shift to 1..K
            emoji_id = top[0] if inserted else 0
            seq.append({"insertion": inserted, "emoji_id": emoji_id,
                        "topk": top, "ins_prob": float(ins_prob[i, j])})
        results.append(seq)
    return results
