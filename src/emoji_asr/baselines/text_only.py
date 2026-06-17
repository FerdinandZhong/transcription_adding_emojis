"""Text-only ablation: the fusion model with its prosody stream switched off."""

from __future__ import annotations

from typing import Dict

from ..models.fusion_model import ModelConfig


def build_text_only_config(num_emoji: int, vocab_size: int, prosody_dim: int,
                           hidden_size: int = 128, text_encoder: Dict = None) -> ModelConfig:
    return ModelConfig(
        num_emoji=num_emoji,
        vocab_size=vocab_size,
        prosody_dim=prosody_dim,
        hidden_size=hidden_size,
        fusion="none",
        use_prosody=False,
        text_encoder=text_encoder or {"backend": "lite"},
    )
