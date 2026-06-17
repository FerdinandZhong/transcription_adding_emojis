"""Models: text encoders, prosody encoder, cross-modal fusion, and the two heads."""

from .fusion_model import EmojiInsertionModel, ModelConfig, decode_predictions

__all__ = ["EmojiInsertionModel", "ModelConfig", "decode_predictions"]
