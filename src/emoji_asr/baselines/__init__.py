"""Baselines and ablations.

* ``text_only`` -- the fusion model with the prosody stream disabled (built via
  ``ModelConfig(use_prosody=False)``); see ``build_text_only_config``.
* ``ser_mapping`` -- Speejis-style fixed VA->emoji lookup, no learned placement.
* ``annotator`` -- LLM / rule-based annotator used directly as a predictor
  (text-only = idea a, fusion = idea c), enabling zero-training baselines.
"""

from .ser_mapping import ser_mapping_predict
from .annotator_baseline import annotator_predict
from .text_only import build_text_only_config

__all__ = ["ser_mapping_predict", "annotator_predict", "build_text_only_config"]
