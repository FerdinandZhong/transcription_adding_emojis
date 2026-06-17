# Paper Outline: Emoji-Augmented Transcription

Target: a 4-6 page conference paper (ICASSP-style) that extends to a journal via a
human-validated benchmark and cross-lingual experiments. Continuity with the prior work,
*Punctuation Restoration: A Case Study of BERT-Based Models' Task-Specific Excellence*
(ICASSP 2025), is explicit: the same token-classification skeleton, now affective and
multimodal.

## Title
Emoji-Augmented Transcription: Multimodal Emoji Insertion for Affective ASR

## Abstract
ASR transcripts strip the emotion carried by the voice. We cast *emoji insertion* as
affective punctuation restoration: a per-token decision of *whether* to emit an emoji and
*which* one, but conditioned on prosody as well as text. We build a silver+human-validated
emoji-speech dataset by conditioning an LLM annotator on both the transcript and detected
speech emotion, and a cross-modal token-classification model with insertion and emoji
heads. On a prosody-divergent test subset (emotional voice, neutral words) the multimodal
model preserves the intended emotion far better than text-only insertion, and a user study
shows a strong preference for augmented transcripts.

## 1. Introduction
- ASR -> readable text via punctuation restoration; but the *affective* channel is lost.
- Contribution statement (3 bullets): multimodal token-level model; idea-(c) dataset;
  emotion-aware evaluation with a prosody-divergent subset + UX study.
- The gap table (see Related Work): no prior work fuses prosody + transcript + learned
  placement + reproducible benchmark.

## 2. Related Work
Use [`related_work.md`](related_work.md): punctuation restoration; text-only emoji
prediction (EmojiLM, SENTIMOJI); VoiceMoji (text-only by design); Speejis (prosody-only,
fixed map, UX study); multimodal ERC + datasets; semantics-preserving evaluation.

## 3. Task Formulation
- Word-aligned utterance; per-word labels: insertion in {0,1} and emoji id in {0..K}.
- Two sub-objectives mirroring punctuation restoration: placement (where) + selection
  (which). Emoji set: ~64 faces grounded in the Kutsuzawa VA circumplex.

## 4. Dataset Construction (idea c)
- Base corpora with aligned audio+text+emotion: MELD, IEMOCAP, CMU-MOSEI (+ M3ED for the
  cross-lingual journal extension).
- Pipeline: Whisper ASR + word timestamps -> wav2vec2 VAD SER -> LLM emoji annotator
  conditioned on transcript *and* detected speech emotion. Text-only (a) and
  speech-mapping (b) are produced as ablations of the same pipeline.
- Human-validated test benchmark: oversample prosody-divergent / sarcasm utterances;
  report inter-annotator agreement (Cohen's kappa); release set + emoji set + VA map.
- Implemented by `data/silver_labels.py`, `data/datasets.py`, `benchmark.py`.

## 5. Model
- Text encoder (ModernBERT by default; any BERT/RoBERTa/DeBERTa; lite Transformer for
  low-resource/offline) -> word states.
- Prosody encoder over word-level VAD + pooled wav2vec2 features (via timestamps).
- Cross-modal fusion (text queries prosody) -> insertion head (+ positional features for
  the boundary decision) and emoji head. `use_prosody=False` recovers text-only exactly.
- Losses: class-balanced BCE for insertion, cross-entropy for emoji at gold positions.
- Implemented by `models/fusion_model.py`.

## 6. Experiments
- Methods: ours (fusion); ablation text-only; Speejis SER-mapping; LLM annotator (a)/(c);
  [real runs] Whisper end-to-end fine-tune; audio-LLM prompting.
- Metrics: placement P/R/F1; emoji top-k; macro-F1; **semantics preservation**;
  **emotion fidelity**; reported for all / congruent / **divergent** groups.
- Headline: multimodal gain on the divergent subset (prosody recovers lost emotion).
- Synthetic harness reproduces the mechanism end-to-end (`outputs/results.md`):
  fusion 0.986 semantics-preservation on divergent vs 0.102 text-only; placement F1 0.705
  vs 0.444; UX preference 85.75%.

## 7. Human Evaluation
- Speejis-style A/B: plain vs augmented transcript; expressiveness (1-7) + preference.
- Tooling: `human_eval.py` (survey export, bootstrap CIs; simulated raters for dry runs).

## 8. Limitations & Ethics
- Silver-label noise; emoji set size vs sparsity; ASR error propagation (report gold vs
  ASR transcripts); cultural variation in emoji interpretation; demographic bias in SER.

## 9. Conclusion
- Prosody-aware, transcript-anchored, learned emoji insertion closes the gap between
  text-only and prosody-only prior art, with a reproducible benchmark and protocol.

## Reproducibility checklist
- Configs in `configs/`; deterministic seeds; offline synthetic harness; unit tests;
  real-corpus adapters and pretrained-model backends documented in the README.
