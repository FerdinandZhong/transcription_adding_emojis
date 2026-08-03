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

## 6. Experiments & Results

Full draft prose: [`results_section.md`](results_section.md).

- Setup: MELD silver-label split (9,989 / 1,109 / 2,610); ModernBERT-base; 30 epochs;
  fusion checkpoint (B1) + freshly trained text-only ablation; zero-shot Speejis + offline
  LLM baselines.
- Metrics: placement P/R/F1; emoji top-1; semantics preservation; emotion fidelity;
  stratified all / congruent / **divergent** (45.4% of test).
- Headline (real MELD): fusion divergent SemPres **0.898** vs text-only **0.383**; Top1
  **0.757** vs **0.330**; EmoFidelity **0.915** vs **0.367**. Placement F1 fusion **0.424**
  vs text-only 0.413 (fusion leads on both sub-tasks).
- Synthetic harness (development only): fusion 0.986 SemPres (divergent) vs 0.102 text-only.

**Table 1.** MELD silver-label test results.

| Method | Place F1 (all) | Top1 (all) | Top1 (congruent) | Top1 (divergent) | SemPres (divergent) | EmoFidelity (divergent) |
|---|---:|---:|---:|---:|---:|---:|
| ser_mapping (Speejis) | 0.321 | 0.097 | 0.022 | 0.131 | 0.298 | 0.280 |
| llm_text_only (idea a) | 0.029 | 0.004 | 0.013 | 0.000 | 0.084 | 0.000 |
| llm_fusion (idea c) | 0.309 | 0.096 | 0.019 | 0.131 | 0.298 | 0.280 |
| text_only (learned) | 0.413 | 0.334 | 0.342 | 0.330 | 0.383 | 0.367 |
| fusion (ours) | 0.424 | 0.760 | 0.766 | 0.757 | 0.898 | 0.915 |


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

## Project status
See [`project_summary.md`](project_summary.md) for a detailed log of completed work,
artifact paths, and the prioritized next-step plan.

## Full draft
Working manuscript (Intro + Method + Results integrated): [`paper_draft.md`](paper_draft.md).
