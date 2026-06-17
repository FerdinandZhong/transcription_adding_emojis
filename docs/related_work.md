# Related Work / Literature Review

This document is the Related Work section for *Emoji-Augmented Transcription: Multimodal
Emoji Insertion for Affective ASR*. It synthesizes prior art into five strands and
articulates the gap this project fills.

## 1. Methodological ancestor: punctuation / boundary restoration

Restoring structure to unpunctuated ASR output is classically framed as **token
classification**: a contextual encoder produces a per-token label (period, comma,
question, none). BERT-based sequence labeling is the dominant approach
(Nguyen et al., *Improving Punctuation Restoration for Speech Transcripts via External
Data*, WNUT 2021), and encoder-decoder variants frame it as seq2seq with T5. Our prior
work, *Punctuation Restoration: A Case Study of BERT-Based Models' Task-Specific
Excellence* (ICASSP 2025), studies exactly this token-classification regime.

**What we reuse.** The token-level "insert-something-after-token-*i*" formulation and its
precision / recall / F1 placement metrics carry over directly to emoji insertion. Emoji
insertion is, structurally, "affective punctuation": instead of choosing among
`{., ?, !, none}` we choose *whether* to emit an emoji and *which* one.

**What is missing.** Punctuation is fully recoverable from text; emotional emoji are not.
The signal that disambiguates *which* emoji belongs (and sometimes *whether* one belongs)
lives partly in the acoustics, which text-only restoration discards.

## 2. Text-only emoji prediction

A large body of work predicts emoji from text alone:

- **DeepMoji** (Felbo et al., 2017) pretrains on emoji-bearing tweets for transfer to
  sentiment / emotion tasks.
- **SemEval-2018 Task 2** and **TweetEval** (Barbieri et al., 2018, 2020) standardize
  *single-emoji* prediction over the 20 most frequent emoji; **MultiEmo** (Lee et al.,
  2022) adds attention-based Bi-LSTMs.
- **EmojiLM / Text2Emoji** (Peng et al., arXiv:2311.01751, 2023) moves beyond single-emoji
  by *synthesizing* a large text<->emoji parallel corpus from an LLM and distilling a
  seq2seq translator, enabling *multi-emoji* generation.
- **SENTIMOJI** (Sci. Rep. 2024) jointly predicts multi-label emoji, emotion, and
  sentiment for code-mixed text.

**Relevance.** These define the "which emoji" sub-problem and provide the silver-labeling
trick we adopt for the text branch (LLM-synthesized emoji). **Limitation.** All operate on
text only, so they inherit the text channel's blindness to prosody (e.g. a flatly worded
sentence delivered sarcastically or excitedly).

## 3. Direct predecessor: emoji insertion into transcripts

**VoiceMoji** (Samsung; Kumar et al., arXiv:2112.12028, IEEE 2021) is the closest task: it
takes a blob of transcribed text and inserts emoji at detected boundaries (a CNN boundary
detector plus an Attention-based Char-Aware LSTM emoji predictor), on-device. It is the
first system to add emoji to transcribed text.

**Critical design choice we challenge.** VoiceMoji is *deliberately text-only*; the authors
explicitly refrain from prosody because "prosody cues are inconsistent ... and differ across
STT engines." This makes the system STT-agnostic but forfeits the very signal that conveys
emotion when the words do not. Our central hypothesis is that this trade-off is wrong for an
*affective* objective: prosody is exactly what recovers emotion lost in transcription.

## 4. Speech-emotion -> emoji (multimodal, but not learned placement)

**Speejis** (de Lacerda Pataca et al., arXiv:2502.05296, 2025) runs paralinguistic Speech
Emotion Recognition (SER) producing continuous valence/arousal/dominance (VAD), then maps
VAD to a facial emoji via a *fixed lookup table* (Kutsuzawa et al., 2022, who provide VAD
values for 74 facial emoji), decorating a voice-message waveform. A user study (N=12) shows
strong UX preference for these "speejis."

The prosody<->emoji link is empirically grounded by recent work showing emoji semantics
systematically shift prosodic realization and that listeners recover emoji intent from
prosody (arXiv:2508.00537, 2025). **EmoSRE** (Curr. Psychol. 2025) couples an LLM emotion
predictor with prosody encoding for emotional speech synthesis / refined recognition.

**Limitations.** Speejis (a) uses a *fixed* VAD->emoji map rather than a learned model,
(b) attaches emoji to the *waveform / whole message*, not to *positions in the transcript*,
and (c) evaluates UX, not emoji-placement / selection accuracy against a benchmark. There is
no learned, transcript-anchored, evaluable model.

## 5. Multimodal emotion recognition and datasets

Multimodal Emotion Recognition in Conversation (ERC) provides our backbone components and
training corpora:

- **Datasets** with aligned audio + transcript + emotion: **IEMOCAP** (dyadic; categorical
  + VAD), **MELD** (multi-party, *Friends*; 7 emotions + 3-way sentiment), **CMU-MOSEI**
  (large-scale sentiment + emotion), and **M3ED** (Chinese) for cross-lingual extension.
- **SER backbone**: Wagner et al. (2023) wav2vec2 / transformer models that close the
  "valence gap" and emit dimensional VAD from paralinguistic features.
- **Fusion architectures**: e.g. **MiSTER-E** (arXiv:2602.23300) fuses speech and text LLM
  embeddings via a mixture-of-experts gate for ERC.

**Relevance.** These give us (i) corpora that already pair speech with transcripts and
emotion labels, and (ii) proven cross-modal fusion machinery to adapt from "classify the
utterance emotion" to "insert the right emoji at the right token."

## 6. Evaluation methodology

Exact-match emoji accuracy is too strict because *several* emoji can faithfully express the
same utterance. **Semantics-Preserving Emoji Recommendation** (Qiu et al.,
arXiv:2409.10760, 2024) introduces a **semantics-preservation score**: a recommendation is
correct if a downstream classifier recovers the same emotion/sentiment from the emoji-laden
text. We adopt this alongside top-k accuracy and macro-F1.

## 7. The gap and our contribution

| Work | Uses prosody | Learned placement in transcript | Chooses which emoji | Reproducible benchmark |
|------|:---:|:---:|:---:|:---:|
| Punctuation restoration (ICASSP'25) | no | yes (punct) | n/a | yes |
| EmojiLM / TweetEval | no | partial (text) | yes | yes |
| VoiceMoji | **no** (by design) | yes | yes | no (private data) |
| Speejis | yes | **no** (waveform, fixed map) | yes (fixed) | **no** (UX only) |
| **This work** | **yes** | **yes** | **yes (learned)** | **yes** |

No prior system *fuses prosody with the transcript to insert emoji at the correct textual
positions with a learned model and a reproducible, emotion-aware benchmark*. Concretely we
contribute:

1. A **multimodal token-level model** extending BERT punctuation restoration with a prosody
   stream and two heads (insertion point + emoji selection).
2. A **silver+human-validated dataset** built by conditioning an LLM annotator on *both*
   transcript and detected speech emotion (idea c), which fixes the "pure text loses
   emotion" failure mode.
3. An **evaluation protocol** (placement P/R/F1, top-k, macro-F1, semantics preservation,
   emotion fidelity) with a dedicated **prosody-divergent** subset that isolates the value
   of prosody, plus a small Speejis-style human UX study.
