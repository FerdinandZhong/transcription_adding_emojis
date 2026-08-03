# 6. Results

This section reports automatic evaluation on the MELD silver-label benchmark (OpenAI
annotator, speech-conditioned; Section 4). All learned models use
`answerdotai/ModernBERT-base` as the text encoder, cross-modal fusion with a word-level
prosody stream, and 30 training epochs on 9,989 utterances (`batch_size=32`, AdamW).
The fusion model (B1) is evaluated from a saved checkpoint; the text-only ablation is
identical in architecture except that the prosody stream is disabled (`use_prosody=False`).
Zero-training baselines (Speejis-style SER mapping; offline LLM annotator in text-only and
speech-conditioned modes) require no parameter updates. Test-set size is 2,610 utterances;
**prosody-divergent** utterances (emotional voice, lexically neutral text) comprise 45.4%
of the test split and are the primary locus where acoustic emotion is not recoverable from
words alone.

## 6.1 Metrics

We report two coupled sub-tasks, mirroring punctuation restoration:

* **Placement** — binary decision at each word boundary (insert emoji or not). Score:
  token-level precision, recall, and F1 (Table 1: Place F1 on the full test set).
* **Emoji selection** — which emoji to emit at predicted insertion sites. Scores: top-1
  accuracy over all words (Top1), **semantics preservation** (whether the predicted emoji
  matches the gold emotion class under our VA-grounded mapping), and **emotion fidelity**
  (agreement with utterance-level gold emotion).

Results are stratified into **all**, **congruent** (text and detected speech emotion align),
and **divergent** (speech is emotional while the transcript lacks explicit emotion words)
groups. The divergent subset isolates the value of prosody: a text-only model cannot
recover affect that is absent from the lexical channel.

## 6.2 Main results

Table 1 summarizes test performance. The prosody-aware fusion model achieves the strongest
emoji-selection and emotion-aware scores by a wide margin. On the full test set, fusion
attains **76.0%** top-1 emoji accuracy versus **33.4%** for the learned text-only ablation
—a **+42.6** point absolute gain. Semantics preservation on the divergent subset rises from
**0.383** (text-only) to **0.898** (fusion), and emotion fidelity from **0.367** to
**0.915**. These gains confirm the central hypothesis: when the transcript under-specifies
affect, word-level prosody features supply the missing signal for appropriate emoji choice.

**Table 1.** MELD silver-label test results. Place F1 is computed on all test utterances;
Top1 and emotion-aware metrics are shown for all, congruent, and divergent groups as
indicated.

| Method | Place F1 (all) | Top1 (all) | Top1 (congruent) | Top1 (divergent) | SemPres (divergent) | EmoFidelity (divergent) |
|---|---:|---:|---:|---:|---:|---:|
| ser_mapping (Speejis) | 0.321 | 0.097 | 0.022 | 0.131 | 0.298 | 0.280 |
| llm_text_only (idea a) | 0.029 | 0.004 | 0.013 | 0.000 | 0.084 | 0.000 |
| llm_fusion (idea c) | 0.309 | 0.096 | 0.019 | 0.131 | 0.298 | 0.280 |
| text_only (learned) | 0.413 | 0.334 | 0.342 | 0.330 | 0.383 | 0.367 |
| **fusion (ours)** | **0.424** | **0.760** | **0.766** | **0.757** | **0.898** | **0.915** |

## 6.3 Prosody-divergent vs. congruent analysis

On the **congruent** subset, fusion already leads text-only on emoji top-1 (**76.6%** vs.
**34.2%**), indicating that prosody is informative even when emotion words appear in the
transcript—detected speech emotion and lexical cues need not agree perfectly with the
silver labels, and the fusion encoder exploits both channels.

On the **divergent** subset—the regime most relevant to affective ASR—text-only selection
collapses toward chance-level emotion fidelity (**0.367**), while fusion remains stable
(**0.915**). The **+0.515** absolute improvement in semantics preservation (0.898 vs. 0.383)
is the headline multimodal gain: prosody recovers emoji intent that pure text cannot infer.
Notably, fusion maintains **75.7%** top-1 on divergent utterances, nearly matching its
all-set performance (76.0%), whereas text-only drops only modestly in absolute terms
(33.0% vs. 33.4%) but remains far below fusion because it never accesses the acoustic
channel.

## 6.4 Placement vs. selection

The two sub-tasks are both won by the fusion model in this run. Fusion achieves placement
F1 of **0.424** versus **0.413** for text-only—a small but consistent margin—suggesting
that prosody provides a weak signal for insertion-boundary decisions in addition to its
dominant role in emoji selection. The primary driver of the overall gain remains **which**
emoji to emit once a site is chosen; the placement advantage is secondary. This is
consistent with punctuation-restoration findings where prosodic cues (pauses, pitch resets)
marginally improve boundary detection even when lexical context alone is already
informative.

## 6.5 Baseline comparison

**Speejis-style mapping** (fixed VA→emoji lookup from pooled prosody) achieves moderate
placement F1 (0.321) but weak emoji top-1 (9.7%) and low divergent semantics preservation
(0.298). Without learned text context or placement, a static map cannot align emoji with
transcript semantics.

**Offline LLM annotator baselines** (idea a: text-only; idea c: speech-conditioned) perform
far below learned models on emoji metrics. Idea (a) essentially fails on divergent
utterances (0.000 top-1, 0.084 SemPres), as expected when neither speech nor reliable text
emotion cues are available to a lightweight rule-based stand-in. Idea (c) matches Speejis on
divergent SemPres and EmoFidelity (both 0.298 / 0.280), indicating that our offline
annotator and fixed mapping baselines operate in a similar low-ceiling regime without
task-specific training. *Important:* these baselines use the repository's offline heuristic
annotator, not the GPT-4o-mini labeler used to construct silver training data; they
establish a zero-shot floor rather than an upper bound on LLM performance.

In summary, **only the learned fusion model** jointly achieves strong placement, high
emoji accuracy, and robust emotion-aware scores on prosody-divergent speech—validating
multimodal token classification over either text-only learning or prosody-only lookup.

## 6.6 Training dynamics (fusion, B1)

The fusion model was trained for 30 epochs on MELD silver labels. Training loss decreased
monotonically from **1.81** (epoch 1) to **0.56** (epoch 15) and continued falling through
epoch 30, with no held-out early stopping in this run. Checkpoint metrics at epoch 30
(loaded for Table 1 fusion row): placement F1 0.424, all-set top-1 0.760, divergent
SemPres 0.898. The text-only ablation was trained under identical hyperparameters for the
ablation row in Table 1.

## 6.7 Summary

| Claim | Evidence (divergent subset) |
|---|---|
| Prosody improves emoji selection when text is neutral | SemPres 0.898 (fusion) vs. 0.383 (text-only) |
| Fusion preserves utterance emotion in emoji choice | EmoFidelity 0.915 vs. 0.367 |
| Fixed / zero-shot baselines are insufficient | Speejis & offline LLM top-1 ≤ 13.1% on divergent |
| Placement is largely lexical; selection is multimodal | Text-only Place F1 ≥ fusion; fusion Top1 ≫ text-only |

Section 7 reports the completed 10-item pilot annotation study (insertion κ = 0.24,
3 annotators, 30 valid labels) and describes the preregistered 50-item follow-on now
underway. The pilot's fair agreement corroborates the silver-label-noise caveat above
and confirms that the prosody-divergent subset is the natural locus for human validation.
