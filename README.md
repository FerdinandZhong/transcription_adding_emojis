# Emoji-Augmented Transcription

**Multimodal emoji insertion for affective ASR.** Given speech and its transcript, insert
emotion-appropriate emoji at the right token positions by *fusing prosody (speech emotion)
with text*. This extends BERT-style punctuation restoration (token classification on ASR
output) with an acoustic stream, so emoji reflect *how* something was said, not only *what*.

> Motivation: prior systems pick one side. VoiceMoji inserts emoji from text only;
> Speejis maps speech emotion to emoji on a waveform with a fixed lookup and no learned
> text placement. Neither *fuses prosody with the transcript to insert emoji at the right
> positions with a learned model and a reproducible benchmark*. See
> [`docs/related_work.md`](docs/related_work.md).

## Key idea

```
 words   ──> TextEncoder (ModernBERT / lite) ─┐
                                        ├─> cross-modal fusion ─> insertion head (where?)
 audio ──> SER (wav2vec2 VAD) ─ pool ───┘                     └─> emoji head     (which?)
```

The prosody stream is what recovers emotion that pure text loses (sarcasm, excitement,
neutral-text-but-emotional-voice). Turning it off recovers a text-only baseline exactly,
giving a clean ablation.

## Install

**Local / CPU (quick):**

```bash
pip install -r requirements.txt          # core (runs everything below offline)
# optional, for real audio / LLM annotators:
pip install -e ".[asr,llm]"
```

**GPU training server:**

```bash
bash setup/install_gpu.sh
source .venv/bin/activate
python setup/verify_env.py
```

See [`setup/README.md`](setup/README.md) for CUDA version options and data transfer steps.

## Git LFS (datasets)

Large files under `datasets/` and `data/processed/meld_silver_openai/` are tracked with
[Git LFS](https://git-lfs.com/). After cloning:

```bash
git lfs install
git lfs pull
```

If you resume silver-label generation and get duplicate `uid` rows in `train.jsonl`, dedupe
with `python setup/dedupe_jsonl.py data/processed/meld_silver_openai/train.jsonl`.

## Quickstart (fully offline, synthetic data)

```bash
# Train fusion + text-only, run all baselines, print the comparison tables:
PYTHONPATH=src python -m emoji_asr.experiment

# Same, plus write outputs/ and run the simulated UX study:
PYTHONPATH=src python -m emoji_asr.evaluate --ux_study --out_dir outputs

# Tests:
PYTHONPATH=src python -m pytest -q
```

## Representative results (synthetic; `outputs/results.md`)

| Method | Place F1 (all) | Top1 (all) | Top1 (congruent) | Top1 (divergent) | SemPres (divergent) | EmoFidelity (divergent) |
|---|---:|---:|---:|---:|---:|---:|
| ser_mapping (Speejis) | 0.929 | 0.120 | 0.107 | 0.136 | 0.667 | 0.667 |
| llm_text_only (idea a) | 0.150 | 0.016 | 0.030 | 0.000 | 0.007 | 0.007 |
| llm_fusion (idea c) | 0.545 | 0.076 | 0.030 | 0.129 | 0.646 | 0.646 |
| text_only (learned) | 0.444 | 0.063 | 0.101 | 0.020 | 0.102 | 0.102 |
| **fusion (ours)** | **0.705** | **0.142** | 0.136 | **0.150** | **0.986** | **0.986** |

Reading the table: exact top-1 emoji match is intentionally strict (many emoji validly
express one emotion), so the headline metric is **semantics preservation** on the
**prosody-divergent** subset. The fusion model reaches **0.986** there vs **0.102** for the
text-only model and **0.007** for the text-only LLM annotator (idea a) -- prosody recovers
the emotion the text cannot. The Speejis-style rule baseline places emoji well on clean
synthetic prosody (0.929 F1) but its fixed VA lookup picks the right *emotion* far less
often (0.667). Simulated UX: 85.75% prefer the emoji-augmented transcript (expressiveness
+1.85 on a 1-7 scale).

> Numbers are from the synthetic generator (no external data/GPU). They demonstrate the
> mechanism and the experimental harness; real-corpus numbers require MELD/IEMOCAP/MOSEI.

## Repository layout

```
configs/default.yaml          experiment configuration
docs/related_work.md          literature review (Related Work section)
docs/paper_outline.md         paper structure (ICASSP-style + journal extension)
src/emoji_asr/
  emoji_set.py                ~64-emoji set + VAD coords + emotion mapping + lexicon
  data/
    schema.py                 Example / Vocab / dataset / collate
    synthetic.py              synthetic affective-speech corpus (congruent + divergent)
    asr.py                    ASR backends (passthrough | whisper)
    ser.py                    SER backends (heuristic | wav2vec2 VAD)
    llm.py                    emoji annotators (offline rule | openai); ideas a/b/c
    silver_labels.py          raw -> ASR -> SER -> annotator -> Example pipeline (idea c)
    datasets.py               MELD/IEMOCAP/MOSEI adapters (real-corpus stubs)
  models/
    text_encoder.py           LiteTextEncoder (offline) + HFTextEncoder (ModernBERT/BERT/RoBERTa)
    fusion_model.py           cross-modal fusion + insertion & emoji heads + loss/decode
  baselines/                  text-only, Speejis SER-mapping, annotator-as-predictor
  eval/metrics.py             placement P/R/F1, top-k, macro-F1, sem-preservation, fidelity
  benchmark.py                human-validated benchmark: sample/export/import/agreement
  human_eval.py               Speejis-style A/B UX study tooling (+ simulated raters)
  train.py / evaluate.py / experiment.py
tests/                        unit + integration tests (offline)
```

## Running on real corpora

1. Download MELD (or IEMOCAP / CMU-MOSEI) under their licenses.
2. Build silver-labeled split files (JSONL):

```bash
PYTHONPATH=src python -m emoji_asr.data.build_meld \
  --meld_root /path/to/MELD \
  --out_dir data/processed/meld_silver \
  --annotator openai \
  --condition_on_speech true \
  --ser_backend wav2vec2 \
  --checkpoint_every 50 \
  --resume true
```

The command writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, and `manifest.json`.
It also writes `progress.json` and checkpoints output every `checkpoint_every`
samples so an interrupted run can continue from where it stopped.
Use `--annotator offline --ser_backend heuristic` for a fast dry run.

3. Sample a benchmark with `emoji_asr.benchmark`, have raters validate it, then
   train/evaluate with the same entry points.

See [`docs/paper_outline.md`](docs/paper_outline.md) for the full method, dataset, and
evaluation write-up.
