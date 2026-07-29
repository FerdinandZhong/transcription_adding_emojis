# Faithful re-run: real wav2vec2 word-level prosody

**Why:** the shipped `data/processed/meld_silver_openai` splits were built with
`ser_backend: heuristic`, so "prosody" was a VAD prior derived from the MELD **gold
emotion label** (centroid + σ≈0.02 noise), not audio. That confounds the fusion-vs-text-only
comparison (fusion effectively saw the gold emotion). This guide regenerates the data from
**real speech** so the paper's claims are substantiated.

The code is already wired (`WordProsodyExtractor`, `--ser_backend wav2vec2` path). You only
need to supply audio + an API key and run the commands below.

## Prerequisites

```bash
# 1. Audio + system tools
brew install ffmpeg
pip install -e '.[asr,llm]'          # librosa, soundfile, openai (+ torch/transformers present)

# 2. Download MELD.Raw (~10 GB) and transcode clips to 16 kHz mono WAV under <split>_wav/
bash scripts/download_meld_raw.sh datasets/MELD

# 3. OpenAI key for re-annotation (13,708 utterances via gpt-4o-mini)
export OPENAI_API_KEY=sk-...
```

## Step 1 — Rebuild silver labels from real audio

```bash
PYTHONPATH=src python -m emoji_asr.data.build_meld \
  --meld_root datasets/MELD/data/MELD \
  --audio_root datasets/MELD \
  --out_dir data/processed/meld_silver_wav2vec2 \
  --annotator openai --condition_on_speech true \
  --ser_backend wav2vec2 \
  --chunk_size 50            # resumable checkpoints; safe to interrupt/rerun
```

**Verify provenance** (the check that would have caught the original issue):

```bash
python -c "import json; m=json.load(open('data/processed/meld_silver_wav2vec2/manifest.json')); \
print({s: st['real_prosody_frac'] for s,st in m['split_stats'].items()})"
# Expect real_prosody_frac ≈ 1.0 per split. If ~0.0, audio isn't resolving — check <split>_wav/ names (dia<D>_utt<U>.wav).
```

## Step 2 — Point the eval config at the new data

Edit `configs/meld_paper_eval.yaml` (and `configs/meld_openai.yaml` for training):

```yaml
data:
  jsonl:
    train: data/processed/meld_silver_wav2vec2/train.jsonl
    dev:   data/processed/meld_silver_wav2vec2/dev.jsonl
    test:  data/processed/meld_silver_wav2vec2/test.jsonl
```

## Step 3 — Retrain fusion + text-only ablation

```bash
PYTHONPATH=src python -m emoji_asr.train --config configs/meld_openai.yaml \
  --out runs/meld_wav2vec2 --epochs 30                 # fusion (use_prosody=true)
PYTHONPATH=src python -m emoji_asr.train --config configs/meld_openai.yaml \
  --out runs/meld_wav2vec2_textonly --epochs 30 --use_prosody false
```

(MPS/Apple Silicon: no CUDA, so expect several hours per run. `get_device` auto-selects MPS.)

## Step 4 — Regenerate tables + figures

```bash
PYTHONPATH=src python -m emoji_asr.evaluate --config configs/meld_paper_eval.yaml \
  --checkpoint runs/meld_wav2vec2/checkpoint.pt --out outputs/meld_paper_wav2vec2
PYTHONPATH=src python scripts/generate_paper_figures.py --results outputs/meld_paper_wav2vec2
```

## Step 5 — Update the paper with real numbers

Replace the Table 1/2 rows, the abstract headline (75.3 vs 33.4 etc.), and §6 prose in both
`paper/main.tex` and `docs/paper_manuscript.md`. Expect the fusion advantage to **shrink**
versus the oracle-prosody run — that smaller-but-real gap is the honest result. Keep the §4.2
description accurate to what ran (`ser_backend=wav2vec2`, word-level VAD from
`WordProsodyExtractor`).

## What the new code does

| Component | Role |
|---|---|
| `data/prosody.py :: WordProsodyExtractor` | audio + words → `[n_words, 32]` real prosody: per-word wav2vec2 VAD (dims 0–2) + numpy acoustic features (dims 3–31) |
| `data/prosody.py :: WordAligner` | char-duration-weighted word spans (stand-in for forced alignment; pass Whisper timestamps via `word_spans` for exact alignment) |
| `data/ser.py :: Wav2Vec2SER.predict_from_samples` | run the dimensional model on in-memory audio (per-word, no temp files) |
| `data/silver_labels.py :: label(overwrite_vad=False)` | preserve real per-word VAD instead of clobbering with one utterance value |
| `data/build_meld.py` | builds the extractor when `--ser_backend wav2vec2`; records `real_prosody_frac` in the manifest |

**Known approximation:** word alignment defaults to char-weighted uniform spans, not true
forced alignment. For exact word timestamps, run `WhisperASR` (`data/asr.py`) and pass its
per-word `start`/`end` into `WordProsodyExtractor.extract(..., word_spans=...)`. Note this in
§4.2 / Limitations if the uniform aligner is used for the reported numbers.
