# GPU training environment setup

Scripts to create a Python virtual environment on a Linux GPU machine for training
the emoji-insertion model (ModernBERT fusion + baselines).

## Prerequisites

- Linux server with NVIDIA GPU
- NVIDIA driver installed (`nvidia-smi` works)
- Python **3.9–3.12** (`python3 --version`)
- Git + this repository cloned

## One-command install

From the repository root:

```bash
bash setup/install_gpu.sh
```

This will:

1. Create `.venv/` in the repo root (override with `VENV_DIR=...`)
2. Install **PyTorch with CUDA** from the official wheel index
3. Install training dependencies (`setup/requirements-gpu.txt`)
4. Install this project in editable mode (`pip install -e .`)
5. Run `setup/verify_env.py` (CUDA smoke test + ModernBERT import check)

Activate later:

```bash
source .venv/bin/activate
```

## CUDA version

Default PyTorch wheel: **CUDA 12.4** (`TORCH_CUDA=cu124`).

Pick the tag that matches your driver/CUDA stack:

| Variable | When to use |
|---|---|
| `TORCH_CUDA=cu124` | CUDA 12.4+ (default) |
| `TORCH_CUDA=cu121` | CUDA 12.1 |
| `TORCH_CUDA=cu118` | CUDA 11.8 (older clusters) |

Example:

```bash
TORCH_CUDA=cu121 bash setup/install_gpu.sh
```

See [PyTorch Get Started](https://pytorch.org/get-started/locally/) if unsure.

## Copy data to the GPU machine

### Option A: clone with Git LFS (recommended)

This repo stores canonical datasets via **Git LFS**:

- `datasets/MELD/` — MELD CSV annotations (+ repo metadata)
- `data/processed/meld_silver_openai/` — OpenAI silver-label train/dev/test JSONL

After cloning:

```bash
git lfs install
git lfs pull
```

### Option B: manual copy

Transfer the processed silver labels:

```
data/processed/meld_silver_openai/
  train.jsonl
  dev.jsonl
  test.jsonl
  manifest.json
```

Optional raw corpus:

```
datasets/MELD/
```

### Deduplicate train if needed

```bash
python setup/dedupe_jsonl.py data/processed/meld_silver_openai/train.jsonl
```

## Verify only

```bash
source .venv/bin/activate
python setup/verify_env.py
```

## Training commands (after setup)

Synthetic sanity check (no external data):

```bash
python -m emoji_asr.experiment
```

Full experiment table (baselines + fusion + text-only):

```bash
python -m emoji_asr.evaluate --out_dir outputs
```

For ModernBERT on real MELD JSONL, set in config:

```yaml
model:
  text_encoder:
    backend: hf
    hf_model: answerdotai/ModernBERT-base
train:
  device: cuda
  batch_size: 16   # reduce to 8 if OOM on 16GB GPU
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA not available` after install | Wrong `TORCH_CUDA` tag; reinstall with matching tag |
| OOM during ModernBERT training | Lower `train.batch_size` to 8 or 4; use gradient accumulation later |
| `ModernBERT not available` | Upgrade transformers: `pip install -U "transformers>=4.48"` |
| Proxy errors for OpenAI annotator | Unset proxy vars before API calls |

## Files

| File | Purpose |
|---|---|
| `install_gpu.sh` | Main installer (venv + CUDA PyTorch + deps) |
| `requirements-gpu.txt` | Python deps excluding torch |
| `verify_env.py` | Post-install GPU / package checks |
