# CAI Workbench jobs — emoji-asr training

CAI Jobs run **`cai/run_training_job.py`** as the entry script. That script:

1. Optionally runs `git lfs pull` (silver-label JSONL)
2. Creates/reuses `.venv` via `setup/install_gpu.sh` (handles `PIP_USER` on CDSW)
3. Runs training with the venv Python — **no manual `source .venv/bin/activate`**

## Create a job in the UI

1. **Project → Jobs → New Job**
2. **Script:** `cai/run_training_job.py`
3. **Runtime:** GPU CUDA (e.g. `ml-runtime-pbj-workbench-python3.10-cuda:2026.01.1-b6`)
4. **Resources:** 1 GPU, 8 CPU, 32 GB RAM (adjust as needed)
5. **Timeout:** 24h+ for full ModernBERT runs

### Project environment variables (recommended)

| Variable | Example | Purpose |
|---|---|---|
| `TRAIN_MODE` | `train` | `train` \| `experiment` \| `evaluate` |
| `TRAIN_CONFIG` | `configs/meld_openai.yaml` | Training config |
| `TORCH_CUDA` | `cu124` | PyTorch CUDA wheel tag |
| `FORCE_REINSTALL` | `false` | Force full venv reinstall |
| `SKIP_SETUP` | `false` | Skip install if venv already OK |

6. **Run** the job.

## Or create jobs from YAML

Inside a CAI session (after `pip install pyyaml requests` if needed):

```bash
python cai/create_jobs.py
```

Uses `cai/jobs_config.yaml` and CDSW API env vars (`CDSW_APIV2_KEY`, `CDSW_DOMAIN`, `CDSW_PROJECT_ID`).

## Job modes

| `TRAIN_MODE` | Command run |
|---|---|
| `train` (default) | `python -m emoji_asr.train --config …` |
| `experiment` | `python -m emoji_asr.experiment --config …` |
| `evaluate` | `python -m emoji_asr.evaluate --config … --out_dir …` |

## Local test (same entry script)

**Prefer running as a Job** (`cai/run_training_job.py`). In an interactive session:

```bash
cd /home/cdsw/transcription_adding_emojis
TRAIN_MODE=train python cai/run_training_job.py
```

If you paste into a **notebook cell** (no `__file__`), set the project root explicitly:

```python
import os
os.environ["PROJECT_ROOT"] = "/home/cdsw/transcription_adding_emojis"
# then: exec(open("cai/run_training_job.py").read())
```

Or from the repo root in a terminal — not `%run` on copied fragments without `PROJECT_ROOT`.

## Logs

Job stdout shows venv bootstrap (`install_gpu.sh`) then epoch losses / metrics.
After training, weights are saved under ``train.out_dir`` (default ``runs/meld_openai/``):

```
runs/meld_openai/
  checkpoint.pt    # model weights + vocab + config snapshot
  config.yaml      # copy of training config
  metrics.json     # test-set summary metrics
  history.json     # per-epoch train loss
```
