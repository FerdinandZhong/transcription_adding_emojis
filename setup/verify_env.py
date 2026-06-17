#!/usr/bin/env python3
"""Verify GPU training environment after setup/install_gpu.sh."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    print(f"Project root: {root}")

    # Python version
    if sys.version_info < (3, 9):
        _fail(f"Python >= 3.9 required, got {sys.version.split()[0]}")
    _ok(f"Python {sys.version.split()[0]}")

    # Core deps
    for pkg in ("numpy", "pandas", "yaml", "sklearn", "tqdm"):
        importlib.import_module(pkg)
    _ok("numpy, pandas, pyyaml, scikit-learn, tqdm")

    import torch

    _ok(f"torch {torch.__version__}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        _ok(f"CUDA available: {n} device(s)")
        for i in range(n):
            print(f"       [{i}] {torch.cuda.get_device_name(i)}")
        # Tiny GPU matmul smoke test
        x = torch.randn(256, 256, device="cuda")
        y = x @ x
        _ok(f"CUDA matmul smoke test passed ({y.shape})")
    else:
        print("[WARN] CUDA not available — training will fall back to CPU/MPS.")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _ok("Apple MPS is available")
        else:
            print("[WARN] No GPU backend detected. Check NVIDIA driver + CUDA PyTorch wheel.")

    import transformers

    _ok(f"transformers {transformers.__version__}")
    try:
        from transformers import ModernBertModel  # noqa: F401
        _ok("ModernBERT class importable (transformers>=4.48)")
    except ImportError as exc:
        _fail(f"ModernBERT not available: {exc}")

    import emoji_asr

    _ok(f"emoji_asr {emoji_asr.__version__} importable")

    data_dir = root / "data" / "processed" / "meld_silver_openai"
    if data_dir.is_dir():
        for split in ("train", "dev", "test"):
            p = data_dir / f"{split}.jsonl"
            if p.exists():
                n = sum(1 for _ in open(p, encoding="utf-8") if _.strip())
                print(f"[INFO] {split}.jsonl: {n} lines")
            else:
                print(f"[WARN] missing {p}")
    else:
        print(f"[INFO] processed dataset not found at {data_dir} (copy or generate on GPU machine)")

    print("\nEnvironment looks ready for training.")


if __name__ == "__main__":
    main()
