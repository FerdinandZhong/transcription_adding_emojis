#!/usr/bin/env python3
"""
CAI Job entry point — bootstrap GPU venv and run emoji-asr training.

Set this file as the Job script in CAI Workbench (path relative to project root):
  cai/run_training_job.py

Environment variables (optional):
  TRAIN_MODE       train | experiment | evaluate  (default: train)
  TRAIN_CONFIG     YAML config path               (default: configs/meld_openai.yaml)
  OUTPUT_DIR       evaluate output dir            (default: outputs/meld_openai)
  TORCH_CUDA       cu124 | cu121 | cu118          (passed to setup/install_gpu.sh)
  FORCE_REINSTALL  true — always re-run install_gpu.sh
  SKIP_SETUP       true — skip install if venv already has numpy+torch+emoji_asr
  SKIP_GIT_LFS     true — skip ``git lfs pull``
  SKIP_VERIFY      1    — skip verify_env at end of install_gpu.sh
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _truthy(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "y")


def project_root() -> Path:
    """Resolve repo root in Jobs, CLI, and Jupyter (where ``__file__`` is absent)."""
    for key in ("PROJECT_ROOT", "CDSW_PROJECT_ROOT"):
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val).resolve()

    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "setup" / "install_gpu.sh").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate

    default = Path("/home/cdsw/transcription_adding_emojis")
    if default.is_dir():
        return default.resolve()
    return cwd


def maybe_git_lfs_pull(root: Path) -> None:
    if _truthy("SKIP_GIT_LFS"):
        print("[INFO] SKIP_GIT_LFS=true — skipping git lfs pull")
        return
    if not (root / ".git").is_dir():
        print("[INFO] Not a git repo — skipping git lfs pull")
        return
    print("==> git lfs pull")
    subprocess.run(["git", "lfs", "pull"], cwd=root, check=False)


def venv_python(root: Path) -> Path:
    return root / ".venv" / "bin" / "python"


def venv_is_ready(py: Path) -> bool:
    if not py.is_file():
        return False
    probe = subprocess.run(
        [str(py), "-c", "import numpy, torch, emoji_asr"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        if probe.stderr.strip():
            print(probe.stderr.strip())
        return False
    return True


def ensure_venv(root: Path) -> Path:
    py = venv_python(root)

    if _truthy("SKIP_SETUP") and venv_is_ready(py):
        print(f"[OK] Reusing venv at {py.parent.parent}")
        return py

    if not _truthy("FORCE_REINSTALL") and venv_is_ready(py):
        print(f"[OK] Existing venv ready at {py.parent.parent}")
        return py

    install_sh = root / "setup" / "install_gpu.sh"
    if not install_sh.is_file():
        print(f"ERROR: missing {install_sh}")
        sys.exit(1)

    print("==> Bootstrapping GPU environment (setup/install_gpu.sh)")
    env = os.environ.copy()
    env.pop("PIP_USER", None)
    env.pop("PYTHONUSERBASE", None)
    if _truthy("SKIP_VERIFY"):
        env["SKIP_VERIFY"] = "1"
    result = subprocess.run(["bash", str(install_sh)], cwd=root, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)
    if not py.is_file():
        print(f"ERROR: venv python not found at {py}")
        sys.exit(1)
    return py


def run_training(root: Path, py: Path) -> None:
    mode = os.environ.get("TRAIN_MODE", "train").strip().lower()
    config = os.environ.get("TRAIN_CONFIG", "configs/meld_openai.yaml")
    config_path = (root / config).resolve()
    if not config_path.is_file():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    cmd = [str(py), "-m"]
    if mode == "experiment":
        cmd += ["emoji_asr.experiment", "--config", str(config_path)]
    elif mode == "evaluate":
        out_dir = os.environ.get("OUTPUT_DIR", "outputs/meld_openai")
        cmd += [
            "emoji_asr.evaluate",
            "--config",
            str(config_path),
            "--out_dir",
            str((root / out_dir).resolve()),
        ]
    elif mode == "train":
        cmd += ["emoji_asr.train", "--config", str(config_path)]
    else:
        print(f"ERROR: unknown TRAIN_MODE={mode!r} (use train, experiment, or evaluate)")
        sys.exit(1)

    print("==> Running:", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, cwd=root, env=env, check=True)


def main() -> None:
    root = project_root()
    os.chdir(root)
    print("=" * 60)
    print("  Emoji-ASR training job (CAI)")
    print("=" * 60)
    print(f"Project root : {root}")
    print(f"TRAIN_MODE   : {os.environ.get('TRAIN_MODE', 'train')}")
    print(f"TRAIN_CONFIG : {os.environ.get('TRAIN_CONFIG', 'configs/meld_openai.yaml')}")
    print()

    maybe_git_lfs_pull(root)
    py = ensure_venv(root)
    run_training(root, py)
    print("\nTraining job finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: command failed with exit code {exc.returncode}")
        sys.exit(exc.returncode)
