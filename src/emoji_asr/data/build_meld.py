"""Build train/dev/test MELD JSONL datasets for model training.

Example:
    PYTHONPATH=src python -m emoji_asr.data.build_meld \
      --meld_root /path/to/MELD \
      --out_dir data/processed/meld_silver \
      --annotator openai \
      --condition_on_speech true \
      --ser_backend wav2vec2
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

from ..emoji_set import EmojiSet
from .datasets import load_meld_split
from .io import example_to_record, load_jsonl, split_stats
from .llm import build_annotator
from .ser import build_ser
from .silver_labels import SilverLabeler


_CSV_CANDIDATES = {
    "train": ["train_sent_emo.csv", "train.csv"],
    "dev": ["dev_sent_emo.csv", "valid_sent_emo.csv", "dev.csv"],
    "test": ["test_sent_emo.csv", "test.csv"],
}


def _str2bool(v: str) -> bool:
    s = v.strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {v}")


def _resolve_split_csvs(meld_root: str) -> Dict[str, str]:
    roots_to_try = [
        meld_root,
        os.path.join(meld_root, "data", "MELD"),
    ]
    out: Dict[str, str] = {}
    for split, cands in _CSV_CANDIDATES.items():
        found = None
        for root in roots_to_try:
            for name in cands:
                path = os.path.join(root, name)
                if os.path.exists(path):
                    found = path
                    break
            if found is not None:
                break
        if found is None:
            raise FileNotFoundError(
                f"Could not find CSV for split '{split}' under {meld_root}; "
                f"tried: {', '.join(cands)}"
            )
        out[split] = found
    return out


def _resolve_audio_dir(base_root: str, split: str) -> Optional[str]:
    candidates = [
        os.path.join(base_root, f"{split}_wav"),
        os.path.join(base_root, f"{split}_audio"),
        os.path.join(base_root, split),
        os.path.join(base_root, "audio", split),
        os.path.join(base_root, "wav", split),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def _count_jsonl_lines(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _save_json(path: str, payload: Dict) -> None:
    """Atomically save JSON to avoid partial corruption on interruption."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def build_meld_dataset(
    meld_root: str,
    out_dir: str,
    annotator_name: str = "offline",
    condition_on_speech: bool = True,
    use_text: bool = True,
    ser_backend: str = "heuristic",
    wav2vec2_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    openai_model: str = "gpt-4o-mini",
    openai_max_retries: int = 3,
    openai_retry_sleep_s: float = 1.5,
    prosody_dim: int = 32,
    seed: int = 13,
    max_rows_per_split: Optional[int] = None,
    audio_root: Optional[str] = None,
    chunk_size: int = 50,
    resume: bool = True,
) -> Dict:
    """Generate and save MELD train/dev/test splits as JSONL.

    Writes checkpointed output per split in chunks, so long OpenAI runs are resumable.
    """
    emoji_set = EmojiSet()
    ser = build_ser(backend=ser_backend, wav2vec2_model=wav2vec2_model)
    annotator = build_annotator(
        annotator=annotator_name,
        emoji_set=emoji_set,
        condition_on_speech=condition_on_speech,
        use_text=use_text,
        openai_model=openai_model,
        openai_max_retries=openai_max_retries,
        openai_retry_sleep_s=openai_retry_sleep_s,
    )
    labeler = SilverLabeler(ser=ser, annotator=annotator, emoji_set=emoji_set)

    split_csvs = _resolve_split_csvs(meld_root)
    audio_root = audio_root or meld_root
    split_audio = {
        split: _resolve_audio_dir(audio_root, split) for split in split_csvs
    }

    os.makedirs(out_dir, exist_ok=True)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    progress_path = os.path.join(out_dir, "progress.json")
    if not resume and os.path.exists(progress_path):
        os.remove(progress_path)

    summary = {
        "settings": {
            "meld_root": meld_root,
            "audio_root": audio_root,
            "annotator": annotator_name,
            "condition_on_speech": condition_on_speech,
            "use_text": use_text,
            "ser_backend": ser_backend,
            "wav2vec2_model": wav2vec2_model,
            "openai_model": openai_model,
            "openai_max_retries": openai_max_retries,
            "openai_retry_sleep_s": openai_retry_sleep_s,
            "prosody_dim": prosody_dim,
            "seed": seed,
            "max_rows_per_split": max_rows_per_split,
            "chunk_size": chunk_size,
            "resume": resume,
        },
        "files": {},
        "split_stats": {},
        "progress_path": progress_path,
    }
    progress = {
        "settings": summary["settings"],
        "splits": {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if resume and os.path.exists(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                progress.update(loaded)
        except Exception:
            # A damaged progress file should not block resumption from JSONL line counts.
            pass

    for split in ("train", "dev", "test"):
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        summary["files"][split] = out_path

        already = _count_jsonl_lines(out_path) if resume else 0
        if not resume and os.path.exists(out_path):
            os.remove(out_path)
        processed = already
        target_max = max_rows_per_split

        mode = "a" if resume else "w"
        progress["splits"].setdefault(split, {})
        progress["splits"][split].update({
            "output_path": out_path,
            "csv_path": split_csvs[split],
            "processed": processed,
            "status": "running",
        })
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(progress_path, progress)
        print(
            f"[{split}] resume={resume} start_from={processed} checkpoint_every={chunk_size}",
            flush=True,
        )

        with open(out_path, mode, encoding="utf-8") as f:
            while True:
                remaining = None
                if target_max is not None:
                    remaining = max(0, target_max - processed)
                    if remaining == 0:
                        break
                take = chunk_size if remaining is None else min(chunk_size, remaining)
                chunk = load_meld_split(
                    csv_path=split_csvs[split],
                    audio_dir=split_audio.get(split),
                    labeler=labeler,
                    emoji_set=emoji_set,
                    prosody_dim=prosody_dim,
                    seed=seed,
                    max_rows=take,
                    mark_divergent=True,
                    start_row=processed,
                )
                if not chunk:
                    break
                for ex in chunk:
                    f.write(json.dumps(example_to_record(ex), ensure_ascii=False) + "\n")
                f.flush()
                processed += len(chunk)
                progress["splits"][split]["processed"] = processed
                progress["updated_at"] = datetime.now(timezone.utc).isoformat()
                _save_json(progress_path, progress)
                print(
                    f"[{split}] processed={processed}"
                    + (f"/{target_max}" if target_max is not None else ""),
                    flush=True,
                )

        # compute stats from final saved split
        examples = load_jsonl(out_path)
        summary["files"][split] = out_path
        summary["split_stats"][split] = split_stats(examples)
        progress["splits"][split]["status"] = "completed"
        progress["splits"][split]["processed"] = len(examples)
        progress["splits"][split]["stats"] = summary["split_stats"][split]
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_json(progress_path, progress)

    summary_path = os.path.join(out_dir, "manifest.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    summary["manifest"] = summary_path
    return summary


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Build MELD training datasets (JSONL)")
    ap.add_argument("--meld_root", required=True, help="Directory containing MELD CSVs")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated files")
    ap.add_argument("--audio_root", default=None, help="Optional root for split audio dirs")
    ap.add_argument("--annotator", default="offline", choices=["offline", "openai"])
    ap.add_argument("--condition_on_speech", type=_str2bool, default=True)
    ap.add_argument("--use_text", type=_str2bool, default=True,
                    help="Set false for speech-only mapping (idea b)")
    ap.add_argument("--ser_backend", default="heuristic", choices=["heuristic", "wav2vec2"])
    ap.add_argument("--wav2vec2_model",
                    default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--openai_max_retries", type=int, default=3)
    ap.add_argument("--openai_retry_sleep_s", type=float, default=1.5)
    ap.add_argument("--prosody_dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_rows_per_split", type=int, default=None)
    ap.add_argument("--chunk_size", type=int, default=50,
                    help="Number of samples to write before checkpointing progress")
    ap.add_argument("--checkpoint_every", type=int, default=None,
                    help="Alias of --chunk_size; takes precedence when set")
    ap.add_argument("--resume", type=_str2bool, default=True)
    args = ap.parse_args()
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else args.chunk_size

    summary = build_meld_dataset(
        meld_root=args.meld_root,
        out_dir=args.out_dir,
        annotator_name=args.annotator,
        condition_on_speech=args.condition_on_speech,
        use_text=args.use_text,
        ser_backend=args.ser_backend,
        wav2vec2_model=args.wav2vec2_model,
        openai_model=args.openai_model,
        openai_max_retries=args.openai_max_retries,
        openai_retry_sleep_s=args.openai_retry_sleep_s,
        prosody_dim=args.prosody_dim,
        seed=args.seed,
        max_rows_per_split=args.max_rows_per_split,
        audio_root=args.audio_root,
        chunk_size=checkpoint_every,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
