import json
import os

import pandas as pd

from emoji_asr.data.build_meld import build_meld_dataset
from emoji_asr.data.io import load_jsonl, save_jsonl
from emoji_asr.data.synthetic import make_splits
from emoji_asr.emoji_set import EmojiSet


def test_jsonl_roundtrip(tmp_path):
    es = EmojiSet()
    train, _, _ = make_splits(8, 2, 2, emoji_set=es)
    path = os.path.join(tmp_path, "train.jsonl")
    save_jsonl(train, path)
    restored = load_jsonl(path)
    assert len(restored) == len(train)
    assert restored[0].uid == train[0].uid
    assert restored[0].prosody.shape == train[0].prosody.shape


def test_build_meld_dataset_minimal(tmp_path):
    meld_root = os.path.join(tmp_path, "meld")
    os.makedirs(meld_root, exist_ok=True)
    rows = [
        {"Dialogue_ID": 1, "Utterance_ID": 1, "Utterance": "i am happy today", "Emotion": "joy"},
        {"Dialogue_ID": 1, "Utterance_ID": 2, "Utterance": "this is fine", "Emotion": "neutral"},
    ]
    for name in ("train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"):
        pd.DataFrame(rows).to_csv(os.path.join(meld_root, name), index=False)

    out_dir = os.path.join(tmp_path, "processed")
    summary = build_meld_dataset(
        meld_root=meld_root,
        out_dir=out_dir,
        annotator_name="offline",
        condition_on_speech=True,
        ser_backend="heuristic",
        prosody_dim=16,
        seed=7,
    )
    for split in ("train", "dev", "test"):
        assert os.path.exists(summary["files"][split])
        examples = load_jsonl(summary["files"][split])
        assert len(examples) == 2
        assert examples[0].prosody.shape[1] == 16
    manifest_path = summary["manifest"]
    assert os.path.exists(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["split_stats"]["train"]["n_examples"] == 2


def test_build_meld_dataset_resume_checkpoint(tmp_path):
    meld_root = os.path.join(tmp_path, "meld")
    os.makedirs(meld_root, exist_ok=True)
    rows = []
    for i in range(9):
        rows.append({
            "Dialogue_ID": 1,
            "Utterance_ID": i + 1,
            "Utterance": f"sample utterance {i}",
            "Emotion": "joy" if i % 2 == 0 else "neutral",
        })
    for name in ("train_sent_emo.csv", "dev_sent_emo.csv", "test_sent_emo.csv"):
        pd.DataFrame(rows).to_csv(os.path.join(meld_root, name), index=False)

    out_dir = os.path.join(tmp_path, "processed_resume")
    build_meld_dataset(
        meld_root=meld_root,
        out_dir=out_dir,
        annotator_name="offline",
        condition_on_speech=True,
        ser_backend="heuristic",
        max_rows_per_split=4,
        chunk_size=2,
        resume=False,
    )
    # Resume to a larger cap; should continue from existing lines.
    summary = build_meld_dataset(
        meld_root=meld_root,
        out_dir=out_dir,
        annotator_name="offline",
        condition_on_speech=True,
        ser_backend="heuristic",
        max_rows_per_split=7,
        chunk_size=2,
        resume=True,
    )
    for split in ("train", "dev", "test"):
        examples = load_jsonl(summary["files"][split])
        assert len(examples) == 7
    progress_path = os.path.join(out_dir, "progress.json")
    assert os.path.exists(progress_path)
    with open(progress_path, "r", encoding="utf-8") as f:
        progress = json.load(f)
    assert progress["splits"]["train"]["processed"] == 7
    assert progress["splits"]["train"]["status"] == "completed"
