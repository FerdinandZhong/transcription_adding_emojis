import os

from emoji_asr.emoji_set import EmojiSet
from emoji_asr.data.synthetic import make_splits
from emoji_asr.benchmark import (
    sample_benchmark, export_for_annotation, import_annotations,
    inter_annotator_agreement,
)
from emoji_asr.human_eval import simulate_ratings, aggregate_ratings
from emoji_asr.baselines.annotator_baseline import annotator_predict


def test_sample_oversamples_divergent():
    es = EmojiSet()
    _, _, te = make_splits(40, 10, 300, divergent_test_fraction=0.5,
                           prosody_dim=16, seed=5, emoji_set=es)
    bench = sample_benchmark(te, n=50, divergent_target=0.5, seed=5)
    frac_div = sum(e.divergent for e in bench) / len(bench)
    assert frac_div >= 0.3


def test_export_import_and_agreement(tmp_path):
    es = EmojiSet()
    _, _, te = make_splits(40, 10, 60, divergent_test_fraction=0.5,
                           prosody_dim=16, seed=6, emoji_set=es)
    bench = sample_benchmark(te, n=20, seed=6)
    path = os.path.join(tmp_path, "task.jsonl")
    export_for_annotation(bench, path, es)
    assert os.path.exists(path)

    # Build two annotator files from the (hidden) silver labels with a tiny perturbation.
    import json
    ann_path_a = os.path.join(tmp_path, "a.jsonl")
    ann_path_b = os.path.join(tmp_path, "b.jsonl")
    with open(ann_path_a, "w") as fa, open(ann_path_b, "w") as fb:
        for ex in bench:
            pos = next((j for j, v in enumerate(ex.insertion) if v == 1), -1)
            ch = next((es.char(e) for e in ex.emoji_ids if e > 0), "")
            fa.write(json.dumps({"uid": ex.uid, "position": pos, "emoji": ch}) + "\n")
            fb.write(json.dumps({"uid": ex.uid, "position": pos, "emoji": ch}) + "\n")
    va = import_annotations(ann_path_a, bench, es)
    vb = import_annotations(ann_path_b, bench, es)
    assert len(va) == len(bench)
    kappa = inter_annotator_agreement({"a": va, "b": vb}, es)
    assert kappa["insertion_kappa"] == 1.0  # identical annotators -> perfect agreement


def test_ux_simulation():
    es = EmojiSet()
    _, _, te = make_splits(40, 10, 40, divergent_test_fraction=0.5,
                           prosody_dim=16, seed=7, emoji_set=es)
    preds = annotator_predict(te, es, condition_on_speech=True)
    ratings = simulate_ratings(te, preds, es)
    summary = aggregate_ratings(ratings)
    assert summary["n"] == len(te)
    assert 0.0 <= summary["preference_augmented"] <= 1.0
