"""CLI to evaluate trained models and zero-training baselines, plus a simulated UX study.

Defaults run fully offline on synthetic data. Outputs the per-group metric tables and
(optionally) a human-eval JSONL survey + simulated rating summary.
"""

from __future__ import annotations

import argparse
import json
import os


def main():  # pragma: no cover - CLI
    from .config import load_config
    from .data.synthetic import make_splits
    from .emoji_set import EmojiSet
    from .experiment import render_table, run_experiment
    from .baselines.annotator_baseline import annotator_predict
    from .human_eval import aggregate_ratings, export_ab_survey, simulate_ratings
    from .utils import set_seed

    ap = argparse.ArgumentParser(description="Evaluate emoji-insertion methods")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ux_study", action="store_true",
                    help="export an A/B survey and print simulated ratings")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out = run_experiment(args.config)
    table = render_table(out["results"])
    print("\n=== Results (synthetic) ===\n")
    print(table)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write("# Results\n\n" + table + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(out["results"], f, indent=2, default=str)

    if args.ux_study:
        cfg = load_config(args.config)
        set_seed(cfg.get("seed", 13))
        es = EmojiSet()
        sc = cfg["data"]["synthetic"]
        _, _, test = make_splits(sc["n_train"], sc["n_dev"], sc["n_test"],
                                 divergent_test_fraction=sc.get("divergent_test_fraction", 0.5),
                                 prosody_dim=cfg["model"].get("prosody_dim", 32),
                                 seed=cfg.get("seed", 13), emoji_set=es)
        preds = annotator_predict(test, es, condition_on_speech=True, topk=3)
        survey_path = os.path.join(args.out_dir, "ux_survey.jsonl")
        export_ab_survey(test, preds, es, survey_path)
        ratings = simulate_ratings(test, preds, es)
        summary = aggregate_ratings(ratings)
        print("\n=== Simulated UX study (replace with real raters) ===\n")
        print(json.dumps(summary, indent=2))
        with open(os.path.join(args.out_dir, "ux_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
