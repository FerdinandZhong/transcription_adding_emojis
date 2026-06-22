"""End-to-end experiment runner.

Trains the multimodal fusion model and the text-only ablation, runs the zero-training
baselines (Speejis-style SER mapping; LLM/rule annotator in text-only and fusion modes),
evaluates everything per group, and renders comparison tables. With the default config
this runs fully offline on synthetic data and reproduces the headline result: the
prosody-aware methods win on the prosody-divergent subset.
"""

from __future__ import annotations

import json
from typing import Dict, List

from .baselines.annotator_baseline import annotator_predict
from .baselines.ser_mapping import ser_mapping_predict
from .config import load_config
from .data.loader import load_train_dev_test
from .data.schema import Vocab
from .emoji_set import EmojiSet
from .eval import evaluate_model, evaluate_predictions
from .models.fusion_model import EmojiInsertionModel, ModelConfig
from .train import load_checkpoint, save_checkpoint, train_model
from .utils import get_device, set_seed

GROUPS = ("all", "congruent", "divergent")


def _resolve_path(cfg: Dict, key: str):
    from pathlib import Path
    tc = cfg.get("train", {})
    val = tc.get(key, "")
    if not val:
        return None
    p = Path(val)
    if not p.is_absolute():
        root = Path(__file__).resolve().parents[2]
        p = (root / p).resolve()
    return p if p.is_file() else None


def _train_eval(cfg, train, dev, test, vocab, es, use_prosody, fusion,
                *, label: str, checkpoint_key: str):
    device = get_device(cfg["train"].get("device", "auto"))
    topk = max(cfg["eval"]["topk"])
    ckpt_path = _resolve_path(cfg, checkpoint_key)

    if ckpt_path is not None:
        print(f"[{label}] loading checkpoint: {ckpt_path}")
        model, ckpt_vocab, meta = load_checkpoint(ckpt_path, device=device)
        if len(ckpt_vocab) != len(vocab):
            print(f"[WARN] {label}: vocab size mismatch (ckpt={len(ckpt_vocab)}, "
                  f"data={len(vocab)}); using checkpoint vocab")
            vocab = ckpt_vocab
        if meta.get("metrics"):
            print(f"[{label}] checkpoint metrics: {meta['metrics']}")
        return evaluate_model(model, test, vocab, es, device=device, topk=topk)

    m = cfg["model"]
    mc = ModelConfig(
        num_emoji=es.num_emoji, vocab_size=len(vocab),
        prosody_dim=m.get("prosody_dim", 32),
        hidden_size=m.get("text_encoder", {}).get("hidden_size", 128),
        fusion=fusion if use_prosody else "none", use_prosody=use_prosody,
        dropout=m.get("dropout", 0.1), text_encoder=m.get("text_encoder", {"backend": "lite"}),
    )
    model = EmojiInsertionModel(mc)
    tc = cfg["train"]
    train_model(model, train, dev, vocab, epochs=tc["epochs"], batch_size=tc["batch_size"],
                lr=tc["lr"], weight_decay=tc["weight_decay"],
                insertion_weight=tc.get("insertion_loss_weight", 1.0),
                emoji_weight=tc.get("emoji_loss_weight", 1.0), device=device,
                seed=cfg.get("seed", 13), verbose=False)
    metrics = evaluate_model(model, test, vocab, es, device=device, topk=topk)
    out_dir = tc.get("out_dir", "runs/default")
    save_key = "fusion_save_as" if use_prosody else "text_only_save_as"
    default_name = "fusion_checkpoint.pt" if use_prosody else "text_only_checkpoint.pt"
    ckpt_name = tc.get(save_key, default_name)
    summary = {g: {"placement_f1": metrics[g]["placement"]["f1"],
                   "emoji_top1": metrics[g]["emoji"]["top1"],
                   "semantics_preservation": metrics[g]["emoji"]["semantics_preservation"]}
               for g in GROUPS}
    save_checkpoint(out_dir, model, vocab, cfg, metrics=summary, filename=ckpt_name)
    return metrics


def run_experiment(config_path: str = None) -> Dict:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 13))
    es = EmojiSet()
    train, dev, test = load_train_dev_test(cfg, emoji_set=es)
    vocab = Vocab.build(train)
    topk = max(cfg["eval"]["topk"])

    results: Dict[str, Dict] = {}

    # Zero-training baselines.
    results["ser_mapping (Speejis)"] = {
        g: evaluate_predictions(test, ser_mapping_predict(test, es, topk=topk), es,
                                group=None if g == "all" else g)
        for g in GROUPS
    }
    results["llm_text_only (idea a)"] = {
        g: evaluate_predictions(
            test, annotator_predict(test, es, condition_on_speech=False, topk=topk), es,
            group=None if g == "all" else g)
        for g in GROUPS
    }
    results["llm_fusion (idea c)"] = {
        g: evaluate_predictions(
            test, annotator_predict(test, es, condition_on_speech=True, topk=topk), es,
            group=None if g == "all" else g)
        for g in GROUPS
    }

    # Trained models (optional checkpoints skip re-training).
    results["text_only (learned)"] = _train_eval(
        cfg, train, dev, test, vocab, es, use_prosody=False, fusion="none",
        label="text_only", checkpoint_key="text_only_checkpoint",
    )
    results["fusion (ours)"] = _train_eval(
        cfg, train, dev, test, vocab, es, use_prosody=True,
        fusion=cfg["model"].get("fusion", "cross_attention"),
        label="fusion", checkpoint_key="fusion_checkpoint",
    )
    return {"results": results, "config": cfg}


def render_table(results: Dict[str, Dict]) -> str:
    """Markdown table: placement F1 + emoji top1 / semantics-preservation per group."""
    header = (
        "| Method | Place F1 (all) | Top1 (all) | Top1 (congruent) | "
        "Top1 (divergent) | SemPres (divergent) | EmoFidelity (divergent) |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for name, by_group in results.items():
        a, c, d = by_group["all"], by_group["congruent"], by_group["divergent"]
        lines.append(
            f"| {name} | {a['placement']['f1']:.3f} | {a['emoji']['top1']:.3f} | "
            f"{c['emoji']['top1']:.3f} | {d['emoji']['top1']:.3f} | "
            f"{d['emoji']['semantics_preservation']:.3f} | "
            f"{d['emoji']['emotion_fidelity']:.3f} |"
        )
    return "\n".join(lines)


def main():  # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="Run the full emoji-insertion experiment")
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", action="store_true", help="dump full metrics as JSON")
    args = ap.parse_args()

    out = run_experiment(args.config)
    source = out["config"].get("data", {}).get("source", "synthetic")
    print(f"\n=== Results ({source}) ===\n")
    print(render_table(out["results"]))
    if args.json:
        print("\n=== Full metrics ===\n")
        print(json.dumps(out["results"], indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
