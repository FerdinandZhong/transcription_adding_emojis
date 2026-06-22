"""Training loop for the emoji-insertion model.

``train_model`` is the importable entry point used by the experiment runner and tests.
A thin ``main`` wraps it as a CLI driven by a YAML config; it builds synthetic data by
default so ``python -m emoji_asr.train`` runs with no external dependencies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from .data.schema import EmojiDataset, Example, Vocab, collate
from .models.fusion_model import EmojiInsertionModel, ModelConfig
from .utils import get_device, set_seed


def _iter_batches(ds: EmojiDataset, batch_size: int, shuffle: bool, rng):
    idx = list(range(len(ds)))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        yield collate([ds[i] for i in chunk])


def train_model(model: EmojiInsertionModel, train_examples: List[Example],
                dev_examples: Optional[List[Example]], vocab: Vocab,
                epochs: int = 8, batch_size: int = 16, lr: float = 5e-4,
                weight_decay: float = 0.01, insertion_weight: float = 1.0,
                emoji_weight: float = 1.0, device=None, seed: int = 13,
                verbose: bool = True) -> Dict:
    import torch

    device = device or get_device("auto")
    model.to(device)
    rng = np.random.default_rng(seed)
    ds = EmojiDataset(train_examples, vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": []}
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in _iter_batches(ds, batch_size, shuffle=True, rng=rng):
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            out = model(batch)
            ld = model.compute_loss(batch, out, insertion_weight, emoji_weight)
            opt.zero_grad()
            ld["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(ld["loss"].detach()))
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history["train_loss"].append(mean_loss)
        if verbose:
            print(f"epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}")
    return history


def save_checkpoint(
    out_dir: Union[str, Path],
    model: EmojiInsertionModel,
    vocab: Vocab,
    cfg: Dict,
    *,
    metrics: Optional[Dict] = None,
    history: Optional[Dict] = None,
    filename: str = "checkpoint.pt",
) -> Path:
    """Persist fine-tuned weights and metadata under ``out_dir``."""
    import torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / filename

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model.cfg),
        "vocab_itos": list(vocab.itos),
        "train_config": cfg,
        "metrics": metrics,
        "history": history,
    }
    torch.save(payload, ckpt_path)

    with (out / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if metrics is not None:
        with (out / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

    if history is not None:
        with (out / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device=None,
) -> Tuple[EmojiInsertionModel, Vocab, Dict]:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    import torch

    device = device or get_device("auto")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = ModelConfig(**ckpt["model_config"])
    vocab = Vocab()
    vocab.itos = list(ckpt["vocab_itos"])
    vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
    model = EmojiInsertionModel(mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    meta = {
        "train_config": ckpt.get("train_config"),
        "metrics": ckpt.get("metrics"),
        "history": ckpt.get("history"),
    }
    return model, vocab, meta


def build_model_from_cfg(cfg: Dict, vocab: Vocab, num_emoji: int) -> EmojiInsertionModel:
    m = cfg["model"]
    use_prosody = m.get("type", "fusion") == "fusion"
    mc = ModelConfig(
        num_emoji=num_emoji,
        vocab_size=len(vocab),
        prosody_dim=m.get("prosody_dim", 32),
        hidden_size=m.get("text_encoder", {}).get("hidden_size", 128),
        fusion=m.get("fusion", "cross_attention") if use_prosody else "none",
        use_prosody=use_prosody,
        dropout=m.get("dropout", 0.1),
        text_encoder=m.get("text_encoder", {"backend": "lite"}),
    )
    return EmojiInsertionModel(mc)


def main():  # pragma: no cover - CLI
    from .config import load_config
    from .data.loader import load_train_dev_test
    from .emoji_set import EmojiSet
    from .eval import evaluate_model

    ap = argparse.ArgumentParser(description="Train the emoji-insertion model")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 13))
    es = EmojiSet()
    train, dev, test = load_train_dev_test(cfg, emoji_set=es)
    vocab = Vocab.build(train)
    model = build_model_from_cfg(cfg, vocab, es.num_emoji)
    tc = cfg["train"]
    device = get_device(tc.get("device", "auto"))
    history = train_model(model, train, dev, vocab, epochs=tc["epochs"],
                batch_size=tc["batch_size"], lr=tc["lr"],
                weight_decay=tc["weight_decay"],
                insertion_weight=tc.get("insertion_loss_weight", 1.0),
                emoji_weight=tc.get("emoji_loss_weight", 1.0), device=device,
                seed=cfg.get("seed", 13))
    metrics = evaluate_model(model, test, vocab, es, device=device,
                             topk=max(cfg["eval"]["topk"]))
    summary = {g: {"placement_f1": metrics[g]["placement"]["f1"],
                   "emoji_top1": metrics[g]["emoji"]["top1"],
                   "semantics_preservation": metrics[g]["emoji"]["semantics_preservation"]}
               for g in ("all", "congruent", "divergent")}
    print(json.dumps(summary, indent=2))

    out_dir = tc.get("out_dir", "runs/default")
    ckpt_name = tc.get("checkpoint", "checkpoint.pt")
    save_checkpoint(out_dir, model, vocab, cfg, metrics=summary, history=history,
                    filename=ckpt_name)


if __name__ == "__main__":  # pragma: no cover
    main()
