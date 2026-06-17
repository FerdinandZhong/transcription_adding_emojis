import torch

from emoji_asr.emoji_set import EmojiSet
from emoji_asr.data.synthetic import make_splits
from emoji_asr.data.schema import Vocab, EmojiDataset, collate
from emoji_asr.models.fusion_model import EmojiInsertionModel, ModelConfig, decode_predictions
from emoji_asr.train import train_model
from emoji_asr.eval import evaluate_model


def _build(use_prosody):
    es = EmojiSet()
    tr, dv, te = make_splits(200, 40, 120, divergent_test_fraction=0.5,
                             prosody_dim=32, seed=3, emoji_set=es)
    vocab = Vocab.build(tr)
    mc = ModelConfig(num_emoji=es.num_emoji, vocab_size=len(vocab), prosody_dim=32,
                     hidden_size=64, fusion="cross_attention" if use_prosody else "none",
                     use_prosody=use_prosody)
    model = EmojiInsertionModel(mc)
    return es, tr, dv, te, vocab, model


def test_forward_shapes_and_decode():
    es, tr, _, _, vocab, model = _build(True)
    ds = EmojiDataset(tr[:8], vocab)
    batch = collate([ds[i] for i in range(8)])
    out = model(batch)
    assert out["insertion_logits"].shape == batch["mask"].shape
    assert out["emoji_logits"].shape[-1] == es.num_emoji
    preds = decode_predictions(out, batch["mask"], topk=3)
    assert len(preds) == 8
    assert all(len(p) == int(batch["mask"][i].sum()) for i, p in enumerate(preds))


def test_fusion_outperforms_text_only_on_divergent():
    torch.manual_seed(0)
    es, tr, dv, te, vocab, fusion = _build(True)
    train_model(fusion, tr, dv, vocab, epochs=12, batch_size=16, lr=5e-4,
                device=torch.device("cpu"), verbose=False)
    rf = evaluate_model(fusion, te, vocab, es, device=torch.device("cpu"), topk=3)

    torch.manual_seed(0)
    es2, tr2, dv2, te2, vocab2, text = _build(False)
    train_model(text, tr2, dv2, vocab2, epochs=12, batch_size=16, lr=5e-4,
                device=torch.device("cpu"), verbose=False)
    rt = evaluate_model(text, te2, vocab2, es2, device=torch.device("cpu"), topk=3)

    fusion_sp = rf["divergent"]["emoji"]["semantics_preservation"]
    text_sp = rt["divergent"]["emoji"]["semantics_preservation"]
    assert fusion_sp > text_sp + 0.2  # prosody clearly helps on divergent cases
