from emoji_asr.emoji_set import EmojiSet
from emoji_asr.eval.metrics import placement_prf, emoji_metrics


def test_placement_prf_perfect():
    r = placement_prf([0, 1, 0, 1], [0, 1, 0, 1])
    assert r["f1"] == 1.0


def test_placement_prf_partial():
    r = placement_prf([1, 1, 0], [1, 0, 1])
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1


def test_emoji_metrics_semantics_preservation():
    es = EmojiSet()
    joy_ids = es.ids_for_emotion("joy")
    assert len(joy_ids) >= 2
    gold = [joy_ids[0]]
    pred_top1 = [joy_ids[1]]            # different emoji, same emotion
    pred_topk = [[joy_ids[1], joy_ids[0]]]
    m = emoji_metrics(gold, pred_top1, pred_topk, ["joy"], es)
    assert m["top1"] == 0.0                       # exact match fails
    assert m["topk"] == 1.0                       # gold appears in top-k
    assert m["semantics_preservation"] == 1.0     # same emotion -> preserved
    assert m["emotion_fidelity"] == 1.0
