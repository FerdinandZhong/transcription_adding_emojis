from emoji_asr.emoji_set import EmojiSet
from emoji_asr.data.synthetic import make_splits
from emoji_asr.data.silver_labels import SilverLabeler, silver_agreement
from emoji_asr.data.llm import build_annotator


def test_synthetic_shapes_and_divergent():
    es = EmojiSet()
    tr, dv, te = make_splits(60, 20, 60, divergent_test_fraction=0.6,
                             prosody_dim=32, seed=1, emoji_set=es)
    assert len(tr) == 60 and len(dv) == 20 and len(te) == 60
    assert any(e.divergent for e in te)
    assert not any(e.divergent for e in tr)   # train is congruent
    for e in tr:
        assert e.prosody.shape == (e.num_words, 32)


def test_fusion_beats_text_only_on_divergent():
    es = EmojiSet()
    _, _, te = make_splits(40, 10, 120, divergent_test_fraction=0.6,
                           prosody_dim=32, seed=2, emoji_set=es)
    fusion = SilverLabeler(annotator=build_annotator("offline", emoji_set=es,
                                                     condition_on_speech=True))
    text = SilverLabeler(annotator=build_annotator("offline", emoji_set=es,
                                                   condition_on_speech=False))
    af = silver_agreement(te, fusion.relabel(te), es)["divergent"]
    at = silver_agreement(te, text.relabel(te), es)["divergent"]
    # Idea (c) recovers emotion on divergent cases; idea (a) cannot.
    assert af["emoji_emotion_acc"] > at["emoji_emotion_acc"]
