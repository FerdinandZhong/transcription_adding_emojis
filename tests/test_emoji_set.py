from emoji_asr.emoji_set import EmojiSet, NO_EMOJI, EMOTIONS, emotion_to_va


def test_label_space():
    es = EmojiSet()
    assert es.num_labels == es.num_emoji + 1
    assert es.char(0) == NO_EMOJI
    assert 50 <= es.num_emoji <= 80  # ~64 curated emoji


def test_emotion_lookup_roundtrip():
    es = EmojiSet()
    for emo in EMOTIONS:
        if emo == "neutral":
            continue
        pid = es.primary_id_for_emotion(emo)
        assert pid > 0
        assert es.emotion_of(pid) == emo


def test_nearest_by_va_returns_real_emoji():
    es = EmojiSet()
    v, a, d = emotion_to_va("joy")
    eid = es.nearest_by_va(v, a, d)
    assert eid >= 1
    assert es.emotion_of(eid) in {"joy", "love"}  # high valence neighbourhood
