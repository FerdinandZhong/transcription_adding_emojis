"""Emoji annotators that turn a transcript (+ optional detected speech emotion) into
word-level insertion / emoji labels.

Three regimes from the plan are supported by a single interface via flags:

* idea (a) text-only        -> ``condition_on_speech=False`` (drops prosody; reproduces
                               the "pure text loses emotion" failure mode);
* idea (b) speech-mapping   -> ``use_text=False`` (Speejis-style VA->emoji at end);
* idea (c) fusion           -> both, the recommended silver-label generator.

``OfflineEmojiAnnotator`` is rule-based and deterministic so the pipeline runs without
an API key. ``OpenAIEmojiAnnotator`` builds a prompt conditioned on transcript + speech
emotion and parses the model output (lazy import; install ``emoji-asr[llm]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

from ..emoji_set import EMOTION_LEXICON, EmojiSet
from .ser import SERResult


@dataclass
class Annotation:
    insertion: List[int]
    emoji_ids: List[int]


class EmojiAnnotator:
    def annotate(self, words: List[str], ser: Optional[SERResult] = None) -> Annotation:
        raise NotImplementedError


class OfflineEmojiAnnotator(EmojiAnnotator):
    """Rule-based annotator implementing text / speech / fusion regimes.

    * Text branch: scan ``words`` for emotion-lexicon hits.
    * Speech branch: use the SER emotion / VAD.
    * Fusion: trust prosody for the emoji choice when the voice is clearly emotional,
      otherwise fall back to the text branch.
    """

    def __init__(self, emoji_set: Optional[EmojiSet] = None,
                 condition_on_speech: bool = True, use_text: bool = True,
                 arousal_threshold: float = 0.25):
        self.emoji_set = emoji_set or EmojiSet()
        self.condition_on_speech = condition_on_speech
        self.use_text = use_text
        self.arousal_threshold = arousal_threshold
        self._lex = {w: emo for emo, ws in EMOTION_LEXICON.items() for w in ws}

    def _text_emotion(self, words: List[str]) -> Tuple[Optional[str], int]:
        for i, w in enumerate(words):
            emo = self._lex.get(w.lower())
            if emo:
                return emo, i
        return None, -1

    def annotate(self, words: List[str], ser: Optional[SERResult] = None) -> Annotation:
        n = len(words)
        insertion = [0] * n
        emoji_ids = [0] * n
        if n == 0:
            return Annotation(insertion, emoji_ids)

        text_emo, text_pos = (self._text_emotion(words) if self.use_text else (None, -1))

        speech_emo = None
        speech_strong = False
        if self.condition_on_speech and ser is not None:
            speech_emo = ser.emotion
            speech_strong = (speech_emo != "neutral" and
                             abs(ser.arousal) >= self.arousal_threshold)

        # Decide emotion and emoji.
        chosen_emo: Optional[str] = None
        if speech_strong:
            chosen_emo = speech_emo  # prosody carries the emotion (incl. divergent text)
        elif text_emo is not None:
            chosen_emo = text_emo

        if chosen_emo is None or chosen_emo == "neutral":
            return Annotation(insertion, emoji_ids)

        # Emoji id: if speech is informative, snap to nearest emoji in VA space;
        # otherwise use the canonical emoji for the (text) emotion.
        if speech_strong and ser is not None:
            emoji_id = self.emoji_set.nearest_by_va(ser.valence, ser.arousal, ser.dominance)
        else:
            emoji_id = self.emoji_set.primary_id_for_emotion(chosen_emo)

        # Placement: end of utterance (consistent with end-of-message tone); if a text
        # cue exists mid-sentence and matches, prefer that boundary.
        pos = n - 1
        if text_emo is not None and text_emo == chosen_emo and text_pos >= 0:
            pos = text_pos
        insertion[pos] = 1
        emoji_ids[pos] = emoji_id
        return Annotation(insertion, emoji_ids)


class OpenAIEmojiAnnotator(EmojiAnnotator):  # pragma: no cover - requires API key
    def __init__(self, emoji_set: Optional[EmojiSet] = None,
                 model: str = "gpt-4o-mini", condition_on_speech: bool = True,
                 max_retries: int = 3, retry_sleep_s: float = 1.5):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.emoji_set = emoji_set or EmojiSet()
        self.condition_on_speech = condition_on_speech
        self.max_retries = max_retries
        self.retry_sleep_s = retry_sleep_s

    def _prompt(self, words: List[str], ser: Optional[SERResult]) -> str:
        palette = " ".join(self.emoji_set.id_to_char[1:])
        speech = ""
        if self.condition_on_speech and ser is not None:
            speech = (f"\nThe speaker's voice sounds {ser.emotion} "
                      f"(valence={ser.valence:.2f}, arousal={ser.arousal:.2f}).")
        numbered = " ".join(f"[{i}]{w}" for i, w in enumerate(words))
        return (
            "Insert at most one emoji to convey the emotion of this spoken utterance. "
            "Choose from this palette: " + palette + speech +
            "\nTokens (index in brackets): " + numbered +
            "\nReply as JSON: {\"position\": <token index to place emoji AFTER, or -1>, "
            "\"emoji\": \"<one emoji from the palette, or empty>\"}."
        )

    def annotate(self, words: List[str], ser: Optional[SERResult] = None) -> Annotation:
        import json
        n = len(words)
        insertion, emoji_ids = [0] * n, [0] * n
        if n == 0:
            return Annotation(insertion, emoji_ids)
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": self._prompt(words, ser)}],
                    temperature=0.0,
                )
                data = json.loads(resp.choices[0].message.content)
                pos = int(data.get("position", -1))
                ch = (data.get("emoji") or "").strip()
                if 0 <= pos < n and ch in self.emoji_set.char_to_id:
                    insertion[pos] = 1
                    emoji_ids[pos] = self.emoji_set.id(ch)
                break
            except Exception:
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_s)
        return Annotation(insertion, emoji_ids)


def build_annotator(annotator: str = "offline", emoji_set: Optional[EmojiSet] = None,
                    condition_on_speech: bool = True, use_text: bool = True,
                    **kw) -> EmojiAnnotator:
    if annotator == "offline":
        return OfflineEmojiAnnotator(emoji_set=emoji_set,
                                     condition_on_speech=condition_on_speech,
                                     use_text=use_text)
    if annotator == "openai":
        return OpenAIEmojiAnnotator(emoji_set=emoji_set,
                                    model=kw.get("openai_model", "gpt-4o-mini"),
                                    condition_on_speech=condition_on_speech,
                                    max_retries=kw.get("openai_max_retries", 3),
                                    retry_sleep_s=kw.get("openai_retry_sleep_s", 1.5))
    raise ValueError(f"unknown annotator: {annotator}")
