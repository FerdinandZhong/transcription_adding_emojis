"""Curated emoji label set with affective coordinates.

The emoji vocabulary is grounded in the valence/arousal (VA) circumplex used by
Speejis (Kutsuzawa et al., 2022, who report VA values for 74 facial emoji). We expose:

* a categorical emotion taxonomy (MELD-compatible, plus a few affective shades),
* per-emoji VA(D) coordinates and an emotion tag,
* an emotion -> emoji index and a VA -> nearest-emoji map (the Speejis-style
  fixed lookup used by one of our baselines),
* a small emotion lexicon used by the offline silver-label annotator and the
  synthetic data generator.

Index 0 is reserved for ``NO_EMOJI`` so the same id space serves the joint
insertion+selection objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

NO_EMOJI = "<none>"

# MELD's seven emotions plus affective shades that benefit from prosody.
EMOTIONS: List[str] = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
    "love",
    "sarcasm",
]


@dataclass(frozen=True)
class EmojiEntry:
    """One emoji and its affective coordinates.

    valence, arousal, dominance are in [-1, 1] following the circumplex
    convention (valence: unpleasant..pleasant, arousal: calm..excited).
    """

    char: str
    name: str
    emotion: str
    valence: float
    arousal: float
    dominance: float = 0.0


# Curated ~64-emoji set. VA values approximate the Kutsuzawa circumplex placement;
# they are intentionally coarse and meant to be replaced by the published table when
# running real experiments.
_ENTRIES: List[EmojiEntry] = [
    # neutral
    EmojiEntry("😐", "neutral_face", "neutral", 0.0, -0.1),
    EmojiEntry("🙂", "slight_smile", "neutral", 0.2, -0.1),
    EmojiEntry("😶", "no_mouth", "neutral", 0.0, -0.3),
    EmojiEntry("🤔", "thinking", "neutral", -0.05, 0.05),
    EmojiEntry("😑", "expressionless", "neutral", -0.1, -0.3),
    # joy
    EmojiEntry("😀", "grinning", "joy", 0.8, 0.5),
    EmojiEntry("😄", "smile_open", "joy", 0.85, 0.55),
    EmojiEntry("😁", "beaming", "joy", 0.85, 0.6),
    EmojiEntry("😆", "laughing", "joy", 0.9, 0.7),
    EmojiEntry("😊", "blush_smile", "joy", 0.8, 0.35),
    EmojiEntry("😂", "tears_of_joy", "joy", 0.85, 0.8),
    EmojiEntry("🤣", "rofl", "joy", 0.9, 0.85),
    EmojiEntry("😃", "smiley", "joy", 0.82, 0.5),
    EmojiEntry("🥳", "partying", "joy", 0.9, 0.8),
    EmojiEntry("😎", "sunglasses", "joy", 0.6, 0.3),
    # sadness
    EmojiEntry("😢", "crying", "sadness", -0.7, 0.1),
    EmojiEntry("😭", "loudly_crying", "sadness", -0.8, 0.5),
    EmojiEntry("😞", "disappointed", "sadness", -0.6, -0.2),
    EmojiEntry("😔", "pensive", "sadness", -0.55, -0.3),
    EmojiEntry("🙁", "slight_frown", "sadness", -0.45, -0.1),
    EmojiEntry("😟", "worried", "sadness", -0.5, 0.1),
    EmojiEntry("😣", "persevere", "sadness", -0.5, 0.2),
    EmojiEntry("💔", "broken_heart", "sadness", -0.8, 0.2),
    # anger
    EmojiEntry("😠", "angry", "anger", -0.6, 0.6),
    EmojiEntry("😡", "rage", "anger", -0.75, 0.8),
    EmojiEntry("🤬", "cursing", "anger", -0.8, 0.85),
    EmojiEntry("😩", "weary", "anger", -0.4, 0.55),
    EmojiEntry("👿", "imp", "anger", -0.7, 0.6),
    # fear
    EmojiEntry("😨", "fearful", "fear", -0.6, 0.7),
    EmojiEntry("😰", "anxious_sweat", "fear", -0.6, 0.65),
    EmojiEntry("😱", "screaming", "fear", -0.65, 0.9),
    EmojiEntry("😬", "grimacing", "fear", -0.3, 0.4),
    EmojiEntry("🥶", "cold_fear", "fear", -0.55, 0.5),
    # disgust
    EmojiEntry("🤢", "nauseated", "disgust", -0.7, 0.4),
    EmojiEntry("🤮", "vomiting", "disgust", -0.8, 0.6),
    EmojiEntry("😖", "confounded", "disgust", -0.55, 0.4),
    EmojiEntry("😒", "unamused", "disgust", -0.45, -0.1),
    EmojiEntry("🙄", "eye_roll", "disgust", -0.4, 0.0),
    # surprise
    EmojiEntry("😮", "open_mouth", "surprise", 0.1, 0.6),
    EmojiEntry("😲", "astonished", "surprise", 0.05, 0.8),
    EmojiEntry("😯", "hushed", "surprise", 0.05, 0.5),
    EmojiEntry("🤯", "mind_blown", "surprise", 0.1, 0.9),
    EmojiEntry("😳", "flushed", "surprise", -0.1, 0.6),
    # love
    EmojiEntry("❤️", "red_heart", "love", 0.85, 0.4),
    EmojiEntry("😍", "heart_eyes", "love", 0.9, 0.6),
    EmojiEntry("🥰", "smiling_hearts", "love", 0.9, 0.45),
    EmojiEntry("😘", "kiss", "love", 0.85, 0.45),
    EmojiEntry("💕", "two_hearts", "love", 0.85, 0.4),
    EmojiEntry("🤗", "hug", "love", 0.75, 0.45),
    # sarcasm / playful (often prosody-only)
    EmojiEntry("😏", "smirk", "sarcasm", 0.2, 0.2),
    EmojiEntry("😜", "winking_tongue", "sarcasm", 0.5, 0.5),
    EmojiEntry("😉", "wink", "sarcasm", 0.45, 0.25),
    EmojiEntry("🙃", "upside_down", "sarcasm", 0.1, 0.2),
    EmojiEntry("😅", "sweat_smile", "sarcasm", 0.35, 0.5),
    # extra commonly-used faces spread across the space
    EmojiEntry("😴", "sleeping", "sadness", -0.2, -0.7),
    EmojiEntry("🥱", "yawn", "neutral", -0.2, -0.6),
    EmojiEntry("😋", "yum", "joy", 0.75, 0.4),
    EmojiEntry("🤩", "star_struck", "joy", 0.9, 0.75),
    EmojiEntry("😤", "triumph", "anger", -0.2, 0.5),
    EmojiEntry("😇", "innocent", "joy", 0.7, 0.2),
    EmojiEntry("🤨", "raised_brow", "neutral", -0.1, 0.1),
    EmojiEntry("😵", "dizzy", "surprise", -0.2, 0.6),
    EmojiEntry("😪", "sleepy_tear", "sadness", -0.4, -0.4),
]


class EmojiSet:
    """Index <-> emoji mappings plus affective lookups.

    Label id 0 is ``NO_EMOJI``; ids 1..K are emoji in ``entries`` order.
    """

    def __init__(self, entries: Optional[List[EmojiEntry]] = None):
        self.entries: List[EmojiEntry] = list(entries if entries is not None else _ENTRIES)
        # id 0 reserved for NO_EMOJI
        self.id_to_char: List[str] = [NO_EMOJI] + [e.char for e in self.entries]
        self.char_to_id: Dict[str, int] = {c: i for i, c in enumerate(self.id_to_char)}
        self._emotion_to_ids: Dict[str, List[int]] = {}
        for i, e in enumerate(self.entries, start=1):
            self._emotion_to_ids.setdefault(e.emotion, []).append(i)

    # --- sizes ---
    @property
    def num_emoji(self) -> int:
        """Number of real emoji (excluding NO_EMOJI)."""
        return len(self.entries)

    @property
    def num_labels(self) -> int:
        """Label space size including NO_EMOJI (= num_emoji + 1)."""
        return len(self.id_to_char)

    # --- lookups ---
    def char(self, label_id: int) -> str:
        return self.id_to_char[label_id]

    def id(self, char: str) -> int:
        return self.char_to_id[char]

    def entry(self, label_id: int) -> Optional[EmojiEntry]:
        if label_id <= 0:
            return None
        return self.entries[label_id - 1]

    def emotion_of(self, label_id: int) -> str:
        if label_id <= 0:
            return "neutral"
        return self.entries[label_id - 1].emotion

    def ids_for_emotion(self, emotion: str) -> List[int]:
        return list(self._emotion_to_ids.get(emotion, []))

    def primary_id_for_emotion(self, emotion: str) -> int:
        """The canonical (first listed) emoji id for an emotion, else NO_EMOJI."""
        ids = self._emotion_to_ids.get(emotion, [])
        return ids[0] if ids else 0

    def nearest_by_va(self, valence: float, arousal: float,
                      dominance: float = 0.0) -> int:
        """Speejis-style fixed mapping: closest emoji in VA(D) space.

        Returns a label id in 1..K (never NO_EMOJI).
        """
        best_id, best_d = 1, float("inf")
        for i, e in enumerate(self.entries, start=1):
            d = (e.valence - valence) ** 2 + (e.arousal - arousal) ** 2
            d += 0.25 * (e.dominance - dominance) ** 2
            if d < best_d:
                best_d, best_id = d, i
        return best_id

    def va_of(self, label_id: int) -> Tuple[float, float, float]:
        e = self.entry(label_id)
        if e is None:
            return (0.0, -0.1, 0.0)
        return (e.valence, e.arousal, e.dominance)


# Emotion -> representative VAD centroid (used to synthesize prosody and to map SER
# categorical output to a point in VA space). Values are coarse circumplex anchors.
EMOTION_VAD: Dict[str, Tuple[float, float, float]] = {
    "neutral": (0.0, -0.1, 0.0),
    "joy": (0.8, 0.55, 0.3),
    "sadness": (-0.65, -0.1, -0.4),
    "anger": (-0.6, 0.7, 0.4),
    "fear": (-0.6, 0.7, -0.5),
    "disgust": (-0.6, 0.3, 0.1),
    "surprise": (0.1, 0.75, -0.1),
    "love": (0.85, 0.45, 0.2),
    "sarcasm": (0.3, 0.35, 0.3),
}


# Emotion lexicon: surface words that *textually* signal an emotion. Used by the
# offline annotator and by the synthetic generator to build text-congruent examples.
EMOTION_LEXICON: Dict[str, List[str]] = {
    "joy": ["happy", "great", "awesome", "love", "wonderful", "yay", "excited", "fun"],
    "sadness": ["sad", "sorry", "unfortunately", "miss", "cry", "lonely", "lost"],
    "anger": ["angry", "furious", "hate", "annoyed", "ridiculous", "unacceptable"],
    "fear": ["scared", "afraid", "terrified", "worried", "nervous", "dangerous"],
    "disgust": ["disgusting", "gross", "nasty", "ew", "revolting", "awful"],
    "surprise": ["wow", "really", "unbelievable", "suddenly", "shocked", "what"],
    "love": ["love", "adore", "sweetheart", "darling", "cherish"],
    "sarcasm": ["sure", "obviously", "whatever", "great", "fantastic", "nice"],
}


def default_emoji_set() -> EmojiSet:
    return EmojiSet()


def emotion_to_va(emotion: str) -> Tuple[float, float, float]:
    return EMOTION_VAD.get(emotion, EMOTION_VAD["neutral"])


def va_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
