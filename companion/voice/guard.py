"""Drop STT text that was probably made up from silence or a breath.

Copied from the Synq backend's stt_guard.py (`check_piece`), so the two projects stay separate.
Whisper sometimes "hears" a stock phrase ("Thank you.", "Thanks for watching") in near silence.
Here the speech length and loudness come from the whole push-to-talk clip.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLIP_SECONDS = 0.2
STOCK_PHRASE_SECONDS = 0.8
QUIET_DB = -45.0
# Pre-roll louder than this means real speech was going on around the segment.
SPEECH_AROUND_DB = -35.0
PRE_ROLL_SECONDS = 0.8

_ALWAYS_NOISE = (
    "thanks for watching", "thank you for watching", "please subscribe", "like and subscribe",
    "subtitles by", "amara.org", "transcribed by",
)
_STOCK_PHRASES = {
    "good bye", "goodbye", "bye", "bye bye", "thank you", "thanks", "thank you so much",
    "thank you very much", "you", "okay bye", "see you", "see you next time",
}
_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)


def normalise(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def check_transcript(text: str, speech_seconds: float, loudness_db: float | None) -> tuple[bool, str]:
    """(keep, reason)."""
    words = normalise(text)
    if not words:
        return False, "empty"
    if any(phrase in words for phrase in _ALWAYS_NOISE):
        return False, "known STT filler phrase"
    if speech_seconds < BLIP_SECONDS and len(words.split()) <= 4:
        return False, f"only {speech_seconds:.2f}s of speech (a blip)"
    if words in _STOCK_PHRASES and (
        speech_seconds < STOCK_PHRASE_SECONDS or (loudness_db is not None and loudness_db < QUIET_DB)
    ):
        return False, "stock phrase from very short or quiet audio (likely made up)"
    return True, "ok"


# --- matching text to speech segments -------------------------------------------------------

MAX_WORDS_PER_SECOND = 7.0


@dataclass
class Segment:
    """One stretch of caller speech between VAD start and stop."""

    ended_at: float
    seconds: float
    loudness_db: float | None
    # Loudness from PRE_ROLL_SECONDS before the segment started until it ended.
    pre_roll_db: float | None = None

    @property
    def speech_around(self) -> bool:
        return self.pre_roll_db is not None and self.pre_roll_db >= SPEECH_AROUND_DB

    @property
    def heard_seconds(self) -> float:
        """How much audio the STT may have heard words in."""
        return self.seconds + (PRE_ROLL_SECONDS if self.speech_around else 0.0)


def word_count(text: str) -> int:
    return len(normalise(text).split())


def match_segment(segments: list[Segment], words: int) -> tuple[Segment | None, int]:
    """(segment the text came from, how many waiting segments to remove).

    The text belongs to the first segment that could hold that many words; segments before it
    produced no text. If none fits, nothing is removed and the segment is None.
    """
    for index, segment in enumerate(segments):
        if words <= MAX_WORDS_PER_SECOND * segment.heard_seconds + 1:
            return segment, index + 1
    return None, 0


def check_with_segments(text: str, segments: list[Segment]) -> tuple[bool, str, Segment | None, int]:
    """Match `text` to a waiting segment and decide whether to keep it.

    Returns (keep, reason, matched segment, how many segments to remove from the queue).
    """
    words = word_count(text)
    segment, used = match_segment(segments, words)
    if segment is None:
        if words <= 4:
            return False, "no matching speech (likely made up)", None, 0
        return True, "ok (no timing)", None, 0
    keep, reason = check_transcript(text, segment.heard_seconds, segment.loudness_db)
    return keep, reason, segment, used
