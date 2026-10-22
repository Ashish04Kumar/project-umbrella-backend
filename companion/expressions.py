"""The shared list of expressions every body (phone, robot, laptop) knows how to show."""

from __future__ import annotations

# Body states: the body sets these itself (mic on, waiting for the brain, playing audio, failure).
BODY_STATES = ("idle", "listening", "thinking", "speaking", "error")

# Reply moods: the LLM picks one per reply with a tag like "[curious]". Bodies draw them on the face.
REPLY_EXPRESSIONS = ("happy", "curious", "concerned", "playful", "surprised", "excited", "sad", "idle")

# Matches any mood tag anywhere in a text (to strip stray ones).
ANY_TAG_PATTERN = r"\[(" + "|".join(REPLY_EXPRESSIONS) + r")\]\s*"

DEFAULT_EXPRESSION = "idle"


def normalize(tag: str | None) -> str:
    tag = (tag or "").strip().lower()
    return tag if tag in REPLY_EXPRESSIONS else DEFAULT_EXPRESSION
