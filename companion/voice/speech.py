"""Turn a streamed LLM reply into sentences ready to speak, as they arrive.

Copied in spirit from Synq Pod 4's stream_talk: a sentence goes to TTS as soon as it is finished,
so the companion starts talking before the whole reply exists. Each sentence is checked first:
no made-up life stories, no mention of the silence in a nudge, no junk, at most ~70 words in all.
"""

from __future__ import annotations

import re

from ..expressions import ANY_TAG_PATTERN, normalize
from ..guard import claims_human_life, looks_broken, mentions_silence

_SENTENCE_END = re.compile(r"([.!?।…]+[\"'”’)\]]*)(\s+|$)")
_TAG = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")
_ANY_TAG = re.compile(ANY_TAG_PATTERN, re.IGNORECASE)
_SPEAKER = re.compile(r"^\s*(assistant|buddy)\s*:\s*", re.IGNORECASE)
MAX_SPOKEN_WORDS = 70


def split_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Cut finished sentences off the front of `buffer`; returns (sentences, rest).

    A sentence ends at . ! ? (or the Hindi danda / an ellipsis) followed by a space. Very short
    pieces ("Oh." "Haan.") are kept with the next sentence so the voice doesn't sound choppy.
    With final=True, whatever is left becomes the last sentence.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        if not match.group(2) and not final:
            break  # the end mark might still be followed by more of the same sentence
        candidate = buffer[start: match.end()].strip()
        if len(candidate.split()) < 3:
            continue  # too short on its own: join it to the next sentence
        sentences.append(candidate)
        start = match.end()
    rest = buffer[start:]
    if final and rest.strip():
        sentences.append(rest.strip())
        rest = ""
    return sentences, rest


class SentenceShaper:
    """Feed it streamed pieces; it returns the sentences that may be spoken."""

    def __init__(self, *, nudge: bool = False, max_words: int | None = MAX_SPOKEN_WORDS) -> None:
        self.nudge = nudge
        self.max_words = max_words
        self.expression: str | None = None
        self.words = 0
        self.full = False            # the word budget is used up: ignore the rest
        self.dropped: list[str] = []  # for logs
        self._buffer = ""
        self._tag_pending = True

    def start_round(self) -> None:
        """A new LLM round (after a tool call) may begin with its own mood tag."""
        self._buffer, self._tag_pending = "", True

    def feed(self, piece: str) -> list[str]:
        self._buffer += piece
        if self._tag_pending:
            stripped = self._buffer.lstrip()
            if stripped.startswith("[") and "]" not in stripped and len(stripped) < 24:
                return []  # the tag is still arriving
            match = _TAG.match(self._buffer)
            if match:
                if self.expression is None:
                    self.expression = normalize(match.group(1))
                self._buffer = self._buffer[match.end():]
            self._tag_pending = False
        sentences, self._buffer = split_sentences(self._buffer)
        return self._accept(sentences)

    def end_round(self) -> list[str]:
        sentences, self._buffer = split_sentences(self._buffer, final=True)
        self._tag_pending = True
        return self._accept(sentences)

    def _accept(self, sentences: list[str]) -> list[str]:
        kept: list[str] = []
        for sentence in sentences:
            if self.full:
                break
            sentence = _ANY_TAG.sub("", _SPEAKER.sub("", sentence)).strip().strip('"').strip()
            if not sentence:
                continue
            if claims_human_life(sentence) or (self.nudge and mentions_silence(sentence)) or looks_broken(sentence):
                self.dropped.append(sentence)
                continue
            count = len(sentence.split())
            if self.max_words is not None and self.words + count > self.max_words and self.words:
                self.full = True
                break
            self.words += count
            kept.append(sentence)
        return kept
