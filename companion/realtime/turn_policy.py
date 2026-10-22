"""Live mode turn taking: when to answer, when to let the user finish, when to stop talking.

Pure rules (no audio, no network) so they can be tested on their own. LiveTurns in agent.py calls them.

End of turn. After the user stops, Umbrella waits before answering. How long depends on two signals:
- Smart Turn v3 (pipecat's small on-device model) listens to the last few seconds of the user's voice
  and says how likely it is that they finished (intonation, rhythm; about 0.2 s on a CPU);
- the words: a turn that ends on "and", "aur", "ki", "umm" or a comma is not finished, whatever the voice says.

    words trail off ("call my mom and")   -> hold up to HOLD_WAIT
    Smart Turn says finished              -> answer after QUICK_WAIT
    Smart Turn says maybe not             -> MEDIUM_WAIT (the words look complete, so not too long)
    no Smart Turn verdict                 -> UNKNOWN_WAIT (the old fixed pause)

Barge-in. Talking over Umbrella stops it only after BARGE_IN_SECONDS of continuous speech, so a cough,
"hmm" or "haan" does not cut it off. Short stop words ("wait", "ruko") stop it as soon as they are heard.
Backchannels said over Umbrella ("ok", "haan", "hmm") are not answered as a new turn.
"""

from __future__ import annotations

import logging
import re
import threading

import numpy as np

log = logging.getLogger("companion.live")

QUICK_WAIT = 0.3
MEDIUM_WAIT = 1.0
HOLD_WAIT = 2.5
UNKNOWN_WAIT = 0.8
MAX_WAIT = 3.0          # never leave the user waiting longer than this after they stopped
COMPLETE_PROBABILITY = 0.5

BARGE_IN_SECONDS = 0.5  # continuous speech over Umbrella before it stops talking
MERGE_WINDOW = 4.0      # going on within this many seconds of a sent turn (before any reply is heard) = one message

# Words that leave a sentence open, in English and Roman Hinglish. Whisper often adds a full stop
# after them ("call my mom and."), so trailing . and ! are ignored when checking.
_OPEN_ENDINGS = {
    "and", "or", "but", "so", "because", "then", "also", "with", "to", "the", "a", "an", "of", "for", "if",
    "that", "which", "like", "my", "your", "is", "are", "was", "um", "umm", "uh", "uhh", "er", "hmm",
    "aur", "ya", "ki", "ke", "ka", "ko", "se", "mein", "me", "par", "pe", "toh", "matlab", "jaise", "phir",
    "lekin", "kyunki", "wo", "woh", "yeh", "ye", "jo", "ek", "bhi",
}
_FILLER_RUN = re.compile(r"\b(u+m+|u+h+|e+r+m*|h+m+)\W*$", re.IGNORECASE)

BACKCHANNELS = {
    "haan", "han", "haa", "ha", "hmm", "hm", "mm", "mhm", "ok", "okay", "acha", "achha", "accha", "right", "yes",
    "yeah", "yep", "sure", "ji", "theek", "thik", "hai", "oh", "ah", "nice", "cool", "got", "it", "uh-huh", "huh",
    "um", "umm", "uh", "uhh", "er", "erm",   # STT often writes a short "hmm" as "um"
}
STOP_WORDS = re.compile(r"^\W*(stop|wait|hold on|one (sec|second)|ruko|ruk ja(o)?|ek (sec|second|minute)|bas)\b", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Zऀ-ॿ'-]+", text.lower())


def sounds_unfinished(text: str) -> bool:
    """The words stop mid-thought: a trailing joining word, a filler, a comma or "..."."""
    stripped = (text or "").strip()
    if not stripped or stripped.endswith("?"):
        return False
    if stripped.endswith((",", "...", "…", "-", "–", "—")):
        return True
    stripped = stripped.rstrip(".!").strip()
    if _FILLER_RUN.search(stripped):
        return True
    words = _words(stripped)
    return bool(words) and words[-1] in _OPEN_ENDINGS


def end_of_turn_wait(probability: float | None, text: str) -> tuple[float, str]:
    """Seconds of silence to wait after the user stopped, and why (shown in the Live log)."""
    if sounds_unfinished(text):
        return HOLD_WAIT, "the words trail off, letting them finish"
    if text.strip().endswith("?"):
        return QUICK_WAIT + 0.2, "a complete question"
    if probability is None:
        return UNKNOWN_WAIT, "no Smart Turn verdict, normal pause"
    if probability >= COMPLETE_PROBABILITY:
        return QUICK_WAIT, f"sounds finished (Smart Turn {probability:.2f})"
    return MEDIUM_WAIT, f"voice sounds unfinished (Smart Turn {probability:.2f}) but the words look complete"


def is_backchannel(text: str) -> bool:
    """A short acknowledgement ("ok", "haan ji", "hmm right"): not a new turn when said over Umbrella."""
    words = _words(text)
    return 0 < len(words) <= 3 and all(word in BACKCHANNELS for word in words)


def is_stop_request(text: str) -> bool:
    return bool(STOP_WORDS.search(text or ""))


class SmartTurn:
    """Pipecat's Smart Turn v3 (bundled ONNX model, onnxruntime), loaded on first use.
    Any problem makes it return None, and the rules fall back to the plain pause."""

    def __init__(self) -> None:
        self._model = None
        self._broken = False
        self._lock = threading.Lock()

    def probability(self, audio: np.ndarray) -> float | None:
        """`audio`: float32 mono at 16 kHz, the user's turn (only the last 8 s are used)."""
        if self._broken or audio.size < 1600:   # under 0.1 s: nothing to judge
            return None
        with self._lock:   # one ONNX session, used from worker threads one call at a time
            try:
                if self._model is None:
                    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

                    self._model = LocalSmartTurnAnalyzerV3()
                    self._model._sample_rate = 16000
                return float(self._model._predict_endpoint(audio.astype(np.float32))["probability"])
            except Exception:
                log.exception("Smart Turn failed; using the plain pause from now on")
                self._broken = True
                return None
