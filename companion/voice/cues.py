"""Copied from the Synq backend (app/voice_cues.py) so the two projects stay separate.

How the caller *sounded* on a turn, measured from the call itself (not just the words).

The STT turns speech into words, and the LLM only sees words. This module adds cheap,
rough measurements of the delivery so the LLM can adapt its tone:

  - pause before answering, and whether they cut the companion off   (VAD timing)
  - speaking time and long pauses mid-answer                (VAD timing)
  - speaking pace (words per minute)                        (transcript words / speaking time)
  - loudness, how much the loudness moves, and whether the
    voice trailed off at the end                            (raw microphone audio)
  - how much the pitch moves (flat vs lively)               (raw microphone audio)
  - filler words like "um", "hmm"                           (transcript)

It never depends on a particular STT or TTS: VAD events and raw audio frames arrive the
same way whatever provider is set in .env, and only the final transcript text is used
from the STT. So swapping Whisper/Piper for a paid provider later needs no change here.

Labels compare the caller with *their own* earlier turns in this call (once there are
two), because voices differ a lot between people. Nothing here is stored: numbers live
only for the call, and audio is discarded as soon as a turn is measured.

These are soft hints, not emotions. The prompts say so, and the companion never mentions them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median

import numpy as np

FRAME_SECONDS = 0.02
MAX_TURN_SECONDS = 30.0
# Words that are just thinking sounds. Some STTs drop them, so this is a lower bound.
_FILLERS = {"um", "umm", "ummm", "uh", "uhh", "uhm", "hmm", "hmmm", "hm", "erm", "er", "aah", "ahh", "mmm"}
_WORD = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class VoiceCues:
    """Measurements for one caller turn plus labels relative to their usual."""

    pause_before_s: float | None = None
    interrupted: bool = False
    speech_s: float = 0.0
    words: int = 0
    words_per_min: float | None = None
    long_pauses: int = 0
    longest_pause_s: float = 0.0
    loudness_db: float | None = None
    loudness_spread_db: float | None = None
    pitch_spread_st: float | None = None
    trailed_off: bool = False
    fillers: int = 0
    # Labels (None when there's not enough to say).
    pace: str | None = None          # slower | faster (than usual)
    volume: str | None = None        # softer | louder (than usual)
    intonation: str | None = None    # flatter | livelier (than usual)
    short_reply: bool = False

    def describe(self) -> str:
        """One short line for the LLM: only what stood out. Empty when nothing did."""
        notes: list[str] = []
        if self.interrupted:
            notes.append("started talking while you were still speaking")
        elif self.pause_before_s is not None and self.pause_before_s >= 3.0:
            notes.append(f"took {self.pause_before_s:.0f}s before answering")
        elif self.pause_before_s is not None and self.pause_before_s <= 0.4 and not self.short_reply:
            notes.append("answered right away")
        if self.short_reply:
            notes.append("very short reply")
        if self.pace:
            notes.append(f"spoke {self.pace} than their usual")
        if self.volume:
            notes.append(f"{self.volume} than their usual")
        if self.intonation:
            notes.append(f"{self.intonation} intonation than their usual")
        if self.trailed_off:
            notes.append("voice trailed off at the end")
        if self.long_pauses:
            notes.append(f"{self.long_pauses} long pause{'s' if self.long_pauses > 1 else ''} mid-answer")
        if self.fillers >= 2:
            notes.append(f"{self.fillers} filler sounds (um/hmm)")
        return "; ".join(notes)

    def log_fields(self) -> dict[str, object]:
        rounded = lambda value: None if value is None else round(value, 1)  # noqa: E731
        return {
            "pause_before_s": rounded(self.pause_before_s),
            "interrupted": self.interrupted,
            "speech_s": rounded(self.speech_s),
            "wpm": rounded(self.words_per_min),
            "long_pauses": self.long_pauses,
            "loudness_db": rounded(self.loudness_db),
            "loudness_spread_db": rounded(self.loudness_spread_db),
            "pitch_spread_st": rounded(self.pitch_spread_st),
            "trailed_off": self.trailed_off,
            "fillers": self.fillers,
        }


# --- audio measurements (pure functions) --------------------------------------------------


def _frames(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    size = int(sample_rate * FRAME_SECONDS)
    count = len(samples) // size
    return samples[: count * size].reshape(count, size) if count else np.zeros((0, size))


def frame_db(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Loudness of each 20 ms frame in dBFS (samples are floats in -1..1)."""
    frames = _frames(samples, sample_rate)
    if not len(frames):
        return np.zeros(0)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def pitch_track(samples: np.ndarray, sample_rate: int, voiced: np.ndarray) -> np.ndarray:
    """Pitch (Hz) of voiced 40 ms windows via normalised autocorrelation; 75-400 Hz voices."""
    size = int(sample_rate * FRAME_SECONDS)
    window = size * 2
    low, high = int(sample_rate / 400), int(sample_rate / 75)
    pitches = []
    for index in np.flatnonzero(voiced):
        start = index * size
        chunk = samples[start : start + window]
        if len(chunk) < window:
            continue
        chunk = chunk - chunk.mean()
        spectrum = np.fft.rfft(chunk, n=window * 2)
        corr = np.fft.irfft(spectrum * np.conj(spectrum))[:window]
        if corr[0] <= 0:
            continue
        corr = corr / corr[0]
        lag = low + int(np.argmax(corr[low:high]))
        if corr[lag] >= 0.5:  # clearly periodic: a voiced sound
            pitches.append(sample_rate / lag)
    return np.array(pitches)


def measure_audio(samples: np.ndarray, sample_rate: int) -> dict[str, float | bool | None]:
    """Loudness, loudness spread, pitch spread and trail-off for one turn's speech audio."""
    db = frame_db(samples, sample_rate)
    if len(db) < 10:
        return {"loudness_db": None, "loudness_spread_db": None, "pitch_spread_st": None, "trailed_off": False}
    voiced = db > max(np.percentile(db, 90) - 25, -55)  # skip near-silent frames between words
    if voiced.sum() < 10:
        return {"loudness_db": None, "loudness_spread_db": None, "pitch_spread_st": None, "trailed_off": False}
    loud = db[voiced]
    pitches = pitch_track(samples, sample_rate, voiced)
    pitch_spread = None
    if len(pitches) >= 12:
        semitones = 12 * np.log2(pitches / np.median(pitches))
        pitch_spread = float(np.std(semitones))
    # Trailing off: the last quarter of the speech is clearly quieter than the rest.
    cut = int(len(loud) * 0.75)
    trailed = len(loud) >= 40 and float(np.mean(loud[cut:])) < float(np.mean(loud[:cut])) - 6
    return {
        "loudness_db": float(np.median(loud)),
        "loudness_spread_db": float(np.std(loud)),
        "pitch_spread_st": pitch_spread,
        "trailed_off": trailed,
    }


def count_fillers(text: str) -> int:
    return sum(1 for word in _WORD.findall(text.lower()) if word in _FILLERS)


# --- per-call tracker ---------------------------------------------------------------------


@dataclass
class _Baseline:
    wpm: list[float] = field(default_factory=list)
    loudness: list[float] = field(default_factory=list)
    pitch_spread: list[float] = field(default_factory=list)

    def label(self, values: list[float], value: float | None, *, low: float, high: float, relative: bool) -> int:
        """-1 below their usual, +1 above, 0 normal or not enough history (needs 2 earlier turns)."""
        if value is None or len(values) < 2:
            return 0
        usual = median(values)
        if relative:
            if value < usual * low:
                return -1
            if value > usual * high:
                return 1
            return 0
        if value < usual - low:
            return -1
        if value > usual + high:
            return 1
        return 0


class VoiceCueTracker:
    """Feed it the call's VAD events and microphone audio; ask for cues when a turn is done.

    Times are in seconds from any monotonic clock. Audio is 16-bit mono PCM.
    """

    def __init__(self) -> None:
        self._bot_stopped_at: float | None = None
        self._bot_speaking = False
        self._turn_started: float | None = None
        self._interrupted = False
        self._segments: list[tuple[float, float | None]] = []
        self._audio: list[bytes] = []
        self._audio_bytes = 0
        self._sample_rate = 16000
        self._user_speaking = False
        self._baseline = _Baseline()

    # Events -------------------------------------------------------------------------------

    def bot_started(self, now: float) -> None:
        self._bot_speaking = True

    def bot_stopped(self, now: float) -> None:
        self._bot_speaking = False
        self._bot_stopped_at = now

    def user_started(self, now: float) -> None:
        if not self._segments:
            self._turn_started = now
            self._interrupted = self._bot_speaking
        self._segments.append((now, None))
        self._user_speaking = True

    def user_stopped(self, now: float) -> None:
        if self._segments and self._segments[-1][1] is None:
            self._segments[-1] = (self._segments[-1][0], now)
        self._user_speaking = False

    def audio(self, pcm: bytes, sample_rate: int) -> None:
        """Microphone audio; only kept while the caller is speaking, capped at 30 s a turn."""
        if not self._user_speaking:
            return
        self._sample_rate = sample_rate
        if self._audio_bytes < MAX_TURN_SECONDS * sample_rate * 2:
            self._audio.append(pcm)
            self._audio_bytes += len(pcm)

    # Results ------------------------------------------------------------------------------

    def discard_turn(self) -> None:
        """Forget the current turn (a sound that gave no words), so it doesn't skew the next one."""
        self._segments, self._audio, self._audio_bytes = [], [], 0
        self._turn_started, self._interrupted = None, False

    def take_turn(self, text: str, now: float) -> "TurnSnapshot":
        """Hand over the turn that just ended and start a fresh one (fast; no maths here)."""
        snapshot = TurnSnapshot(
            text=text,
            segments=[(start, end if end is not None else now) for start, end in self._segments],
            pcm=b"".join(self._audio),
            sample_rate=self._sample_rate,
            pause_before=None if self._turn_started is None or self._bot_stopped_at is None or self._interrupted
            else max(0.0, self._turn_started - self._bot_stopped_at),
            interrupted=self._interrupted,
            baseline=self._baseline,
        )
        self._segments, self._audio, self._audio_bytes = [], [], 0
        self._turn_started, self._interrupted = None, False
        return snapshot


@dataclass
class TurnSnapshot:
    """Everything needed to measure one turn; `measure()` is safe to run in a worker thread."""

    text: str
    segments: list[tuple[float, float]]
    pcm: bytes
    sample_rate: int
    pause_before: float | None
    interrupted: bool
    baseline: _Baseline

    def measure(self) -> VoiceCues:
        return compute_cues(
            text=self.text,
            segments=self.segments,
            pcm=self.pcm,
            sample_rate=self.sample_rate,
            pause_before=self.pause_before,
            interrupted=self.interrupted,
            baseline=self.baseline,
        )


def compute_cues(
    *,
    text: str,
    segments: list[tuple[float, float]],
    pcm: bytes,
    sample_rate: int,
    pause_before: float | None,
    interrupted: bool,
    baseline: _Baseline | None = None,
) -> VoiceCues:
    """Pure: turn timings + audio + words into cues; updates `baseline` with this turn."""
    baseline = baseline or _Baseline()
    speech_s = sum(max(0.0, end - start) for start, end in segments)
    gaps = [later[0] - earlier[1] for earlier, later in zip(segments, segments[1:])]
    long_gaps = [gap for gap in gaps if gap >= 0.8]
    words = len(_WORD.findall(text))
    short = words <= 3 or speech_s < 1.2
    wpm = words / speech_s * 60 if speech_s >= 1.2 and words >= 4 else None

    samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    audio = measure_audio(samples, sample_rate) if len(samples) else {
        "loudness_db": None, "loudness_spread_db": None, "pitch_spread_st": None, "trailed_off": False,
    }

    cues = VoiceCues(
        pause_before_s=pause_before,
        interrupted=interrupted,
        speech_s=speech_s,
        words=words,
        words_per_min=wpm,
        long_pauses=len(long_gaps),
        longest_pause_s=max(long_gaps, default=0.0),
        loudness_db=audio["loudness_db"],  # type: ignore[arg-type]
        loudness_spread_db=audio["loudness_spread_db"],  # type: ignore[arg-type]
        pitch_spread_st=audio["pitch_spread_st"],  # type: ignore[arg-type]
        trailed_off=bool(audio["trailed_off"]) and not short,
        fillers=count_fillers(text),
        short_reply=short,
    )
    # Labels against their own usual (short replies are too short to judge pace or pitch).
    if not short:
        pace = baseline.label(baseline.wpm, wpm, low=0.78, high=1.25, relative=True)
        cues.pace = {-1: "slower", 1: "faster"}.get(pace)
        tone = baseline.label(baseline.pitch_spread, cues.pitch_spread_st, low=0.6, high=1.6, relative=True)
        cues.intonation = {-1: "flatter", 1: "livelier"}.get(tone)
    volume = baseline.label(baseline.loudness, cues.loudness_db, low=5.0, high=5.0, relative=False)
    cues.volume = {-1: "softer", 1: "louder"}.get(volume)
    # This turn becomes part of their usual for the next ones.
    if wpm is not None:
        baseline.wpm.append(wpm)
    if cues.loudness_db is not None:
        baseline.loudness.append(cues.loudness_db)
    if cues.pitch_spread_st is not None and not short:
        baseline.pitch_spread.append(cues.pitch_spread_st)
    return cues
