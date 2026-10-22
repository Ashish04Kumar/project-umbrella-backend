"""Voice cues for a recorded clip (push-to-talk with cues).

Live mode gets speech start/stop times from real-time VAD. A clip has none, so Silero VAD (the
copy inside faster-whisper) finds the stretches of speech in it afterwards. The rest is the same
`compute_cues` as live mode, with the same per-user "usual" (baseline).
"""

from __future__ import annotations

import numpy as np

from .audio import STT_RATE
from .cues import VoiceCues, _Baseline, compute_cues

# Pauses of 0.3 s or more split the speech, so "long pauses" (0.8 s+) inside an answer are seen.
MIN_SILENCE_MS = 300
SPEECH_PAD_MS = 60


def speech_segments(samples_16k: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) in seconds of each stretch of speech in the clip, from Silero VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=MIN_SILENCE_MS, speech_pad_ms=SPEECH_PAD_MS)
    stamps = get_speech_timestamps(samples_16k.astype(np.float32), options, sampling_rate=STT_RATE)
    return [(stamp["start"] / STT_RATE, stamp["end"] / STT_RATE) for stamp in stamps]


def measure_clip(
    samples_16k: np.ndarray,
    text: str,
    *,
    gap_before_recording_s: float | None,
    interrupted: bool,
    baseline: _Baseline,
) -> tuple[VoiceCues, list[tuple[float, float]]]:
    """Cues for one clip. `gap_before_recording_s`: from the end of the companion's voice to the
    moment recording started (the body measures it); the silence at the start of the clip is added."""
    segments = speech_segments(samples_16k)
    pause_before = None
    if gap_before_recording_s is not None and segments and not interrupted:
        pause_before = max(0.0, gap_before_recording_s) + segments[0][0]
    # Only the speech is used for loudness and pitch, like live mode (which keeps audio only while speaking).
    pcm = b"".join(
        (np.clip(samples_16k[int(start * STT_RATE): int(end * STT_RATE)], -1, 1) * 32767).astype(np.int16).tobytes()
        for start, end in segments
    )
    cues = compute_cues(
        text=text, segments=segments, pcm=pcm, sample_rate=STT_RATE,
        pause_before=pause_before, interrupted=interrupted, baseline=baseline,
    )
    return cues, segments
