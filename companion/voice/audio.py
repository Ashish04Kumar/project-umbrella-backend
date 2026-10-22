"""Small audio helpers with numpy only: WAV in and out, resampling, loudness, speech length."""

from __future__ import annotations

import io
import wave

import numpy as np

STT_RATE = 16_000  # what speech-to-text models expect


class AudioError(ValueError):
    """The uploaded audio could not be read (wrong format, empty, too long)."""


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono float32 samples in -1..1, sample rate). Only 16-bit PCM WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError("Send 16-bit PCM WAV audio.") from exc
    if width != 2:
        raise AudioError("Send 16-bit PCM WAV audio.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size == 0:
        raise AudioError("The audio is empty.")
    return samples, rate


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    """float32 (-1..1) or int16 samples -> mono 16-bit WAV bytes."""
    if samples.dtype != np.int16:
        samples = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear resampling. Good enough for speech going into STT."""
    if from_rate == to_rate or samples.size == 0:
        return samples
    count = int(round(samples.size * to_rate / from_rate))
    old_positions = np.arange(samples.size) / from_rate
    new_positions = np.arange(count) / to_rate
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def _frame_db(samples: np.ndarray, rate: int, frame_ms: int = 20) -> np.ndarray:
    size = max(1, rate * frame_ms // 1000)
    usable = samples[: samples.size - samples.size % size]
    if usable.size == 0:
        return np.array([-120.0])
    frames = usable.reshape(-1, size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def speech_seconds(samples: np.ndarray, rate: int, threshold_db: float = -40.0) -> float:
    """Roughly how long someone was speaking: 20 ms frames louder than the threshold."""
    return float(np.sum(_frame_db(samples, rate) > threshold_db) * 0.02)


def loudness_db(samples: np.ndarray, rate: int) -> float:
    """Average loudness of the louder half of the frames (ignores the silence around speech)."""
    db = np.sort(_frame_db(samples, rate))
    return float(np.mean(db[db.size // 2:]))
