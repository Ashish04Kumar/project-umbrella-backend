"""Voice for the companion: speech in (STT) and speech out (TTS), with fallbacks.

    voice = Voice(settings)
    heard, provider = await voice.transcribe(samples_16k)
    wav, provider = await voice.speak("Kal subah 9 baje yaad dila dunga!")
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np

from ..config import Settings
from .audio import to_wav
from .providers import Heard, STTProvider, TTSProvider, VoiceError, make_stt, make_tts

log = logging.getLogger("companion.voice")

_URL = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN = re.compile(r"[*_#`>|~]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)


def speakable(text: str) -> str:
    """Text a voice can read: no links, markdown, bullets or emojis."""
    text = _URL.sub("", text)
    text = _BULLET.sub("", text)
    text = _MARKDOWN.sub("", text)
    # Emojis and other symbols (Unicode category "So") and joiners are silent.
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Cf"))
    # Each line becomes its own sentence, unless it already ends with punctuation.
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    sentences = [line if line[-1] in ".!?:,;।" else line + "." for line in lines[:-1]] + lines[-1:]
    return " ".join(sentences)


def _build(names: list[str], factory, settings: Settings, kind: str) -> list:
    providers = []
    for name in dict.fromkeys(names):
        try:
            providers.append(factory(name, settings))
        except VoiceError as exc:  # e.g. Sarvam without a key: skip it, keep the others
            log.info("%s %s not available: %s", kind, name, exc)
    return providers


class Voice:
    def __init__(self, settings: Settings, stt: list[STTProvider] | None = None, tts: list[TTSProvider] | None = None) -> None:
        self.settings = settings
        self.stt = stt if stt is not None else _build(settings.stt_order, make_stt, settings, "STT")
        self.tts = tts if tts is not None else _build(settings.tts_order, make_tts, settings, "TTS")

    def warm_up(self) -> None:
        """Load the laptop models now (blocking), so the first voice turn, or the first fallback, is not slow."""
        for provider in [*self.stt, *self.tts]:
            warm = getattr(provider, "warm_up", None)
            if warm:
                try:
                    warm()
                    log.info("voice %s ready", provider.name)
                except Exception as exc:
                    log.warning("voice %s could not load: %s", provider.name, exc)

    async def transcribe(self, samples_16k: np.ndarray) -> tuple[Heard, str]:
        errors = []
        for provider in self.stt:
            try:
                return await provider.transcribe(samples_16k), provider.name
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                log.warning("STT %s failed (%s); trying the next one", provider.name, exc)
        raise VoiceError("No speech-to-text worked: " + "; ".join(errors or ["none configured"]))

    async def speak(self, text: str, speed: float | None = None) -> tuple[bytes, str]:
        """Text -> WAV bytes."""
        words = speakable(text)
        if not words:
            raise VoiceError("Nothing to say")
        errors = []
        for provider in self.tts:
            try:
                spoken = await provider.synthesize(words, speed or self.settings.tts_speed)
                return to_wav(spoken.samples, spoken.rate), provider.name
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                log.warning("TTS %s failed (%s); trying the next one", provider.name, exc)
        raise VoiceError("No text-to-speech worked: " + "; ".join(errors or ["none configured"]))


__all__ = ["Heard", "Voice", "VoiceError", "speakable"]
