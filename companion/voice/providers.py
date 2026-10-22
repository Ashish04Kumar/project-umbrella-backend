"""Plug-and-play speech-to-text and text-to-speech, chosen in .env (like the LLMs):

    COMPANION_STT=groq      COMPANION_STT_FALLBACK=whisper,sarvam   (Groq Whisper, laptop faster-whisper, Sarvam Saarika)
    COMPANION_TTS=piper     COMPANION_TTS_FALLBACK=sarvam   (laptop Piper, or Sarvam Bulbul)

Laptop models load once (the first call, or warm_up()) and run in a worker thread so the async
loop never blocks. Sarvam is plain HTTPS with the standard library; its key stays on the backend.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from ..config import PROJECT_DIR, Settings
from .audio import STT_RATE, read_wav, resample, to_wav


class VoiceError(RuntimeError):
    """A speech provider could not run (missing model or key, network error)."""


@dataclass
class Heard:
    text: str
    language: str = ""


@dataclass
class Spoken:
    samples: np.ndarray  # int16 mono
    rate: int


class STTProvider(Protocol):
    name: str

    async def transcribe(self, samples: np.ndarray) -> Heard:
        """`samples`: mono float32 at 16 kHz."""
        ...


class TTSProvider(Protocol):
    name: str

    async def synthesize(self, text: str, speed: float = 1.0) -> Spoken: ...


# --- laptop: faster-whisper and Piper ----------------------------------------------------------

class WhisperSTT:
    name = "whisper"

    def __init__(self, model: str, language: str | None) -> None:
        self.model_name = model
        self.language = language or None  # None: Whisper detects Hindi or English per clip
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise VoiceError("faster-whisper is not installed (pip install -e '.[voice]')") from exc
                self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def warm_up(self) -> None:
        self._load()

    def _run(self, samples: np.ndarray) -> Heard:
        model = self._load()
        # vad_filter: faster-whisper's built-in Silero VAD skips the silence before and after speech.
        segments, info = model.transcribe(samples, language=self.language, vad_filter=True, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments if segment.no_speech_prob < 0.6)
        return Heard(text=text.strip(), language=info.language or "")

    async def transcribe(self, samples: np.ndarray) -> Heard:
        return await asyncio.to_thread(self._run, samples)


class PiperTTS:
    name = "piper"

    def __init__(self, voice_path: str) -> None:
        path = Path(voice_path)
        self.path = path if path.is_absolute() else PROJECT_DIR / path
        self._voice = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._voice is None:
                if not self.path.exists():
                    raise VoiceError(f"Piper voice not found: {self.path}")
                try:
                    from piper import PiperVoice
                except ImportError as exc:
                    raise VoiceError("piper-tts is not installed (pip install -e '.[voice]')") from exc
                self._voice = PiperVoice.load(str(self.path))
        return self._voice

    def warm_up(self) -> None:
        self._load()

    def _run(self, text: str, speed: float) -> Spoken:
        from piper import SynthesisConfig

        voice = self._load()
        # length_scale > 1 is slower speech; speed 1.2 means 20% faster.
        chunks = list(voice.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0 / max(0.5, min(speed, 2.0)))))
        if not chunks:
            raise VoiceError("Piper produced no audio")
        samples = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        return Spoken(samples=samples, rate=chunks[0].sample_rate)

    async def synthesize(self, text: str, speed: float = 1.0) -> Spoken:
        return await asyncio.to_thread(self._run, text, speed)


# --- cloud: shared multipart upload ----------------------------------------------------------------

def _multipart(fields: dict[str, str], wav: bytes) -> tuple[bytes, str]:
    """Build a multipart/form-data body with text fields and one WAV file (standard library only)."""
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        for name, value in fields.items()
    ]
    parts += [
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(),
        wav, f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --- cloud: Groq Whisper (fast, large-v3-turbo; the audio is sent to Groq) ----------------------------

# Whisper follows the style of its prompt, so a Roman-script Hinglish prompt keeps Hindi words in
# English letters (what the LLM and the Piper voice read best) instead of Devanagari or Urdu script.
HINGLISH_PROMPT = "Hinglish baat-cheet, Roman script mein: haan yaar, kal subah gym jaana hai, theek hai, kya plan hai?"


class GroqSTT:
    name = "groq"

    def __init__(self, api_key: str, model: str, language: str | None) -> None:
        if not api_key:
            raise VoiceError("GROQ_API_KEY is not set")
        self.api_key, self.model, self.language = api_key, model, language or None

    def _run(self, samples: np.ndarray) -> Heard:
        fields = {"model": self.model, "response_format": "verbose_json", "temperature": "0", "prompt": HINGLISH_PROMPT}
        if self.language:
            fields["language"] = self.language
        body, content_type = _multipart(fields, to_wav(samples, STT_RATE))
        request = Request(
            "https://api.groq.com/openai/v1/audio/transcriptions", data=body, method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": content_type, "User-Agent": "companion-umbrella"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise VoiceError(f"Groq STT returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise VoiceError(f"Groq STT request failed: {type(exc).__name__}") from exc
        # Drop segments Whisper itself thinks are not speech (same rule as the laptop Whisper).
        segments = payload.get("segments") or []
        if segments:
            text = " ".join(s.get("text", "").strip() for s in segments if s.get("no_speech_prob", 0) < 0.6)
        else:
            text = payload.get("text", "")
        return Heard(text=text.strip(), language=str(payload.get("language") or ""))

    async def transcribe(self, samples: np.ndarray) -> Heard:
        return await asyncio.to_thread(self._run, samples)


# --- cloud: Sarvam (Indian languages, good for Hinglish). Not tested yet: no key. ---------------

def _sarvam_post(url: str, api_key: str, body: bytes, content_type: str) -> dict:
    request = Request(url, data=body, method="POST", headers={"api-subscription-key": api_key, "Content-Type": content_type})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise VoiceError(f"Sarvam returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise VoiceError(f"Sarvam request failed: {type(exc).__name__}") from exc


class SarvamSTT:
    name = "sarvam"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise VoiceError("SARVAM_API_KEY is not set")
        self.api_key, self.model = api_key, model

    def _run(self, samples: np.ndarray) -> Heard:
        body, content_type = _multipart({"model": self.model, "language_code": "unknown"}, to_wav(samples, STT_RATE))
        payload = _sarvam_post("https://api.sarvam.ai/speech-to-text", self.api_key, body, content_type)
        return Heard(text=str(payload.get("transcript", "")).strip(), language=str(payload.get("language_code") or ""))

    async def transcribe(self, samples: np.ndarray) -> Heard:
        return await asyncio.to_thread(self._run, samples)


class SarvamTTS:
    name = "sarvam"

    def __init__(self, api_key: str, model: str, speaker: str, language: str) -> None:
        if not api_key:
            raise VoiceError("SARVAM_API_KEY is not set")
        self.api_key, self.model, self.speaker, self.language = api_key, model, speaker, language

    def _run(self, text: str, speed: float) -> Spoken:
        body = {
            "text": text[:1500], "target_language_code": self.language, "speaker": self.speaker,
            "model": self.model, "pace": max(0.5, min(speed, 2.0)), "speech_sample_rate": 22050,
        }
        payload = _sarvam_post("https://api.sarvam.ai/text-to-speech", self.api_key, json.dumps(body).encode(), "application/json")
        audios = payload.get("audios") or []
        if not audios:
            raise VoiceError("Sarvam returned no audio")
        samples, rate = read_wav(base64.b64decode(audios[0]))
        return Spoken(samples=(samples * 32767).astype(np.int16), rate=rate)

    async def synthesize(self, text: str, speed: float = 1.0) -> Spoken:
        return await asyncio.to_thread(self._run, text, speed)


def make_stt(name: str, settings: Settings) -> STTProvider:
    if name == "whisper":
        return WhisperSTT(settings.whisper_model, settings.whisper_language)
    if name == "groq":
        return GroqSTT(settings.groq_api_key, settings.groq_stt_model, settings.whisper_language)
    if name == "sarvam":
        return SarvamSTT(settings.sarvam_api_key, settings.sarvam_stt_model)
    raise VoiceError(f"Unknown STT provider '{name}' (use groq, whisper or sarvam)")


def make_tts(name: str, settings: Settings) -> TTSProvider:
    if name == "piper":
        return PiperTTS(settings.piper_voice)
    if name == "sarvam":
        return SarvamTTS(settings.sarvam_api_key, settings.sarvam_tts_model, settings.sarvam_speaker, settings.sarvam_language)
    raise VoiceError(f"Unknown TTS provider '{name}' (use piper or sarvam)")


def resample_for_stt(samples: np.ndarray, rate: int) -> np.ndarray:
    return resample(samples, rate, STT_RATE)
