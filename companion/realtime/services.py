"""Pipecat wrappers around the companion's own STT and TTS (companion.voice).

Live calls and push-to-talk then share one Whisper and one Piper (loaded once), the same .env
choices, the same Sarvam fallback and the same voice speed.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601

from ..voice import Voice, speakable
from ..voice.audio import STT_RATE, AudioError, read_wav, resample

CHUNK_SECONDS = 0.2  # TTS audio goes out in small pieces so a barge-in can cut it quickly


class CompanionSTTService(SegmentedSTTService):
    """Pipecat collects one stretch of speech (between VAD start and stop) and hands it here as WAV."""

    def __init__(self, voice: Voice, **kwargs) -> None:
        super().__init__(settings=STTSettings(model=None, language=None), **kwargs)
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            samples, rate = read_wav(audio)
        except AudioError:
            return
        await self.start_processing_metrics()
        try:
            heard, _ = await self._voice.transcribe(resample(samples, rate, STT_RATE))
        except Exception as exc:
            yield ErrorFrame(f"Speech-to-text failed: {exc}")
            return
        finally:
            await self.stop_processing_metrics()
        if heard.text:
            yield TranscriptionFrame(heard.text, self._user_id, time_now_iso8601(), None)


class CompanionTTSService(TTSService):
    """Speaks one sentence at a time with companion.voice (Piper, then Sarvam)."""

    def __init__(self, voice: Voice, **kwargs) -> None:
        super().__init__(
            push_start_frame=True, push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None), **kwargs,
        )
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        if not speakable(text):
            return  # e.g. a lone emoji at the end of a reply: nothing to say, not an error
        try:
            await self.start_tts_usage_metrics(text)
            wav, _ = await self._voice.speak(text)
        except Exception as exc:
            yield ErrorFrame(error=f"Text-to-speech failed: {exc}")
            return
        samples, rate = read_wav(wav)
        pcm = (samples * 32767).astype(np.int16).tobytes()
        step = int(rate * CHUNK_SECONDS) * 2

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(pcm), step):
                yield pcm[start: start + step]

        async for frame in self._stream_audio_frames_from_iterator(chunks(), in_sample_rate=rate, context_id=context_id):
            await self.stop_ttfb_metrics()
            yield frame
