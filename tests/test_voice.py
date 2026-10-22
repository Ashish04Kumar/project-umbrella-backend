"""Voice: audio helpers, the STT guard, provider fallback, a full voice turn, and a real
Piper -> Whisper round trip (skipped when the models or libraries are missing)."""

from __future__ import annotations

import asyncio
import importlib.util
import unittest

import numpy as np

from companion.brain import Companion
from companion.config import PROJECT_DIR, Settings
from companion.llm.base import LLMResult
from companion.memory.postgres import PostgresMemory
from companion.persona import SPOKEN_RULES
from companion.store.postgres import PostgresStore
from companion.voice import Voice, VoiceError, speakable
from companion.voice.audio import AudioError, read_wav, resample, speech_seconds, to_wav
from companion.voice.guard import check_transcript
from companion.voice.providers import Heard, Spoken
from tests.test_postgres import SETTINGS, ScriptedLLM, _drop, _fresh_pool


def tone(seconds: float, rate: int = 16_000, loud: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (loud * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeSTT:
    def __init__(self, name: str, text: str | None) -> None:
        self.name, self.text = name, text

    async def transcribe(self, samples):
        if self.text is None:
            raise VoiceError("down")
        return Heard(text=self.text, language="en")


class FakeTTS:
    name = "fake-tts"

    async def synthesize(self, text, speed=1.0):
        self.last = (text, speed)
        return Spoken(samples=(tone(0.5, 22050) * 32767).astype(np.int16), rate=22050)


class AudioTests(unittest.TestCase):
    def test_wav_round_trip_and_resample(self):
        samples, rate = read_wav(to_wav(tone(1.0, 44100), 44100))
        self.assertEqual(rate, 44100)
        self.assertEqual(resample(samples, 44100, 16000).size, 16000)

    def test_speech_seconds(self):
        clip = np.concatenate([np.zeros(16000, np.float32), tone(1.5), np.zeros(8000, np.float32)])
        self.assertAlmostEqual(speech_seconds(clip, 16000), 1.5, delta=0.05)
        self.assertEqual(speech_seconds(np.zeros(16000, np.float32), 16000), 0.0)

    def test_bad_audio(self):
        with self.assertRaises(AudioError):
            read_wav(b"not a wav")

    def test_guard(self):
        self.assertEqual(check_transcript("Thanks for watching!", 3.0, -20)[0], False)
        self.assertEqual(check_transcript("Thank you.", 0.4, -20)[0], False)
        self.assertEqual(check_transcript("Thank you so much Umbrella, that helped", 2.0, -20)[0], True)
        self.assertEqual(check_transcript("haan", 0.1, -20)[0], False)
        self.assertEqual(check_transcript("", 2.0, -20)[0], False)

    def test_speakable(self):
        self.assertEqual(speakable("**Done!** 😊 Reminder set:\n- dawai 9 AM\n- gym https://x.com"), "Done! Reminder set: dawai 9 AM. gym")
        self.assertEqual(speakable("😊"), "")


class FallbackTests(unittest.TestCase):
    def test_stt_and_tts_fall_back(self):
        voice = Voice(Settings(), stt=[FakeSTT("broken", None), FakeSTT("works", "hello")], tts=[FakeTTS()])
        heard, provider = asyncio.run(voice.transcribe(tone(1)))
        self.assertEqual((heard.text, provider), ("hello", "works"))
        wav, provider = asyncio.run(voice.speak("Hi! 😊", speed=1.2))
        self.assertEqual(provider, "fake-tts")
        self.assertEqual(read_wav(wav)[1], 22050)
        with self.assertRaises(VoiceError):
            asyncio.run(Voice(Settings(), stt=[FakeSTT("broken", None)], tts=[]).transcribe(tone(1)))
        with self.assertRaises(VoiceError):
            asyncio.run(voice.speak("🙂"))



class GroqSTTTests(unittest.TestCase):
    def test_request_and_no_speech_filter(self):
        import io
        import json as jsonlib
        from unittest import mock

        from companion.voice import providers

        sent = {}

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout):
            sent["url"], sent["body"], sent["headers"] = request.full_url, request.data, dict(request.headers)
            return Response(jsonlib.dumps({"language": "english", "text": "ignored", "segments": [
                {"text": " Kal subah gym jaana hai.", "no_speech_prob": 0.05},
                {"text": " Thank you.", "no_speech_prob": 0.9},
            ]}).encode())

        stt = providers.GroqSTT("key", "whisper-large-v3-turbo", None)
        with mock.patch.object(providers, "urlopen", fake_urlopen):
            heard = asyncio.run(stt.transcribe(tone(1.0)))
        self.assertEqual((heard.text, heard.language), ("Kal subah gym jaana hai.", "english"))
        self.assertTrue(sent["url"].endswith("/audio/transcriptions"))
        self.assertEqual(sent["headers"]["Authorization"], "Bearer key")
        self.assertIn(b'name="model"\r\n\r\nwhisper-large-v3-turbo', sent["body"])
        self.assertIn(b"Roman script", sent["body"])
        self.assertIn(b"RIFF", sent["body"])

    def test_without_key_falls_back_to_laptop_whisper(self):
        voice = Voice(Settings(groq_api_key="", stt_order=["groq", "whisper"], sarvam_api_key=""))
        self.assertEqual([p.name for p in voice.stt], ["whisper"])

@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class VoiceTurnTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    def test_voice_turn_kept_and_dropped(self):
        async def run():
            pool = await _fresh_pool()
            llm = ScriptedLLM([LLMResult(text="[happy] Main badhiya hoon! 😊 Tum batao?")])
            tts = FakeTTS()
            voice = Voice(SETTINGS, stt=[FakeSTT("fake-stt", "Hi Umbrella, kaise ho aaj?")], tts=[tts])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool), voice=voice)
            clip = to_wav(np.concatenate([np.zeros(8000, np.float32), tone(1.5)]), 16000)
            turn = await bot.voice_turn("u", clip, speed=1.1)
            await bot.wait_background()
            self.assertTrue(turn.kept)
            self.assertEqual(turn.heard, "Hi Umbrella, kaise ho aaj?")
            self.assertEqual(turn.reply.text, "Main badhiya hoon! 😊 Tum batao?")
            self.assertEqual(tts.last, ("Main badhiya hoon! Tum batao?", 1.1))   # emoji not spoken
            self.assertTrue(turn.audio_wav.startswith(b"RIFF"))
            prompt = [c for c in llm.calls if c.get("tools") is not None][0]["messages"][0].content
            self.assertIn(SPOKEN_RULES, prompt)

            # A made-up "Thank you." from a short blip is ignored: no LLM call, no reply.
            bot.voice = Voice(SETTINGS, stt=[FakeSTT("fake-stt", "Thank you.")], tts=[tts])
            calls = len(llm.calls)
            dropped = await bot.voice_turn("u", to_wav(tone(0.3), 16000))
            self.assertFalse(dropped.kept)
            self.assertIsNone(dropped.reply)
            self.assertEqual(len(llm.calls), calls)
            await pool.close()
        asyncio.run(run())


MODELS_READY = (
    importlib.util.find_spec("faster_whisper") is not None
    and importlib.util.find_spec("piper") is not None
    and (PROJECT_DIR / Settings().piper_voice).exists()
)


@unittest.skipUnless(MODELS_READY, "faster-whisper, piper-tts or the Piper voice is missing")
class RealModelsTests(unittest.TestCase):
    def test_piper_to_whisper_round_trip(self):
        voice = Voice(Settings(stt_order=["whisper"], tts_order=["piper"]))
        wav, _ = asyncio.run(voice.speak("Please remind me to go to the gym tomorrow evening."))
        samples, rate = read_wav(wav)
        heard, provider = asyncio.run(voice.transcribe(resample(samples, rate, 16000)))
        self.assertEqual(provider, "whisper")
        self.assertIn("remind me to go to the gym tomorrow", heard.text.lower())


if __name__ == "__main__":
    unittest.main()
