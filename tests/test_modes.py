"""Input modes switch the voice cues: on for live and push-to-talk-with-cues, off for text and push-to-talk."""

from __future__ import annotations

import asyncio
import time
import unittest

import numpy as np

from companion.brain import Companion
from companion.llm.base import LLMResult
from companion.memory.postgres import PostgresMemory
from companion.modes import LIVE, MODES, PUSH_TO_TALK, PUSH_TO_TALK_CUES, TEXT, InputMode
from companion.store.postgres import PostgresStore
from companion.voice import Voice
from companion.voice.audio import read_wav, resample, to_wav
from companion.voice.cues import _Baseline
from companion.voice.providers import Heard
from tests.test_postgres import SETTINGS, ScriptedLLM, _drop, _fresh_pool
from tests.test_voice import MODELS_READY, FakeTTS


class ModeConstantTests(unittest.TestCase):
    def test_which_modes_send_voice_cues(self):
        self.assertEqual({name: mode.voice_cues for name, mode in MODES.items()}, {
            "text": False, "push_to_talk": False, "push_to_talk_cues": True, "live": True,
        })
        self.assertIs(MODES["live"], LIVE)
        self.assertFalse(TEXT.voice_cues)


async def piper_clip(text: str, silence_before: float = 0.0, silence_mid: float = 0.0) -> np.ndarray:
    """Real speech from Piper at 16 kHz, with optional silence before and in the middle."""
    voice = Voice(SETTINGS.__class__(stt_order=[], tts_order=["piper"]))
    halves = text.split("|")
    parts = [np.zeros(int(silence_before * 16000), np.float32)]
    for index, half in enumerate(halves):
        wav, _ = await voice.speak(half)
        samples, rate = read_wav(wav)
        parts.append(resample(samples, rate, 16000))
        if index < len(halves) - 1:
            parts.append(np.zeros(int(silence_mid * 16000), np.float32))
    return np.concatenate(parts)


class FixedSTT:
    name = "fixed"

    def __init__(self, text: str) -> None:
        self.text = text

    async def transcribe(self, samples):
        return Heard(text=self.text, language="en")


@unittest.skipUnless(MODELS_READY, "Piper voice or faster-whisper missing")
class ClipCueTests(unittest.TestCase):
    def test_vad_finds_speech_pause_and_gap(self):
        from companion.voice.clip_cues import measure_clip

        clip = asyncio.run(piper_clip("I have an interview tomorrow morning.|I am a little nervous about it.", silence_before=1.0, silence_mid=1.2))
        cues, segments = measure_clip(clip, "I have an interview tomorrow morning. I am a little nervous about it.",
                                      gap_before_recording_s=3.0, interrupted=False, baseline=_Baseline())
        self.assertGreaterEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0][0], 1.0, delta=0.25)        # speech starts after the 1 s of silence
        self.assertAlmostEqual(cues.pause_before_s, 4.0, delta=0.25)   # 3 s before recording + 1 s in the clip
        self.assertEqual(cues.long_pauses, 1)                          # the 1.2 s gap in the middle
        self.assertIn("took 4s before answering", cues.describe())
        self.assertIn("1 long pause mid-answer", cues.describe())

    def test_interrupted_means_no_pause(self):
        from companion.voice.clip_cues import measure_clip

        clip = asyncio.run(piper_clip("Wait, wait, tell me the time instead."))
        cues, _ = measure_clip(clip, "Wait, wait, tell me the time instead.", gap_before_recording_s=None, interrupted=True, baseline=_Baseline())
        self.assertIsNone(cues.pause_before_s)
        self.assertIn("started talking while you were still speaking", cues.describe())


@unittest.skipUnless(SETTINGS.database_url and MODELS_READY, "needs Postgres and the Piper voice")
class VoiceTurnModeTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    def test_push_to_talk_sends_no_cues_and_cue_mode_does(self):
        async def run():
            text = "I have an interview tomorrow morning. I am a little nervous about it."
            clip = to_wav(await piper_clip("I have an interview tomorrow morning.|I am a little nervous about it.", 1.0, 1.2), 16000)
            pool = await _fresh_pool()
            llm = ScriptedLLM([LLMResult(text="[concerned] Koi baat nahi, aaram se batao."), LLMResult(text="[concerned] Achha, kis cheez ka interview hai?")])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool),
                            voice=Voice(SETTINGS, stt=[FixedSTT(text)], tts=[FakeTTS()]))

            plain = await bot.voice_turn("u", clip, mode=PUSH_TO_TALK, gap_before_recording_s=3.0)
            await bot.wait_background()
            self.assertIsNone(plain.voice_cues)
            self.assertEqual(plain.voice_note, "")
            prompt = [c for c in llm.calls if c.get("tools") is not None][-1]["messages"][0].content
            self.assertNotIn("How they sounded", prompt)

            with_cues = await bot.voice_turn("u", clip, mode=PUSH_TO_TALK_CUES, gap_before_recording_s=3.0)
            await bot.wait_background()
            self.assertIsNotNone(with_cues.voice_cues)
            self.assertIn("took 4s before answering", with_cues.voice_note)
            prompt = [c for c in llm.calls if c.get("tools") is not None][-1]["messages"][0].content
            self.assertIn("How they sounded just now", prompt)
            self.assertIn("took 4s before answering", prompt)
            self.assertIn("voice_cues", [e["kind"] for e in bot.trace.recent("u")])
            await pool.close()
        asyncio.run(run())


class LiveModeSwitchTests(unittest.TestCase):
    def test_live_with_cues_off_sends_no_note(self):
        from tests.test_live import FakeBrain, make_turns

        async def run():
            notes = []

            class Recorder(FakeBrain):
                async def chat_stream(self, user_id, text="", *, cue="", voice_note="", spoken=True, voice_cues=None, turn=""):
                    notes.append((voice_note, voice_cues))
                    async for event in super().chat_stream(user_id, text, cue=cue):
                        yield event

            for mode, expect_cues in ((LIVE, True), (InputMode("live_no_cues", voice_cues=False), False)):
                notes.clear()
                turns, _ = make_turns(Recorder())
                turns._mode = mode
                turns._bot_speaking = True                      # the user cuts in: a cue live mode would report
                turns._cues.bot_started(time.monotonic())
                await turns.on_user_started(time.monotonic())
                turns._bot_speaking = False
                pcm = (np.sin(np.arange(320) / 3) * 8000).astype(np.int16).tobytes()
                for _ in range(75):
                    turns.on_mic_audio(pcm, 16000, time.monotonic())
                turns.on_user_stopped(time.monotonic())
                turns._on_transcript("wait wait tell me the time instead please", time.monotonic())
                await asyncio.sleep(1.2)
                self.assertEqual(len(notes), 1, mode)
                note, fields = notes[0]
                if expect_cues:
                    self.assertIn("started talking while you were still speaking", note)
                    self.assertTrue(fields["interrupted"])
                else:
                    self.assertEqual((note, fields), ("", None))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()


class ConfirmedTextTests(unittest.TestCase):
    def test_confirmed_text_skips_stt_and_transcribe_only_returns_words(self):
        from companion.voice import VoiceError

        class NoSTT:
            name = "should-not-run"

            async def transcribe(self, samples):
                raise VoiceError("STT must not run when the text was confirmed")

        async def run():
            clip = to_wav(np.sin(np.arange(16000) / 5).astype(np.float32) * 0.3, 16000)
            bot = Companion(settings=SETTINGS, llm=ScriptedLLM([]), store=None, memory=None, voice=Voice(SETTINGS, stt=[NoSTT()], tts=[]))
            heard = await bot.hear("u", clip, mode=PUSH_TO_TALK, text="  kal 9 baje dawai yaad dilana ")
            self.assertEqual((heard.heard, heard.kept, heard.stt_provider), ("kal 9 baje dawai yaad dilana", True, "confirmed by the user"))
            bot.voice = Voice(SETTINGS, stt=[FixedSTT("haan bolo")], tts=[])
            words = await bot.transcribe(clip)
            self.assertEqual((words["text"], words["provider"]), ("haan bolo", "fixed"))
        asyncio.run(run())
