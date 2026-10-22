"""Live voice: sentence shaping, streaming replies, barge-in saving, and the live turn taking
(ported from Synq Pod 4) with a fake transport and a fake brain."""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from contextlib import aclosing
from datetime import datetime, timedelta

from companion.brain import Companion
from companion.config import Settings
from companion.guard import mentions_silence, sounds_like_crisis
from companion.llm.base import LLMError, LLMResult, ToolCall
from companion.memory.postgres import PostgresMemory
from companion.store.postgres import PostgresStore
from companion.voice.speech import SentenceShaper, split_sentences
from tests.test_postgres import SETTINGS, TZ, ScriptedLLM, _drop, _fresh_pool


class StreamingLLM(ScriptedLLM):
    """Streams each scripted reply in small pieces; `fail_streams` makes the first N streams fail."""

    def __init__(self, replies, memories=None, fail_streams: int = 0):
        super().__init__(replies, memories)
        self.fail_streams = fail_streams

    async def stream(self, messages, *, tools=None, **options):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": False, "stream": True})
        if self.fail_streams:
            self.fail_streams -= 1
            raise LLMError("rate limited")
        result = self.replies.pop(0)
        for start in range(0, len(result.text), 7):
            await asyncio.sleep(0)
            yield "text", result.text[start: start + 7], "fake"
        if result.tool_calls:
            yield "tool_calls", result.tool_calls, "fake"


class ShaperTests(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(split_sentences("Haan. Main theek hoon yaar! Aur tum"), (["Haan. Main theek hoon yaar!"], "Aur tum"))
        self.assertEqual(split_sentences("Aur tum kaise ho", final=True), (["Aur tum kaise ho"], ""))

    def test_tag_split_across_pieces_and_filters(self):
        shaper = SentenceShaper(nudge=True)
        out = []
        for piece in ["[cur", "ious] Haan", " yaar! Kal tumne gym ka bola tha. ", "Yesterday I went there too. ",
                      "No rush, take your time. Kaisa raha aaj?"]:
            out += shaper.feed(piece)
        out += shaper.end_round()
        self.assertEqual(shaper.expression, "curious")
        self.assertEqual(out, ["Haan yaar! Kal tumne gym ka bola tha.", "Kaisa raha aaj?"])
        self.assertEqual(len(shaper.dropped), 2)

    def test_word_budget(self):
        shaper = SentenceShaper(max_words=8)
        out = shaper.feed("[idle] One two three four five. Six seven eight nine ten. ") + shaper.end_round()
        self.assertEqual(out, ["One two three four five."])
        self.assertTrue(shaper.full)

    def test_safety_words(self):
        self.assertTrue(sounds_like_crisis("sach mein ab jeena nahi chahta"))
        self.assertTrue(sounds_like_crisis("I want to die"))
        self.assertFalse(sounds_like_crisis("main gym jaana chahta hoon"))
        self.assertTrue(mentions_silence("Koi baat nahi, take your time."))


@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class ChatStreamTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    async def _bot(self, llm):
        pool = await _fresh_pool()
        return Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool), pool=pool)

    async def _collect(self, stream, stop_after: int | None = None):
        events = []
        async with aclosing(stream) as events_in:  # like the live agent: closing = "the rest was not heard"
            async for event in events_in:
                events.append(event)
                if stop_after is not None and sum(e["type"] == "sentence" for e in events) >= stop_after:
                    break
        return events

    def test_stream_with_tool_round_then_history(self):
        async def run():
            llm = StreamingLLM([
                # A read tool: its result goes back to the LLM, which then streams the reply.
                LLMResult(tool_calls=[ToolCall(id="c1", name="reminder_list", arguments=json.dumps({"query": "walk"}))]),
                LLMResult(text="[happy] Done yaar, kal subah saat baje yaad dila dunga. Aur kuch?"),
            ])
            bot = await self._bot(llm)
            events = await self._collect(bot.chat_stream("u", "kal 7 baje walk yaad dilana", voice_note="Long pause before answering."))
            await bot.wait_background()
            self.assertEqual(events[0], {"type": "expression", "expression": "happy"})
            self.assertEqual([e["text"] for e in events if e["type"] == "sentence"], ["Done yaar, kal subah saat baje yaad dila dunga.", "Aur kuch?"])
            self.assertEqual(events[-1]["reply"].tools_used, ["reminder_list"])
            self.assertIn("Long pause before answering.", llm.calls[0]["messages"][0].content)
            conversation = await bot.store.current_conversation("u")
            history, _, _ = await bot.store.recent_messages(conversation, 10)
            self.assertEqual([(m.role, m.content) for m in history], [
                ("user", "kal 7 baje walk yaad dilana"), ("assistant", "Done yaar, kal subah saat baje yaad dila dunga. Aur kuch?"),
            ])
            await bot.close()
        asyncio.run(run())

    def test_barge_in_saves_only_what_was_said(self):
        async def run():
            llm = StreamingLLM([LLMResult(text="[idle] Pehli baat ye hai ki aaram karo. Doosri baat ye hai ki paani piyo. Teesri baat neend lo.")])
            bot = await self._bot(llm)
            events = await self._collect(bot.chat_stream("u", "thak gaya hoon yaar aaj"), stop_after=1)
            await bot.wait_background()
            self.assertEqual(events[-1], {"type": "sentence", "text": "Pehli baat ye hai ki aaram karo."})
            conversation = await bot.store.current_conversation("u")
            history, _, _ = await bot.store.recent_messages(conversation, 10)
            self.assertEqual(history[-1].content, "Pehli baat ye hai ki aaram karo.")
            await bot.close()
        asyncio.run(run())

    def test_cue_turn_is_not_saved_as_user_words(self):
        async def run():
            llm = StreamingLLM([LLMResult(text="[curious] Aur batao, gym ke baad kya plan hai?")])
            bot = await self._bot(llm)
            events = await self._collect(bot.chat_stream("u", cue="They went quiet. Invite them to share more."))
            await bot.wait_background()
            self.assertEqual([e["text"] for e in events if e["type"] == "sentence"], ["Aur batao, gym ke baad kya plan hai?"])
            self.assertIsNone(llm.calls[0]["tools"])  # no tools on a nudge
            self.assertFalse(any(c.get("json_mode") for c in llm.calls))  # no memory extraction
            conversation = await bot.store.current_conversation("u")
            history, _, _ = await bot.store.recent_messages(conversation, 10)
            self.assertEqual([m.role for m in history], ["assistant"])
            await bot.close()
        asyncio.run(run())

    def test_tool_done_then_llm_fails_still_speaks_confirmation(self):
        async def run():
            tomorrow = (datetime.now(TZ) + timedelta(days=(4 - datetime.now(TZ).weekday()) % 7 or 7)).strftime("%Y-%m-%dT09:00")  # next Friday (the messages say "Friday")
            llm = StreamingLLM([LLMResult(tool_calls=[ToolCall(id="c1", name="reminder_add", arguments=json.dumps({"text": "dawai", "when": tomorrow}))])])

            original = llm.stream

            async def stream(messages, **options):
                if len([c for c in llm.calls if c.get("stream")]) >= 1:
                    llm.calls.append({"stream": True})
                    raise LLMError("rate limited")
                async for event in original(messages, **options):
                    yield event

            llm.stream = stream
            bot = await self._bot(llm)
            events = await self._collect(bot.chat_stream("u", "Friday 9 baje dawai yaad dilana"))
            await bot.wait_background()
            spoken = [e["text"] for e in events if e["type"] == "sentence"]
            self.assertEqual(len(spoken), 1)
            self.assertTrue(spoken[0].startswith("Reminder set: dawai,"), spoken)
            await bot.close()
        asyncio.run(run())


# --- live turn taking with a fake transport and a fake brain ---------------------------------------

class FakeTransport:
    def __init__(self):
        self.messages = []

    async def send_message(self, text):
        self.messages.append(json.loads(text))


class FakeBrain:
    def __init__(self, nudge_seconds=0.3, think_seconds=0.0):
        self.settings = Settings(silence_nudge_seconds=nudge_seconds)
        self.turns: list[tuple[str, str]] = []
        self.discarded: list[str] = []
        self.think_seconds = think_seconds

    async def chat_stream(self, user_id, text="", *, cue="", voice_note="", spoken=True, voice_cues=None, turn=""):
        self.turns.append((text, cue))
        await asyncio.sleep(self.think_seconds)   # the LLM thinking
        yield {"type": "expression", "expression": "curious"}
        yield {"type": "sentence", "text": "Achha, aur batao?" if text else "Kya chal raha hai?"}

    def discard_turn(self, turn):
        self.discarded.append(turn)


class StubSmartTurn:
    """Smart Turn with a fixed verdict, so the rules can be tested without audio."""

    def __init__(self, probability):
        self.value = probability

    def probability(self, audio):
        return self.value


def make_turns(brain, probability=0.9):
    from companion.realtime.agent import LiveTurns

    transport = FakeTransport()
    turns = LiveTurns(brain, "u", transport)
    turns._smart = StubSmartTurn(probability)
    turns.pushed = []

    async def push_frame(frame, direction=None):
        turns.pushed.append(frame)

    async def interrupt():
        turns.interrupted = True

    turns.push_frame = push_frame
    turns.broadcast_interruption = interrupt
    turns.create_task = lambda coro, name=None: asyncio.get_running_loop().create_task(coro)
    turns.interrupted = False
    return turns, transport


async def speak(turns, seconds):
    """The user talks for `seconds` (VAD start, mic audio, VAD stop)."""
    import numpy as np

    pcm = (np.sin(np.arange(320) / 3) * 8000).astype(np.int16).tobytes()
    await turns.on_user_started(time.monotonic())
    for _ in range(int(seconds / 0.02)):
        turns.on_mic_audio(pcm, 16000, time.monotonic())
    await asyncio.sleep(seconds)
    turns.on_user_stopped(time.monotonic())


class LiveTurnTests(unittest.TestCase):
    def test_pieces_join_into_one_turn_and_made_up_text_is_dropped(self):
        async def run():
            import numpy as np

            brain = FakeBrain()
            turns, transport = make_turns(brain)
            pcm = (np.sin(np.arange(320) / 3) * 8000).astype(np.int16).tobytes()
            await turns.on_user_started(time.monotonic())
            for _ in range(75):  # 1.5 s of speech
                turns.on_mic_audio(pcm, 16000, time.monotonic())
            turns.on_user_stopped(time.monotonic())
            turns._on_transcript("kal mera interview hai", time.monotonic())
            turns._on_transcript("aur thoda darr lag raha hai", time.monotonic())
            turns._on_transcript("Thank you.", time.monotonic())   # no speech left to hold it: made up
            await asyncio.sleep(1.2)
            self.assertEqual(brain.turns, [("kal mera interview hai aur thoda darr lag raha hai", "")])
            types = [m["type"] for m in transport.messages]
            self.assertIn("caption", types)
            self.assertIn({"type": "expression", "expression": "curious"}, transport.messages)
            self.assertEqual([f.text for f in turns.pushed if type(f).__name__ == "TTSSpeakFrame"], ["Achha, aur batao?"])
        asyncio.run(run())

    def test_barge_in_needs_half_a_second_of_speech(self):
        async def run():
            turns, _ = make_turns(FakeBrain())
            turns._bot_speaking = True
            # A cough or "hmm" over Buddy: under half a second, Buddy keeps talking.
            await turns.on_user_started(time.monotonic())
            await asyncio.sleep(0.2)
            turns.on_user_stopped(time.monotonic())
            await asyncio.sleep(0.5)
            self.assertFalse(turns.interrupted)
            # Real talking over Buddy: it stops after half a second and the stream is cut.
            await turns.on_user_started(time.monotonic())
            await asyncio.sleep(0.65)
            self.assertTrue(turns.interrupted)
            self.assertTrue(turns._cut_in)
        asyncio.run(run())

    def test_stop_words_stop_at_once_and_backchannels_are_not_answered(self):
        async def run():
            brain = FakeBrain()
            turns, _ = make_turns(brain)
            turns._bot_speaking = True
            await speak(turns, 0.3)                      # short: no barge-in by time
            turns._on_transcript("ok haan", time.monotonic())
            await asyncio.sleep(1.0)
            self.assertEqual(brain.turns, [])            # "ok haan" over Buddy is not a turn
            self.assertFalse(turns.interrupted)
            await speak(turns, 0.3)
            turns._on_transcript("wait", time.monotonic())
            await asyncio.sleep(0.2)
            self.assertTrue(turns.interrupted)           # "wait" stops Buddy even though it was short
            self.assertEqual(brain.turns, [])
        asyncio.run(run())

    def test_unfinished_words_wait_finished_ones_go_quickly(self):
        async def run():
            brain = FakeBrain()
            turns, _ = make_turns(brain, probability=0.9)
            await speak(turns, 0.6)
            turns._on_transcript("remind me to call my mom and", time.monotonic())
            await asyncio.sleep(1.2)
            self.assertEqual(brain.turns, [])            # "…and": holding for them to go on
            await speak(turns, 0.6)
            turns._on_transcript("tonight at eight.", time.monotonic())
            await asyncio.sleep(0.9)
            self.assertEqual(brain.turns, [("remind me to call my mom and tonight at eight.", "")])
        asyncio.run(run())

    def test_unsure_voice_waits_a_little_longer(self):
        async def run():
            brain = FakeBrain()
            turns, _ = make_turns(brain, probability=0.1)
            await speak(turns, 0.6)
            turns._on_transcript("mera naam Rohan hai", time.monotonic())
            await asyncio.sleep(0.6)
            self.assertEqual(brain.turns, [])            # voice sounds unfinished: about 1 s
            await asyncio.sleep(0.8)
            self.assertEqual(brain.turns, [("mera naam Rohan hai", "")])
        asyncio.run(run())

    def test_going_on_before_the_reply_is_heard_merges_into_one_message(self):
        async def run():
            brain = FakeBrain(think_seconds=1.0)
            turns, _ = make_turns(brain, probability=0.9)
            await speak(turns, 0.6)
            turns._on_transcript("add milk to my shopping list.", time.monotonic())
            await asyncio.sleep(0.8)                     # sent; Buddy is thinking (nothing heard yet)
            self.assertEqual(len(brain.turns), 1)
            await speak(turns, 0.6)                      # they go on talking
            turns._on_transcript("and eggs too.", time.monotonic())
            await asyncio.sleep(2.0)
            self.assertEqual(len(brain.discarded), 1)    # the first reply is dropped, not saved
            self.assertEqual(brain.turns[-1], ("add milk to my shopping list. and eggs too.", ""))
            self.assertEqual(len(brain.turns), 2)
        asyncio.run(run())

    def test_a_new_message_long_after_is_not_merged(self):
        async def run():
            from companion.realtime import turn_policy

            brain = FakeBrain(think_seconds=1.5)
            turns, _ = make_turns(brain, probability=0.9)
            await speak(turns, 0.6)
            turns._on_transcript("what's the weather?", time.monotonic())
            await asyncio.sleep(0.8)
            turns._inflight_sent_at -= turn_policy.MERGE_WINDOW + 1   # pretend the LLM has been slow for a while
            await speak(turns, 0.6)
            turns._on_transcript("also add eggs to my list.", time.monotonic())
            await asyncio.sleep(3.0)
            self.assertEqual(brain.discarded, [])
            self.assertEqual([t for t, _ in brain.turns], ["what's the weather?", "also add eggs to my list."])
        asyncio.run(run())

    def test_one_silence_nudge_per_user_turn(self):
        async def run():
            brain = FakeBrain(nudge_seconds=0.2)
            turns, _ = make_turns(brain)
            turns._start_nudge_timer()
            await asyncio.sleep(0.5)
            self.assertEqual(len(brain.turns), 1)
            self.assertTrue(brain.turns[0][1])  # a cue, no user words
            turns._start_nudge_timer()       # bot spoke again, but the user said nothing since
            await asyncio.sleep(0.4)
            self.assertEqual(len(brain.turns), 1)
            turns.safety_mode = True
            turns._nudged_since_user = False
            turns._start_nudge_timer()
            await asyncio.sleep(0.4)
            self.assertEqual(len(brain.turns), 1)  # never in safety mode
        asyncio.run(run())

    def test_crisis_words_send_support_resources(self):
        async def run():
            brain = FakeBrain()
            turns, transport = make_turns(brain)
            await turns._answer("sach mein ab jeena nahi chahta", None)
            self.assertTrue(turns.safety_mode)
            self.assertIn("safety_resources", [m["type"] for m in transport.messages])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
