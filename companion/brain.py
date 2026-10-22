"""The public entry point. Embedded use:

    bot = await Companion.create()
    reply = await bot.chat("ashish", "kal 9 baje dawai yaad dilana")
    print(reply.text, reply.expression)
    await bot.close()
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import aclosing
import re
import time
import uuid
from datetime import datetime
from collections.abc import AsyncIterator
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .config import Settings
from .graph import build_graph
from .llm.base import LLMError
from .llm import usage as llm_usage
from .llm.router import LLMRouter
from .llm.usage import TurnUsage
from .memory import extractor
from .memory.in_memory import InMemoryMemory
from .memory.interface import MemoryStore
from .memory.postgres import PostgresMemory
from .store import db
from .store.postgres import PostgresStore, data_owner
from .tools.builtin import build_registry, when_text
from .trace import TraceHub, clip
from .voice import Heard, Voice, VoiceError
from .voice.audio import STT_RATE, loudness_db, read_wav, resample, speech_seconds
from .modes import PUSH_TO_TALK, InputMode
from .voice.clip_cues import measure_clip
from .voice.cues import _Baseline
from .voice.guard import check_transcript
from .voice.speech import MAX_SPOKEN_WORDS, SentenceShaper, split_sentences
from .llm.base import ChatMessage

log = logging.getLogger("companion")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


_LEADING_TAG = re.compile(r"^\s*\[[a-zA-Z_]*\]?\s*")


def _visible(round_text: str) -> str | None:
    """The reply text without its leading mood tag, or None while the tag may still be arriving."""
    stripped = round_text.lstrip()
    if stripped.startswith("[") and "]" not in stripped and len(stripped) < 24:
        return None
    return _LEADING_TAG.sub("", round_text, count=1) if stripped.startswith("[") else round_text


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


class Reply(BaseModel):
    text: str
    expression: str
    tools_used: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    provider: str = ""
    db_changes: list[dict[str, Any]] = Field(default_factory=list)   # what this turn changed in the database
    # LLM tokens this turn used: {"input", "output", "reasoning", "total", "calls": [per LLM call]} (llm/usage.py)
    usage: dict[str, Any] = Field(default_factory=dict)


class VoiceTurn(BaseModel):
    """One push-to-talk turn: what was heard, the reply, and the reply as audio."""

    heard: str = ""
    language: str = ""
    kept: bool = False
    reason: str = ""               # why the audio was ignored, when kept is False
    reply: Reply | None = None
    audio_wav: bytes | None = None  # the reply spoken; None if TTS failed
    stt_provider: str = ""
    tts_provider: str = ""
    stt_ms: int = 0
    tts_ms: int = 0
    mode: str = "push_to_talk"
    audio_seconds: float = 0.0
    speech_seconds: float = 0.0
    loudness_db: float | None = None
    voice_cues: dict[str, Any] | None = None  # measured cues (only in modes with voice_cues=True)
    voice_note: str = ""                       # the line the LLM got about how they sounded


class Companion:
    def __init__(
        self, *, settings: Settings, llm: Any, store: Any, memory: MemoryStore, pool: Any = None,
        background_llm: Any = None, voice: Voice | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.background_llm = background_llm or llm   # memory extraction and summaries
        self.store = store
        self.memory = memory
        self.tools = build_registry(settings)
        self._discarded: set[str] = set()   # live turns merged into the next message (see discard_turn)
        self.voice = voice or Voice(settings)  # models load on first use or warm_up()
        self.trace = TraceHub()  # live events for the playground's Trace panel
        self._cue_baselines: dict[str, _Baseline] = {}  # each user's "usual" pace/loudness/pitch, shared by all voice modes
        self._pool = pool
        self._graph = build_graph(
            llm=llm, store=store, memory=memory, tools=self.tools, settings=settings,
            trace=self.trace,
        )
        self._background: set[asyncio.Task[None]] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def create(cls, settings: Settings | None = None, *, llm: Any = None, memory: MemoryStore | None = None) -> "Companion":
        settings = settings or Settings()
        pool = await db.connect(settings.database_url, settings.db_schema)
        if memory is None:
            memory = InMemoryMemory() if settings.memory_backend == "in_memory" else PostgresMemory(pool)
        background = llm or LLMRouter.from_settings(settings, settings.background_llm_order)
        return cls(
            settings=settings, llm=llm or LLMRouter.from_settings(settings), background_llm=background,
            store=PostgresStore(pool), memory=memory, pool=pool,
        )

    async def close(self) -> None:
        await self.wait_background()
        if self._pool is not None:
            await self._pool.close()

    async def wait_background(self) -> None:
        """Wait for memory extraction and summaries still running (tests, shutdown)."""
        while self._background:
            await asyncio.gather(*list(self._background), return_exceptions=True)

    async def chat(
        self, user_id: str, text: str, *, spoken: bool = False, turn: str = "", voice_note: str = "", mode_label: str = "",
        thread: str = "main",
    ) -> Reply:
        """One message in, one reply out. `spoken=True` asks for a short reply meant to be read aloud."""
        text = text.strip()
        if not text:
            return Reply(text="", expression="idle")
        turn = turn or uuid.uuid4().hex[:8]
        started = time.perf_counter()
        self.trace.emit(user_id, "turn_start", turn, text=text, mode=mode_label or ("voice (push-to-talk)" if spoken else "text"))
        # One message at a time per user, so the history stays in order.
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            timezone = await self.store.ensure_user(user_id, self.settings.timezone)
            tz = ZoneInfo(timezone)
            counted = TurnUsage()
            token = llm_usage.CURRENT.set(counted)   # every LLM call in this turn adds its tokens here
            try:
                state = await self._graph.ainvoke(
                    {"turn": turn, "thread": thread, "user_id": user_id, "user_text": text, "timezone": timezone, "now": datetime.now(tz), "spoken": spoken,
                     "voice_note": voice_note},
                    {"recursion_limit": 4 * self.settings.max_tool_rounds + 8},
                )
            finally:
                llm_usage.CURRENT.reset(token)   # background work after the reply is not part of this turn
            await self.store.add_messages(state["conversation_id"], state["new_messages"], state["expression"])
        self.trace.emit(
            user_id, "turn_end", turn, reply=state["reply"], expression=state["expression"], ms=_ms_since(started),
            tools=[line.split(" ", 1)[0] for line in state["tool_actions"]], saved="user message + reply (+ tool steps) in short-term memory",
        )
        task = asyncio.create_task(self._after_turn(user_id, tz, state["conversation_id"], text, state["reply"], state["tool_actions"], turn, data_owner(user_id, thread)))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return Reply(
            text=state["reply"], expression=state["expression"],
            tools_used=[line.split(" ", 1)[0] for line in state["tool_actions"]],
            sources=list(dict.fromkeys(state["sources"])), provider=state.get("provider", ""), db_changes=state.get("db_changes", []),
            usage=counted.summary(),
        )

    async def chat_stream(
        self, user_id: str, text: str = "", *, cue: str = "", voice_note: str = "", spoken: bool = True,
        voice_cues: dict[str, Any] | None = None, turn: str = "", mode_label: str = "", thread: str = "main",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a reply as it is written (all modes: text, push-to-talk, live).

        Yields {"type": "delta", "text"} (new reply text as the model writes it, for typing on screen),
        {"type": "expression", "expression"}, {"type": "sentence", "text"} (a finished, checked sentence,
        ready to speak) ... and finally {"type": "done", "reply": Reply}. With spoken=False (text chat)
        there is no word limit and the saved reply keeps its line breaks. `cue` (instead of `text`) makes a turn with no new user
        words: a greeting or a silence nudge. If the reader stops early (the user talked over it),
        only the sentences already handed out are saved as the reply, because only those were heard.
        Read it inside `contextlib.aclosing(...)` so stopping early closes it at once.
        """
        text = text.strip()
        queue: asyncio.Queue = asyncio.Queue()
        handed_out: list[str] = []
        reader_done = asyncio.Event()
        read_all = asyncio.Event()  # set only if every sentence was read (no barge-in)
        shaper = SentenceShaper(nudge=bool(cue) and not text, max_words=None if not spoken else MAX_SPOKEN_WORDS)
        turn = turn or uuid.uuid4().hex[:8]
        label = mode_label or ("live voice" if text else "live voice, companion speaks first")
        self.trace.emit(user_id, "turn_start", turn, text=text or f"(no words: {cue})", mode=label)
        if voice_cues is not None:
            self.trace.emit(user_id, "voice_cues", turn, fields=voice_cues, note=voice_note or "(nothing unusual: no note sent)")
        runner = asyncio.create_task(
            self._stream_turn(user_id, text, cue, voice_note, spoken, queue, handed_out, reader_done, shaper, turn, read_all, thread)
        )
        try:
            expression_sent = False
            round_text, shown = "", 0  # raw text of this LLM round, and how much of it is already on screen
            while True:
                event = await queue.get()
                if event is None:
                    break
                if event["type"] == "round_start":
                    shaper.start_round()
                    round_text, shown = "", 0
                    continue
                if event["type"] == "piece":
                    sentences = shaper.feed(event["text"])
                    if shaper.expression and not expression_sent:
                        expression_sent = True  # the mood tag is read: the face can change before any words
                        yield {"type": "expression", "expression": shaper.expression}
                    round_text += event["text"]
                    visible = _visible(round_text)
                    if visible is not None and len(visible) > shown:
                        yield {"type": "delta", "text": visible[shown:]}
                        shown = len(visible)
                elif event["type"] == "round_end":
                    sentences = shaper.end_round()
                elif event["type"] == "final":
                    # Nothing was streamed (every LLM failed, or junk): speak the guard's reply instead,
                    # e.g. the confirmation built from tool results. A nudge with nothing usable stays quiet.
                    sentences = [] if handed_out or shaper.nudge else split_sentences(event["reply"], final=True)[0]
                    shaper.expression = shaper.expression or event["expression"]
                else:
                    continue
                if sentences and not expression_sent:
                    expression_sent = True
                    yield {"type": "expression", "expression": shaper.expression or "idle"}
                for sentence in sentences:
                    handed_out.append(sentence)
                    self.trace.emit(user_id, "speak", turn, sentence=sentence)
                    yield {"type": "sentence", "text": sentence}
            read_all.set()
            reader_done.set()  # everything was handed out; the runner may save now
            reply = await runner
            yield {"type": "done", "reply": reply}
        finally:
            reader_done.set()
            if not runner.done():
                # Stopped early: let the turn finish in the background (tools still run, history is saved).
                self._background.add(runner)
                runner.add_done_callback(self._background.discard)

    async def _stream_turn(
        self, user_id: str, text: str, cue: str, voice_note: str, spoken: bool,
        queue: asyncio.Queue, handed_out: list[str], reader_done: asyncio.Event, shaper: "SentenceShaper", turn: str = "",
        read_all: asyncio.Event | None = None, thread: str = "main",
    ) -> Reply:
        read_all = read_all or asyncio.Event()
        started = time.perf_counter()
        counted = TurnUsage()
        token = llm_usage.CURRENT.set(counted)   # every LLM call in this turn adds its tokens here
        try:
            async with self._locks.setdefault(user_id, asyncio.Lock()):
                timezone = await self.store.ensure_user(user_id, self.settings.timezone)
                tz = ZoneInfo(timezone)
                state: dict[str, Any] = {}
                async for mode, chunk in self._graph.astream(
                    {
                        "turn": turn, "thread": thread, "user_id": user_id, "user_text": text, "timezone": timezone, "now": datetime.now(tz),
                        "spoken": spoken, "stream": True, "cue": cue if not text else "", "voice_note": voice_note,
                    },
                    {"recursion_limit": 4 * self.settings.max_tool_rounds + 8},
                    stream_mode=["custom", "values"],
                ):
                    if mode == "custom":
                        await queue.put(chunk)
                    else:
                        state = chunk
                await queue.put({"type": "final", "reply": state.get("reply", ""), "expression": state.get("expression", "idle")})
                await queue.put(None)
                # Save only what was actually handed out to be spoken.
                try:
                    await asyncio.wait_for(reader_done.wait(), 120)
                except TimeoutError:
                    pass
                if spoken:
                    # Save only what was actually handed out to be spoken (a barge-in drops the rest).
                    said = " ".join(handed_out)
                    messages = [m for m in state["new_messages"] if not (m.role == "assistant" and not m.tool_calls)]
                    if said:
                        messages.append(ChatMessage(role="assistant", content=said))
                else:
                    # Text chat: the guard's checked reply, with its line breaks.
                    said, messages = state.get("reply", ""), state["new_messages"]
                expression = shaper.expression or state.get("expression", "idle")
                if turn in self._discarded:
                    # Live mode merged this turn into the next one (the user went on talking before hearing
                    # a reply): keep only what changed data, never the half message or the unheard reply.
                    self._discarded.discard(turn)
                    wrote = any(self.tools.is_write(line.split(" ", 1)[0]) for line in state["tool_actions"])
                    messages = [m for m in messages if m.role != "assistant" or m.tool_calls] if wrote else []
                    said = ""
                    self.trace.emit(user_id, "turn_discarded", turn, kept_tool_steps=wrote,
                                    note="merged into the next message: the user went on talking before hearing a reply")
                    if not messages:
                        return Reply(text="", expression=expression, usage=counted.summary())
                await self.store.add_messages(state["conversation_id"], messages, expression)
                cut = not read_all.is_set()  # the reader stopped before the end: the user talked over the rest
                self.trace.emit(
                    user_id, "turn_end", turn, reply=said, expression=expression, ms=_ms_since(started),
                    tools=[line.split(" ", 1)[0] for line in state["tool_actions"]],
                    dropped_sentences=shaper.dropped,
                    saved=("only the sentences actually spoken" + (" (the user talked over the rest)" if cut else "")
                           if spoken else "user message + reply (+ tool steps)")
                    + ("; no user message (companion spoke first)" if not text else ""),
                )
        except Exception:
            await queue.put(None)
            raise
        finally:
            llm_usage.CURRENT.reset(token)   # background work after the reply is not part of this turn
        if text:
            task = asyncio.create_task(self._after_turn(user_id, tz, state["conversation_id"], text, said, state["tool_actions"], turn, data_owner(user_id, thread)))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
        return Reply(
            text=said, expression=expression,
            tools_used=[line.split(" ", 1)[0] for line in state["tool_actions"]],
            sources=list(dict.fromkeys(state["sources"])), provider=state.get("provider", ""), db_changes=state.get("db_changes", []),
            usage=counted.summary(),
        )

    async def _after_turn(
        self, user_id: str, tz: ZoneInfo, conversation_id: str, text: str, reply: str, tool_actions: list[str], turn: str = "",
        owner: str = "",
    ) -> None:
        """Background work after a reply: remember what matters, fold old chat into the summary."""
        now = datetime.now(tz)
        # Memory is saved only when the user asks (tools), unless auto_memory is on.
        # "ok", "haha", "hmm": nothing to remember, so skip the extractor and save tokens.
        if not self.settings.auto_memory:
            pass
        elif len(text.split()) >= 3 or tool_actions:
            self.trace.emit(user_id, "memory_extract_start", turn, note="background: reading the turn for things worth remembering")
            started = time.perf_counter()
            try:
                known = await self.memory.list_all(owner or user_id)
                items = await extractor.extract(self.background_llm, text, reply, tool_actions, known, now.date())
                decisions = await extractor.apply(self.memory, owner or user_id, items, tz, now)
                for decision, item, _ in decisions:
                    if decision != "ignore":
                        log.info("memory %s: [%s] %s", decision, item.type, item.text)
                self.trace.emit(
                    user_id, "memory_extract", turn, ms=_ms_since(started),
                    items=[
                        {"decision": d, "type": i.type, "key": i.key, "text": i.text, "confidence": i.confidence, "why": why,
                         "until": str(i.until or "")}
                        for d, i, why in decisions
                    ],
                )
            except LLMError as exc:
                log.warning("memory extraction skipped: %s", exc)
                self.trace.emit(user_id, "memory_skip", turn, why=f"no LLM available: {exc}")
            except Exception as exc:
                log.exception("memory extraction failed")
                self.trace.emit(user_id, "memory_skip", turn, why=f"failed: {exc}")
        else:
            self.trace.emit(user_id, "memory_skip", turn, why="message under 3 words with no action: nothing to remember")
        if not self.settings.auto_memory:
            self.trace.emit(user_id, "memory_skip", turn, why="saving only when you ask (auto memory is off)")
        try:
            await self._maybe_summarize(conversation_id, user_id, turn)
        except LLMError as exc:
            log.warning("summary skipped: %s", exc)
        except Exception:
            log.exception("summary failed")

    async def _summarize(self, existing: str, messages: list) -> str:
        return _THINK.sub("", await extractor.summarize(self.background_llm, existing, messages)).strip()

    async def _maybe_summarize(self, conversation_id: str, user_id: str = "", turn: str = "") -> None:
        summary, done, pending = await self.store.unsummarized(conversation_id)
        if len(pending) <= self.settings.summarize_after:
            return
        fold = pending[: len(pending) - self.settings.recent_messages]
        new_summary = await self._summarize(summary, fold)
        if new_summary:
            await self.store.save_summary(conversation_id, new_summary, done + len(fold))
            self.trace.emit(
                user_id, "summary", turn, scope="this conversation", folded=len(fold),
                kept_whole=self.settings.recent_messages, summary=clip(new_summary),
            )

    def cue_baseline(self, user_id: str) -> _Baseline:
        """The user's "usual" for voice cues (kept in memory while the server runs)."""
        return self._cue_baselines.setdefault(user_id, _Baseline())

    async def hear(
        self, user_id: str, wav: bytes, *, mode: InputMode = PUSH_TO_TALK, max_seconds: float = 60.0,
        gap_before_recording_s: float | None = None, interrupted: bool = False, turn: str = "", text: str | None = None,
    ) -> VoiceTurn:
        """Speech in: STT, the made-up-words guard and, in modes with voice cues, how they sounded.
        Raises AudioError for bad audio. The reply is not made here (see voice_turn / voice_stream).

        `text`: the words the user already saw and confirmed (or edited) in the input box. Then STT is
        skipped and that text is used as is; the audio is still used for the voice cues."""
        samples, rate = read_wav(wav)
        if samples.size / rate > max_seconds:
            raise VoiceError(f"Speak for at most {max_seconds:.0f} seconds at a time.")
        spoken_for, loudness = speech_seconds(samples, rate), loudness_db(samples, rate)
        started = time.perf_counter()
        samples_16k = resample(samples, rate, STT_RATE)
        if text is not None and text.strip():
            heard, stt_provider = Heard(text=text.strip()), "confirmed by the user"
        else:
            heard, stt_provider = await self.voice.transcribe(samples_16k)
        result = VoiceTurn(
            heard=heard.text, language=heard.language, stt_provider=stt_provider, stt_ms=_ms_since(started), mode=mode.name,
            audio_seconds=round(samples.size / rate, 2), speech_seconds=round(spoken_for, 2), loudness_db=round(loudness, 1),
        )
        keep, reason = (True, "confirmed in the input box") if text and text.strip() else check_transcript(heard.text, spoken_for, loudness)
        self.trace.emit(
            user_id, "stt", turn, provider=stt_provider, ms=result.stt_ms, heard=heard.text, language=heard.language,
            speech_seconds=result.speech_seconds, loudness_db=result.loudness_db, kept=keep, why=reason,
        )
        log.info("heard (%s, %.1fs speech, %.0f dB, %d ms): %r -> %s", stt_provider, spoken_for, loudness, result.stt_ms, heard.text, reason)
        if not keep:
            return result.model_copy(update={"reason": reason})
        if mode.voice_cues:
            cues, segments = await asyncio.to_thread(
                measure_clip, samples_16k, heard.text, gap_before_recording_s=gap_before_recording_s,
                interrupted=interrupted, baseline=self.cue_baseline(user_id),
            )
            fields = {**cues.log_fields(), "pace": cues.pace, "volume": cues.volume, "intonation": cues.intonation,
                      "short_reply": cues.short_reply, "speech_segments": [[round(a, 2), round(b, 2)] for a, b in segments]}
            result = result.model_copy(update={"voice_cues": fields, "voice_note": cues.describe()})
            self.trace.emit(user_id, "voice_cues", turn, fields=fields, note=result.voice_note or "(nothing unusual: no note sent)")
        return result.model_copy(update={"kept": True, "reason": "ok"})

    async def voice_turn(
        self, user_id: str, wav: bytes, *, speed: float | None = None, max_seconds: float = 60.0,
        mode: InputMode = PUSH_TO_TALK, gap_before_recording_s: float | None = None, interrupted: bool = False,
        thread: str = "main",
    ) -> VoiceTurn:
        """Speech in, speech out, all at once (no streaming). Raises AudioError for bad audio."""
        turn_id = uuid.uuid4().hex[:8]
        turn = await self.hear(
            user_id, wav, mode=mode, max_seconds=max_seconds, gap_before_recording_s=gap_before_recording_s,
            interrupted=interrupted, turn=turn_id,
        )
        if not turn.kept:
            return turn
        reply = await self.chat(
            user_id, turn.heard, spoken=True, turn=turn_id, voice_note=turn.voice_note,
            mode_label="voice (push-to-talk + voice cues)" if mode.voice_cues else "voice (push-to-talk)", thread=thread,
        )
        started = time.perf_counter()
        try:
            audio, tts_provider = await self.voice.speak(reply.text, speed)
        except VoiceError as exc:
            log.warning("could not speak the reply: %s", exc)
            audio, tts_provider = None, ""
        return turn.model_copy(update={"reply": reply, "audio_wav": audio, "tts_provider": tts_provider, "tts_ms": _ms_since(started)})

    async def transcribe(self, wav: bytes, max_seconds: float = 60.0) -> dict[str, Any]:
        """Just speech to text (for showing the words while the user is still talking). No reply, no memory."""
        samples, rate = read_wav(wav)
        if samples.size / rate > max_seconds:
            raise VoiceError(f"Speak for at most {max_seconds:.0f} seconds at a time.")
        started = time.perf_counter()
        heard, provider = await self.voice.transcribe(resample(samples, rate, STT_RATE))
        return {"text": heard.text, "language": heard.language, "provider": provider, "ms": _ms_since(started)}

    async def voice_stream(
        self, user_id: str, wav: bytes, *, speed: float | None = None, mode: InputMode = PUSH_TO_TALK,
        gap_before_recording_s: float | None = None, interrupted: bool = False, turn: str = "", thread: str = "main",
        text: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Push-to-talk, streamed: {"type": "heard", "turn": VoiceTurn}, then the chat_stream events,
        with {"type": "audio", "text", "wav": bytes, "tts_provider"} right after each sentence.
        Read it inside contextlib.aclosing(...): stopping early (barge-in) keeps only what was spoken."""
        turn = turn or uuid.uuid4().hex[:8]
        heard = await self.hear(
            user_id, wav, mode=mode, gap_before_recording_s=gap_before_recording_s, interrupted=interrupted, turn=turn, text=text,
        )
        yield {"type": "heard", "turn": heard}
        if not heard.kept:
            yield {"type": "done", "reply": None}
            return
        events = self.chat_stream(
            user_id, heard.heard, voice_note=heard.voice_note, voice_cues=heard.voice_cues, spoken=True, turn=turn,
            mode_label="voice (push-to-talk + voice cues)" if mode.voice_cues else "voice (push-to-talk)", thread=thread,
        )
        async with aclosing(events) as stream:
            async for event in stream:
                yield event
                if event["type"] == "sentence":
                    try:
                        audio, provider = await self.voice.speak(event["text"], speed)
                        yield {"type": "audio", "text": event["text"], "wav": audio, "tts_provider": provider}
                    except VoiceError as exc:
                        log.warning("could not speak a sentence: %s", exc)

    def discard_turn(self, turn: str) -> None:
        """Live mode: the user went on talking before hearing this turn's reply, so it is merged into the
        next message. Its half message and unheard reply are not saved (tool steps that changed data are)."""
        self._discarded.add(turn)

    async def due_reminders(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Reminders whose time has come (each is returned once). Bodies call this on a timer."""
        due = await self.store.take_due_reminders(user_id)
        for reminder in due:
            reminder["due"] = when_text(reminder["due_at"], ZoneInfo(self.settings.timezone))
        return due
