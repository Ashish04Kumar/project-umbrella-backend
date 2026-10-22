"""HTTP server around the Companion, plus a browser playground for testing.

    .venv/bin/python -m companion.server          # http://127.0.0.1:8300

Local testing only for now: there is no login, so it listens on 127.0.0.1 and must not be
exposed to the internet. Auth comes with the real server (milestone 2).
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import uuid
from collections import Counter
from contextlib import aclosing
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo
import time
from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..brain import Companion
from ..config import cors_origins
from ..graph import MEMORY_IN_PROMPT
from ..modes import MODES, PUSH_TO_TALK, PUSH_TO_TALK_CUES, TEXT
from ..store.postgres import data_owner, short
from ..voice import VoiceError
from ..voice.audio import AudioError

try:  # live voice is optional: it needs pipecat and LiveKit (the "realtime" extra)
    from ..realtime.sessions import LiveError, LiveSessions
except ImportError:  # pragma: no cover
    LiveSessions = None  # type: ignore[assignment,misc]

    class LiveError(RuntimeError):  # type: ignore[no-redef]
        pass

PLAYGROUND = Path(__file__).parent / "playground.html"
PLUGINS_DIR = Path(__file__).parent / "plugins"


MAX_AUDIO_BYTES = 60 * 48_000 * 2 + 1024  # 60 s of 48 kHz 16-bit mono


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    speed: float | None = Field(default=None, ge=0.5, le=2.0)


class LiveIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)


class LiveStopIn(BaseModel):
    room: str = Field(min_length=1, max_length=120)


THREAD_PATTERN = r"^(main|t-[0-9a-f]{8})$"


class ClearIn(BaseModel):
    store: Literal["short", "long", "notes", "todos", "reminders", "list"]
    day: date | None = None
    list: str | None = None
    thread: str = Field(default="main", pattern=THREAD_PATTERN)


class EditIn(BaseModel):
    """What the playground's ✏️ form sends; only the fields given are changed."""
    text: str | None = Field(default=None, min_length=1, max_length=4000)
    title: str | None = Field(default=None, max_length=60)
    subject: str | None = Field(default=None, min_length=1, max_length=60)
    topic: str | None = Field(default=None, min_length=1, max_length=40)
    when: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")   # reminders, local time
    repeat: Literal["none", "daily", "weekly", "monthly"] | None = None
    all_day: bool | None = None
    day: date | None = None
    priority: Literal["low", "normal", "high"] | None = None
    done: bool | None = None


EDIT_KINDS = r"^(memory|note|reminder|todo|list_item)$"
EDIT_TABLES = {"note": "notes", "reminder": "reminders", "todo": "todos"}


class TestStartIn(BaseModel):
    user_id: str = Field(default="consumer1", min_length=1, max_length=80)
    turns: int = Field(default=30, ge=1, le=60)
    gap_seconds: float = Field(default=20, ge=0, le=120)


class ChatIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=4000)
    thread: str = Field(default="main", pattern=THREAD_PATTERN)  # "main": the user's own ongoing chat


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(make_bot: Callable[[], Awaitable[Companion]] | None = None, warm_up_voice: bool = True) -> FastAPI:
    """`make_bot` lets tests build their own Companion; normally Companion.create() runs on startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bot = await (make_bot or Companion.create)()
        app.state.live = LiveSessions(app.state.bot) if LiveSessions else None
        # Load Whisper and Piper in the background so the first voice turn is quick.
        warm = asyncio.create_task(asyncio.to_thread(app.state.bot.voice.warm_up)) if warm_up_voice else None
        yield
        if app.state.live:
            await app.state.live.stop_all()
        if warm:
            await asyncio.gather(warm, return_exceptions=True)
        await app.state.bot.close()

    app = FastAPI(title="Companion", lifespan=lifespan)
    # The umbrella-d dashboard (a separate Next.js app) calls this API from its own address.
    origins = cors_origins()
    if origins:
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def brain() -> Companion:
        return app.state.bot

    @app.get("/")
    async def playground() -> FileResponse:
        return FileResponse(PLAYGROUND)

    @app.get("/plugins/{name}/{path:path}")
    async def plugin_file(name: str, path: str) -> FileResponse:
        """Files of a front-end plugin (e.g. the 3D robot), only for plugins switched on in COMPANION_UI_PLUGINS."""
        if name not in brain().settings.ui_plugins:
            raise HTTPException(404, "No such plugin")
        root = (PLUGINS_DIR / name).resolve()
        target = (root / path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "No such file")
        media = {".js": "text/javascript", ".glb": "model/gltf-binary"}.get(target.suffix)
        return FileResponse(target, media_type=media)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        voice = brain().voice
        return {
            "ok": True, "name": brain().settings.name, "tools": brain().tools.names(),
            "stt": [p.name for p in voice.stt], "tts": [p.name for p in voice.tts],
            "live": bool(app.state.live and app.state.live.configured()),
            "plugins": [name for name in brain().settings.ui_plugins if (PLUGINS_DIR / name / "plugin.js").is_file()],
        }

    def log_request(user_id: str, turn: str, obj: dict[str, Any]) -> dict[str, Any]:
        """The request object of a turn: kept in the trace and sent back to the body (the chat's { } button)."""
        brain().trace.emit(user_id, "request", turn, object=obj)
        return obj

    def log_response(user_id: str, turn: str, obj: dict[str, Any]) -> dict[str, Any]:
        brain().trace.emit(user_id, "response", turn, object=obj)
        return obj

    @app.post("/v1/chat")
    async def chat(body: ChatIn) -> dict[str, Any]:
        started = time.perf_counter()
        turn = uuid.uuid4().hex[:8]
        log_request(body.user_id, turn, {"endpoint": "POST /v1/chat", "mode": TEXT.name, "turn": turn, "body": body.model_dump(), "at": _now()})
        reply = await brain().chat(body.user_id, body.text, turn=turn, thread=body.thread)
        result = {**reply.model_dump(), "ms": round((time.perf_counter() - started) * 1000)}
        log_response(body.user_id, turn, {"turn": turn, "reply": result})
        return {**result, "turn": turn}

    @app.post("/v1/chat/stream")
    async def chat_stream(body: ChatIn) -> StreamingResponse:
        """Text chat, streamed (server-sent events): delta (new text as it is written), expression,
        sentence, and finally done with the whole reply object."""
        turn = uuid.uuid4().hex[:8]
        request_obj = log_request(body.user_id, turn, {"endpoint": "POST /v1/chat/stream", "mode": TEXT.name, "turn": turn, "body": body.model_dump(), "at": _now()})

        async def events():
            started, counts, reply = time.perf_counter(), Counter(), None
            yield _sse({"type": "request", "object": request_obj})
            stream = brain().chat_stream(body.user_id, body.text, spoken=False, turn=turn, mode_label="text", thread=body.thread)
            async with aclosing(stream) as items:
                async for event in items:
                    counts[event["type"]] += 1
                    if event["type"] == "done":
                        reply = {**event["reply"].model_dump(), "ms": round((time.perf_counter() - started) * 1000)}
                        yield _sse({"type": "done", "reply": reply})
                    else:
                        yield _sse(event)
            response_obj = log_response(body.user_id, turn, {"turn": turn, "reply": reply, "stream_events": dict(counts)})
            yield _sse({"type": "response", "object": response_obj})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/v1/voice")
    async def voice(
        request: Request, user_id: str, speed: float | None = None, mode: str = PUSH_TO_TALK.name,
        gap_ms: int | None = None, interrupted: bool = False, thread: str = Query("main", pattern=THREAD_PATTERN),
    ) -> dict[str, Any]:
        """Push-to-talk: the body is a 16-bit WAV clip. Returns what was heard, the reply and its audio.

        `mode=push_to_talk_cues` also measures voice cues (see companion/modes.py). Then `gap_ms` is the time
        from the end of the companion's voice to the start of recording, and `interrupted` says the user
        pressed the mic while the companion was still talking."""
        input_mode = MODES.get(mode)
        if input_mode is None or input_mode not in (PUSH_TO_TALK, PUSH_TO_TALK_CUES):
            raise HTTPException(422, "mode must be push_to_talk or push_to_talk_cues")
        audio = await request.body()
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413 if audio else 400, "Send one WAV clip of at most 60 seconds.")
        started = time.perf_counter()
        try:
            turn = await brain().voice_turn(
                user_id, audio, speed=speed, mode=input_mode, interrupted=interrupted, thread=thread,
                gap_before_recording_s=gap_ms / 1000 if gap_ms is not None and 0 <= gap_ms <= 120_000 else None,
            )
        except (AudioError, VoiceError) as exc:
            raise HTTPException(422, str(exc)) from exc
        result = turn.model_dump(exclude={"audio_wav"})
        result["audio_b64"] = base64.b64encode(turn.audio_wav).decode() if turn.audio_wav else None
        result["ms"] = round((time.perf_counter() - started) * 1000)
        return result

    @app.post("/v1/voice/stream")
    async def voice_stream(
        request: Request, user_id: str, speed: float | None = None, mode: str = PUSH_TO_TALK.name,
        gap_ms: int | None = None, interrupted: bool = False, thread: str = Query("main", pattern=THREAD_PATTERN),
        text: str | None = Query(None, max_length=4000),
    ) -> StreamingResponse:
        """Push-to-talk, streamed (server-sent events): heard (what was understood, voice cues), then
        expression, delta, sentence and audio (each sentence's voice as base64 WAV), and done."""
        input_mode = MODES.get(mode)
        if input_mode is None or input_mode not in (PUSH_TO_TALK, PUSH_TO_TALK_CUES):
            raise HTTPException(422, "mode must be push_to_talk or push_to_talk_cues")
        audio = await request.body()
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413 if audio else 400, "Send one WAV clip of at most 60 seconds.")
        turn = uuid.uuid4().hex[:8]
        gap = gap_ms / 1000 if gap_ms is not None and 0 <= gap_ms <= 120_000 else None
        query = {"user_id": user_id, "thread": thread, "mode": mode, "speed": speed, "gap_ms": gap_ms, "interrupted": interrupted, "text": text}

        async def events():
            started, counts, reply, spoken_audio = time.perf_counter(), Counter(), None, {"sentences": 0, "bytes": 0, "tts_provider": ""}
            stream = brain().voice_stream(
                user_id, audio, speed=speed, mode=input_mode, gap_before_recording_s=gap, interrupted=interrupted, turn=turn, thread=thread,
                text=text,
            )
            try:
                async with aclosing(stream) as items:
                    async for event in items:
                        counts[event["type"]] += 1
                        if event["type"] == "heard":
                            heard = event["turn"]
                            request_obj = log_request(user_id, turn, {
                                "endpoint": "POST /v1/voice/stream", "mode": mode, "turn": turn, "query": query, "at": _now(),
                                "audio": {"wav_bytes": len(audio), "seconds": heard.audio_seconds,
                                          "speech_seconds": heard.speech_seconds, "loudness_db": heard.loudness_db},
                                "stt": {"provider": heard.stt_provider, "ms": heard.stt_ms, "text": heard.heard,
                                        "language": heard.language, "kept": heard.kept, "guard": heard.reason},
                                "voice_cues": heard.voice_cues, "voice_note_for_llm": heard.voice_note,
                            })
                            yield _sse({"type": "heard", "turn": heard.model_dump(exclude={"audio_wav", "reply"})})
                            yield _sse({"type": "request", "object": request_obj})
                        elif event["type"] == "audio":
                            spoken_audio["sentences"] += 1
                            spoken_audio["bytes"] += len(event["wav"])
                            spoken_audio["tts_provider"] = event["tts_provider"]
                            yield _sse({"type": "audio", "text": event["text"], "tts_provider": event["tts_provider"],
                                        "wav_b64": base64.b64encode(event["wav"]).decode()})
                        elif event["type"] == "done":
                            reply = {**event["reply"].model_dump(), "ms": round((time.perf_counter() - started) * 1000)} if event["reply"] else None
                            yield _sse({"type": "done", "reply": reply})
                        else:
                            yield _sse(event)
            except (AudioError, VoiceError) as exc:
                yield _sse({"type": "error", "error": str(exc)})
            response_obj = log_response(user_id, turn, {"turn": turn, "reply": reply, "spoken_audio": spoken_audio, "stream_events": dict(counts)})
            yield _sse({"type": "response", "object": response_obj})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/v1/stt")
    async def stt(request: Request) -> dict[str, Any]:
        """Speech to text only (the body is a 16-bit WAV clip): the playground calls it while you are
        still talking, to show your words in the input box. Nothing is saved."""
        audio = await request.body()
        if not audio or len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413 if audio else 400, "Send one WAV clip of at most 60 seconds.")
        try:
            return await brain().transcribe(audio)
        except (AudioError, VoiceError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/tts")
    async def tts(body: SpeakIn) -> Response:
        """Speak any text (for trying voices and speeds)."""
        try:
            audio, provider = await brain().voice.speak(body.text, body.speed)
        except VoiceError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(audio, media_type="audio/wav", headers={"X-TTS-Provider": provider})

    @app.post("/v1/live/start")
    async def live_start(body: LiveIn) -> dict[str, Any]:
        """Start a live voice session; the body joins the returned LiveKit room with the token."""
        if app.state.live is None:
            raise HTTPException(501, "Live voice needs the realtime extra: pip install -e '.[realtime]'")
        try:
            return await app.state.live.start(body.user_id)
        except LiveError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post("/v1/live/stop")
    async def live_stop(body: LiveStopIn) -> dict[str, Any]:
        return {"stopped": bool(app.state.live) and await app.state.live.stop(body.room)}

    @app.get("/v1/users/{user_id}/trace")
    async def trace_stream(user_id: str, request: Request) -> StreamingResponse:
        """Live trace (server-sent events): recent events first, then each new one as it happens."""
        hub = brain().trace

        async def events():
            queue = hub.subscribe(user_id)
            try:
                for event in hub.recent(user_id):
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), 15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            finally:
                hub.unsubscribe(user_id, queue)

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    def memory_shape(item) -> dict[str, Any]:
        return {
            "id": short(item.id or ""), "scope": item.scope, "category": item.category, "subject": item.subject,
            "topic": item.topic, "key": item.key, "text": item.text, "pinned": item.pinned,
            "date": (item.event_date.strftime("%d %b") if item.yearly else str(item.event_date)) if item.event_date else None,
            "keywords": item.keywords, "confidence": round(item.confidence, 2), "status": item.status,
            "asked": bool(item.data.get("asked")),
            "expires_at": item.expires_at.isoformat() if item.expires_at else None,
            "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        }

    @app.get("/v1/users/{user_id}/memory/state")
    async def memory_state(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        """Short-term and long-term memory side by side, with the rules that move data between them."""
        bot = brain()
        owner = data_owner(user_id, thread)
        s = bot.settings
        return {
            "short_term": {
                **await bot.store.memory_snapshot(user_id, s.recent_messages, thread),
                "working_memory": [memory_shape(i) for i in await bot.memory.list_all(owner, scope="short")],
            },
            "long_term": {
                "memories": [memory_shape(i) for i in await bot.memory.list_all(owner, scope="long")],
                "pending": [memory_shape(i) for i in await bot.memory.list_all(owner, status="pending")],
            },
            "rules": {
                "window": s.recent_messages, "summarize_after": s.summarize_after,
                "thread": thread, "memory_in_prompt": MEMORY_IN_PROMPT,
                "memory_backend": type(bot.memory).__name__, "auto_memory": s.auto_memory,
            },
        }

    @app.get("/v1/users/{user_id}/memory/raw")
    async def memory_raw(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        """Short-term and long-term memory exactly as stored in Postgres: table schema and rows."""
        return await brain().store.raw_dump(user_id, thread)

    @app.get("/v1/users/{user_id}/history")
    async def history(user_id: str, limit: int = 40, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        rows = await brain().store.latest_messages(user_id, min(limit, 200), thread)
        return {"messages": [{**row, "created_at": row["created_at"].isoformat()} for row in rows]}

    @app.post("/v1/users/{user_id}/threads")
    async def new_thread(user_id: str) -> dict[str, Any]:
        """New conversation = a separate thread (chat). The main thread is never reset or replaced.
        Listing, naming and closing threads come later."""
        return {"thread": await brain().store.create_thread(user_id)}

    # --- Test tab: a simulated consumer chats in a new thread; the tab watches the trace --------------
    from ..testing.consumer import TestRun, run as run_test

    tests: dict[str, TestRun] = {}

    @app.post("/v1/test/start")
    async def test_start(body: TestStartIn) -> dict[str, Any]:
        current = tests.get("current")
        if current and current.running:
            raise HTTPException(409, f"A test is already running for {current.user_id}.")
        # One consumer = one thread: the test chats in the consumer's own (main) thread.
        test = TestRun(user_id=body.user_id, thread="main", turns=body.turns, gap_seconds=body.gap_seconds)
        test.task = asyncio.create_task(run_test(brain(), test))
        tests["current"] = test
        return test.status()

    @app.post("/v1/test/stop")
    async def test_stop() -> dict[str, Any]:
        current = tests.get("current")
        if not current:
            return {"running": False}
        current.running = False   # stops after the turn in progress
        return current.status()

    @app.get("/v1/test/status")
    async def test_status() -> dict[str, Any]:
        current = tests.get("current")
        return current.status() if current else {"running": False}

    @app.get("/v1/users")
    async def users() -> dict[str, Any]:
        """Every user in the database, newest activity first (the playground's user dropdown)."""
        rows = await brain().store.list_users()
        return {"users": [{**row, "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None} for row in rows]}

    @app.get("/v1/users/{user_id}/threads")
    async def threads(user_id: str) -> dict[str, Any]:
        """All of a user's threads (main = thread 1), for the playground's thread switcher."""
        rows = await brain().store.list_threads(user_id)
        return {"threads": [
            {**row, "started_at": row["started_at"].isoformat() if row["started_at"] else None,
             "last_message_at": row["last_message_at"].isoformat() if row["last_message_at"] else None}
            for row in rows
        ]}

    # The side panels show and change the data of one thread (each thread has its own).
    @app.get("/v1/users/{user_id}/memory")
    async def memory(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        bot, owner = brain(), data_owner(user_id, thread)
        return {
            "short_term": [memory_shape(i) for i in await bot.memory.list_all(owner, scope="short")],
            "long_term": [memory_shape(i) for i in await bot.memory.list_all(owner, scope="long")],
            "pending": [memory_shape(i) for i in await bot.memory.list_all(owner, status="pending")],
        }

    @app.delete("/v1/users/{user_id}/memory/{item_id}")
    async def forget(user_id: str, item_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        if not await brain().memory.forget(data_owner(user_id, thread), item_id):
            raise HTTPException(404, "No such memory")
        return {"ok": True}

    @app.get("/v1/users/{user_id}/reminders")
    async def reminders(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        found = await brain().store.reminder_list(data_owner(user_id, thread), status="upcoming")
        return {"reminders": [{**row, "id": short(row["id"]), "due_at": row["due_at"].isoformat()} for row in found["reminders"]]}

    @app.get("/v1/users/{user_id}/reminders/due")
    async def due(user_id: str) -> dict[str, Any]:
        """Bodies poll this; each due reminder is returned once (and once more after 10 minutes if nobody
        answered). The body says `speak` out loud and offers Done / Snooze."""
        rows = await brain().due_reminders(user_id)
        return {"due": [
            {"id": r["id"], "text": r["text"], "due": r["due"], "repeat": r["repeat"], "again": r["reminded_count"] > 0,
             "thread": r["user_id"].split("/", 1)[1] if "/" in r["user_id"] else "main",
             "all_day": r.get("all_day", False),
             "speak": (f"Today, all day: {r['text']}." if r.get("all_day") else
                       f"{'Reminder again' if r['reminded_count'] > 0 else 'Reminder'}: {r['text']}.")}
            for r in rows
        ]}

    @app.post("/v1/users/{user_id}/reminders/{reminder_id}/done")
    async def reminder_done(user_id: str, reminder_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        if not await brain().store.reminder_done(data_owner(user_id, thread), reminder_id):
            raise HTTPException(404, "No such reminder")
        return {"ok": True}

    @app.post("/v1/users/{user_id}/reminders/{reminder_id}/snooze")
    async def reminder_snooze(
        user_id: str, reminder_id: str, minutes: int = Query(10, ge=1, le=1440), thread: str = Query("main", pattern=THREAD_PATTERN),
    ) -> dict[str, Any]:
        bot = brain()
        tz = ZoneInfo(await bot.store.ensure_user(user_id, bot.settings.timezone))
        due_at = datetime.now(tz) + timedelta(minutes=minutes)
        if not await bot.store.reminder_reschedule(data_owner(user_id, thread), reminder_id, due_at, status="snoozed"):
            raise HTTPException(404, "No such reminder")
        return {"ok": True, "due_at": due_at.isoformat()}

    @app.get("/v1/users/{user_id}/lists")
    async def lists(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        store, owner = brain().store, data_owner(user_id, thread)
        names = [row["name"] for row in (await store.list_show(owner, None))["lists"]]
        shown = {name: (await store.list_show(owner, name))["items"] for name in names}
        tz = ZoneInfo(await store.ensure_user(user_id, brain().settings.timezone))
        todos = await store.todo_list(owner, status="all", today=datetime.now(tz).date())
        return {
            "lists": {name: [{"id": short(i["id"]), "text": i["text"], "done": i["done"]} for i in items] for name, items in shown.items()},
            "todos": [{**t, "id": short(t["id"]), "day": str(t["day"])} for t in todos["todos"]],
        }

    @app.get("/v1/users/{user_id}/notes")
    async def notes(user_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        rows = await brain().store.note_all(data_owner(user_id, thread))
        return {"notes": [{**row, "id": short(row["id"]), "created_at": row["created_at"].isoformat()} for row in rows]}

    @app.patch("/v1/users/{user_id}/items/{kind}/{item_id}")
    async def edit_item(user_id: str, kind: str, item_id: str, body: EditIn, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        """Edit one memory, note, reminder, to-do or list item from the side panels."""
        if not re.match(EDIT_KINDS, kind):
            raise HTTPException(404, "Unknown kind")
        bot, owner = brain(), data_owner(user_id, thread)
        if kind == "memory":
            outcome = await bot.memory.edit(owner, item_id, text=body.text, subject=body.subject, topic=body.topic)
            if outcome == "conflict":
                raise HTTPException(409, "Another memory already has this subject and topic.")
            ok = outcome == "ok"
        elif kind == "list_item":
            ok = await bot.store.list_item_edit(owner, item_id, body.text, body.done)
        else:
            values = body.model_dump()
            if body.when:
                tz = ZoneInfo(await bot.store.ensure_user(user_id, bot.settings.timezone))
                due = datetime.fromisoformat(body.when).replace(tzinfo=tz)
                if body.all_day:
                    # A full-day reminder is said at 9:00 that morning (or soon, if that has passed today).
                    due = max(due.replace(hour=9, minute=0), datetime.now(tz) + timedelta(minutes=1))
                if due < datetime.now(tz) - timedelta(minutes=1):
                    raise HTTPException(422, "That time has already passed.")
                values["due_at"] = due
            ok = await bot.store.edit_row(EDIT_TABLES[kind], owner, item_id, values)
        if not ok:
            raise HTTPException(404, "Nothing to change, or no such item")
        return {"ok": True}

    @app.delete("/v1/users/{user_id}/items/{kind}/{item_id}")
    async def delete_item(user_id: str, kind: str, item_id: str, thread: str = Query("main", pattern=THREAD_PATTERN)) -> dict[str, Any]:
        if not re.match(EDIT_KINDS, kind):
            raise HTTPException(404, "Unknown kind")
        bot, owner = brain(), data_owner(user_id, thread)
        if kind == "memory":
            ok = await bot.memory.forget(owner, item_id)
        elif kind == "list_item":
            ok = await bot.store.list_item_delete(owner, item_id)
        else:
            ok = await bot.store.delete_row(EDIT_TABLES[kind], owner, item_id)
        if not ok:
            raise HTTPException(404, "No such item")
        return {"ok": True}

    @app.post("/v1/users/{user_id}/clear")
    async def clear(user_id: str, body: ClearIn) -> dict[str, Any]:
        """Manual override from the playground panels (the page asks "are you sure?" first)."""
        store, owner = brain().store, data_owner(user_id, body.thread)
        if body.store == "list":
            if not body.list:
                raise HTTPException(422, "Give the list name.")
            done = await store.list_clear(owner, body.list, checked_only=False)
            return {"removed": done.get("removed", 0)}
        return {"removed": await store.store_clear(owner, body.store, body.day)}

    return app
