"""Live voice with the companion, over LiveKit (WebRTC). Ported from Synq Pod 4's voice agent.

    Body mic -> LiveKit -> Silero VAD -> VadTap -> STT -> LiveTurns -> TTS -> LiveKit -> Body speaker

LiveTurns does what Pod 4's turn processor does, with the companion brain instead of the call
Conductor, plus smarter turn taking (rules in turn_policy.py):
- turn taking: answer only when the user stopped talking and every Whisper piece has arrived; how long
  to wait depends on Smart Turn v3 (does the voice sound finished?) and the words ("…and" = not done);
- merge: if the user goes on talking before hearing the reply, that reply is dropped and both parts
  become one message;
- smart barge-in: talking over the companion stops it only after half a second of speech (not for a
  cough or "hmm"); "wait"/"ruko" stops it at once; "ok"/"haan" said over it is not a new turn;
- STT guard: text is matched to the stretch of speech it came from, made-up words are dropped;
- voice cues: pauses, pace, loudness, intonation, fillers -> a soft hint for the LLM;
- streaming: each sentence goes to TTS as soon as the LLM has written it;
- barge-in: when the user talks over the companion, its audio stops and the rest of the reply is dropped;
- one gentle silence nudge per user turn, grounded in what they last said;
- a greeting when the user joins, support resources if they sound in danger, a time limit.
It also sends the body small JSON messages: caption_partial (the user's words heard so far in this
turn, for the input bar), caption_delta (Umbrella's words as they are written),
captions (the full turn), the reply's expression and the state
(listening / thinking / speaking / idle) so a face can react.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import TYPE_CHECKING

import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from ..guard import SUPPORT_RESOURCES, sounds_like_crisis
from ..modes import LIVE, InputMode
from ..voice.cues import VoiceCues, VoiceCueTracker
from ..voice.guard import BLIP_SECONDS, PRE_ROLL_SECONDS, Segment, check_with_segments
from .services import CompanionSTTService, CompanionTTSService
from .turn_policy import BARGE_IN_SECONDS, MAX_WAIT, MERGE_WINDOW, SmartTurn, end_of_turn_wait, is_backchannel, is_stop_request

if TYPE_CHECKING:
    from ..brain import Companion

log = logging.getLogger("companion.live")

# Same timings as Synq Pod 4. Whisper sends one transcript per stretch of speech, each 1-3.5 s after
# it ended, so one answer can arrive in pieces. Answer only when the user stopped talking, the last
# piece is at least UTTERANCE_GAP old, and no real stretch is still waiting for its text.
UTTERANCE_GAP_SECONDS = 0.6
STT_WAIT_MAX_SECONDS = 4.0
END_SILENCE_SECONDS = 0.7   # used only when Smart Turn is off (see turn_policy.end_of_turn_wait)
TURN_AUDIO_SECONDS = 8      # Smart Turn hears the last 8 s of the user's voice
SMART_TURN_WAIT = 0.6       # how long to wait for Smart Turn's verdict before deciding without it

GREETING_CUE = (
    "They just joined a voice chat with you. Greet them warmly and briefly, by name if you know it. If something "
    "is coming up for them (see what you know), you may ask about it; otherwise ask one easy question about their day."
)
NUDGE_CUE = (
    "They have been quiet for a while after your reply. Warmly invite them to share more about what they last said, "
    "with one easy question, using only their real words. Don't mention the silence or the waiting."
)
GOODBYE_CUE = "The voice chat time is up. Say a short, warm goodbye in one or two sentences. Do not ask a question."


class LiveTurns(FrameProcessor):
    def __init__(self, bot: "Companion", user_id: str, transport: LiveKitTransport, mode: InputMode = LIVE, smart_turn: bool = True) -> None:
        super().__init__()
        self._bot = bot
        self._mode = mode  # mode.voice_cues switches the voice cues on or off (companion/modes.py)
        self._user_id = user_id
        self._transport = transport
        self._lock = asyncio.Lock()
        self._bot_speaking = False
        self._user_speaking = False
        self._pending: list[str] = []
        self._answer_task: asyncio.Task | None = None
        self._last_piece_at = 0.0
        # Finished stretches of user speech still waiting for their STT text, oldest first.
        self._segment_started_at: float | None = None
        self._segment_energy = 0.0
        self._segment_samples = 0
        self._segments: list[Segment] = []
        self._last_speech_end = 0.0
        # Recent mic energy per 20 ms frame (time, energy, samples) for the pre-roll loudness.
        self._recent_audio: deque[tuple[float, float, int]] = deque(maxlen=150)
        self._nudge_handle: asyncio.TimerHandle | None = None
        self._nudged_since_user = False
        self._cues = VoiceCueTracker()
        if mode.voice_cues and hasattr(bot, "cue_baseline"):
            self._cues._baseline = bot.cue_baseline(user_id)  # the same "usual" as push-to-talk with cues
        self._cut_in = False       # the user started talking: a streaming reply stops sending sentences
        self.safety_mode = False   # after crisis words: no small-talk nudges
        self.closing = False       # the goodbye has started: no more nudges
        self.last_expression, self.last_cut, self.last_sentences = "idle", False, []  # the latest reply, for the trace
        # Smart turn taking (turn_policy.py).
        self._smart = SmartTurn() if smart_turn else None
        self._turn_audio: deque[np.ndarray] = deque()   # recent mic audio at 16 kHz, for Smart Turn
        self._turn_audio_samples = 0
        self._turn_probability: float | None = None     # Smart Turn's verdict for the latest pause
        self._probability_pending = False
        self._probability_seq = 0
        self._waiting_reason = ""                       # last "still waiting" reason sent to the Live log
        self._segment_over_bot = False                  # the current stretch of speech began while Umbrella talked
        self._over_bot: list[bool] = []                 # same order as self._segments
        self._barge_handle: asyncio.TimerHandle | None = None
        self._inflight_task: asyncio.Task | None = None # the reply being prepared or spoken
        self._inflight_text = ""
        self._inflight_turn = ""
        self._reply_audible = False                     # its audio has started playing
        self._inflight_sent_at = 0.0
        self._merging = False

    # --- messages to the body ----------------------------------------------------------------

    async def _send(self, payload: dict) -> None:
        try:
            await self._transport.send_message(json.dumps(payload, ensure_ascii=False))
        except Exception as exc:  # a dropped data message must never break the call
            log.debug("could not send %s: %s", payload.get("type"), exc)

    async def _state(self, state: str) -> None:
        await self._send({"type": "state", "state": state})

    def _decide(self, decision: str, reason: str, **data) -> None:
        """One turn-taking decision, for the playground's Live log (trace event "live_turn") and the log file."""
        log.info("turn taking: %s (%s)", decision, reason)
        trace = getattr(self._bot, "trace", None)
        if trace is not None:
            trace.emit(self._user_id, "live_turn", "", decision=decision, reason=reason, **data)

    # --- speaking ----------------------------------------------------------------------------------

    async def _speak(self, events: AsyncIterator[dict]) -> str:
        """Send each streamed sentence to TTS as it arrives; stop if the user starts talking.
        Returns what was handed to TTS."""
        self._cut_in = False
        said: list[str] = []
        self.last_expression, self.last_cut = "idle", False
        self.last_usage: dict = {}   # LLM tokens of this reply (none when the user talked over it before the end)
        async with aclosing(events) as stream:
            async for event in stream:
                if self._cut_in:
                    log.info("user talked over the reply: rest dropped")
                    self.last_cut = True
                    break
                if event["type"] == "delta":
                    # The words as the model writes them, so the body can show them at once.
                    await self._send({"type": "caption_delta", "role": "assistant", "text": event["text"]})
                elif event["type"] == "expression":
                    self.last_expression = event["expression"]
                    await self._send({"type": "expression", "expression": event["expression"]})
                elif event["type"] == "sentence":
                    said.append(event["text"])
                    await self.push_frame(TTSSpeakFrame(event["text"]), FrameDirection.DOWNSTREAM)
                elif event["type"] == "done" and event.get("reply") is not None:
                    self.last_usage = event["reply"].usage
        self.last_sentences = said
        reply = " ".join(said)
        if reply:
            await self._send({"type": "caption", "role": "assistant", "text": reply})
        return reply

    async def _log_objects(self, turn: str, request: dict, reply: str, started: float) -> None:
        """Request and response objects of one live turn: kept in the trace and sent to the body
        (a `turn_objects` message), so a chat can show what went to the backend and what came back."""
        request_obj = {"mode": self._mode.name, "turn": turn, **request}
        response_obj = {
            "turn": turn, "reply": {"text": reply, "expression": self.last_expression},
            "spoken_sentences": self.last_sentences, "cut_by_barge_in": self.last_cut, "usage": getattr(self, "last_usage", {}),
            "data_messages_to_body": ["state", "expression", "caption_delta", "caption", "turn_objects"],
            "ms": round((time.monotonic() - started) * 1000),
        }
        trace = getattr(self._bot, "trace", None)
        if trace is not None:
            trace.emit(self._user_id, "request", turn, object=request_obj)
            trace.emit(self._user_id, "response", turn, object=response_obj)
        await self._send({"type": "turn_objects", "request": request_obj, "response": response_obj})

    async def say_cue(self, cue: str) -> str:
        """Greeting, nudge or goodbye: a reply with no new words from the user. Call with the lock held."""
        await self._state("thinking")
        self._reply_audible = False
        turn, started = uuid.uuid4().hex[:8], time.monotonic()
        reply = await self._speak(self._bot.chat_stream(self._user_id, cue=cue, turn=turn))
        await self._log_objects(turn, {"text": None, "cue": cue, "transport": "LiveKit"}, reply, started)
        return reply

    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # --- silence nudge --------------------------------------------------------------------------

    def _cancel_nudge(self) -> None:
        if self._nudge_handle is not None:
            self._nudge_handle.cancel()
        self._nudge_handle = None

    def _start_nudge_timer(self) -> None:
        if self._nudged_since_user or self.safety_mode or self.closing:
            return
        self._cancel_nudge()
        # A timer, not a sleeping task (Pod 4 learned this: no "never awaited" warnings when cancelled).
        self._nudge_handle = asyncio.get_running_loop().call_later(self._bot.settings.silence_nudge_seconds, self._fire_nudge)

    def _fire_nudge(self) -> None:
        self._nudge_handle = None
        if self._bot_speaking or self._user_speaking:
            return
        self.create_task(self._nudge())

    async def _nudge(self) -> None:
        if self._lock.locked() or self.safety_mode or self.closing:
            return
        self._nudged_since_user = True
        log.info("silence nudge")
        async with self._lock:
            await self.say_cue(NUDGE_CUE)

    # --- frames ----------------------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._reply_audible = True
            self._cues.bot_started(now)
            self._cancel_nudge()
            await self._state("speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._cues.bot_stopped(now)
            self._start_nudge_timer()
            if not self._lock.locked():
                await self._state("idle")
        if isinstance(frame, TranscriptionFrame):
            self._on_transcript(frame.text.strip(), now)
            return
        await self.push_frame(frame, direction)

    # --- user speech, called by VadTap in real time ------------------------------------------------

    def on_mic_audio(self, pcm: bytes, sample_rate: int, now: float) -> None:
        if self._mode.voice_cues:
            self._cues.audio(pcm, sample_rate)
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        energy = float(np.sum(samples * samples))
        self._recent_audio.append((now, energy, len(samples)))
        if self._smart is not None:
            mono = samples if sample_rate == 16000 else _resample(samples, sample_rate)
            self._turn_audio.append(mono)
            self._turn_audio_samples += len(mono)
            while self._turn_audio and self._turn_audio_samples - len(self._turn_audio[0]) >= TURN_AUDIO_SECONDS * 16000:
                self._turn_audio_samples -= len(self._turn_audio.popleft())
        if self._user_speaking:
            self._segment_energy += energy
            self._segment_samples += len(samples)

    def _loudness_since(self, start: float) -> float | None:
        energy = samples = 0
        for at, frame_energy, frame_samples in self._recent_audio:
            if at >= start:
                energy += frame_energy
                samples += frame_samples
        return 10 * float(np.log10(max(energy / samples, 1e-12))) if samples else None

    async def on_user_started(self, now: float) -> None:
        self._user_speaking = True
        self._segment_started_at = now
        self._segment_energy, self._segment_samples = 0.0, 0
        self._segment_over_bot = self._bot_speaking
        self._probability_seq += 1          # a verdict about the previous pause no longer counts
        self._turn_probability, self._probability_pending = None, False
        self._cues.user_started(now)
        self._cancel_nudge()
        await self._state("listening")
        if self._bot_speaking:
            # Smart barge-in: stop the companion only if this lasts BARGE_IN_SECONDS (not a cough or "hmm").
            # (Earphones keep its own voice from triggering this.)
            self._cancel_barge_in()
            self._barge_handle = asyncio.get_running_loop().call_later(
                BARGE_IN_SECONDS, lambda: self.create_task(self._confirm_barge_in(now)))
        elif self._inflight_task is not None and not self._inflight_task.done() and not self._reply_audible:
            if now - self._inflight_sent_at <= MERGE_WINDOW:
                await self._merge()
            else:
                # Long after it was sent (a slow LLM): a new message, not the rest of the old one. The old reply
                # is not spoken over them; its question stays in the history for the next answer.
                self._cut_in = True
                self._decide("reply dropped", f"they started a new message {now - self._inflight_sent_at:.0f}s after the last one, before its reply")
        elif self._lock.locked() and not self._reply_audible:
            self._cut_in = True   # a greeting or nudge not heard yet: drop it, they are talking

    def _show_heard(self) -> None:
        """What the user has said so far in this turn, for the body's input bar (cleared by the user caption)."""
        self.create_task(self._send({"type": "caption_partial", "role": "user", "text": " ".join(self._pending)}))

    def _cancel_barge_in(self) -> None:
        if self._barge_handle is not None:
            self._barge_handle.cancel()
            self._barge_handle = None

    async def _confirm_barge_in(self, started: float) -> None:
        self._barge_handle = None
        if self._user_speaking and self._segment_started_at == started and (self._bot_speaking or self._lock.locked()):
            await self._interrupt("barge-in", f"talked over Umbrella for {BARGE_IN_SECONDS:g}s")

    async def _interrupt(self, decision: str, reason: str) -> None:
        self._cut_in = True
        self._decide(decision, reason)
        await self.broadcast_interruption()

    async def _merge(self) -> None:
        """The user went on talking before hearing the reply: drop that reply, join both parts into one turn."""
        self._merging = True
        self._cut_in = True
        if hasattr(self._bot, "discard_turn"):
            self._bot.discard_turn(self._inflight_turn)
        self._pending.insert(0, self._inflight_text)
        self._show_heard()
        self._decide("merge", "they went on talking before hearing the reply: both parts become one message",
                     text=self._inflight_text)
        self._inflight_task.cancel()
        await self.broadcast_interruption()   # drop any sentence already queued for the voice

    def on_user_stopped(self, now: float) -> None:
        self._user_speaking = False
        self._cues.user_stopped(now)
        if self._barge_handle is not None:
            self._cancel_barge_in()
            self._decide("barge-in ignored", "a short sound over Umbrella (under half a second)")
        if self._smart is not None and self._turn_audio:
            self._probability_seq += 1
            self._probability_pending = True
            self.create_task(self._judge_turn(self._probability_seq, np.concatenate(list(self._turn_audio))))
        if self._segment_started_at is not None:
            loudness = (
                10 * float(np.log10(max(self._segment_energy / self._segment_samples, 1e-12)))
                if self._segment_samples else None
            )
            pre_roll = self._loudness_since(self._segment_started_at - PRE_ROLL_SECONDS)
            self._segments.append(Segment(now, now - self._segment_started_at, loudness, pre_roll))
            self._over_bot.append(self._segment_over_bot)
            self._segment_started_at = None
        self._last_speech_end = now
        self._ensure_answer_loop()

    # --- turn taking --------------------------------------------------------------------------------

    async def _judge_turn(self, seq: int, audio: np.ndarray) -> None:
        started = time.monotonic()
        probability = await asyncio.to_thread(self._smart.probability, audio)
        if seq != self._probability_seq:
            return   # they spoke again meanwhile
        self._turn_probability, self._probability_pending = probability, False
        if probability is not None:
            self._decide("smart turn", f"finished-ness {probability:.2f}", probability=round(probability, 2),
                         ms=round((time.monotonic() - started) * 1000))

    def _on_transcript(self, text: str, now: float) -> None:
        keep, reason, segment, used = check_with_segments(text, self._segments)
        del self._segments[:used]
        over_bot = any(self._over_bot[: max(used, 1)]) if self._over_bot else False
        del self._over_bot[:used]
        if not keep:
            log.info("STT piece dropped %r: %s", text, reason)
            self._ensure_answer_loop()
            return
        if over_bot and is_stop_request(text) and len(text.split()) <= 3:
            # "wait", "ruko": stop talking and listen; nothing to answer yet.
            if self._bot_speaking or self._lock.locked():
                self.create_task(self._interrupt("stopped on request", f"they said {text!r}"))
            self._ensure_answer_loop()
            return
        if over_bot and is_backchannel(text) and not self._cut_in:
            self._decide("backchannel ignored", f"{text!r} said over Umbrella is not a new turn")
            self._ensure_answer_loop()
            return
        log.info("heard %r (%s)", text, f"{segment.seconds:.1f}s of speech" if segment else "no timing")
        self._pending.append(text)
        self._show_heard()
        self._last_piece_at = now
        self._cancel_nudge()
        self._ensure_answer_loop()

    def _stt_still_working(self, now: float) -> bool:
        self._segments = [segment for segment in self._segments if now - segment.ended_at < STT_WAIT_MAX_SECONDS]
        return any(segment.seconds >= BLIP_SECONDS for segment in self._segments)

    def _ready_to_answer(self, now: float) -> bool:
        if not self._pending or self._user_speaking or self._stt_still_working(now):
            return False
        silence = now - self._last_speech_end
        if self._smart is None:
            ready = now - self._last_piece_at >= UTTERANCE_GAP_SECONDS and silence >= END_SILENCE_SECONDS
            self._send_reason = "fixed pause (Smart Turn off)"
            return ready
        if self._probability_pending and silence < SMART_TURN_WAIT:
            return False   # Smart Turn is still listening (about 0.2 s)
        wait, reason = end_of_turn_wait(self._turn_probability, " ".join(self._pending))
        if silence >= MAX_WAIT:
            wait, reason = 0.0, f"waited the longest allowed ({MAX_WAIT:g}s)"
        ready = silence >= wait and now - self._last_piece_at >= min(UTTERANCE_GAP_SECONDS, wait)
        if not ready and reason != self._waiting_reason and wait > 0.5:
            self._waiting_reason = reason
            self._decide("waiting", reason, wait_s=wait, text=" ".join(self._pending))
        self._send_reason = f"{reason}; {silence:.1f}s pause"
        return ready

    def _ensure_answer_loop(self) -> None:
        if self._answer_task is None or self._answer_task.done():
            self._answer_task = self.create_task(self._answer_loop())

    async def _answer_loop(self) -> None:
        """One loop answers turns in order; new pieces of text never cancel it."""
        answered = False
        while True:
            await asyncio.sleep(0.15)
            now = time.monotonic()
            if not self._pending:
                if self._user_speaking or self._stt_still_working(now):
                    continue
                # A sound with no usable words (a cough, "hmm"): count it as their turn, re-arm the nudge.
                if not answered and not self._bot_speaking and not self._lock.locked():
                    self._cues.discard_turn()
                    self._nudged_since_user = False
                    self._start_nudge_timer()
                return
            if not self._ready_to_answer(now):
                continue
            await self._answer_pending()
            answered = True

    async def _answer_pending(self) -> None:
        text = " ".join(self._pending)
        self._pending.clear()
        self._waiting_reason = ""
        self._decide("sent", getattr(self, "_send_reason", ""), text=text)
        snapshot = self._cues.take_turn(text, time.monotonic())  # always taken, so the next turn starts clean
        cues: VoiceCues | None = None
        if self._mode.voice_cues:
            try:
                cues = await asyncio.to_thread(snapshot.measure)
            except Exception:
                log.exception("voice cues failed")
        # The reply runs as its own task so a merge can cancel it (the user went on talking).
        self._merging = False
        self._inflight_text, self._reply_audible = text, False
        self._inflight_sent_at = time.monotonic()
        self._inflight_task = asyncio.get_running_loop().create_task(self._answer(text, cues))
        try:
            await self._inflight_task
        except asyncio.CancelledError:
            if not self._merging:
                raise
            self._merging = False   # merged: the joined text is pending again and the loop goes on

    async def _answer(self, text: str, cues: VoiceCues | None) -> None:
        self._cancel_nudge()
        self._nudged_since_user = False
        await self._send({"type": "caption", "role": "user", "text": text})
        voice_note = cues.describe() if cues else ""
        if cues:
            log.info("voice cues: %s | note: %s", cues.log_fields(), voice_note or "-")
        if sounds_like_crisis(text) and not self.safety_mode:
            self.safety_mode = True
            await self._send({"type": "safety_resources", "resources": SUPPORT_RESOURCES})
        started = time.monotonic()
        async with self._lock:
            await self._state("thinking")
            fields = {**cues.log_fields(), "pace": cues.pace, "volume": cues.volume, "intonation": cues.intonation} if cues else None
            turn = uuid.uuid4().hex[:8]
            self._inflight_turn = turn
            reply = await self._speak(self._bot.chat_stream(self._user_id, text, voice_note=voice_note, voice_cues=fields, turn=turn))
        await self._log_objects(turn, {
            "text": text, "transport": "LiveKit", "stt": "pieces joined after the turn-taking wait",
            "voice_cues": fields, "voice_note_for_llm": voice_note, "safety_mode": self.safety_mode,
        }, reply, started)
        log.info("turn answered in %d ms: %r", (time.monotonic() - started) * 1000, reply)


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    import soxr   # installed with pipecat; only needed if the mic is not at 16 kHz

    return soxr.resample(samples, rate, 16000)


class VadTap(FrameProcessor):
    """Between VAD and STT: passes speech events and mic audio to LiveTurns at once.

    Whisper transcribes inside its frame handling, so VAD frames behind it arrive up to a second
    late. Watching them here keeps "is the user talking right now?" accurate for turn taking,
    barge-in and the voice cues, with any STT provider.
    """

    def __init__(self, turns: LiveTurns) -> None:
        super().__init__()
        self._turns = turns

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()
        if isinstance(frame, InputAudioRawFrame) and frame.num_channels == 1:
            self._turns.on_mic_audio(frame.audio, frame.sample_rate, now)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._turns.on_user_started(now)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turns.on_user_stopped(now)
        await self.push_frame(frame, direction)


async def run_live_session(
    bot: "Companion",
    user_id: str,
    *,
    room_name: str,
    token: str,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    """Join `room_name` as the companion and talk until the user leaves or time is up."""
    settings = bot.settings
    transport = LiveKitTransport(
        url=settings.livekit_url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    turns = LiveTurns(bot, user_id, transport, LIVE)
    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            VadTap(turns),
            CompanionSTTService(bot.voice),
            turns,
            CompanionTTSService(bot.voice),
            transport.output(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True),
        conversation_id=room_name,
    )
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)
    log.info("live session starting | user=%s | room=%s | stt=%s | tts=%s", user_id, room_name,
             [p.name for p in bot.voice.stt], [p.name for p in bot.voice.tts])

    @transport.event_handler("on_connected")
    async def connected(_: LiveKitTransport) -> None:
        log.info("companion joined room %s", room_name)
        if on_connected:
            on_connected.set()

    @transport.event_handler("on_first_participant_joined")
    async def greet(_: LiveKitTransport, __: str) -> None:
        log.info("user joined room %s", room_name)
        async with turns._lock:
            await turns.say_cue(GREETING_CUE)

    @transport.event_handler("on_participant_disconnected")
    async def left(_: LiveKitTransport, participant_id: str) -> None:
        log.info("user left room %s", room_name)
        await runner.cancel()

    async def goodbye_on_time() -> None:
        await asyncio.sleep(settings.live_max_minutes * 60)
        turns.closing = True
        for _ in range(15):  # let the current sentence finish first
            if not turns.bot_speaking and not turns.busy:
                break
            await asyncio.sleep(1)
        async with turns._lock:
            await turns.say_cue(GOODBYE_CUE)
        await asyncio.sleep(8)  # time to hear the goodbye
        await runner.cancel()

    goodbye_task = asyncio.create_task(goodbye_on_time())
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        await runner.cancel()
        raise
    except Exception as exc:
        if on_error:
            on_error(exc)
        raise
    finally:
        goodbye_task.cancel()
        await runner.cleanup()
        log.info("live session ended | room=%s", room_name)
