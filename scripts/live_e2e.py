"""Headless end-to-end check of live voice over the real LiveKit server.

A Python LiveKit client joins as the "user", speaks Piper-generated sentences into the room like a
microphone, and records what comes back: Umbrella's audio, captions, expression and state messages.
It checks the greeting, a normal turn, a pause in the middle of a sentence (one message, not two), a
merge (going on before the reply is heard), a short "hmm" over Umbrella (it keeps talking), barge-in
(talking over a long reply) and the silence nudge.

    .venv/bin/python scripts/live_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import numpy as np
from livekit import rtc

from companion import Companion, Settings
from companion.realtime.sessions import LiveSessions
from companion.voice.audio import read_wav, resample

RATE = 48_000
FRAME = RATE // 100  # 10 ms
T0 = time.monotonic()


def stamp() -> str:
    return f"{time.monotonic() - T0:6.1f}s"


class User:
    def __init__(self) -> None:
        self.events: list[tuple[float, dict]] = []
        self.audio_frames = 0
        self.last_audio_at = 0.0
        self.first_audio_after: float | None = None
        self.source = rtc.AudioSource(RATE, 1)

    def on_data(self, packet: rtc.DataPacket) -> None:
        message = json.loads(packet.data.decode())
        self.events.append((time.monotonic(), message))
        if message["type"] == "caption":
            print(f"{stamp()}  CAPTION {message['role']:9}: {message['text']}")
        else:
            print(f"{stamp()}  {message['type']}: {message.get('state') or message.get('expression') or message}")

    async def listen(self, track: rtc.Track) -> None:
        async for event in rtc.AudioStream(track):
            samples = np.frombuffer(bytes(event.frame.data), dtype=np.int16)
            if np.abs(samples).max(initial=0) > 300:  # actual sound, not silence padding
                self.audio_frames += 1
                self.last_audio_at = time.monotonic()
                if self.first_audio_after is None:
                    self.first_audio_after = time.monotonic()

    async def say(self, pcm: np.ndarray) -> None:
        """Push audio in real time, 10 ms at a time, like a microphone."""
        for start in range(0, len(pcm), FRAME):
            chunk = pcm[start: start + FRAME]
            if len(chunk) < FRAME:
                chunk = np.pad(chunk, (0, FRAME - len(chunk)))
            await self.source.capture_frame(rtc.AudioFrame(chunk.tobytes(), RATE, 1, FRAME))

    async def silence(self, seconds: float) -> None:
        await self.say(np.zeros(int(seconds * RATE), dtype=np.int16))

    def captions(self, role: str) -> list[str]:
        return [m["text"] for _, m in self.events if m["type"] == "caption" and m["role"] == role]


async def speech(bot: Companion, text: str) -> np.ndarray:
    wav, _ = await bot.voice.speak(text)
    samples, rate = read_wav(wav)
    return (resample(samples, rate, RATE) * 32767).astype(np.int16)


async def wait_for(check, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if check():
            return True
        await asyncio.sleep(0.1)
    return False


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format=f"         · %(name)s: %(message)s")
    for noisy in ("companion.db", "livekit", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = Settings(db_schema="companion_live", silence_nudge_seconds=8)
    bot = await Companion.create(settings)
    await asyncio.to_thread(bot.voice.warm_up)
    sessions = LiveSessions(bot)
    results: dict[str, object] = {}

    lines = {
        "turn": await speech(bot, "Hi Umbrella, I am a bit tired today. Can you suggest one quick way to relax?"),
        "long": await speech(bot, "Tell me a long story about a brave little robot who travels across the whole world."),
        "cut": await speech(bot, "Wait wait, stop. Just tell me the time instead."),
        "half": await speech(bot, "Can you remind me to call my mom and"),
        "rest": await speech(bot, "tonight at eight."),
        "milk": await speech(bot, "Add milk to my shopping list."),
        "eggs": await speech(bot, "and eggs too."),
        "hmm": await speech(bot, "Hmm."),
    }

    print(f"{stamp()}  starting session")
    session = await sessions.start("e2e-user")
    user = User()
    room = rtc.Room()
    room.on("data_received", user.on_data)
    room.on("track_subscribed", lambda track, *_: asyncio.ensure_future(user.listen(track)) if track.kind == rtc.TrackKind.KIND_AUDIO else None)
    await room.connect(session["url"], session["token"])
    track = rtc.LocalAudioTrack.create_audio_track("mic", user.source)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    print(f"{stamp()}  user joined {session['room']}")

    # 1. Greeting
    results["greeting"] = await wait_for(lambda: user.captions("assistant"), 25)
    await wait_for(lambda: time.monotonic() - user.last_audio_at > 1.5 and user.audio_frames > 0, 30)
    results["greeting_audio_frames"] = user.audio_frames

    # 2. A normal turn: speak, then stay quiet so VAD ends the turn
    await user.silence(0.5)
    user.first_audio_after = None
    await user.say(lines["turn"])
    spoke_at = time.monotonic()
    await user.silence(1.0)
    results["heard_turn"] = await wait_for(lambda: len(user.captions("user")) >= 1, 20)
    await wait_for(lambda: user.first_audio_after is not None, 25)
    if user.first_audio_after:
        results["reply_audio_latency_s"] = round(user.first_audio_after - spoke_at, 2)
    await wait_for(lambda: len(user.captions("assistant")) >= 2, 25)
    await wait_for(lambda: time.monotonic() - user.last_audio_at > 1.5, 40)

    # 2b. A pause in the middle of a sentence: "...call my mom and" [1.5 s] "tonight at eight."
    await user.silence(0.5)
    before_user, before_bot = len(user.captions("user")), len(user.captions("assistant"))
    await user.say(lines["half"])
    await user.silence(1.5)
    results["mid_pause_no_reply_yet"] = len(user.captions("assistant")) == before_bot
    await user.say(lines["rest"])
    await user.silence(1.0)
    await wait_for(lambda: len(user.captions("user")) > before_user, 20)
    joined = user.captions("user")[before_user:]
    results["mid_pause_one_message"] = len(joined) == 1 and "mom" in joined[0].lower() and "eight" in joined[0].lower()
    await wait_for(lambda: len(user.captions("assistant")) > before_bot, 30)
    await wait_for(lambda: time.monotonic() - user.last_audio_at > 1.5, 40)

    # 2c. Merge: the turn is sent, and the user goes on before hearing the reply
    await user.silence(0.5)
    before_user, before_bot = len(user.captions("user")), len(user.captions("assistant"))
    user.first_audio_after = None
    await user.say(lines["milk"])
    await user.silence(1.6)          # long enough to be sent, too short for a spoken reply
    sent_alone = len(user.captions("user")) > before_user
    heard_reply = user.first_audio_after is not None
    await user.say(lines["eggs"])
    await user.silence(1.0)
    await wait_for(lambda: len(user.captions("assistant")) > before_bot, 30)
    after = user.captions("user")[before_user:]
    results["merge_first_part_was_sent"] = sent_alone and not heard_reply
    results["merge_joined_text"] = after[-1] if after else None
    results["merge_ok"] = bool(after) and "milk" in after[-1].lower() and "eggs" in after[-1].lower()
    await wait_for(lambda: time.monotonic() - user.last_audio_at > 1.5, 40)

    # 3. Barge-in: ask for something long, then talk over the reply
    await user.silence(0.5)
    user.first_audio_after = None
    await user.say(lines["long"])
    await user.silence(1.0)
    await wait_for(lambda: user.first_audio_after is not None, 25)
    await asyncio.sleep(2.0)  # let Umbrella talk for a moment
    # A short "hmm" over Umbrella: it must keep talking.
    await user.say(lines["hmm"])
    await asyncio.sleep(1.5)
    results["short_hmm_umbrella_kept_talking"] = time.monotonic() - user.last_audio_at < 0.5
    talking_before = time.monotonic() - user.last_audio_at < 0.5
    cut_at = time.monotonic()
    await user.say(lines["cut"])
    states_after = [m.get("state") for at, m in user.events if at >= cut_at and m["type"] == "state"]
    results["barge_in_umbrella_was_talking"] = talking_before
    results["barge_in_listening_state"] = "listening" in states_after
    await user.silence(1.0)
    await wait_for(lambda: len(user.captions("user")) >= 3, 25)
    await wait_for(lambda: len(user.captions("assistant")) >= 4, 30)
    await wait_for(lambda: time.monotonic() - user.last_audio_at > 1.5, 40)

    # 4. Silence nudge: say nothing
    before = len(user.captions("assistant"))
    quiet_from = time.monotonic()
    nudged = await wait_for(lambda: len(user.captions("assistant")) > before, settings.silence_nudge_seconds + 20)
    results["nudge"] = nudged
    if nudged:
        results["nudge_after_s"] = round(time.monotonic() - quiet_from, 1)
    # ...and only once
    await asyncio.sleep(settings.silence_nudge_seconds + 4)
    results["nudge_only_once"] = len(user.captions("assistant")) == before + 1

    await room.disconnect()
    await sessions.stop_all()
    await bot.close()
    print("\nRESULTS")
    for key, value in results.items():
        print(f"  {key}: {value}")
    print("  user captions:", user.captions("user"))


if __name__ == "__main__":
    asyncio.run(main())
