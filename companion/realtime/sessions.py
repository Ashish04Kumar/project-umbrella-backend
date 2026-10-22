"""Start and stop live voice sessions: one LiveKit room per session, the companion joins as an agent.

The LiveKit API secret stays on this server; the body only gets a short-lived room token.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

from livekit import api

from .agent import run_live_session

if TYPE_CHECKING:
    from ..brain import Companion

log = logging.getLogger("companion.live")
STARTUP_SECONDS = 20


class LiveError(RuntimeError):
    """A live session could not start (LiveKit not configured or not reachable)."""


def _token(bot: "Companion", room: str, identity: str, minutes: float) -> str:
    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True, can_publish_data=True)
    return (
        api.AccessToken(bot.settings.livekit_api_key, bot.settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=minutes))
        .to_jwt()
    )


class LiveSessions:
    def __init__(self, bot: "Companion") -> None:
        self.bot = bot
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._rooms_by_user: dict[str, str] = {}

    def configured(self) -> bool:
        s = self.bot.settings
        return bool(s.livekit_url and s.livekit_api_key and s.livekit_api_secret)

    async def start(self, user_id: str) -> dict[str, str]:
        """Start the companion in a fresh room and return what the body needs to join it."""
        if not self.configured():
            raise LiveError("Set COMPANION_LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env")
        old = self._rooms_by_user.get(user_id)
        if old:
            await self.stop(old)  # one live session per user
        safe_user = re.sub(r"[^a-zA-Z0-9_-]", "", user_id)[:24] or "user"
        room = f"companion-{safe_user}-{uuid.uuid4().hex[:8]}"
        minutes = self.bot.settings.live_max_minutes + 5
        agent_token = _token(self.bot, room, f"companion-{uuid.uuid4().hex[:6]}", minutes)
        connected = asyncio.Event()
        task = asyncio.create_task(run_live_session(self.bot, user_id, room_name=room, token=agent_token, on_connected=connected))
        self._tasks[room] = task
        self._rooms_by_user[user_id] = room
        task.add_done_callback(lambda done: self._forget(room, user_id, done))
        waiter = asyncio.create_task(connected.wait())
        try:
            done, _ = await asyncio.wait({waiter, task}, timeout=STARTUP_SECONDS, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        if not connected.is_set():
            await self.stop(room)
            if task.done() and task.exception():
                raise LiveError(f"The companion could not join LiveKit: {task.exception()}")
            raise LiveError("The companion did not join LiveKit in time")
        return {
            "room": room,
            "url": self.bot.settings.livekit_url,
            "token": _token(self.bot, room, f"user-{safe_user}-{uuid.uuid4().hex[:4]}", minutes),
        }

    def _forget(self, room: str, user_id: str, task: asyncio.Task) -> None:
        if self._tasks.get(room) is task:
            self._tasks.pop(room, None)
        if self._rooms_by_user.get(user_id) == room:
            self._rooms_by_user.pop(user_id, None)
        if not task.cancelled() and task.exception():
            log.error("live session %s failed: %s", room, task.exception())

    async def stop(self, room: str) -> bool:
        task = self._tasks.pop(room, None)
        if task is None:
            return False
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return True

    async def stop_all(self) -> None:
        for room in list(self._tasks):
            await self.stop(room)

    def active(self) -> list[str]:
        return [room for room, task in self._tasks.items() if not task.done()]
