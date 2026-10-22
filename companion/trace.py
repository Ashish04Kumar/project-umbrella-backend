"""Live trace of what the brain does on each turn, for the playground's Trace panel.

Every step (context loaded, the model's reasoning, tool chosen, tool result, guard, memory saved,
summary) is published as a small event. Subscribers (the server's live stream) get them as they
happen; the last events per user are kept so a panel opened late still sees the latest turns.
This is for local testing: events contain the user's words, so never expose it publicly.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import defaultdict, deque
from typing import Any

KEEP_PER_USER = 1500   # enough for a 30-turn test run with full LLM requests


class TraceHub:
    def __init__(self) -> None:
        self._recent: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=KEEP_PER_USER))
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._ids = itertools.count(1)

    def emit(self, user_id: str, kind: str, turn: str = "", **data: Any) -> None:
        """Publish one event. Never raises: tracing must not break a reply."""
        try:
            event = {"id": next(self._ids), "at": time.time(), "user_id": user_id, "turn": turn, "kind": kind, "data": data}
            self._recent[user_id].append(event)
            for queue in list(self._subscribers.get(user_id, ())):
                if queue.qsize() < 1000:
                    queue.put_nowait(event)
        except Exception:
            pass

    def recent(self, user_id: str) -> list[dict[str, Any]]:
        return list(self._recent.get(user_id, ()))

    def subscribe(self, user_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[user_id].add(queue)
        return queue

    def unsubscribe(self, user_id: str, queue: asyncio.Queue) -> None:
        self._subscribers.get(user_id, set()).discard(queue)


def clip(value: Any, limit: int = 600) -> Any:
    """Shorten long strings so events stay small."""
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value
