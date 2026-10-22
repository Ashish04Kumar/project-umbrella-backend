"""MemoryStore kept in a Python dict: for tests and quick embedded runs. Lost on restart.
Search here is simple word overlap; the Postgres adapter does the real ranking."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from ..search import clean_words
from .interface import MemoryItem, key_for


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _alive(item: MemoryItem) -> bool:
    return item.expires_at is None or item.expires_at > _now()


class InMemoryMemory:
    def __init__(self) -> None:
        self._items: dict[str, dict[str, MemoryItem]] = {}

    def _user(self, user_id: str) -> dict[str, MemoryItem]:
        return self._items.setdefault(user_id, {})

    async def remember(self, user_id: str, item: MemoryItem) -> str:
        key = key_for(item)
        for existing in self._user(user_id).values():
            if existing.key == key and existing.status == item.status:
                await self.update(user_id, existing.id or "", item.model_copy(update={"key": key}))
                return existing.id or ""
        item_id = str(uuid.uuid4())
        self._user(user_id)[item_id] = item.model_copy(update={"id": item_id, "key": key, "created_at": _now(), "updated_at": _now()})
        return item_id

    async def search(
        self, user_id: str, *, query: str | None = None, subject: str | None = None, category: str | None = None,
        entities: list[str] | None = None, date_from: date | None = None, date_to: date | None = None,
        month: int | None = None, scope: str | None = None, limit: int = 8,
    ) -> list[MemoryItem]:
        words = clean_words(query)
        people = {name.lower() for name in entities or []}
        found = []
        for item in self._user(user_id).values():
            if item.status != "active" or not _alive(item):
                continue
            if (scope and item.scope != scope) or (category and item.category != category):
                continue
            if subject and subject.lower() not in (item.subject.lower(), *[e.lower() for e in item.entities]) and subject.lower() not in item.text.lower():
                continue
            if people and not people & {item.subject.lower(), *[e.lower() for e in item.entities]}:
                continue
            day = item.event_date or (item.created_at.date() if item.created_at else None)
            if (date_from and (not day or day < date_from)) or (date_to and (not day or day > date_to)):
                continue
            if month and (not item.event_date or item.event_date.month != month):
                continue
            haystack = " ".join([item.subject, item.topic, item.text, *item.keywords, *item.entities]).lower()
            score = sum(1 for word in words if word in haystack)
            if words and not score:
                continue
            found.append((score, item.updated_at or _now(), item))
        found.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [item.model_copy(update={"score": float(score)}) for score, _, item in found[:limit]]

    async def recall(self, user_id: str, query: str, limit: int = 8) -> list[MemoryItem]:
        return await self.search(user_id, query=query, limit=limit)

    async def update(self, user_id: str, item_id: str, item: MemoryItem) -> None:
        old = self._user(user_id).get(item_id)
        if old is not None:
            self._user(user_id)[item_id] = item.model_copy(
                update={"id": item_id, "key": key_for(item), "created_at": old.created_at, "updated_at": _now()}
            )

    async def forget(self, user_id: str, item_id: str) -> bool:
        for existing_id in list(self._user(user_id)):
            if existing_id.startswith(item_id.strip()):
                del self._user(user_id)[existing_id]
                return True
        return False

    async def list_all(
        self, user_id: str, kind: str | None = None, status: str = "active", *, scope: str | None = None,
        pinned: bool | None = None,
    ) -> list[MemoryItem]:
        items = [
            item for item in self._user(user_id).values()
            if item.status == status and _alive(item) and (kind is None or item.category == kind)
            and (scope is None or item.scope == scope) and (pinned is None or item.pinned == pinned)
        ]
        return sorted(items, key=lambda item: item.updated_at or _now(), reverse=True)

    async def clear(self, user_id: str, scope: str) -> int:
        doomed = [item_id for item_id, item in self._user(user_id).items() if item.scope == scope]
        for item_id in doomed:
            del self._user(user_id)[item_id]
        return len(doomed)
