"""The memory interface. Any backend (local Postgres, in-memory, the owner's cloud memory later)
implements MemoryStore, and the rest of the brain only talks to this interface.

Two scopes (docs/personal-data-spec.md):
- short: things for the next few days ("parson Rahul ko 2500 diye"); they expire on their own.
- long: lasting facts, organised by category, subject and topic ("Riya / birthday: 3 May").
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Scope = Literal["short", "long"]
# Long-term categories. "event" is only used by the old automatic extractor for short-term items.
CATEGORIES = ("profile", "person", "date", "preference", "project", "other", "event")


class MemoryItem(BaseModel):
    id: str | None = None
    scope: Scope = "long"
    category: str = "other"
    subject: str = ""             # who or what it is about: "Riya", "me", "Project Synq"
    topic: str = ""               # short: "birthday", "city", "job"
    text: str                     # one plain sentence: "Sister, birthday 3 May"
    key: str = ""                 # filled by key_for(); same key = update, never a duplicate
    event_date: date | None = None   # the day it is about (short) or a date fact (long)
    yearly: bool = False          # birthdays, anniversaries: the year does not matter
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    pinned: bool = False          # profile facts that go with every message
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    status: Literal["active", "pending"] = "active"   # pending = Buddy should ask before keeping it
    sensitive: bool = False
    expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    score: float | None = None    # set by search()


def normalize_key(key: str) -> str:
    """'Person: Riya ' -> 'person:riya' so the same thing always gets the same key."""
    key = key.strip().lower()
    key = re.sub(r"\s*:\s*", ":", key)
    return re.sub(r"\s+", "_", key)[:80]


def key_for(item: MemoryItem) -> str:
    """Long-term: category:subject:topic (an update replaces the old fact).
    Short-term: a hash of the text (saying the same thing twice does not duplicate it)."""
    if item.key:
        return normalize_key(item.key)
    if item.scope == "short":
        return "short:" + hashlib.sha1(" ".join(item.text.lower().split()).encode()).hexdigest()[:12]
    return normalize_key(f"{item.category}:{item.subject or 'me'}:{item.topic or 'general'}")


def search_words(query: str) -> list[str]:
    """Words worth searching for (kept for older callers; see companion.search.clean_words)."""
    from ..search import clean_words

    return clean_words(query)


class MemoryStore(Protocol):
    async def remember(self, user_id: str, item: MemoryItem) -> str:
        """Save an item; an item with the same key and status is updated instead. Returns its id."""
        ...

    async def search(
        self, user_id: str, *, query: str | None = None, subject: str | None = None, category: str | None = None,
        entities: list[str] | None = None, date_from: date | None = None, date_to: date | None = None,
        month: int | None = None, scope: str | None = None, limit: int = 8,
    ) -> list[MemoryItem]:
        """Active, unexpired items matching the filters, best first (item.score set)."""
        ...

    async def recall(self, user_id: str, query: str, limit: int = 8) -> list[MemoryItem]:
        """Shortcut for search(query=...)."""
        ...

    async def update(self, user_id: str, item_id: str, item: MemoryItem) -> None: ...

    async def forget(self, user_id: str, item_id: str) -> bool:
        """Delete one item (id or its first 8 characters). Returns False if not found."""
        ...

    async def list_all(
        self, user_id: str, kind: str | None = None, status: str = "active", *, scope: str | None = None,
        pinned: bool | None = None,
    ) -> list[MemoryItem]:
        """Unexpired items, newest first. `kind` filters by category (name kept for older callers)."""
        ...

    async def clear(self, user_id: str, scope: str) -> int:
        """Delete every item of one scope. Returns how many were deleted."""
        ...
