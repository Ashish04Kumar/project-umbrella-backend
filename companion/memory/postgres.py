"""Local MemoryStore on Postgres (table `memories`).

Search = SQL filters, then full-text (weighted: subject/topic > keywords/entities > text) and fuzzy
trigram matching, fused with RRF (companion/search.py). See docs/search-and-retrieval-spec.md.
"""

from __future__ import annotations

import json
from datetime import date

import asyncpg

from ..search import clean_words, fuse, to_tsquery
from ..store.audit import arrow, describe, record
from .interface import MemoryItem, key_for

_COLUMNS = (
    "id::text AS id, scope, category, subject, topic, text, key, event_date, yearly, keywords, entities, pinned, "
    "data, confidence, status, sensitive, expires_at, created_at, updated_at"
)
_ALIVE = "(expires_at IS NULL OR expires_at > now())"


def _item(row: asyncpg.Record, score: float | None = None) -> MemoryItem:
    values = {key: row[key] for key in row.keys() if key not in ("fts", "fuzzy", "boost")}
    values["data"] = json.loads(values["data"]) if isinstance(values["data"], str) else values["data"]
    return MemoryItem(**values, score=score)


class PostgresMemory:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def remember(self, user_id: str, item: MemoryItem) -> str:
        old = await self.pool.fetchrow(
            "SELECT text FROM memories WHERE user_id = $1 AND key = $2 AND status = $3", user_id, key_for(item), item.status
        )
        row = await self.pool.fetchrow(
            """
            INSERT INTO memories (user_id, kind, key, text, data, confidence, status, sensitive, expires_at,
                                  scope, category, subject, topic, event_date, yearly, keywords, entities, pinned)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            ON CONFLICT (user_id, key, status) DO UPDATE SET
                kind = EXCLUDED.kind, text = EXCLUDED.text, data = EXCLUDED.data, confidence = EXCLUDED.confidence,
                sensitive = EXCLUDED.sensitive, expires_at = EXCLUDED.expires_at, scope = EXCLUDED.scope,
                category = EXCLUDED.category, subject = EXCLUDED.subject, topic = EXCLUDED.topic,
                event_date = EXCLUDED.event_date, yearly = EXCLUDED.yearly, keywords = EXCLUDED.keywords,
                entities = EXCLUDED.entities, pinned = EXCLUDED.pinned, updated_at = now()
            RETURNING id::text
            """,
            user_id, item.category, key_for(item), item.text.strip(), json.dumps(item.data, ensure_ascii=False),
            item.confidence, item.status, item.sensitive, item.expires_at, item.scope, item.category,
            item.subject.strip(), item.topic.strip(), item.event_date, item.yearly,
            [k.strip().lower() for k in item.keywords if k.strip()], [e.strip() for e in item.entities if e.strip()],
            item.pinned,
        )
        where = f"{item.scope}-term · {item.category}" + (f" · {item.subject} {item.topic}".rstrip() if item.subject else "")
        if old:
            record("UPDATE", "memories", row["id"], where, arrow("text", repr(old["text"]), repr(item.text.strip())))
        else:
            record("INSERT", "memories", row["id"], where, f"text {item.text.strip()!r}"
                   + (f", expires {item.expires_at.date()}" if item.expires_at else "") + (", pinned" if item.pinned else ""))
        return row["id"]

    async def search(
        self, user_id: str, *, query: str | None = None, subject: str | None = None, category: str | None = None,
        entities: list[str] | None = None, date_from: date | None = None, date_to: date | None = None,
        month: int | None = None, scope: str | None = None, limit: int = 8,
    ) -> list[MemoryItem]:
        words = clean_words(query)
        tsquery, fuzzy_text = to_tsquery(words), " ".join(words)
        people = [name.strip().lower() for name in (entities or []) if name.strip()] or None
        rows = await self.pool.fetch(
            f"""
            SELECT {_COLUMNS},
                CASE WHEN $2 = '' THEN 0 ELSE ts_rank_cd(search, to_tsquery('simple', $2)) END AS fts,
                CASE WHEN $3 = '' THEN 0 ELSE greatest(public.word_similarity($3, text), public.word_similarity($3, subject)) END AS fuzzy,
                (CASE WHEN $4::text IS NOT NULL AND lower(subject) = lower($4) THEN 1 ELSE 0 END
                 + CASE WHEN pinned THEN 0.3 ELSE 0 END)::float AS boost
            FROM memories
            WHERE user_id = $1 AND status = 'active' AND {_ALIVE}
              AND ($5::text IS NULL OR scope = $5)
              AND ($6::text IS NULL OR category = $6)
              AND ($4::text IS NULL OR lower(subject) = lower($4) OR public.similarity(lower(subject), lower($4)) > 0.45
                   OR EXISTS (SELECT 1 FROM unnest(entities) e WHERE lower(e) = lower($4)) OR text ILIKE '%' || $4 || '%')
              AND ($7::text[] IS NULL OR lower(subject) = ANY($7)
                   OR EXISTS (SELECT 1 FROM unnest(entities) e WHERE lower(e) = ANY($7))
                   OR EXISTS (SELECT 1 FROM unnest($7::text[]) p WHERE text ILIKE '%' || p || '%'))
              AND ($8::date IS NULL OR coalesce(event_date, created_at::date) >= $8)
              AND ($9::date IS NULL OR coalesce(event_date, created_at::date) <= $9)
              AND ($10::int IS NULL OR extract(month FROM event_date) = $10)
              AND ($2 = '' OR search @@ to_tsquery('simple', $2)
                   OR public.word_similarity($3, text) > 0.35 OR public.word_similarity($3, subject) > 0.45)
            ORDER BY coalesce(event_date, created_at::date) DESC, updated_at DESC
            LIMIT 60
            """,
            user_id, tsquery, fuzzy_text, subject, scope, category, people, date_from, date_to, month,
        )
        if not words:
            return [_item(row, None) for row in rows[:limit]]
        ranked = fuse([dict(row) for row in rows])
        by_id = {row["id"]: row for row in rows}
        return [_item(by_id[row["id"]], row["score"]) for row in ranked[:limit]]

    async def recall(self, user_id: str, query: str, limit: int = 8) -> list[MemoryItem]:
        return await self.search(user_id, query=query, limit=limit)

    async def update(self, user_id: str, item_id: str, item: MemoryItem) -> None:
        await self.pool.execute(
            """
            UPDATE memories SET text = $3, data = $4::jsonb, confidence = $5, status = $6, sensitive = $7,
                expires_at = $8, scope = $9, category = $10, subject = $11, topic = $12, event_date = $13,
                yearly = $14, keywords = $15, entities = $16, pinned = $17, updated_at = now()
            WHERE user_id = $1 AND id::text = $2
            """,
            user_id, item_id, item.text.strip(), json.dumps(item.data, ensure_ascii=False), item.confidence, item.status,
            item.sensitive, item.expires_at, item.scope, item.category, item.subject, item.topic, item.event_date,
            item.yearly, item.keywords, item.entities, item.pinned,
        )

    async def edit(self, user_id: str, item_id: str, *, text: str | None, subject: str | None, topic: str | None) -> str:
        """Change one item's words from the playground. Returns "ok", "missing" or "conflict" (another item
        already has that subject and topic). The key follows the new subject/topic (or text, short term)."""
        row = await self.pool.fetchrow(
            "SELECT id::text AS id, scope, category, subject, topic, text FROM memories WHERE user_id = $1 AND id::text LIKE $2 || '%'",
            user_id, item_id.strip()[:36],
        )
        if not row:
            return "missing"
        item = MemoryItem(scope=row["scope"], category=row["category"], subject=subject if subject is not None else row["subject"],
                          topic=topic if topic is not None else row["topic"], text=(text if text is not None else row["text"]).strip())
        try:
            await self.pool.execute(
                "UPDATE memories SET text = $3, subject = $4, topic = $5, key = $6, updated_at = now() WHERE user_id = $1 AND id::text = $2",
                user_id, row["id"], item.text, item.subject, item.topic, key_for(item),
            )
        except asyncpg.UniqueViolationError:
            return "conflict"
        record("UPDATE", "memories", row["id"], f"{row['scope']}-term (edited in the panel)", arrow("text", repr(row["text"]), repr(item.text)))
        return "ok"

    async def forget(self, user_id: str, item_id: str) -> bool:
        rows = await self.pool.fetch(
            "DELETE FROM memories WHERE user_id = $1 AND id::text LIKE $2 || '%' RETURNING id::text, scope, text", user_id, item_id.strip()[:36]
        )
        for row in rows:
            record("DELETE", "memories", row["id"], f"{row['scope']}-term", f"text {row['text']!r}")
        return bool(rows)

    async def list_all(
        self, user_id: str, kind: str | None = None, status: str = "active", *, scope: str | None = None,
        pinned: bool | None = None,
    ) -> list[MemoryItem]:
        rows = await self.pool.fetch(
            f"""
            SELECT {_COLUMNS} FROM memories
            WHERE user_id = $1 AND status = $2 AND {_ALIVE} AND ($3::text IS NULL OR category = $3)
              AND ($4::text IS NULL OR scope = $4) AND ($5::boolean IS NULL OR pinned = $5)
            ORDER BY updated_at DESC LIMIT 300
            """,
            user_id, status, kind, scope, pinned,
        )
        return [_item(row) for row in rows]

    async def clear(self, user_id: str, scope: str) -> int:
        result = await self.pool.execute("DELETE FROM memories WHERE user_id = $1 AND scope = $2", user_id, scope)
        count = int(result.split()[-1])
        record("DELETE", "memories", user_id, f"all {scope}-term", f"{count} rows removed")
        return count
