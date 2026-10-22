"""Reminders, to-dos, notes and lists in Postgres (docs/personal-data-spec.md, section 3).

The LLM never writes SQL: tools pass validated fields and these methods run fixed, parameterised
queries. Counts are computed here, so numbers in answers are always right. "Match by words"
lookups return candidates; companion.search.pick() decides between "one clear hit" and "ask which".
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .audit import arrow, describe, record
from ..search import clean_words, fuse, to_tsquery

# Minimum fuzzy similarity for a "match by words" candidate.
MATCH_MIN = 0.4


def _rows(records) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def _exact_first(rows: list[dict[str, Any]], words: str) -> list[dict[str, Any]]:
    """When some rows contain a searched word as written, drop the ones that only matched fuzzily
    ('bank' must not bring 'banana'). Fuzzy hits stay when nothing matches exactly (typos)."""
    wanted = words.lower().split()
    exact = [row for row in rows if any(word in row["text"].lower() for word in wanted)]
    return exact or rows


class PersonalData:
    """Mixed into PostgresStore; uses self.pool."""

    # --- matching by words ---------------------------------------------------------------------

    async def _match(self, table: str, user_id: str, words: str, extra: str = "", columns: str = "") -> list[dict[str, Any]]:
        """Rows of `table` whose text matches `words` (contains or fuzzy), best first, with score `s`."""
        cleaned = " ".join(clean_words(words)) or words.strip().lower()
        if not cleaned:
            return []
        rows = await self.pool.fetch(
            f"""
            SELECT id::text AS id, text{columns}, greatest(public.word_similarity($2, text),
                        CASE WHEN text ILIKE '%' || $2 || '%' THEN 1 ELSE 0 END) AS s
            FROM {table} WHERE user_id = $1 {extra}
              AND (text ILIKE '%' || $2 || '%' OR public.word_similarity($2, text) > {MATCH_MIN})
            ORDER BY s DESC, created_at DESC LIMIT 6
            """,
            user_id, cleaned,
        )
        return _rows(rows)

    # --- reminders -----------------------------------------------------------------------------

    async def reminder_add(
        self, user_id: str, text: str, due_at: datetime, repeat: str, entities: list[str], all_day: bool = False,
    ) -> dict[str, Any]:
        # The same reminder (words and time) already waiting: return it instead of a second copy.
        existing = await self.pool.fetchrow(
            "SELECT id::text, text, due_at, repeat, all_day FROM reminders WHERE user_id = $1 AND lower(text) = lower($2) AND due_at = $3 "
            "AND status IN ('pending', 'snoozed') LIMIT 1",
            user_id, text, due_at,
        )
        if existing:
            return {**dict(existing), "existing": True}
        row = await self.pool.fetchrow(
            "INSERT INTO reminders (user_id, text, due_at, repeat, entities, all_day) VALUES ($1, $2, $3, $4, $5, $6) "
            "RETURNING id::text, text, due_at, repeat, all_day",
            user_id, text, due_at, repeat, entities, all_day,
        )
        record("INSERT", "reminders", row["id"], describe(text, due_at.date() if all_day else due_at),
               f"status pending, repeat {repeat}" + (", all day" if all_day else ""))
        return dict(row)

    async def reminder_list(
        self, user_id: str, *, date_from: datetime | None = None, date_to: datetime | None = None, status: str = "upcoming",
        repeat: str | None = None, query: str | None = None, limit: int = 50,
    ) -> dict[str, Any]:
        """status: upcoming (pending/snoozed/fired) | done | cancelled | all."""
        statuses = {"upcoming": ["pending", "snoozed", "fired"], "done": ["done"], "cancelled": ["cancelled"]}.get(status)
        words = " ".join(clean_words(query))
        rows = await self.pool.fetch(
            """
            SELECT id::text AS id, text, due_at, repeat, status, all_day FROM reminders
            WHERE user_id = $1 AND ($2::text[] IS NULL OR status = ANY($2))
              AND ($3::timestamptz IS NULL OR due_at >= $3) AND ($4::timestamptz IS NULL OR due_at <= $4)
              AND ($5::text IS NULL OR repeat = $5)
              AND ($6 = '' OR text ILIKE '%' || $6 || '%' OR public.word_similarity($6, text) > 0.4)
            ORDER BY due_at LIMIT $7
            """,
            user_id, statuses, date_from, date_to, repeat, words, limit,
        )
        found = _exact_first(_rows(rows), words)
        return {"reminders": found, "count": len(found)}

    async def reminder_occurrences(self, rows: list[dict[str, Any]], date_from: datetime, date_to: datetime) -> int:
        """How many times reminders fire in a window, counting each repeat (for "is hafte kitne reminders")."""
        total = 0
        for row in rows:
            step = {"daily": 1, "weekly": 7}.get(row["repeat"])
            due = row["due_at"]
            if row["repeat"] == "monthly":
                total += 1 if date_from <= due <= date_to else 0   # windows asked about are at most a few weeks
            elif step:
                if due <= date_to:
                    total += int((date_to - max(due, date_from)).total_seconds() // (step * 86400)) + 1
            elif date_from <= due <= date_to:
                total += 1
        return total

    async def reminder_find(self, user_id: str, words: str) -> list[dict[str, Any]]:
        return await self._match("reminders", user_id, words, "AND status IN ('pending', 'snoozed', 'fired')", ", due_at, repeat")

    async def reminders_waiting(self, user_id: str, reminder_id: str | None = None) -> list[dict[str, Any]]:
        """Every reminder still waiting (or one by its id, also the short id a list showed), oldest first."""
        rows = await self.pool.fetch(
            "SELECT id::text AS id, text, due_at, repeat, all_day, 1.0::float AS s FROM reminders "
            "WHERE user_id = $1 AND status IN ('pending', 'snoozed', 'fired') AND ($2::text IS NULL OR id::text LIKE $2 || '%') "
            "ORDER BY due_at, created_at",
            user_id, reminder_id,
        )
        return _rows(rows)

    async def _reminder_row(self, user_id: str, reminder_id: str) -> Any:
        return await self.pool.fetchrow(
            "SELECT id::text, text, due_at, status FROM reminders WHERE user_id = $1 AND id::text LIKE $2 || '%'", user_id, reminder_id
        )

    async def reminder_set_status(self, user_id: str, reminder_id: str, status: str) -> None:
        old = await self._reminder_row(user_id, reminder_id)
        await self.pool.execute(
            "UPDATE reminders SET status = $3 WHERE user_id = $1 AND id::text = $2", user_id, reminder_id, status
        )
        if old:
            record("UPDATE", "reminders", old["id"], describe(old["text"], old["due_at"]), arrow("status", old["status"], status))

    async def reminder_reschedule(self, user_id: str, reminder_id: str, due_at: datetime, status: str = "pending") -> bool:
        """`reminder_id` may be the short id shown in panels (its first characters)."""
        old = await self._reminder_row(user_id, reminder_id)
        result = await self.pool.execute(
            "UPDATE reminders SET due_at = $3, status = $4, reminded_count = 0, fired_at = NULL WHERE user_id = $1 AND id::text LIKE $2 || '%'",
            user_id, reminder_id, due_at, status,
        )
        if old:
            change = arrow("due_at", old["due_at"], due_at) + ("" if old["status"] == status else "; " + arrow("status", old["status"], status))
            record("UPDATE", "reminders", old["id"], describe(old["text"]), change)
        return result != "UPDATE 0"

    async def reminder_done(self, user_id: str, reminder_id: str) -> bool:
        """'Ho gaya': a one-off reminder is done; a repeating one just stops waiting for an answer."""
        old = await self._reminder_row(user_id, reminder_id)
        result = await self.pool.execute(
            """
            UPDATE reminders SET status = CASE WHEN repeat = 'none' THEN 'done' ELSE 'pending' END, reminded_count = 0
            WHERE user_id = $1 AND id::text LIKE $2 || '%' AND status IN ('fired', 'pending', 'snoozed')
            """,
            user_id, reminder_id,
        )
        if old and not result.endswith(" 0"):
            new = await self._reminder_row(user_id, reminder_id)
            record("UPDATE", "reminders", old["id"], describe(old["text"], old["due_at"]), arrow("status", old["status"], new["status"] if new else "?"))
        return not result.endswith(" 0")

    async def take_due_reminders(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Reminders to say out loud now: newly due ones, and fired ones nobody answered for 10 minutes
        (said once more). One-off reminders become 'fired'; repeating ones move to their next time."""
        async with self.pool.acquire() as connection, connection.transaction():
            rows = await connection.fetch(
                """
                SELECT id::text AS id, user_id, text, due_at, repeat, status, reminded_count, all_day FROM reminders
                WHERE ($1::text IS NULL OR user_id = $1 OR user_id LIKE $1 || '/%') AND (   -- the user's threads too
                    (status IN ('pending', 'snoozed') AND due_at <= now())
                    OR (status = 'fired' AND reminded_count < 2 AND fired_at <= now() - interval '10 minutes'))
                FOR UPDATE SKIP LOCKED
                """,
                user_id,
            )
            for row in rows:
                if row["repeat"] == "none" or row["status"] == "fired":
                    await connection.execute(
                        "UPDATE reminders SET status = 'fired', fired_at = now(), reminded_count = reminded_count + 1 WHERE id::text = $1",
                        row["id"],
                    )
                else:
                    step = {"daily": "1 day", "weekly": "7 days", "monthly": "1 month"}[row["repeat"]]
                    # Skip missed repeats so a laptop that was off does not fire a backlog.
                    await connection.execute(
                        f"""
                        UPDATE reminders SET status = 'pending', due_at = due_at + interval '{step}' * greatest(1, ceil(
                            extract(epoch FROM now() - due_at) / extract(epoch FROM interval '{step}')))
                        WHERE id::text = $1
                        """,
                        row["id"],
                    )
        return _rows(rows)

    # --- to-dos --------------------------------------------------------------------------------

    async def todo_add(self, user_id: str, items: list[str], day: date, priority: str, entities: list[str]) -> list[dict[str, Any]]:
        added = []
        async with self.pool.acquire() as connection, connection.transaction():
            for text in items:
                # The same open to-do on the same day is updated (priority), not added twice.
                existing = await connection.fetchrow(
                    "SELECT id::text AS id FROM todos WHERE user_id = $1 AND day = $2 AND NOT done AND lower(text) = lower($3)",
                    user_id, day, text,
                )
                if existing:
                    await connection.execute("UPDATE todos SET priority = $2 WHERE id::text = $1", existing["id"], priority)
                    added.append({"id": existing["id"], "text": text, "updated": True})
                    record("UPDATE", "todos", existing["id"], describe(text, day), f"priority → {priority} (already on that day's list)")
                    continue
                todo_id = await connection.fetchval(
                    "INSERT INTO todos (user_id, text, day, priority, entities) VALUES ($1, $2, $3, $4, $5) RETURNING id::text",
                    user_id, text, day, priority, entities,
                )
                added.append({"id": todo_id, "text": text})
                record("INSERT", "todos", todo_id, describe(text, day), f"priority {priority}, not done")
        return added

    async def todo_list(
        self, user_id: str, *, day: date | None = None, date_from: date | None = None, date_to: date | None = None,
        status: str = "pending", priority: str | None = None, query: str | None = None, today: date | None = None,
    ) -> dict[str, Any]:
        """status: pending | done | all. Returns items, counts and (when asking about today) overdue ones."""
        if day:
            date_from = date_to = day
        done = {"pending": False, "done": True}.get(status)
        words = " ".join(clean_words(query))
        rows = await self.pool.fetch(
            """
            SELECT id::text AS id, text, day, priority, done, done_at FROM todos
            WHERE user_id = $1 AND ($2::date IS NULL OR day >= $2) AND ($3::date IS NULL OR day <= $3)
              AND ($4::boolean IS NULL OR done = $4) AND ($5::text IS NULL OR priority = $5)
              AND ($6 = '' OR text ILIKE '%' || $6 || '%' OR public.word_similarity($6, text) > 0.4)
            ORDER BY day, done, CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END, created_at
            LIMIT 100
            """,
            user_id, date_from, date_to, done, priority, words,
        )
        items = _exact_first(_rows(rows), words)
        result: dict[str, Any] = {
            "todos": items,
            "count": {"total": len(items), "done": sum(1 for i in items if i["done"]), "pending": sum(1 for i in items if not i["done"])},
        }
        if today and day == today and status != "done":
            overdue = await self.pool.fetch(
                "SELECT id::text AS id, text, day, priority FROM todos WHERE user_id = $1 AND day < $2 AND NOT done ORDER BY day LIMIT 20",
                user_id, today,
            )
            result["overdue"] = _rows(overdue)
        return result

    async def todo_find(self, user_id: str, words: str, *, pending_only: bool = True) -> list[dict[str, Any]]:
        return await self._match("todos", user_id, words, "AND NOT done" if pending_only else "", ", day, priority")

    async def todo_mark_done(self, user_id: str, todo_ids: list[str]) -> int:
        rows = await self.pool.fetch(
            "UPDATE todos SET done = true, done_at = now() WHERE user_id = $1 AND id::text = ANY($2) AND NOT done "
            "RETURNING id::text, text, day", user_id, todo_ids,
        )
        for row in rows:
            record("UPDATE", "todos", row["id"], describe(row["text"], row["day"]), "done false → true")
        return len(rows)

    async def todo_ids_for_day(self, user_id: str, day: date, pending_only: bool = True) -> list[dict[str, Any]]:
        rows = await self.pool.fetch(
            "SELECT id::text AS id, text FROM todos WHERE user_id = $1 AND day = $2 AND ($3 = false OR NOT done) ORDER BY created_at",
            user_id, day, pending_only,
        )
        return _rows(rows)

    async def todo_delete(self, user_id: str, todo_ids: list[str]) -> int:
        rows = await self.pool.fetch("DELETE FROM todos WHERE user_id = $1 AND id::text = ANY($2) RETURNING id::text, text, day", user_id, todo_ids)
        for row in rows:
            record("DELETE", "todos", row["id"], describe(row["text"], row["day"]))
        return len(rows)

    async def todo_set_day(self, user_id: str, todo_ids: list[str], day: date) -> int:
        old = await self.pool.fetch("SELECT id::text, text, day FROM todos WHERE user_id = $1 AND id::text = ANY($2)", user_id, todo_ids)
        result = await self.pool.execute("UPDATE todos SET day = $3 WHERE user_id = $1 AND id::text = ANY($2)", user_id, todo_ids, day)
        for row in old:
            record("UPDATE", "todos", row["id"], describe(row["text"]), arrow("day", row["day"], day))
        return int(result.split()[-1])

    # --- notes ---------------------------------------------------------------------------------

    async def note_save(self, user_id: str, text: str, title: str, keywords: list[str], entities: list[str]) -> str:
        note_id = await self.pool.fetchval(
            "INSERT INTO notes (user_id, text, title, keywords, entities) VALUES ($1, $2, $3, $4, $5) RETURNING id::text",
            user_id, text, title, [k.strip().lower() for k in keywords if k.strip()], entities,
        )
        record("INSERT", "notes", note_id, describe(title or text[:40]), f"text {text[:60]!r}")
        return note_id

    async def note_search(
        self, user_id: str, *, query: str | None = None, entities: list[str] | None = None,
        date_from: date | None = None, date_to: date | None = None, limit: int = 8,
    ) -> list[dict[str, Any]]:
        words = clean_words(query)
        tsquery, fuzzy_text = to_tsquery(words), " ".join(words)
        people = [name.strip().lower() for name in (entities or []) if name.strip()] or None
        rows = await self.pool.fetch(
            """
            SELECT id::text AS id, title, text, keywords, entities, created_at,
                CASE WHEN $2 = '' THEN 0 ELSE ts_rank_cd(search, to_tsquery('simple', $2)) END AS fts,
                CASE WHEN $3 = '' THEN 0 ELSE greatest(public.word_similarity($3, text), public.word_similarity($3, title)) END AS fuzzy,
                0::float AS boost
            FROM notes
            WHERE user_id = $1
              AND ($4::text[] IS NULL OR EXISTS (SELECT 1 FROM unnest(entities) e WHERE lower(e) = ANY($4))
                   OR EXISTS (SELECT 1 FROM unnest($4::text[]) p WHERE text ILIKE '%' || p || '%' OR title ILIKE '%' || p || '%'))
              AND ($5::date IS NULL OR created_at::date >= $5) AND ($6::date IS NULL OR created_at::date <= $6)
              AND ($2 = '' OR search @@ to_tsquery('simple', $2)
                   OR public.word_similarity($3, text) > 0.35 OR public.word_similarity($3, title) > 0.45)
            ORDER BY created_at DESC LIMIT 60
            """,
            user_id, tsquery, fuzzy_text, people, date_from, date_to,
        )
        items = _rows(rows)
        return (fuse(items) if words else items)[:limit]

    async def note_all(self, user_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Every note, newest first (the playground's Notes tab)."""
        rows = await self.pool.fetch(
            "SELECT id::text AS id, title, text, keywords, created_at FROM notes WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id, limit,
        )
        return _rows(rows)

    async def note_titles(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """Newest note titles: the last fallback when nothing matched (the LLM picks by meaning)."""
        rows = await self.pool.fetch(
            "SELECT id::text AS id, title, left(text, 160) AS preview, created_at FROM notes WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id, limit,
        )
        return _rows(rows)

    async def note_find(self, user_id: str, words: str) -> list[dict[str, Any]]:
        cleaned = " ".join(clean_words(words)) or words.strip().lower()
        rows = await self.pool.fetch(
            """
            SELECT id::text AS id, title, text, greatest(public.word_similarity($2, title), public.word_similarity($2, text),
                CASE WHEN title ILIKE '%' || $2 || '%' THEN 1 ELSE 0 END) AS s
            FROM notes WHERE user_id = $1
              AND (title ILIKE '%' || $2 || '%' OR text ILIKE '%' || $2 || '%'
                   OR public.word_similarity($2, title) > 0.4 OR public.word_similarity($2, text) > 0.4)
            ORDER BY s DESC, created_at DESC LIMIT 6
            """,
            user_id, cleaned,
        )
        return _rows(rows)

    async def note_delete(self, user_id: str, note_ids: list[str]) -> int:
        rows = await self.pool.fetch("DELETE FROM notes WHERE user_id = $1 AND id::text = ANY($2) RETURNING id::text, title, text", user_id, note_ids)
        for row in rows:
            record("DELETE", "notes", row["id"], describe(row["title"] or row["text"][:40]))
        return len(rows)

    # --- editing one row from the playground panels ---------------------------------------------
    # Ids may be the short ids the panels show (their first characters). Every query is limited to the owner.

    _EDITABLE = {"notes": ("title", "text"), "reminders": ("text", "due_at", "repeat", "all_day"), "todos": ("text", "day", "priority", "done")}

    async def edit_row(self, table: str, owner: str, row_id: str, values: dict[str, Any]) -> bool:
        changes = {column: values[column] for column in self._EDITABLE[table] if values.get(column) is not None}
        if not changes:
            return False
        sets = [f"{column} = ${index}" for index, column in enumerate(changes, 3)]
        if table == "reminders" and "due_at" in changes:
            # A new time: the reminder waits again (it may have fired or been snoozed already).
            sets.append("status = CASE WHEN status IN ('fired', 'snoozed') THEN 'pending' ELSE status END, reminded_count = 0, fired_at = NULL")
        if table == "todos" and "done" in changes:
            sets.append(f"done_at = CASE WHEN ${3 + list(changes).index('done')} THEN coalesce(done_at, now()) ELSE NULL END")
        result = await self.pool.execute(
            f"UPDATE {table} SET {', '.join(sets)} WHERE user_id = $1 AND id::text LIKE $2 || '%'",
            owner, row_id, *changes.values(),
        )
        if result != "UPDATE 0":
            record("UPDATE", table, row_id, "edited in the panel", ", ".join(f"{k} → {v}" for k, v in changes.items()))
        return result != "UPDATE 0"

    async def delete_row(self, table: str, owner: str, row_id: str) -> bool:
        assert table in self._EDITABLE
        result = await self.pool.execute(f"DELETE FROM {table} WHERE user_id = $1 AND id::text LIKE $2 || '%'", owner, row_id)
        if result != "DELETE 0":
            record("DELETE", table, row_id, "deleted in the panel")
        return result != "DELETE 0"

    async def list_item_edit(self, owner: str, item_id: str, text: str | None, done: bool | None) -> bool:
        result = await self.pool.execute(
            """
            UPDATE list_items i SET text = coalesce($3, i.text), done = coalesce($4, i.done)
            FROM lists l WHERE l.id = i.list_id AND l.user_id = $1 AND i.id::text LIKE $2 || '%'
            """,
            owner, item_id, text, done,
        )
        return result != "UPDATE 0"

    async def list_item_delete(self, owner: str, item_id: str) -> bool:
        result = await self.pool.execute(
            "DELETE FROM list_items i USING lists l WHERE l.id = i.list_id AND l.user_id = $1 AND i.id::text LIKE $2 || '%'",
            owner, item_id,
        )
        return result != "DELETE 0"

    # --- lists ---------------------------------------------------------------------------------

    async def list_resolve(self, user_id: str, name: str) -> dict[str, Any] | None:
        """The user's list with this name, also when misspelt ("shoping" -> shopping)."""
        row = await self.pool.fetchrow(
            """
            SELECT id, name FROM lists WHERE user_id = $1 AND (lower(name) = lower($2) OR public.similarity(lower(name), lower($2)) > 0.45)
            ORDER BY lower(name) = lower($2) DESC, public.similarity(lower(name), lower($2)) DESC LIMIT 1
            """,
            user_id, name.strip(),
        )
        return dict(row) if row else None

    async def list_add(self, user_id: str, name: str, items: list[str]) -> dict[str, Any]:
        existing = await self.list_resolve(user_id, name)
        async with self.pool.acquire() as connection, connection.transaction():
            list_id = existing["id"] if existing else await connection.fetchval(
                "INSERT INTO lists (user_id, name) VALUES ($1, $2) RETURNING id", user_id, name.strip().lower()
            )
            added, already = [], []
            for text in items:
                found = await connection.fetchval(
                    "SELECT id FROM list_items WHERE list_id = $1 AND lower(text) = lower($2) AND NOT done", list_id, text
                )
                if found:
                    already.append(text)
                    continue
                item_id = await connection.fetchval("INSERT INTO list_items (list_id, text) VALUES ($1, $2) RETURNING id::text", list_id, text)
                added.append(text)
                record("INSERT", "list_items", item_id, describe(text), f"on the {existing['name'] if existing else name.strip().lower()} list")
        return {"list": existing["name"] if existing else name.strip().lower(), "added": added, "already_there": already}

    async def list_show(self, user_id: str, name: str | None) -> dict[str, Any]:
        if name:
            found = await self.list_resolve(user_id, name)
            if not found:
                return {"found": False, "list": name}
            rows = await self.pool.fetch(
                "SELECT id::text AS id, text, done FROM list_items WHERE list_id = $1 ORDER BY done, created_at", found["id"]
            )
            items = _rows(rows)
            return {"found": True, "list": found["name"], "items": items,
                    "count": {"total": len(items), "left": sum(1 for i in items if not i["done"]), "checked": sum(1 for i in items if i["done"])}}
        rows = await self.pool.fetch(
            """
            SELECT l.name, count(i.id) AS total, count(i.id) FILTER (WHERE NOT i.done) AS left
            FROM lists l LEFT JOIN list_items i ON i.list_id = l.id WHERE l.user_id = $1 GROUP BY l.name ORDER BY l.name
            """,
            user_id,
        )
        return {"lists": _rows(rows)}

    async def list_items_match(self, user_id: str, name: str, words: str, *, unchecked_only: bool) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        found = await self.list_resolve(user_id, name)
        if not found:
            return None, []
        cleaned = words.strip().lower()
        rows = await self.pool.fetch(
            f"""
            SELECT id::text AS id, text, done, greatest(public.word_similarity($2, text), CASE WHEN text ILIKE '%' || $2 || '%' THEN 1 ELSE 0 END) AS s
            FROM list_items WHERE list_id = $1 {"AND NOT done" if unchecked_only else ""}
              AND (text ILIKE '%' || $2 || '%' OR public.word_similarity($2, text) > {MATCH_MIN})
            ORDER BY s DESC LIMIT 6
            """,
            found["id"], cleaned,
        )
        return found, _rows(rows)

    async def list_items_update(self, item_ids: list[str], *, check: bool | None = None, delete: bool = False) -> int:
        if delete:
            rows = await self.pool.fetch("DELETE FROM list_items WHERE id::text = ANY($1) RETURNING id::text, text", item_ids)
            for row in rows:
                record("DELETE", "list_items", row["id"], describe(row["text"]))
        else:
            rows = await self.pool.fetch("UPDATE list_items SET done = $2 WHERE id::text = ANY($1) RETURNING id::text, text", item_ids, bool(check))
            for row in rows:
                record("UPDATE", "list_items", row["id"], describe(row["text"]), arrow("done", not check, bool(check)))
        return len(rows)

    async def list_clear(self, user_id: str, name: str, checked_only: bool) -> dict[str, Any]:
        found = await self.list_resolve(user_id, name)
        if not found:
            return {"found": False, "list": name}
        result = await self.pool.execute(
            f"DELETE FROM list_items WHERE list_id = $1 {'AND done' if checked_only else ''}", found["id"]
        )
        removed = int(result.split()[-1])
        record("DELETE", "list_items", found["id"], f'the {found["name"]} list', f"{removed} {'ticked ' if checked_only else ''}items removed")
        return {"found": True, "list": found["name"], "removed": removed}

    # --- clearing a whole store and the day overview ------------------------------------------------

    async def store_counts(self, user_id: str, store: str, day: date | None = None) -> dict[str, Any]:
        """How many rows a clear would remove, with a short preview (for the confirm question)."""
        queries = {
            "short": "SELECT text FROM memories WHERE user_id = $1 AND scope = 'short'",
            "long": "SELECT text FROM memories WHERE user_id = $1 AND scope = 'long'",
            "notes": "SELECT coalesce(nullif(title, ''), left(text, 40)) AS text FROM notes WHERE user_id = $1",
            "todos": "SELECT text FROM todos WHERE user_id = $1 AND ($2::date IS NULL OR day = $2)",
            "reminders": "SELECT text FROM reminders WHERE user_id = $1 AND status IN ('pending', 'snoozed', 'fired')",
        }
        sql = queries[store]
        rows = await self.pool.fetch(sql, user_id, day) if store == "todos" else await self.pool.fetch(sql, user_id)
        return {"count": len(rows), "preview": [row["text"] for row in rows[:5]]}

    async def store_clear(self, user_id: str, store: str, day: date | None = None) -> int:
        statements = {
            "short": ("DELETE FROM memories WHERE user_id = $1 AND scope = 'short'", ()),
            "long": ("DELETE FROM memories WHERE user_id = $1 AND scope = 'long'", ()),
            "notes": ("DELETE FROM notes WHERE user_id = $1", ()),
            "todos": ("DELETE FROM todos WHERE user_id = $1 AND ($2::date IS NULL OR day = $2)", (day,)),
            "reminders": ("UPDATE reminders SET status = 'cancelled' WHERE user_id = $1 AND status IN ('pending', 'snoozed', 'fired')", ()),
        }
        sql, extra = statements[store]
        result = await self.pool.execute(sql, user_id, *extra)
        count = int(result.split()[-1])
        table = {"short": "memories", "long": "memories", "notes": "notes", "todos": "todos", "reminders": "reminders"}[store]
        record("UPDATE" if store == "reminders" else "DELETE", table, user_id, f"all {store}" + (f" of {day}" if day else ""),
               f"{count} rows" + (" → cancelled" if store == "reminders" else " removed"))
        return count
