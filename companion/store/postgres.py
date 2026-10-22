"""Everything the brain saves besides memory: users, chat, reminders, todos, notes, lists, tool log.

Ids shown to the LLM are the first 8 characters of a uuid; every lookup accepts that prefix.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import asyncpg

from ..llm.base import ChatMessage
from .personal import PersonalData


MAIN_THREAD = "main"  # the user's own ongoing chat with the companion


def data_owner(user_id: str, thread: str = MAIN_THREAD) -> str:
    """Who owns the saved data (memory, notes, reminders, to-dos, lists) of a thread. Each thread is its
    own world: the main thread uses the user id, every other thread "<user>/<thread>", so thread 2 starts
    empty and never sees or changes thread 1's data. Chat history is per thread through `conversations`."""
    return user_id if thread == MAIN_THREAD else f"{user_id}/{thread}"


def short(item_id: str) -> str:
    return item_id[:8]


class PostgresStore(PersonalData):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    # --- users ---------------------------------------------------------------------------------

    async def ensure_user(self, user_id: str, timezone: str) -> str:
        """Create the user on first contact. Returns the user's timezone."""
        row = await self.pool.fetchrow(
            """
            INSERT INTO companion_users (id, timezone) VALUES ($1, $2)
            ON CONFLICT (id) DO UPDATE SET id = EXCLUDED.id
            RETURNING timezone
            """,
            user_id, timezone,
        )
        return row["timezone"]

    # --- conversations and messages --------------------------------------------------------------

    async def current_conversation(self, user_id: str, thread: str = MAIN_THREAD) -> str:
        """The thread's conversation, created on first use. A thread never rotates or resets:
        older messages are folded into its running summary instead."""
        row = await self.pool.fetchval(
            "SELECT id::text FROM conversations WHERE user_id = $1 AND thread = $2 ORDER BY last_message_at DESC LIMIT 1",
            user_id, thread,
        )
        if row:
            return row
        return await self.pool.fetchval(
            "INSERT INTO conversations (user_id, thread) VALUES ($1, $2) RETURNING id::text", user_id, thread
        )

    async def create_thread(self, user_id: str) -> str:
        """A separate chat (thread). The main thread is not touched. Returns the new thread id."""
        thread = f"t-{uuid.uuid4().hex[:8]}"
        await self.pool.execute("INSERT INTO conversations (user_id, thread) VALUES ($1, $2)", user_id, thread)
        return thread

    async def list_threads(self, user_id: str) -> list[dict[str, Any]]:
        """The user's threads: main first, then the others in the order they were opened, numbered 1, 2, 3..."""
        rows = await self.pool.fetch(
            """
            SELECT c.thread, min(c.started_at) AS started_at, max(c.last_message_at) AS last_message_at,
                   count(m.id) FILTER (WHERE m.role IN ('user', 'assistant') AND m.content <> '') AS messages,
                   (SELECT m2.content FROM messages m2 JOIN conversations c2 ON c2.id = m2.conversation_id
                    WHERE c2.user_id = $1 AND c2.thread = c.thread AND m2.role = 'user' ORDER BY m2.id LIMIT 1) AS first_message
            FROM conversations c LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_id = $1 GROUP BY c.thread ORDER BY (c.thread <> 'main'), min(c.started_at)
            """,
            user_id,
        )
        threads = [dict(row) for row in rows]
        if not threads or threads[0]["thread"] != MAIN_THREAD:
            threads.insert(0, {"thread": MAIN_THREAD, "started_at": None, "last_message_at": None, "messages": 0, "first_message": None})
        for number, thread in enumerate(threads, 1):
            thread["number"] = number
        return threads

    async def add_messages(self, conversation_id: str, messages: list[ChatMessage], expression: str | None = None) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            for message in messages:
                await connection.execute(
                    """
                    INSERT INTO messages (conversation_id, role, content, expression, tool_calls, tool_call_id, name)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
                    """,
                    conversation_id, message.role, message.content,
                    expression if message.role == "assistant" and not message.tool_calls else None,
                    json.dumps([call.model_dump() for call in message.tool_calls]) if message.tool_calls else None,
                    message.tool_call_id, message.name,
                )
            await connection.execute("UPDATE conversations SET last_message_at = now() WHERE id = $1", conversation_id)

    async def recent_messages(self, conversation_id: str, limit: int) -> tuple[list[ChatMessage], str, int]:
        """The last `limit` user/assistant messages not yet folded into the summary,
        plus the summary and the total message count."""
        conversation = await self.pool.fetchrow(
            "SELECT summary, summarized_count FROM conversations WHERE id = $1", conversation_id
        )
        rows = await self.pool.fetch(
            """
            SELECT role, content FROM (
                SELECT id, role, content, row_number() OVER (ORDER BY id) AS position
                FROM messages WHERE conversation_id = $1 AND role IN ('user', 'assistant') AND content <> ''
            ) numbered
            WHERE position > $2 ORDER BY id DESC LIMIT $3
            """,
            conversation_id, conversation["summarized_count"], limit,
        )
        total = await self.pool.fetchval(
            "SELECT count(*) FROM messages WHERE conversation_id = $1 AND role IN ('user', 'assistant') AND content <> ''",
            conversation_id,
        )
        history = [ChatMessage(role=row["role"], content=row["content"]) for row in reversed(rows)]
        return history, conversation["summary"], int(total)

    async def latest_messages(self, user_id: str, limit: int = 40, thread: str = MAIN_THREAD) -> list[dict[str, Any]]:
        """Read-only: the user's latest chat lines across conversations (for a body to redraw its chat)."""
        rows = await self.pool.fetch(
            """
            SELECT m.role, m.content, m.expression, m.created_at, m.conversation_id::text AS conversation_id
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.user_id = $1 AND c.thread = $3 AND m.role IN ('user', 'assistant') AND m.content <> ''
            ORDER BY m.id DESC LIMIT $2
            """,
            user_id, limit, thread,
        )
        return [dict(row) for row in reversed(rows)]

    async def memory_snapshot(self, user_id: str, window: int, thread: str = MAIN_THREAD) -> dict[str, Any]:
        """Read-only view of short-term memory for the Trace panel (never starts a conversation)."""
        rows = await self.pool.fetch(
            "SELECT id::text AS id, thread, started_at, last_message_at, summary, summarized_count FROM conversations "
            "WHERE user_id = $1 AND thread = $2 ORDER BY last_message_at DESC LIMIT 1",
            user_id, thread,
        )
        if not rows:
            return {"conversation": None, "messages": [], "total": 0}
        current = rows[0]
        messages = await self.pool.fetch(
            """
            SELECT role, content, expression, created_at FROM messages
            WHERE conversation_id = $1 AND role IN ('user', 'assistant') AND content <> ''
            ORDER BY id
            """,
            current["id"],
        )
        total = len(messages)
        summarized = current["summarized_count"]
        start_of_window = max(summarized, total - window)
        return {
            "conversation": {
                "id": current["id"], "thread": current["thread"], "started_at": current["started_at"].isoformat(),
                "last_message_at": current["last_message_at"].isoformat(),
                "summary": current["summary"], "summarized_count": summarized,
            },
            "total": total,
            "messages": [
                {
                    "role": m["role"], "content": m["content"], "expression": m["expression"],
                    "at": m["created_at"].isoformat(),
                    # where this message lives now: folded into the summary, sent whole to the LLM, or neither
                    "where": "summary" if index < summarized else "window" if index >= start_of_window else "stored",
                }
                for index, m in enumerate(messages)
            ][-60:],
        }

    async def list_users(self) -> list[dict[str, Any]]:
        """Every user that has chatted or has a profile (thread data owners "<user>/<thread>" are not users)."""
        rows = await self.pool.fetch(
            """
            SELECT u.id, count(DISTINCT c.thread) AS threads, max(c.last_message_at) AS last_message_at
            FROM (SELECT id FROM companion_users UNION SELECT user_id FROM conversations) u
            LEFT JOIN conversations c ON c.user_id = u.id
            WHERE position('/' in u.id) = 0
            GROUP BY u.id ORDER BY max(c.last_message_at) DESC NULLS LAST, u.id
            """
        )
        return [dict(row) for row in rows]

    async def raw_dump(self, user_id: str, thread: str = MAIN_THREAD) -> dict[str, Any]:
        """One thread's data exactly as stored in Postgres (schema + rows), for the Trace panel: the thread's
        conversation and messages, and the memory, reminders, to-dos, notes and lists it owns."""
        owner = data_owner(user_id, thread)
        tables = ["companion_users", "conversations", "messages", "memories", "reminders", "todos", "notes", "lists", "list_items"]
        columns = await self.pool.fetch(
            """
            SELECT table_name, column_name, data_type, is_nullable, column_default FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = ANY($1::text[]) ORDER BY table_name, ordinal_position
            """,
            tables,
        )
        schema: dict[str, list[dict[str, Any]]] = {name: [] for name in tables}
        for column in columns:
            if column["column_name"] == "search":
                continue   # the generated search vector: long and not useful to read
            schema[column["table_name"]].append({
                "column": column["column_name"], "type": column["data_type"],
                "nullable": column["is_nullable"] == "YES", "default": column["column_default"],
            })
        user = await self.pool.fetchrow("SELECT * FROM companion_users WHERE id = $1", user_id)
        conversations = await self.pool.fetch(
            "SELECT id::text AS id, user_id, thread, started_at, last_message_at, summary, summarized_count FROM conversations "
            "WHERE user_id = $1 AND thread = $2 ORDER BY last_message_at DESC",
            user_id, thread,
        )
        messages = []
        if conversations:
            messages = await self.pool.fetch(
                "SELECT id, conversation_id::text AS conversation_id, role, content, expression, tool_calls, tool_call_id, name, created_at "
                "FROM messages WHERE conversation_id = $1::uuid ORDER BY id DESC LIMIT 40",
                conversations[0]["id"],
            )

        async def owned(table: str, order: str) -> list[Any]:
            names = ", ".join(c["column"] if c["type"] != "uuid" else f"{c['column']}::text AS {c['column']}" for c in schema[table])
            return await self.pool.fetch(f"SELECT {names} FROM {table} WHERE user_id = $1 ORDER BY {order} LIMIT 200", owner)

        memories = await owned("memories", "scope, updated_at DESC")
        reminders = await owned("reminders", "due_at")
        todos = await owned("todos", "day, created_at")
        notes = await owned("notes", "created_at DESC")
        lists = await owned("lists", "name")
        items = await self.pool.fetch(
            "SELECT i.id::text AS id, i.list_id::text AS list_id, i.text, i.done, i.created_at FROM list_items i "
            "JOIN lists l ON l.id = i.list_id WHERE l.user_id = $1 ORDER BY l.name, i.created_at",
            owner,
        )

        def plain(row: Any) -> dict[str, Any]:
            values = dict(row)
            for key in ("data", "tool_calls", "settings"):
                if isinstance(values.get(key), str):
                    try:
                        values[key] = json.loads(values[key])
                    except ValueError:
                        pass
            return values

        return {
            "schema": schema,
            "owner": owner,
            "rows": {
                "companion_users": [plain(user)] if user else [],
                "conversations (this thread)": [plain(row) for row in conversations],
                "messages (this thread, newest first)": [plain(row) for row in messages],
                f"memories (owner {owner})": [plain(row) for row in memories],
                f"reminders (owner {owner})": [plain(row) for row in reminders],
                f"todos (owner {owner})": [plain(row) for row in todos],
                f"notes (owner {owner})": [plain(row) for row in notes],
                f"lists (owner {owner})": [plain(row) for row in lists],
                f"list_items (owner {owner})": [plain(row) for row in items],
            },
        }

    async def unsummarized(self, conversation_id: str) -> tuple[str, int, list[ChatMessage]]:
        conversation = await self.pool.fetchrow(
            "SELECT summary, summarized_count FROM conversations WHERE id = $1", conversation_id
        )
        rows = await self.pool.fetch(
            """
            SELECT role, content FROM messages
            WHERE conversation_id = $1 AND role IN ('user', 'assistant') AND content <> ''
            ORDER BY id OFFSET $2
            """,
            conversation_id, conversation["summarized_count"],
        )
        return conversation["summary"], conversation["summarized_count"], [ChatMessage(role=r["role"], content=r["content"]) for r in rows]

    async def save_summary(self, conversation_id: str, summary: str, summarized_count: int) -> None:
        await self.pool.execute(
            "UPDATE conversations SET summary = $2, summarized_count = $3 WHERE id = $1",
            conversation_id, summary, summarized_count,
        )

    # --- tool log ------------------------------------------------------------------------------

    async def log_tool(self, user_id: str, tool: str, args: Any, result: Any, ok: bool) -> None:
        await self.pool.execute(
            "INSERT INTO tool_log (user_id, tool, args, result, ok) VALUES ($1, $2, $3::jsonb, $4::jsonb, $5)",
            user_id, tool, json.dumps(args, ensure_ascii=False, default=str), json.dumps(result, ensure_ascii=False, default=str), ok,
        )
