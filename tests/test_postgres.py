"""Tests against a real Postgres, in a throwaway schema "companion_test" (dropped afterwards).
A scripted fake LLM stands in for Groq, so no network is used. Skipped without COMPANION_DATABASE_URL."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import asyncpg

from companion.brain import Companion
from companion.config import Settings
from companion.llm.base import LLMResult, ToolCall
from companion.memory.interface import MemoryItem
from companion.memory.postgres import PostgresMemory
from companion.store import db
from companion.store.postgres import PostgresStore

SETTINGS = Settings(db_schema="companion_test", tavily_api_key="", llm_order=["groq"])
TZ = ZoneInfo(SETTINGS.timezone)


class ScriptedLLM:
    """Returns the scripted chat replies in order; extractor (json_mode) calls get `memories`."""

    def __init__(self, replies: list[LLMResult], memories: list[dict] | None = None) -> None:
        self.replies = list(replies)
        self.memories = memories or []
        self.calls: list[dict] = []

    async def chat(self, messages, *, tools=None, json_mode=False, **options):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        if json_mode:
            return LLMResult(text=json.dumps({"items": self.memories}), provider="fake")
        return self.replies.pop(0)


async def _fresh_pool() -> asyncpg.Pool:
    setup = await asyncpg.connect(SETTINGS.database_url)
    await setup.execute("DROP SCHEMA IF EXISTS companion_test CASCADE")
    await setup.close()
    return await db.connect(SETTINGS.database_url, "companion_test")


async def _drop() -> None:
    setup = await asyncpg.connect(SETTINGS.database_url)
    await setup.execute("DROP SCHEMA IF EXISTS companion_test CASCADE")
    await setup.close()


@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class PostgresTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    def test_memory_upsert_search_expiry(self):
        async def run():
            pool = await _fresh_pool()
            memory = PostgresMemory(pool)
            paneer = dict(scope="long", category="preference", subject="me", topic="food")
            first = await memory.remember("u", MemoryItem(**paneer, text="Likes paneer a lot"))
            again = await memory.remember("u", MemoryItem(**paneer, text="Does not like paneer any more"))
            self.assertEqual(first, again)                                      # same category/subject/topic: updated
            await memory.remember("u", MemoryItem(scope="long", category="date", subject="Riya", topic="birthday",
                                                  text="Riya's birthday", event_date=datetime(2026, 5, 3).date(), yearly=True))
            await memory.remember("u", MemoryItem(scope="short", text="Exam last week", expires_at=datetime.now(TZ) - timedelta(hours=1)))
            await memory.remember("other", MemoryItem(scope="long", category="profile", subject="me", topic="city", text="Lives in Riya town"))
            found = await memory.recall("u", "Riya ka birthday kab hai?")
            self.assertEqual([i.text for i in found], ["Riya's birthday"])     # another user's memory never shows
            self.assertEqual(len(await memory.list_all("u")), 2)               # expired short-term item hidden
            self.assertEqual([i.text for i in await memory.recall("u", "paneer")], ["Does not like paneer any more"])
            self.assertTrue(await memory.forget("u", first[:8]))
            self.assertFalse(await memory.forget("u", "deadbeef"))
            await pool.close()
        asyncio.run(run())

    def test_due_reminders_once_repeat_and_said_again(self):
        async def run():
            pool = await _fresh_pool()
            store = PostgresStore(pool)
            past = datetime.now(TZ) - timedelta(minutes=5)
            await store.reminder_add("u", "dawai", past, "none", [])
            await store.reminder_add("u", "paani", past - timedelta(days=3), "daily", [])
            due = await store.take_due_reminders("u")
            self.assertEqual(sorted(r["text"] for r in due), ["dawai", "paani"])
            self.assertEqual(await store.take_due_reminders("u"), [])          # each fires once
            upcoming = (await store.reminder_list("u"))["reminders"]
            self.assertEqual([(r["text"], r["status"]) for r in upcoming], [("dawai", "fired"), ("paani", "pending")])
            self.assertGreater(upcoming[1]["due_at"], datetime.now(TZ))        # the daily one moved forward
            self.assertLess(upcoming[1]["due_at"], datetime.now(TZ) + timedelta(days=1))
            # Nobody pressed Done: said once more after 10 minutes, then never again.
            await pool.execute("UPDATE reminders SET fired_at = now() - interval '11 minutes' WHERE text = 'dawai'")
            again = await store.take_due_reminders("u")
            self.assertEqual([(r["text"], r["reminded_count"]) for r in again], [("dawai", 1)])
            await pool.execute("UPDATE reminders SET fired_at = now() - interval '11 minutes' WHERE text = 'dawai'")
            self.assertEqual(await store.take_due_reminders("u"), [])
            # The same reminder again (same words, same time): no second copy.
            first = await store.reminder_add("u", "gym", past + timedelta(days=2), "none", [])
            again = await store.reminder_add("u", "Gym", past + timedelta(days=2), "none", [])
            self.assertEqual((again["id"], again.get("existing")), (first["id"], True))
            await store.reminder_set_status("u", first["id"], "cancelled")
            # Every change is recorded: log line, and the turn's list when a tool runs inside a turn.
            from companion.store.audit import CURRENT, TurnChanges

            class Trace:
                events = []

                def emit(self, user_id, kind, turn, **data):
                    Trace.events.append((kind, data))

            turn = TurnChanges(trace=Trace(), user_id="u", turn="t1")
            token = CURRENT.set(turn)
            gym = await store.reminder_add("u", "gym class", past + timedelta(days=3), "none", [])
            await store.reminder_set_status("u", gym["id"], "cancelled")
            CURRENT.reset(token)
            self.assertEqual([(c["op"], c["table"]) for c in turn.changes], [("INSERT", "reminders"), ("UPDATE", "reminders")])
            self.assertEqual(turn.changes[1]["change"], "status pending → cancelled")
            self.assertIn('"gym class"', turn.changes[1]["row"])
            self.assertEqual([kind for kind, _ in Trace.events], ["db_change", "db_change"])
            # Done and snooze from the buttons.
            dawai = upcoming[0]["id"]
            await store.reminder_reschedule("u", dawai, datetime.now(TZ) - timedelta(seconds=1), status="snoozed")
            self.assertEqual([r["text"] for r in await store.take_due_reminders("u")], ["dawai"])
            self.assertTrue(await store.reminder_done("u", dawai))
            self.assertEqual([r["text"] for r in (await store.reminder_list("u", status="done"))["reminders"]], ["dawai"])
            await pool.close()
        asyncio.run(run())

    def test_chat_turns_save_only_when_asked(self):
        async def run():
            pool = await _fresh_pool()
            tomorrow = (datetime.now(TZ) + timedelta(days=(4 - datetime.now(TZ).weekday()) % 7 or 7)).strftime("%Y-%m-%dT09:00")  # next Friday (the messages say "Friday")
            llm = ScriptedLLM([
                LLMResult(tool_calls=[ToolCall(id="c1", name="reminder_add", arguments=json.dumps({"text": "dawai lena", "when": tomorrow}))]),
                LLMResult(tool_calls=[ToolCall(id="c2", name="ltm_save", arguments=json.dumps({"facts": [
                    {"category": "profile", "subject": "me", "topic": "name", "text": "Name is Ashish"},
                    {"category": "profile", "subject": "Ashish", "topic": "city", "text": "Lives in Pune."}]}))]),
                LLMResult(text="[curious] Aur batao, kya chal raha hai?"),
            ])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            reply = await bot.chat("u", "remind me on Friday at 9 to take medicine")
            # A write tool's confirmation is built in code: no second LLM call.
            self.assertTrue(reply.text.startswith("Reminder set: dawai lena, "), reply.text)
            self.assertIn("9:00 AM", reply.text)
            self.assertEqual(reply.tools_used, ["reminder_add"])
            self.assertEqual(len(llm.calls), 1)
            offered = {t["function"]["name"] for t in llm.calls[0]["tools"]}
            self.assertIn("reminder_add", offered)
            self.assertNotIn("note_save", offered)                              # only the groups the words point to

            reply = await bot.chat("u", "remember long term: my name is Ashish and I live in Pune")
            self.assertEqual(reply.tools_used, ["ltm_save"])
            await bot.chat("u", "kya chal raha hai?")
            await bot.wait_background()
            self.assertFalse(any(c["json_mode"] for c in llm.calls))           # the extractor is off by default
            system = llm.calls[-1]["messages"][0].content
            self.assertIn("Name is Ashish", system)                             # the pinned profile goes with every message
            self.assertNotIn("reminder_add", {t["function"]["name"] for t in llm.calls[-1]["tools"]})
            reminders = (await bot.store.reminder_list("u"))["reminders"]
            self.assertEqual([r["text"] for r in reminders], ["dawai lena"])
            self.assertEqual(reminders[0]["due_at"].astimezone(TZ).hour, 9)
            saved = await bot.memory.list_all("u")
            self.assertEqual(sorted((i.subject, i.topic, i.text, i.pinned) for i in saved),
                             [("me", "city", "Lives in Pune", True), ("me", "name", "Name is Ashish", True)])   # two facts, one call
            conversation_id = await bot.store.current_conversation("u")
            history, _, total = await bot.store.recent_messages(conversation_id, 20)
            self.assertEqual(total, 6)
            self.assertEqual(history[0].content, "remind me on Friday at 9 to take medicine")
            await pool.close()
        asyncio.run(run())

    def test_tool_done_but_llm_failed_still_confirms(self):
        async def run():
            from companion.llm.base import LLMError

            pool = await _fresh_pool()
            tomorrow = (datetime.now(TZ) + timedelta(days=(4 - datetime.now(TZ).weekday()) % 7 or 7)).strftime("%Y-%m-%dT09:00")  # next Friday (the messages say "Friday")

            class FailsAfterTool(ScriptedLLM):
                async def chat(self, messages, **options):
                    if not options.get("json_mode") and len(self.calls) >= 1:
                        self.calls.append({})
                        raise LLMError("rate limited")
                    return await super().chat(messages, **options)

            llm = FailsAfterTool([LLMResult(tool_calls=[ToolCall(id="c1", name="reminder_add", arguments=json.dumps({"text": "dawai", "when": tomorrow}))])])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            reply = await bot.chat("u", "Friday 9 baje dawai yaad dilana")
            await bot.wait_background()
            self.assertTrue(reply.text.startswith("Reminder set: dawai, "), reply.text)
            self.assertIn("9:00 AM", reply.text)
            await pool.close()
        asyncio.run(run())

    def test_long_chat_is_summarized(self):
        async def run():
            pool = await _fresh_pool()
            settings = Settings(db_schema="companion_test", tavily_api_key="", recent_messages=4, summarize_after=6)
            replies = [LLMResult(text=f"[idle] reply {i}") for i in range(4)] + [LLMResult(text="They chatted about four things.")]
            llm = ScriptedLLM(replies)
            bot = Companion(settings=settings, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            for i in range(4):
                await bot.chat("u", f"message {i}")
                await bot.wait_background()
            conversation_id = await bot.store.current_conversation("u")
            history, summary, _ = await bot.store.recent_messages(conversation_id, 4)
            self.assertEqual(summary, "They chatted about four things.")
            self.assertEqual([m.content for m in history], ["message 2", "reply 2", "message 3", "reply 3"])
            await pool.close()
        asyncio.run(run())

    def test_new_thread_never_touches_the_main_thread(self):
        async def run():
            pool = await _fresh_pool()
            llm = ScriptedLLM([
                LLMResult(text="[idle] Achha, all the best!"),
                LLMResult(text="[curious] Naya thread, kya baat karni hai?"),
                LLMResult(text="[happy] Haan, Friday wala exam yaad hai."),
            ])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            await bot.chat("u", "mera exam Friday ko hai")
            await bot.wait_background()
            main_before = await bot.store.current_conversation("u")

            thread = await bot.store.create_thread("u")
            self.assertRegex(thread, r"^t-[0-9a-f]{8}$")
            await bot.chat("u", "ye alag baat hai", thread=thread)
            await bot.wait_background()
            self.assertNotEqual(await bot.store.current_conversation("u", thread), main_before)
            # The separate thread's prompt does not contain the main thread's chat.
            thread_prompt = [c for c in llm.calls if c.get("tools") is not None][-1]["messages"]
            self.assertNotIn("mera exam Friday ko hai", [m.content for m in thread_prompt])

            # Back on main: same conversation as before, with its whole history.
            await bot.chat("u", "exam yaad hai na?")
            await bot.wait_background()
            self.assertEqual(await bot.store.current_conversation("u"), main_before)
            main_rows = await bot.store.latest_messages("u", 20)
            self.assertEqual([m["content"] for m in main_rows if m["role"] == "user"], ["mera exam Friday ko hai", "exam yaad hai na?"])
            thread_rows = await bot.store.latest_messages("u", 20, thread)
            self.assertEqual([m["content"] for m in thread_rows if m["role"] == "user"], ["ye alag baat hai"])
            await pool.close()

class ThreadIsolationTests(unittest.TestCase):
    """Each thread is its own world: memory, notes, reminders, to-dos and lists never cross threads."""

    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    @unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
    def test_thread_two_starts_empty_and_stays_apart(self):
        async def run():
            from companion.store.postgres import data_owner

            pool = await _fresh_pool()
            past = (datetime.now(TZ) - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M")
            llm = ScriptedLLM([
                LLMResult(tool_calls=[ToolCall(id="a", name="ltm_save", arguments=json.dumps(
                    {"category": "profile", "subject": "me", "topic": "name", "text": "Name is Ashish"}))]),
                LLMResult(tool_calls=[ToolCall(id="b", name="note_save", arguments=json.dumps(
                    {"title": "Wifi", "text": "Wifi password abc123", "keywords": ["wifi"]}))]),
                LLMResult(tool_calls=[ToolCall(id="c", name="find_saved", arguments=json.dumps({"query": "wifi password"}))]),
                LLMResult(text="[idle] Nothing saved here."),
                LLMResult(tool_calls=[ToolCall(id="d", name="ltm_save", arguments=json.dumps(
                    {"category": "profile", "subject": "me", "topic": "name", "text": "Name is Rohan"}))]),
            ])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            await bot.chat("u", "remember long term: my name is Ashish")
            await bot.chat("u", "note down the wifi password abc123")
            thread = await bot.store.create_thread("u")
            await bot.chat("u", "what's the wifi password?", thread=thread)
            # Thread 2's prompt has no pinned profile from main, and its search found nothing.
            thread_system = [c for c in llm.calls if c.get("tools") is not None][-2]["messages"][0].content
            self.assertNotIn("Name is Ashish", thread_system)
            searched = [m for m in [c for c in llm.calls if c.get("tools") is not None][-1]["messages"] if m.role == "tool"]
            self.assertIn('"count": 0', searched[-1].content)
            await bot.chat("u", "remember long term: my name is Rohan", thread=thread)
            self.assertEqual([i.text for i in await bot.memory.list_all("u")], ["Name is Ashish"])
            self.assertEqual([i.text for i in await bot.memory.list_all(data_owner("u", thread))], ["Name is Rohan"])
            # Due reminders of every thread still reach the user.
            await bot.store.reminder_add(data_owner("u", thread), "water", datetime.now(TZ) - timedelta(minutes=1), "none", [])
            self.assertEqual([r["text"] for r in await bot.store.take_due_reminders("u")], ["water"])
            numbers = [(t["thread"], t["number"]) for t in await bot.store.list_threads("u")]
            self.assertEqual(numbers, [("main", 1), (thread, 2)])
            await pool.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
