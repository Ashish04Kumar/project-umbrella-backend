"""The live trace: the hub itself, the events of a full turn, and the memory state endpoint."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from companion.brain import Companion
from companion.llm.base import LLMResult, ToolCall
from companion.memory.postgres import PostgresMemory
from companion.server.app import create_app
from companion.store import db
from companion.store.postgres import PostgresStore
from companion.trace import TraceHub
from tests.test_postgres import SETTINGS, TZ, ScriptedLLM, _drop, _fresh_pool


class HubTests(unittest.TestCase):
    def test_recent_and_subscribers_per_user(self):
        async def run():
            hub = TraceHub()
            queue = hub.subscribe("a")
            hub.emit("a", "turn_start", "t1", text="hi")
            hub.emit("b", "turn_start", "t2", text="other user")
            self.assertEqual((await queue.get())["data"], {"text": "hi"})
            self.assertTrue(queue.empty())
            self.assertEqual([e["turn"] for e in hub.recent("a")], ["t1"])
            hub.unsubscribe("a", queue)
            hub.emit("a", "turn_end", "t1")
            self.assertTrue(queue.empty())
        asyncio.run(run())


@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class TurnTraceTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    def test_a_tool_turn_emits_every_step(self):
        async def run():
            pool = await _fresh_pool()
            tomorrow = (datetime.now(TZ) + timedelta(days=(4 - datetime.now(TZ).weekday()) % 7 or 7)).strftime("%Y-%m-%dT08:00")  # next Friday (the messages say "Friday")
            llm = ScriptedLLM([
                LLMResult(
                    tool_calls=[ToolCall(id="c1", name="reminder_add", arguments=json.dumps({"text": "gift order", "when": tomorrow}))],
                    reasoning="Need a reminder tomorrow 8 AM.", usage={"prompt": 900, "completion": 40, "reasoning": 12, "total": 940},
                ),
            ])
            bot = Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool))
            await bot.chat("u", "remind me on Friday at 8 to order the gift")
            await bot.wait_background()
            events = bot.trace.recent("u")
            kinds = [e["kind"] for e in events]
            # A write tool confirms in code: one LLM call, then straight to the guard.
            self.assertEqual(kinds, ["turn_start", "context", "llm_call", "llm_request", "llm_result", "db_change", "tool_result", "code_reply", "guard", "turn_end", "memory_skip"])
            self.assertEqual(len({e["turn"] for e in events}), 1)  # one turn id for the whole turn
            by_kind = {e["kind"]: e["data"] for e in events}
            first_llm = by_kind["llm_result"]
            self.assertEqual(first_llm["reasoning"], "Need a reminder tomorrow 8 AM.")
            self.assertEqual(first_llm["tool_calls"][0]["name"], "reminder_add")
            self.assertEqual(first_llm["usage"]["total"], 940)
            self.assertTrue(by_kind["tool_result"]["ok"])
            self.assertIn("est_prompt_tokens", by_kind["context"])
            self.assertEqual(by_kind["context"]["groups"], ["read", "reminder"])
            self.assertIn("auto memory is off", by_kind["memory_skip"]["why"])
            # The Test tab sees the whole request: system prompt first, the user's message last.
            sent = by_kind["llm_request"]
            self.assertEqual((sent["messages"][0]["role"], sent["messages"][-1]["content"]), ("system", "remind me on Friday at 8 to order the gift"))
            self.assertIn("reminder_add", sent["tools"])
            await pool.close()
        asyncio.run(run())

    def test_memory_state_endpoint(self):
        llm = ScriptedLLM([
            LLMResult(tool_calls=[
                ToolCall(id="c1", name="stm_add", arguments=json.dumps({"text": "Exam on Friday", "keywords": ["exam"]})),
                ToolCall(id="c2", name="ltm_save", arguments=json.dumps(
                    {"category": "preference", "subject": "me", "topic": "food", "text": "Likes paneer"})),
            ]),
        ])

        async def make_bot():
            await _drop()
            pool = await db.connect(SETTINGS.database_url, SETTINGS.db_schema)
            return Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool), pool=pool)

        with TestClient(create_app(make_bot, warm_up_voice=False)) as client:
            client.post("/v1/chat", json={"user_id": "u", "text": "remember: I like paneer long term, and short term: exam on Friday"})
            state = client.get("/v1/users/u/memory/state").json()
            self.assertEqual([(m["category"], m["subject"], m["text"]) for m in state["long_term"]["memories"]], [("preference", "me", "Likes paneer")])
            self.assertEqual([(m["scope"], m["text"]) for m in state["short_term"]["working_memory"]], [("short", "Exam on Friday")])
            self.assertIsNotNone(state["short_term"]["working_memory"][0]["expires_at"])   # 10 days
            self.assertEqual([(m["role"], m["where"]) for m in state["short_term"]["messages"]], [("user", "window"), ("assistant", "window")])
            self.assertEqual(state["rules"]["window"], SETTINGS.recent_messages)
            self.assertFalse(state["rules"]["auto_memory"])
            # The side panel and the clear button.
            self.assertEqual(len(client.get("/v1/users/u/memory").json()["short_term"]), 1)
            self.assertEqual(client.post("/v1/users/u/clear", json={"store": "short"}).json(), {"removed": 1})
            self.assertEqual(client.get("/v1/users/u/memory").json()["short_term"], [])
            self.assertEqual(client.get("/v1/users/nobody/memory/state").json()["short_term"]["conversation"], None)


if __name__ == "__main__":
    unittest.main()
