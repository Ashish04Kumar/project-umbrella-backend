"""The HTTP server and playground, with a scripted LLM and a throwaway Postgres schema."""

from __future__ import annotations

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
from tests.test_postgres import SETTINGS, TZ, ScriptedLLM, _drop


@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class ServerTests(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        import asyncio
        asyncio.run(_drop())

    def test_chat_and_side_panels(self):
        tomorrow = (datetime.now(TZ) + timedelta(days=(4 - datetime.now(TZ).weekday()) % 7 or 7)).strftime("%Y-%m-%dT09:00")  # next Friday (the messages say "Friday")
        llm = ScriptedLLM(
            [
                LLMResult(tool_calls=[ToolCall(id="c1", name="reminder_add", arguments=json.dumps({"text": "dawai", "when": tomorrow}))]),
                LLMResult(tool_calls=[
                    ToolCall(id="c2", name="todo_add", arguments=json.dumps({"items": ["call the bank"]})),
                    ToolCall(id="c3", name="list_add", arguments=json.dumps({"list": "shopping", "items": ["milk", "eggs"]})),
                ]),
            ],
        )

        async def make_bot():
            await _drop()
            pool = await db.connect(SETTINGS.database_url, SETTINGS.db_schema)
            return Companion(settings=SETTINGS, llm=llm, store=PostgresStore(pool), memory=PostgresMemory(pool), pool=pool)

        with TestClient(create_app(make_bot, warm_up_voice=False)) as client:
            self.assertIn("Companion Playground", client.get("/").text)
            self.assertTrue(client.get("/health").json()["ok"])
            reply = client.post("/v1/chat", json={"user_id": "u", "text": "remind me on Friday at 9 to take medicine"}).json()
            self.assertTrue(reply["text"].startswith("Reminder set: dawai, "), reply["text"])
            self.assertEqual((reply["expression"], reply["tools_used"]), ("happy", ["reminder_add"]))
            self.assertIsInstance(reply["ms"], int)
            history = client.get("/v1/users/u/history").json()["messages"]
            self.assertEqual([m["role"] for m in history], ["user", "assistant"])
            self.assertEqual(history[1]["expression"], "happy")
            self.assertEqual([r["text"] for r in client.get("/v1/users/u/reminders").json()["reminders"]], ["dawai"])
            self.assertEqual(client.get("/v1/users/u/reminders/due").json()["due"], [])
            self.assertEqual(client.get("/v1/users/u/lists").json(), {"lists": {}, "todos": []})
            client.post("/v1/chat", json={"user_id": "u", "text": "add call the bank to my todos and milk, eggs to my shopping list"})
            lists = client.get("/v1/users/u/lists").json()
            self.assertEqual([t["text"] for t in lists["todos"]], ["call the bank"])
            self.assertEqual([i["text"] for i in lists["lists"]["shopping"]], ["milk", "eggs"])
            # Clear buttons (the page asks "are you sure?" before calling these).
            self.assertEqual(client.post("/v1/users/u/clear", json={"store": "list", "list": "shopping"}).json(), {"removed": 2})
            self.assertEqual(client.post("/v1/users/u/clear", json={"store": "todos"}).json(), {"removed": 1})
            self.assertEqual(client.post("/v1/users/u/clear", json={"store": "everything"}).status_code, 422)
            # A due reminder: said once, then Done / Snooze from its buttons.
            reminder_id = client.get("/v1/users/u/reminders").json()["reminders"][0]["id"]
            self.assertEqual(client.post("/v1/users/u/reminders/nope/done").status_code, 404)
            snoozed = client.post(f"/v1/users/u/reminders/{reminder_id}/snooze?minutes=10")
            self.assertEqual(snoozed.status_code, 200)
            self.assertEqual(client.get("/v1/users/u/reminders").json()["reminders"][0]["status"], "snoozed")
            self.assertEqual(client.post(f"/v1/users/u/reminders/{reminder_id}/done").json(), {"ok": True})
            self.assertEqual(client.get("/v1/users/u/reminders").json()["reminders"], [])
            self.assertEqual(client.post("/v1/chat", json={"user_id": "u", "text": ""}).status_code, 422)
            self.assertEqual(client.delete("/v1/users/u/memory/deadbeef").status_code, 404)
            # Front-end plugins: served only when switched on, and never outside their folder.
            self.assertIn("robot", client.get("/health").json()["plugins"])
            robot = client.get("/plugins/robot/plugin.js")
            self.assertEqual((robot.status_code, robot.headers["content-type"].split(";")[0]), (200, "text/javascript"))
            self.assertIn("export function mount", robot.text)
            self.assertEqual(client.get("/plugins/robot/gg2-head.glb").status_code, 200)
            self.assertEqual(client.get("/plugins/other/plugin.js").status_code, 404)
            self.assertEqual(client.get("/plugins/robot/..%2F..%2Fapp.py").status_code, 404)


    def test_edit_and_delete_from_the_panels(self):
        from companion.memory.interface import MemoryItem

        async def make_bot():
            await _drop()
            pool = await db.connect(SETTINGS.database_url, SETTINGS.db_schema)
            return Companion(settings=SETTINGS, llm=ScriptedLLM([]), store=PostgresStore(pool), memory=PostgresMemory(pool), pool=pool)

        with TestClient(create_app(make_bot, warm_up_voice=False)) as client:
            bot = client.app.state.bot
            call = client.portal.call
            call(bot.store.ensure_user, "u", SETTINGS.timezone)
            memory_id = call(bot.memory.remember, "u", MemoryItem(scope="long", category="person", subject="Riya", topic="city", text="Lives in Mumbai"))
            note_id = call(bot.store.note_save, "u", "Wifi password abc123", "Wifi", ["wifi"], [])
            tomorrow = datetime.now(TZ) + timedelta(days=1)
            reminder = call(bot.store.reminder_add, "u", "gym", tomorrow, "none", [])
            todo = call(bot.store.todo_add, "u", ["call the bank"], tomorrow.date(), "normal", [])[0]
            call(bot.store.list_add, "u", "shopping", ["milk"])
            item_id = client.get("/v1/users/u/lists").json()["lists"]["shopping"][0]["id"]
            url = lambda kind, item: f"/v1/users/u/items/{kind}/{item[:8]}"   # noqa: E731  (the panels use short ids)

            self.assertEqual(client.patch(url("memory", memory_id), json={"text": "Lives in Bangalore"}).json(), {"ok": True})
            self.assertEqual([m["text"] for m in client.get("/v1/users/u/memory").json()["long_term"]], ["Lives in Bangalore"])
            client.patch(url("note", note_id), json={"title": "Home wifi", "text": "Wifi password xyz789"})
            self.assertEqual([(n["title"], n["text"]) for n in client.get("/v1/users/u/notes").json()["notes"]], [("Home wifi", "Wifi password xyz789")])
            later = (tomorrow + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M")
            client.patch(url("reminder", reminder["id"]), json={"text": "gym and swim", "when": later, "repeat": "weekly"})
            shown = client.get("/v1/users/u/reminders").json()["reminders"][0]
            due = datetime.fromisoformat(shown["due_at"]).astimezone(TZ).strftime("%Y-%m-%dT%H:%M")   # the API sends UTC
            self.assertEqual((shown["text"], shown["repeat"], due), ("gym and swim", "weekly", later))
            self.assertEqual(client.patch(url("reminder", reminder["id"]), json={"when": "2020-01-01T09:00"}).status_code, 422)
            client.patch(url("todo", todo["id"]), json={"text": "call the bank at 10", "priority": "high", "done": True})
            self.assertEqual([(t["text"], t["priority"], t["done"]) for t in client.get("/v1/users/u/lists").json()["todos"]],
                             [("call the bank at 10", "high", True)])
            client.patch(url("list_item", item_id), json={"text": "milk 2 L", "done": True})
            self.assertEqual(client.get("/v1/users/u/lists").json()["lists"]["shopping"], [{"id": item_id, "text": "milk 2 L", "done": True}])

            for kind, row_id in (("memory", memory_id), ("note", note_id), ("reminder", reminder["id"]), ("todo", todo["id"]), ("list_item", item_id)):
                self.assertEqual(client.delete(url(kind, row_id)).status_code, 200, kind)
                self.assertEqual(client.delete(url(kind, row_id)).status_code, 404, kind)
            self.assertEqual(client.patch("/v1/users/u/items/other/abc", json={"text": "x"}).status_code, 404)


if __name__ == "__main__":
    unittest.main()
