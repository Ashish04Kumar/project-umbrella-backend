"""Golden set: the sample data from docs/search-and-retrieval-spec.md (section 5) and the tool calls from
its question tables (section 6), run against a real Postgres. The LLM is not involved: this checks that the
tool the LLM would call finds the right rows. Skipped without COMPANION_DATABASE_URL."""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import date, datetime, time, timedelta

from companion.llm.base import ToolCall
from companion.memory.interface import MemoryItem
from companion.memory.postgres import PostgresMemory
from companion.store.postgres import PostgresStore
from companion.tools.builtin import build_registry
from companion.tools.registry import ToolContext

from .test_postgres import SETTINGS, TZ, _drop, _fresh_pool

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=TZ)   # the spec's "today": Monday 19 Oct 2026
U = "golden"


def day(month: int, d: int) -> date:
    return date(2026, month, d)


async def seed(store: PostgresStore, memory: PostgresMemory) -> None:
    expires = datetime.combine(day(10, 30), time(), tzinfo=TZ)
    short = [
        ("Paid Rahul 2500", day(10, 17), ["payment", "money"], ["Rahul"]),
        ("Car given for service", day(10, 17), ["car", "service"], []),
        ("Amit's parcel arrives", day(10, 20), ["parcel", "delivery"], ["Amit"]),
        ("On a diet this week", day(10, 19), ["diet", "food"], []),
        ("Passport office visit", day(10, 24), ["passport", "office"], []),
        ("Rent pending to landlord", None, ["rent", "payment", "baaki"], ["landlord"]),
    ]
    for text, when, keywords, entities in short:
        await memory.remember(U, MemoryItem(scope="short", category="other", text=text, event_date=when,
                                            keywords=keywords, entities=entities, expires_at=expires))
    long = [
        ("profile", "me", "name", "Name is Ashish", None, False, True),
        ("profile", "me", "city", "Lives in Pune", None, False, True),
        ("profile", "me", "blood group", "Blood group B+", None, False, True),
        ("person", "Riya", "relation", "Sister, lives in Mumbai", None, False, False),
        ("date", "Riya", "birthday", "Riya's birthday", day(5, 3), True, False),
        ("person", "Amit", "relation", "College friend, lives in Delhi", None, False, False),
        ("person", "Rahul", "relation", "Colleague, designer", None, False, False),
        ("preference", "Mummy", "food", "Likes rasmalai", None, False, False),
        ("date", "Parents", "anniversary", "Parents' wedding anniversary", day(12, 12), True, False),
        ("project", "Project Synq", "about", "Project Synq, a dating app", None, False, False),
        ("project", "Companion Umbrella", "about", "Companion Umbrella, an AI buddy", None, False, False),
        ("preference", "me", "food", "Likes paneer", None, False, False),
    ]
    for category, subject, topic, text, when, yearly, pinned in long:
        await memory.remember(U, MemoryItem(scope="long", category=category, subject=subject, topic=topic, text=text,
                                            event_date=when, yearly=yearly, pinned=pinned, entities=[subject] if subject != "me" else []))
    at = lambda month, d, h, m=0: datetime(2026, month, d, h, m, tzinfo=TZ)   # noqa: E731
    await store.reminder_add(U, "Dawai", at(10, 20, 9), "daily", [])
    await store.reminder_add(U, "Gym", at(10, 23, 18), "none", [])
    await store.reminder_add(U, "Team call", at(10, 26, 10), "weekly", [])
    await store.reminder_add(U, "Rent", at(11, 5, 10), "monthly", [])
    todos = [
        (day(10, 18), ["Bank jaana", "Groceries"], "normal"),
        (day(10, 19), ["Doodh lana", "Amit ko call"], "normal"),
        (day(10, 19), ["Bill bharna"], "high"),
        (day(10, 20), ["Passport documents ready"], "normal"),
        (day(10, 23), ["Presentation banana"], "high"),
        (day(10, 23), ["Riya ke liye gift order"], "normal"),
    ]
    for when, items, priority in todos:
        await store.todo_add(U, items, when, priority, [])
    for text in ("Groceries", "Doodh lana"):
        found = await store.todo_find(U, text)
        await store.todo_mark_done(U, [found[0]["id"]])
    notes = [
        ("Wifi", "Wifi password abc123", ["wifi", "internet", "password", "code"], []),
        ("App idea", "Voice se kharcha add karna", ["app", "idea", "expense"], []),
        ("Amit's address", "21 MG Road, Delhi", ["address", "ghar"], ["Amit"]),
        ("Books", "Atomic Habits, Deep Work", ["books", "reading"], []),
        ("Car insurance", "XYZ123", ["car", "insurance", "policy"], []),
        ("Meeting", "Launch 1 Nov; Rahul design; action: logo final, pricing page", ["meeting", "launch"], ["Rahul"]),
        ("Doctor", "Dawai din mein 2 baar", ["doctor", "medicine", "dawai"], []),
    ]
    for title, text, keywords, entities in notes:
        await store.note_save(U, text, title, keywords, entities)
    # The spec's sample data was saved on its "today" (19 Oct); undated items are searched by that day.
    await store.pool.execute("UPDATE memories SET created_at = $2, updated_at = $2 WHERE user_id = $1", U, NOW)
    await store.list_add(U, "shopping", ["ande", "bread", "aloo 2 kg", "doodh"])
    await store.list_add(U, "watchlist", ["Dune", "Oppenheimer"])
    await store.list_add(U, "goa packing", ["charger", "sunscreen"])
    await store.list_add(U, "books", ["Atomic Habits"])
    for name, item in (("shopping", "doodh"), ("watchlist", "Dune"), ("goa packing", "sunscreen")):
        _, rows = await store.list_items_match(U, name, item, unchecked_only=True)
        await store.list_items_update([rows[0]["id"]], check=True)


@unittest.skipUnless(SETTINGS.database_url, "COMPANION_DATABASE_URL is not set")
class GoldenSearchTests(unittest.TestCase):
    """Each check is one row of the spec's question tables: the tool call and what must come back."""

    @classmethod
    def tearDownClass(cls):
        asyncio.run(_drop())

    def test_golden_set(self):
        async def run():
            pool = await _fresh_pool()
            store, memory = PostgresStore(pool), PostgresMemory(pool)
            await store.ensure_user(U, SETTINGS.timezone)
            await seed(store, memory)
            ctx = ToolContext(user_id=U, tz=TZ, now=NOW, store=store, memory=memory, settings=SETTINGS)
            registry = build_registry(SETTINGS)

            async def call(name: str, **args):
                result = await registry.run_tool(ToolCall(id="g", name=name, arguments=json.dumps(args)), ctx)
                self.assertNotIn("error", result, f"{name}({args}): {result}")
                return result

            async def texts(**args) -> list[str]:
                return [r["text"] for r in (await call("find_saved", **args))["results"]]

            # 6.1 short-term memory
            self.assertEqual(len(await texts(**{"in": ["short"]})), 6)
            self.assertEqual((await texts(entities=["Rahul"], **{"in": ["short"]}))[0], "Paid Rahul 2500")
            self.assertEqual(sorted(await texts(date_from="2026-10-17", date_to="2026-10-17", **{"in": ["short"]})),
                             ["Car given for service", "Paid Rahul 2500"])
            self.assertEqual(await texts(date_from="2026-10-20", date_to="2026-10-20", **{"in": ["short"]}), ["Amit's parcel arrives"])
            self.assertEqual((await texts(query="passport"))[0], "Passport office visit")
            self.assertIn("Rent pending to landlord", await texts(query="payment rent baaki", **{"in": ["short"]}))
            self.assertEqual((await texts(query="car service"))[0], "Car given for service")
            # 6.2 long-term memory
            riya = await texts(subject="Riya")
            self.assertIn("Riya's birthday", riya)
            self.assertIn("Sister, lives in Mumbai", riya)
            self.assertEqual(await texts(category="date", month=10, **{"in": ["long"]}), [])
            self.assertEqual(sorted(await texts(category="date", month=5, **{"in": ["long"]})), ["Riya's birthday"])
            me = await texts(subject="me", **{"in": ["long"]})
            self.assertTrue({"Name is Ashish", "Lives in Pune", "Blood group B+", "Likes paneer"} <= set(me), me)
            self.assertEqual(sorted(await texts(category="project", **{"in": ["long"]})),
                             ["Companion Umbrella, an AI buddy", "Project Synq, a dating app"])
            self.assertEqual(await texts(subject="Mummy", category="preference"), ["Likes rasmalai"])
            self.assertEqual((await texts(query="anniversary"))[0], "Parents' wedding anniversary")
            self.assertEqual((await texts(subject="me", query="blood group"))[0], "Blood group B+")
            delhi = await texts(query="Delhi")
            self.assertIn("College friend, lives in Delhi", delhi)
            self.assertIn("21 MG Road, Delhi", delhi)
            # 6.5 notes (through the same search)
            self.assertEqual((await texts(query="wifi password"))[0], "Wifi password abc123")
            self.assertEqual((await texts(query="internet code"))[0], "Wifi password abc123")      # keywords hit
            self.assertEqual((await texts(query="Amit address"))[0], "21 MG Road, Delhi")
            self.assertEqual((await texts(query="book", **{"in": ["notes"]}))[0], "Atomic Habits, Deep Work")
            self.assertEqual((await texts(query="car insurance"))[0], "XYZ123")
            self.assertEqual((await texts(query="doctor"))[0], "Dawai din mein 2 baar")
            self.assertIn("Launch 1 Nov", (await texts(query="launch"))[0])
            self.assertEqual(len(await texts(**{"in": ["notes"]})), 7)
            self.assertIn("Wifi password abc123", await texts(query="wifi pasword"))               # typo still found

            # 6.3 reminders
            rem = lambda r: [x["text"] for x in r["reminders"]]   # noqa: E731
            self.assertEqual(rem(await call("reminder_list")), ["Dawai", "Gym", "Team call", "Rent"])
            self.assertEqual(rem(await call("reminder_list", query="gym")), ["Gym"])
            self.assertEqual(rem(await call("reminder_list", repeat="daily")), ["Dawai"])
            self.assertEqual(rem(await call("reminder_list", query="rent")), ["Rent"])
            # 6.4 to-dos
            todo = lambda r: [x["text"] for x in r["todos"]]   # noqa: E731
            today = await call("todo_list", day="2026-10-19", status="pending")
            self.assertEqual(todo(today), ["Bill bharna", "Amit ko call"])                         # high first
            self.assertIn("Bank jaana", [x["text"] for x in today.get("overdue_from_earlier_days", [])])
            self.assertEqual(sorted(todo(await call("todo_list", day="2026-10-18", status="all"))), ["Bank jaana", "Groceries"])
            self.assertEqual(todo(await call("todo_list", day="2026-10-19", status="done")), ["Doodh lana"])
            self.assertEqual(todo(await call("todo_list", status="pending", priority="high")), ["Bill bharna", "Presentation banana"])
            self.assertEqual(todo(await call("todo_list", query="bank")), ["Bank jaana"])
            self.assertEqual(todo(await call("todo_list", query="Riya")), ["Riya ke liye gift order"])
            # 6.6 lists
            shopping = await call("list_show", list="shopping")
            self.assertEqual(shopping["count"], {"total": 4, "left": 3, "checked": 1})
            self.assertEqual(len((await call("list_show"))["lists"]), 4)
            self.assertEqual((await call("list_show", list="shoping"))["list"], "shopping")         # typo in the name
            watch = await call("list_show", list="watchlist")
            self.assertEqual([i["text"] for i in watch["items"] if not i["checked"]], ["Oppenheimer"])
            await pool.close()
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
