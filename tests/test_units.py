"""Unit tests that need no database or network. Run: .venv/bin/python -m unittest discover tests"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from companion.config import Settings
from companion.guard import FALLBACK_REPLY, clean_reply, confirm_from_tools, looks_broken
from companion.llm.base import LLMError, LLMResult, ToolCall
from companion.llm.router import LLMRouter
from companion.memory.extractor import ExtractedItem, decide, parse_items
from companion.memory.in_memory import InMemoryMemory
from companion.memory.interface import MemoryItem, normalize_key
from companion.tools.builtin import build_registry, safe_eval
from companion.tools.registry import ToolContext, ToolError

TZ = ZoneInfo("Asia/Kolkata")


class GuardTests(unittest.TestCase):
    def test_a_denial_is_not_an_action_claim(self):
        from companion.guard import claims_an_action

        self.assertFalse(claims_an_action("Could you specify which office? I don't have details about your workplace saved in your memory or notes."))
        self.assertFalse(claims_an_action("I have nothing saved about that."))
        self.assertTrue(claims_an_action("Saved it in your notes."))
        self.assertTrue(claims_an_action("Reminder set for 7 pm."))
        self.assertTrue(claims_an_action("No problem, reminder set for 7 pm."))
        self.assertTrue(claims_an_action("Don't worry, I saved it in your notes."))

    def test_tag_is_pulled_out(self):
        self.assertEqual(clean_reply("[curious] Achha, phir kya hua?"), ("Achha, phir kya hua?", "curious"))

    def test_new_moods(self):
        self.assertEqual(clean_reply("[surprised] Arre wah!")[1], "surprised")
        self.assertEqual(clean_reply("[excited] Badhai ho! [happy] Party?"), ("Badhai ho! Party?", "excited"))
        self.assertEqual(clean_reply("[sad] Ye sunke bura laga.")[1], "sad")

    def test_missing_or_unknown_tag_is_idle(self):
        self.assertEqual(clean_reply("Theek hai.")[1], "idle")
        self.assertEqual(clean_reply("[angry] Theek hai.")[1], "idle")

    def test_hidden_reasoning_removed(self):
        self.assertEqual(clean_reply("<think>plan</think>[happy] Wah!"), ("Wah!", "happy"))

    def test_human_life_claim_dropped(self):
        text, _ = clean_reply("[happy] Nice! Yesterday I went to that cafe too. Kaisa laga?")
        self.assertEqual(text, "Nice! Kaisa laga?")

    def test_unpaired_quote_removed(self):
        self.assertEqual(clean_reply('[idle] Soup bhi badiya rahega."')[0], "Soup bhi badiya rahega.")
        self.assertEqual(clean_reply('[idle] Usne kaha "haan" tha.')[0], 'Usne kaha "haan" tha.')

    def test_empty_reply_gets_fallback(self):
        self.assertEqual(clean_reply("[idle] ")[0], FALLBACK_REPLY)

    def test_broken_output_detected(self):
        self.assertTrue(looks_broken("[curious] Parson (2024? actually   …  ) ? ? ? ... ... ... Oops… … ..."))
        self.assertFalse(looks_broken("[happy] Haha, Ferrari ka sapna toh mast hai! Plan banayein?"))
        self.assertFalse(looks_broken("[curious] ब्लैक होल अंतरिक्ष में ऐसा क्षेत्र है जहाँ गुरुत्वाकर्षण बहुत तेज़ होता है।"))

    def test_confirmation_from_tool_results(self):
        results = [
            {"name": "reminder_add", "result": {"ok": True, "say": "Reminder set: dawai lena, Tue 20 Oct, 9:00 AM."}},
            {"name": "todo_add", "result": {"ok": False, "error": "Invalid arguments"}},
            {"name": "reminder_add", "result": {"ok": True, "ask": "When should I remind you about 'gym'?"}},
        ]
        self.assertEqual(
            confirm_from_tools(results),
            "Reminder set: dawai lena, Tue 20 Oct, 9:00 AM. That didn't work: Invalid arguments "
            "When should I remind you about 'gym'?",
        )
        self.assertIsNone(confirm_from_tools([]))

    def test_code_reply_keeps_the_users_words(self):
        said = "[happy] Saved in long-term memory: Riya · birthday: My sister Riya's birthday is 3 May."
        self.assertEqual(clean_reply(said)[0], FALLBACK_REPLY)                 # from the LLM this reads as a made-up family
        self.assertIn("My sister Riya", clean_reply(said, from_code=True)[0])  # built by code from the tool: kept

    def test_made_up_action_detected(self):
        from companion.guard import claims_an_action
        for text in ('Saved to notes: "Wifi password".', "Added to your shopping list: milk.", "Reminder set for 9.", "I've noted that down.",
                     "Updated: Riya now lives in Bangalore.", "Sure. Moved 'gym' to Saturday."):
            self.assertTrue(claims_an_action(text), text)
        self.assertTrue(claims_an_action("Badhiya! Milk aur bananas list se tick off kar diye, ab bas bread bacha hai."))
        self.assertFalse(claims_an_action("Tune Karan ko 500 rupees udhaar diye the."))
        for text in ("Riya's birthday is 3 May.", "Want me to add it to your list?", "You have 3 things on your list."):
            self.assertFalse(claims_an_action(text), text)

    def test_messy_output_salvaged(self):
        """Real gpt-oss outputs from the live test (19 Oct): junk and leaked planning before the answer."""
        from companion.guard import looks_messy, salvage
        junk = [
            ('[curious] What\xa0?\xa0?\xa0?? \n\n[curious] Your shopping list still has: milk, eggs.', "Your shopping list still has: milk, eggs."),
            ("[curly\xa0?] \n\n[\xa0?\xa0] \n\nWe need to respond with proper format: start with mood tag. "
             "We need correct answer.Today you have two to-dos: pay the bill and call Amit.", "Today you have two to-dos: pay the bill and call Amit."),
        ]
        for raw, answer in junk:
            self.assertTrue(looks_messy(raw))
            self.assertEqual(clean_reply(salvage(raw))[0], answer)
        for clean in ("[happy] Riya's birthday is 3 May. Want to plan something?", "[curious] Is it 9 am? Or 10?"):
            self.assertFalse(looks_messy(clean))
            self.assertEqual(salvage(clean), clean)

    def test_our_instruction_note_never_reaches_the_user(self):
        raw = "(Their latest message is in English: reply in English, not Hinglish.)\n\nThat's true! 8k steps is great."
        self.assertEqual(clean_reply(raw)[0], "That's true! 8k steps is great.")

    def test_made_up_change_gets_an_honest_reply(self):
        from companion.guard import UNCHANGED_REPLY, UNSAVED_REPLY, unsaved_reply
        self.assertEqual(unsaved_reply("no no please clear the latest reminder"), UNCHANGED_REPLY)
        self.assertEqual(unsaved_reply("my mom's birthday is 14 Nov"), UNSAVED_REPLY)

    def test_line_breaks_kept(self):
        self.assertEqual(clean_reply("[idle] Steps:\n1. Uthao\n2. Chalo")[0], "Steps:\n1. Uthao\n2. Chalo")


class ExtractorDecisionTests(unittest.TestCase):
    def item(self, **values):
        base = {"type": "preference", "key": "food:paneer", "text": "Likes paneer", "confidence": 0.9}
        return ExtractedItem(**{**base, **values})

    def test_rules(self):
        self.assertEqual(decide(self.item()), "save")
        self.assertEqual(decide(self.item(confidence=0.6)), "ask")
        self.assertEqual(decide(self.item(confidence=0.3)), "ignore")
        self.assertEqual(decide(self.item(confidence=0.3, explicit=True)), "save")
        self.assertEqual(decide(self.item(hypothetical=True, explicit=True)), "ignore")
        self.assertEqual(decide(self.item(sensitive=True)), "ignore")
        self.assertEqual(decide(self.item(sensitive=True, explicit=True)), "save")
        self.assertEqual(decide(self.item(type="none")), "ignore")

    def test_parse_skips_broken_items(self):
        raw = 'Sure: {"items": [{"type": "person", "key": "person:riya", "text": "Sister Riya", "confidence": 0.9}, {"type": "weird"}]}'
        items = parse_items(raw)
        self.assertEqual([i.key for i in items], ["person:riya"])
        self.assertEqual(parse_items("not json"), [])


class ToolTests(unittest.TestCase):
    def test_safe_eval(self):
        self.assertEqual(safe_eval("(1200*12)/100"), 144)
        self.assertEqual(safe_eval("2^10"), 1024)
        with self.assertRaises(ToolError):
            safe_eval("__import__('os')")

    def test_normalize_key(self):
        self.assertEqual(normalize_key(" Person : Riya  Sharma "), "person:riya_sharma")

    def test_reminder_in_past_is_refused_and_bad_args_reported(self):
        class Store:
            async def reminder_add(self, *a):  # pragma: no cover - must not be called
                raise AssertionError("should not save")

            async def log_tool(self, *a):
                pass

        now = datetime(2026, 10, 19, 18, 0, tzinfo=TZ)
        ctx = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(),
                          user_text="remind me to take dawai today at 9")
        registry = build_registry(Settings())
        # A time in the past and a missing time are questions back to the user, not saves.
        past = asyncio.run(registry.run_tool(ToolCall(id="1", name="reminder_add", arguments='{"text": "dawai", "when": "2026-10-19T09:00"}'), ctx))
        self.assertIn("already passed", past["ask"])
        broken = asyncio.run(registry.run_tool(ToolCall(id="2", name="reminder_add", arguments='{"text": "dawai"'), ctx))
        self.assertIn("not valid JSON", broken["error"])
        missing = asyncio.run(registry.run_tool(ToolCall(id="3", name="reminder_add", arguments='{"text": "dawai"}'), ctx))
        self.assertIn("fill `when`", missing["error"])   # they said "today at 9": the model must fill it, not ask
        unknown = asyncio.run(registry.run_tool(ToolCall(id="4", name="fly", arguments="{}"), ctx))
        self.assertFalse(unknown["ok"])

    def test_last_years_date_is_sent_back_with_this_weeks_dates(self):
        class Store:
            async def reminder_add(self, *a):  # pragma: no cover - must not be called
                raise AssertionError("should not save")

            async def log_tool(self, *a):
                pass

        now = datetime(2026, 10, 19, 18, 0, tzinfo=TZ)
        ctx = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(),
                          user_text="remind me to go to the gym on Friday at 6")
        result = asyncio.run(build_registry(Settings()).run_tool(
            ToolCall(id="1", name="reminder_add", arguments='{"text": "gym", "when": "2025-05-29T18:00"}'), ctx))
        self.assertFalse(result["ok"])
        self.assertIn("past year", result["error"])
        self.assertIn("Friday: Fri 23 Oct 2026", result["error"])

    def test_questions_with_last_years_date_are_sent_back_too(self):
        class Store:
            async def log_tool(self, *a):
                pass

        now = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        ctx = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings())
        registry = build_registry(Settings())
        for name, args in (("day_overview", '{"day": "2025-03-21"}'), ("todo_list", '{"day": "2025-03-22"}'),
                           ("reminder_list", '{"date_from": "2025-03-21", "date_to": "2025-03-21"}'),
                           ("clear_saved", '{"store": "todos", "day": "2025-03-21"}')):
            result = asyncio.run(registry.run_tool(ToolCall(id="1", name=name, arguments=args), ctx))
            self.assertIn("past year", result["error"], name)

    def test_close_last_year_dates_are_fixed_quietly(self):
        """Gemini Lite wrote "tomorrow" as 2025-10-21 five times in one run (English test, 20 Oct)."""
        from datetime import date

        from companion.tools.dates import fix_past_years

        today = date(2026, 10, 20)
        args = {"when": "2025-10-21T18:00", "days": ["2025-10-13", "2025-05-29"], "text": "2025-10-21 call mom", "n": 3}
        self.assertEqual(fix_past_years(args, today), {"when": "2026-10-21T18:00", "days": ["2026-10-13", "2025-05-29"],
                                                       "text": "2026-10-21 call mom", "n": 3})
        self.assertEqual(fix_past_years("2026-10-21", today), "2026-10-21")
        self.assertEqual(fix_past_years("2023-02-29", today), "2023-02-29")   # not a real date: left for the tool to refuse

        class Store:
            saved = None

            async def reminder_add(self, user_id, text, due, *rest, **kw):
                Store.saved = due
                return {"id": "r1", "text": text, "due_at": due, "repeat": "none", "duplicate": False, "all_day": False}

            async def reminders_waiting(self, *a, **kw):
                return []

            async def log_tool(self, *a):
                pass

        ctx = ToolContext(user_id="u", tz=TZ, now=datetime(2026, 10, 20, 12, 0, tzinfo=TZ), store=Store(), memory=InMemoryMemory(),
                          settings=Settings(), user_text="set a reminder for tomorrow 6 pm to call mom")
        result = asyncio.run(build_registry(Settings()).run_tool(
            ToolCall(id="1", name="reminder_add", arguments='{"text": "call mom", "when": "2025-10-21T18:00", "confidence": 1}'), ctx))
        self.assertTrue(result["ok"], result)
        self.assertEqual(Store.saved.date().isoformat(), "2026-10-21")

    def test_own_day_word_is_not_mixed_with_the_message_before(self):
        """'…open on Friday?' then Buddy asks a question, then 'add pay rent to my to-dos for tomorrow':
        tomorrow must not be checked against the old Friday (English test, 20 Oct)."""
        from companion.tools.builtin import _check_relative_day

        ctx = ToolContext(user_id="u", tz=TZ, now=datetime(2026, 10, 20, 12, 0, tzinfo=TZ), store=None, memory=InMemoryMemory(),
                          settings=Settings(), user_text="can u add pay rent to my to-dos for tomorrow?",
                          previous_user_text="is it actually open on Friday then?",
                          last_assistant="Do you want me to check your notes, or save it as a reminder?")
        self.assertIsNone(_check_relative_day(datetime(2026, 10, 21).date(), ctx, "pay rent"))

    def test_bulk_clear_needs_our_yes_no_question_first(self):
        class Store:
            cleared = False

            async def list_show(self, user_id, name):
                return {"found": True, "list": "shopping", "items": [], "count": {"total": 4, "left": 4, "checked": 0}}

            async def list_clear(self, user_id, name, checked_only):
                Store.cleared = True
                return {"found": True, "list": "shopping", "removed": 4}

            async def log_tool(self, *a):
                pass

        registry = build_registry(Settings())
        call = ToolCall(id="1", name="list_clear", arguments='{"list": "shopping", "confirm": true}')
        now = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        first = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(), last_assistant="Sure!")
        self.assertIn("(yes/no)", asyncio.run(registry.run_tool(call, first))["ask"])   # the model's confirm=true alone is not a yes
        self.assertFalse(Store.cleared)
        after = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(),
                            last_assistant="Clear all 4 items from the shopping list? (yes/no)")
        self.assertIn("Cleared 4", asyncio.run(registry.run_tool(call, after))["say"])

    def _reminder(self, now, user_text, when, last_assistant="", previous=""):
        class Store:
            async def reminder_add(self, user_id, text, due, repeat, entities, all_day=False):
                return {"id": "abcd1234", "text": text, "due_at": due, "repeat": repeat, "all_day": all_day}

            async def log_tool(self, *a):
                pass

        ctx = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(),
                          user_text=user_text, last_assistant=last_assistant, previous_user_text=previous)
        call = ToolCall(id="1", name="reminder_add", arguments=json.dumps({"text": "study", "when": when}))
        return asyncio.run(build_registry(Settings()).run_tool(call, ctx))

    def test_tomorrow_must_be_tomorrow(self):
        day = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)   # Tue morning
        wrong = self._reminder(day, "remind me tomorrow at 9 to study", "2026-10-22T09:00")
        self.assertIn("Wed 21 Oct", wrong["error"])                  # sent back with the right day
        self.assertIn("Reminder set", self._reminder(day, "remind me tomorrow at 9 to study", "2026-10-21T09:00")["say"])
        self.assertIn("Reminder set", self._reminder(day, "kal 9 baje study yaad dilana", "2026-10-21T09:00")["say"])
        self.assertIn("error", self._reminder(day, "aaj shaam 6 baje yaad dilana", "2026-10-21T18:00"))
        self.assertIn("Reminder set", self._reminder(day, "remind me on Friday at 9", "2026-10-23T09:00")["say"])

    def test_after_midnight_tomorrow_is_asked(self):
        night = datetime(2026, 10, 20, 2, 23, tzinfo=TZ)
        asked = self._reminder(night, "Schedule a reminder for tomorrow at 9 for study.", "2026-10-21T09:00")
        self.assertIn("Which day do you mean", asked["ask"])
        self.assertIn("Tue 20 Oct", asked["ask"])
        answered = self._reminder(night, "Wednesday", "2026-10-21T09:00", last_assistant=asked["ask"],
                                  previous="Schedule a reminder for tomorrow at 9 for study.")
        self.assertIn("Reminder set", answered["say"])

    def test_one_to_five_oclock_asks_am_or_pm(self):
        day = datetime(2026, 10, 20, 10, 0, tzinfo=TZ)
        asked = self._reminder(day, "kal subah do baje gym ka reminder", "2026-10-21T02:00")
        self.assertIn("2:00 AM (night) or 2:00 PM", asked["ask"])
        self.assertIn("Reminder set", self._reminder(day, "remind me tomorrow at 2 am", "2026-10-21T02:00")["say"])
        self.assertIn("Reminder set", self._reminder(day, "AM", "2026-10-21T02:00", last_assistant=asked["ask"],
                                                     previous="kal subah do baje gym ka reminder")["say"])

    def _save(self, name, args, user_text, last_assistant="", previous=""):
        class Store:
            async def todo_add(self, user_id, items, day, priority, entities):
                return [{"id": "t1", "text": item} for item in items]

            async def reminder_add(self, user_id, text, due, repeat, entities):
                return {"id": "r1", "text": text, "due_at": due, "repeat": repeat}

            async def log_tool(self, *a):
                pass

        ctx = ToolContext(user_id="u", tz=TZ, now=datetime(2026, 10, 20, 12, 0, tzinfo=TZ), store=Store(), memory=InMemoryMemory(),
                          settings=Settings(), user_text=user_text, last_assistant=last_assistant, previous_user_text=previous)
        return asyncio.run(build_registry(Settings()).run_tool(ToolCall(id="1", name=name, arguments=json.dumps(args)), ctx))

    def test_where_to_save(self):
        meeting = {"items": ["meeting"], "day": "2026-10-21"}
        # Not said where, not sure: ask, with the places.
        asked = self._save("todo_add", {**meeting, "confidence": 0.5}, "kal meeting hai")
        self.assertIn("Where should I save “meeting”? (reminder / to-do", asked["ask"])
        # Not said where, but sure enough: saved.
        self.assertIn("Added", self._save("todo_add", {**meeting, "confidence": 0.9}, "kal meeting hai")["say"])
        # Said where: saved there, whatever the confidence.
        self.assertIn("Added", self._save("todo_add", {**meeting, "confidence": 0.3}, "add to my to-do: meeting tomorrow")["say"])
        # Said where, but the model picked another place: sent back.
        wrong = self._save("todo_add", {**meeting, "confidence": 0.9}, "remind me about the meeting tomorrow at 10")
        self.assertIn("use reminder_add", wrong["error"])
        # The answer to "Where should I save" decides.
        answer = self._save("reminder_add", {"text": "meeting", "when": "2026-10-21T10:00"}, "Save it in reminder, at 10.", last_assistant=asked["ask"],
                            previous="kal meeting hai")
        self.assertIn("what time",   # no day or time in their words: asked, not guessed
                      self._save("reminder_add", {"text": "meeting", "when": "2026-10-21T09:00"}, "Save it in reminder.", last_assistant=asked["ask"])["ask"])
        self.assertIn("Reminder set", answer["say"])

    def test_an_update_stays_in_long_term(self):
        memory = InMemoryMemory()
        asyncio.run(memory.remember("u", MemoryItem(scope="long", category="person", subject="Riya", topic="details", text="Sister, lives in Mumbai")))

        class Store:
            async def log_tool(self, *a):
                pass

        ctx = ToolContext(user_id="u", tz=TZ, now=datetime(2026, 10, 20, 12, 0, tzinfo=TZ), store=Store(), memory=memory, settings=Settings(),
                          user_text="ok update it to Bangalore. aur short term me note kar de ki electrician ko 800 diye")
        call = ToolCall(id="1", name="ltm_save", arguments=json.dumps(
            {"facts": [{"category": "person", "subject": "Riya", "topic": "city", "text": "Lives in Bangalore"}], "confidence": 0.5}))
        result = asyncio.run(build_registry(Settings()).run_tool(call, ctx))
        self.assertIn("Saved in long-term memory", result["say"])   # not "use stm_add", not "Where should I save"

    def test_named_places(self):
        from companion.tools.builtin import named_places
        self.assertEqual(named_places("remember long term: mom's birthday"), {"long", "memory"})
        self.assertEqual(named_places("remember this: I paid Rahul"), {"memory", "short", "long"})
        self.assertEqual(named_places("add call the bank to my todos and milk to my shopping list"), {"todo", "list"})
        self.assertEqual(named_places("add to my to\u2011do list: call the bank"), {"todo"})
        self.assertEqual(named_places("kal meeting hai"), set())

    def test_reminder_asks_for_the_missing_day_or_time(self):
        day = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)   # Tue noon
        both = self._reminder(day, "set a reminder to call the bank", "2026-10-21T09:00")
        self.assertIn("Which day and what time", both["ask"])
        self.assertIn("full day", both["ask"])
        only_day = self._reminder(day, "remind me on Monday to call the bank", "2026-10-26T09:00")
        self.assertIn("At what time on Mon 26 Oct", only_day["ask"])
        only_time = self._reminder(day, "remind me at 5 pm to call the bank", "2026-10-20T17:00")
        self.assertIn("Which day should I remind you", only_time["ask"])
        # Answers build on the message before: "5 pm" after "remind me on Monday…".
        done = self._reminder(day, "5 pm", "2026-10-26T17:00", last_assistant=only_day["ask"], previous="remind me on Monday to call the bank")
        self.assertIn("Reminder set: study, Mon 26 Oct, 5:00 PM", done["say"])

    def test_misspelt_tomorrow_and_no_repeated_question(self):
        day = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        question = "Which day and what time should I remind you about 'study'? (or say full day)"
        # "tommorow 10 pm" is a day and a time.
        self.assertIn("Reminder set", self._reminder(day, "tommorow 10 pm", "2026-10-21T22:00", last_assistant=question,
                                                     previous="set a reminder of gym")["say"])
        # Only "tommorow": asks the time, not the day again.
        self.assertIn("At what time", self._reminder(day, "tommorow please", "2026-10-21T09:00", last_assistant=question,
                                                     previous="set a reminder of gym")["ask"])
        # Whatever the words, the same question is never asked twice in a row: the model's reading is taken.
        again = "Which day should I remind you about 'study'?"
        self.assertIn("Reminder set", self._reminder(day, "the usual day", "2026-10-21T17:00", last_assistant=again,
                                                     previous="remind me at 5 pm to study")["say"])

    def test_the_message_before_counts_after_a_save_question(self):
        day = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        offer = "Your exam is on Friday (23 Oct 2026). Would you like me to save that in your long-term memory or set a reminder?"
        asked = self._reminder(day, "add in reminder", "2026-10-23T09:00", last_assistant=offer, previous="my exam is on Friday")
        self.assertIn("At what time on Fri 23 Oct", asked["ask"])   # Friday is known, only the time is asked
        # "full day" now: the day is the one our question named.
        self.assertIn("Full-day reminder set", self._reminder(day, "full day", "2026-10-23", last_assistant=asked["ask"].replace("study", "study"),
                                                              previous="add in reminder")["say"])
        # A new reminder does not take the previous one's "full day".
        new = self._save("reminder_add", {"text": "gym", "when": "2026-10-21T09:00"}, "set a reminder of gym",
                         last_assistant=asked["ask"], previous="full day")
        self.assertIn("Which day and what time", new["ask"])

    def test_unsaid_midnight_and_abhi_are_not_a_time(self):
        day = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        asked = self._reminder(day, "Monday ko bank ko call karne ka reminder, time abhi fix nahi hai", "2026-10-26T00:00")
        self.assertIn("At what time on Mon 26 Oct", asked["ask"])
        self.assertIn("Reminder set", self._reminder(day, "remind me Monday at 12 am to call the bank", "2026-10-26T00:00")["say"])

    def test_full_day_reminder(self):
        day = datetime(2026, 10, 20, 12, 0, tzinfo=TZ)
        saved = {}

        class Store:
            async def reminder_add(self, user_id, text, due, repeat, entities, all_day=False):
                saved.update(due=due, all_day=all_day)
                return {"id": "r1", "text": text, "due_at": due, "repeat": repeat, "all_day": all_day}

            async def log_tool(self, *a):
                pass

        ctx = ToolContext(user_id="u", tz=TZ, now=day, store=Store(), memory=InMemoryMemory(), settings=Settings(),
                          user_text="set a full day reminder on Monday: Riya's exam")
        call = ToolCall(id="1", name="reminder_add", arguments=json.dumps({"text": "Riya's exam", "when": "2026-10-26", "all_day": True}))
        result = asyncio.run(build_registry(Settings()).run_tool(call, ctx))
        self.assertEqual(result["say"], "Full-day reminder set: Riya's exam, Mon 26 Oct (all day).")
        self.assertEqual((saved["all_day"], saved["due"].hour, saved["due"].date().isoformat()), (True, 9, "2026-10-26"))

    def test_weekday_must_match_the_date(self):
        day = datetime(2026, 10, 20, 10, 23, tzinfo=TZ)   # Tue
        question = "Which day should I remind you about 'study'?"
        wrong = self._reminder(day, "Wednesday", "2026-10-23T14:00", last_assistant=question, previous="remind me at 2 pm to study")
        self.assertIn("Wed 21 Oct", wrong["error"])
        right = self._reminder(day, "Wednesday", "2026-10-21T14:00", last_assistant=question, previous="remind me at 2 pm to study")
        self.assertIn("Reminder set", right["say"])
        self.assertIn("Reminder set", self._reminder(day, "remind me to buy sunscreen tomorrow at 9", "2026-10-21T09:00")["say"])

    def test_cancel_picks_by_number_date_or_all(self):
        now = datetime(2026, 10, 20, 10, 27, tzinfo=TZ)
        rows = [
            {"id": "c8fde644aa", "text": "Study", "due_at": datetime(2026, 10, 21, 9, 0, tzinfo=TZ), "repeat": "none", "all_day": False, "s": 1.0},
            {"id": "38340ae5bb", "text": "Study", "due_at": datetime(2026, 10, 21, 9, 0, tzinfo=TZ), "repeat": "none", "all_day": False, "s": 1.0},
            {"id": "fc10bbd1cc", "text": "Study", "due_at": datetime(2026, 10, 22, 9, 0, tzinfo=TZ), "repeat": "none", "all_day": False, "s": 1.0},
            {"id": "758c6c9ddd", "text": "Gym jana hai", "due_at": datetime(2026, 10, 21, 2, 0, tzinfo=TZ), "repeat": "none", "all_day": False, "s": 0.3},
        ]
        cancelled: list[str] = []

        class Store:
            async def reminder_find(self, user, words):
                return [dict(r) for r in rows if r["id"][:8] not in cancelled]

            async def reminders_waiting(self, user, rid=None):
                return [dict(r) for r in rows if (rid is None or r["id"].startswith(rid)) and r["id"][:8] not in cancelled]

            async def reminder_set_status(self, user, rid, status):
                cancelled.append(rid[:8])

            async def log_tool(self, *a):
                pass

        def cancel(args, text, last=""):
            ctx = ToolContext(user_id="u", tz=TZ, now=now, store=Store(), memory=InMemoryMemory(), settings=Settings(), user_text=text, last_assistant=last)
            result = asyncio.run(build_registry(Settings()).run_tool(ToolCall(id="1", name="reminder_cancel", arguments=json.dumps(args)), ctx))
            return result.get("say") or result.get("ask") or result.get("error")

        question = cancel({"match": "study"}, "cancel the study reminder")
        # The choices show day and time; exact copies count once; the fuzzy "Gym" is not offered.
        self.assertEqual(question, "Which reminder do you mean? 1) Study, Wed 21 Oct, 9:00 AM 2) Study, Thu 22 Oct, 9:00 AM. "
                                   "Say the number or the date, or 'all'.")
        self.assertIn("Thu 22 Oct", cancel({"match": "study", "number": 2}, "the 2nd", question))
        self.assertEqual(cancelled, ["fc10bbd1"])
        cancelled.clear()
        self.assertIn("(and 1 duplicate copy)", cancel({"match": "study", "day": "2026-10-21"}, "the one on Wednesday", question))
        self.assertEqual(sorted(cancelled), ["38340ae5", "c8fde644"])
        cancelled.clear()
        asked = cancel({"match": "study", "all_matching": True}, "all")
        self.assertIn("(yes/no)", asked)
        self.assertEqual(cancelled, [])                                       # nothing before the yes
        # A yes for one list never cancels another: here the confirm call lost its words, so every reminder fits.
        wider = cancel({"all_matching": True, "confirm": True}, "yes", asked)
        self.assertIn("Cancel all 3 of these?", wider)
        self.assertEqual(cancelled, [])
        self.assertIn("Cancelled 2 reminders", cancel({"match": "study", "all_matching": True, "confirm": True}, "yes", asked))
        self.assertEqual(len(cancelled), 3)
        cancelled.clear()
        # An id picked by the model does not skip the question when the user's words fit several reminders.
        by_id = cancel({"match": "study", "reminder_id": "fc10bbd1"}, "study wala reminder cancel kar do")
        self.assertIn("Which reminder do you mean?", by_id)
        self.assertEqual(cancelled, [])
        # ...but it does when they named the date.
        self.assertIn("Thu 22 Oct", cancel({"match": "study", "reminder_id": "fc10bbd1"}, "cancel the Thursday study reminder"))

    def test_web_search_only_with_key(self):
        self.assertNotIn("web_search", build_registry(Settings(tavily_api_key="")).names())
        self.assertIn("web_search", build_registry(Settings(tavily_api_key="x")).names())


class InMemoryTests(unittest.TestCase):
    def test_upsert_recall_expiry_forget(self):
        async def run():
            memory = InMemoryMemory()
            first = await memory.remember("u", MemoryItem(kind="preference", key="drink:chai", text="Loves chai"))
            again = await memory.remember("u", MemoryItem(kind="preference", key="Drink: Chai", text="Quit chai, drinks coffee now"))
            self.assertEqual(first, again)
            self.assertEqual([i.text for i in await memory.recall("u", "chai pi?")], ["Quit chai, drinks coffee now"])
            await memory.remember("u", MemoryItem(kind="event", key="event:old", text="Old exam", expires_at=datetime.now(TZ) - timedelta(days=1)))
            self.assertEqual(len(await memory.list_all("u")), 1)
            self.assertTrue(await memory.forget("u", first[:8]))
            self.assertEqual(await memory.list_all("u"), [])
        asyncio.run(run())


class RouterTests(unittest.TestCase):
    def test_falls_back_in_order(self):
        class Broken:
            name = "broken"

            async def chat(self, messages, **options):
                raise LLMError("down")

        class Works:
            name = "works"

            async def chat(self, messages, **options):
                return LLMResult(text="hi", provider="works")

        result = asyncio.run(LLMRouter([Broken(), Works()]).chat([]))
        self.assertEqual(result.provider, "works")
        with self.assertRaises(LLMError):
            asyncio.run(LLMRouter([Broken()]).chat([]))


class RouterTests(unittest.TestCase):
    """Keywords offer tool groups; the read tools are always there. (docs/personal-data-spec.md, section 2)"""

    def test_groups(self):
        from companion.router import tool_groups
        self.assertEqual(tool_groups("Riya ka birthday kab hai?"), {"read"})
        self.assertEqual(tool_groups("remind me tomorrow at 9 to call mom"), {"read", "reminder"})
        self.assertTrue({"short", "long", "reminder"} <= tool_groups("remember Riya gave me a reminder about the bill"))
        self.assertTrue({"clear", "note"} <= tool_groups("clear my notes"))
        # An add with no place named offers the six save tools, not only the read tools (English test, 20 Oct).
        self.assertEqual(tool_groups("also add one for saturday 6 pm to call mom"), {"read", "save"})
        self.assertEqual(tool_groups("add to my calendar that im on leave til friday"), {"read", "save"})
        self.assertEqual(tool_groups("add milk to the shopping list"), {"read", "list"})
        self.assertIn("todo", tool_groups("add to my todo: call the bank"))
        self.assertIn("list", tool_groups("add milk to my shopping list"))
        self.assertIn("note", tool_groups("note down the wifi password: abc123"))
        self.assertIn("long", tool_groups("remember long term: my sister Riya's birthday is 3 May"))
        self.assertIn("short", tool_groups("short term: exam on Friday"))

    def test_other_dash_characters(self):
        from companion.router import tool_groups
        self.assertIn("todo", tool_groups("add to tomorrow's to\u2011do: book the car service"))   # non-breaking hyphen
        self.assertIn("todo", tool_groups("my to\u2013do list"))

    def test_a_confirmation_is_not_a_question(self):
        from companion.router import tool_groups
        self.assertEqual(tool_groups("hi", "Reminder set: Study, Wed 21 Oct, 9:00 AM."), {"read"})   # set it again on "hi" once

    def test_follow_up_uses_last_question(self):
        from companion.router import tool_groups
        self.assertIn("reminder", tool_groups("9 baje", "When should I remind you about 'gym'?"))
        self.assertEqual(tool_groups("9 baje"), {"read"})


class LanguageHintTests(unittest.TestCase):
    def test_english_in_english_out(self):
        from companion.persona import script_hint
        self.assertIn("reply in English", script_hint("what is the wifi password?"))
        self.assertIn("Hinglish", script_hint("Riya ka birthday kab hai?"))
        self.assertIn("Hinglish", script_hint("remind me kal 9 baje"))
        self.assertIn("Devanagari", script_hint("मेरा नाम आशीष है"))


class RateLimitWaitTests(unittest.TestCase):
    def test_waits_once_when_every_model_is_rate_limited(self):
        from companion.llm import router as llm_router

        class Limited:
            def __init__(self, name, wait, then=None):
                self.name, self.wait, self.then, self.calls = name, wait, then, 0

            async def chat(self, messages, **options):
                self.calls += 1
                if self.then and self.calls > 1:
                    return self.then
                error = LLMError(f"{self.name} 429")
                error.retry_after = self.wait
                raise error

        original = llm_router.MAX_RATE_WAIT
        main = Limited("main", 0.01, then=LLMResult(text="ok", provider="main"))
        router = LLMRouter([main, Limited("backup", 0.02)])
        self.assertEqual(asyncio.run(router.chat([])).text, "ok")
        self.assertEqual(main.calls, 2)
        # The main model is out for the day, the backup is only per-minute limited: wait for the backup,
        # and skip the main model on the next call instead of asking it again.
        class Daily(Limited):
            async def chat(self, messages, **options):
                self.calls += 1
                error = LLMError("daily")
                error.daily_reset = 600
                raise error
        daily = Daily("main", None)
        backup = Limited("backup", 0.01, then=LLMResult(text="from backup", provider="backup"))
        mixed = LLMRouter([daily, backup])
        self.assertEqual(asyncio.run(mixed.chat([])).text, "from backup")
        asyncio.run(mixed.chat([]))
        self.assertEqual(daily.calls, 1)
        # A wait longer than the cap (or a daily limit, retry_after None) fails at once.
        slow = LLMRouter([Limited("main", original + 5), Limited("backup", original + 5)])
        with self.assertRaises(LLMError):
            asyncio.run(slow.chat([]))


class StreamTimeoutTests(unittest.TestCase):
    def test_a_silent_model_is_skipped_on_the_next_call(self):
        class Silent:
            name, timeout, calls = "gemini", 20, 0

            async def stream(self, messages, **options):
                Silent.calls += 1
                raise TimeoutError()
                yield  # pragma: no cover

        class Quick:
            name = "gemini_lite"

            async def stream(self, messages, **options):
                yield "text", "hi"

        async def run(router):
            return [value async for _, value, _ in router.stream([])]

        router = LLMRouter([Silent(), Quick()])
        with self.assertLogs("companion.llm", "WARNING") as logs:
            self.assertEqual(asyncio.run(run(router)), ["hi"])
        self.assertIn("sent nothing for 20s (TimeoutError)", logs.output[0])
        self.assertIn("→ trying gemini_lite", logs.output[0])
        asyncio.run(run(router))
        self.assertEqual(Silent.calls, 1)   # paused, not waited on again


class TurnPolicyTests(unittest.TestCase):
    """Live mode: how long to wait after the user stops (companion/realtime/turn_policy.py)."""

    def test_unfinished_words(self):
        from companion.realtime.turn_policy import sounds_unfinished
        for text in ("call my mom and.", "add milk, eggs and", "I was thinking, um...", "kal bank jaana hai aur", "so,"):
            self.assertTrue(sounds_unfinished(text), text)
        for text in ("What's on my list?", "Remind me at 8 pm.", "mera naam Rohan hai", ""):
            self.assertFalse(sounds_unfinished(text), text)

    def test_wait_follows_words_then_voice(self):
        from companion.realtime.turn_policy import HOLD_WAIT, MEDIUM_WAIT, QUICK_WAIT, UNKNOWN_WAIT, end_of_turn_wait
        self.assertEqual(end_of_turn_wait(0.95, "call my mom and")[0], HOLD_WAIT)   # words win over the voice
        self.assertEqual(end_of_turn_wait(0.9, "Remind me at 8.")[0], QUICK_WAIT)
        self.assertEqual(end_of_turn_wait(0.1, "Remind me at 8.")[0], MEDIUM_WAIT)
        self.assertEqual(end_of_turn_wait(None, "Remind me at 8.")[0], UNKNOWN_WAIT)
        self.assertLess(end_of_turn_wait(0.1, "What's on my list?")[0], MEDIUM_WAIT)

    def test_backchannels_and_stop_words(self):
        from companion.realtime.turn_policy import is_backchannel, is_stop_request
        self.assertTrue(all(is_backchannel(t) for t in ("haan ji", "ok", "hmm right", "acha theek hai", "Um.")))
        self.assertFalse(is_backchannel("ok remind me at 9"))
        self.assertTrue(all(is_stop_request(t) for t in ("Wait.", "ruko", "stop", "one sec")))
        self.assertFalse(is_stop_request("can you remind me"))


class EmptyStreamTests(unittest.TestCase):
    def test_an_empty_stream_falls_back_to_the_next_model(self):
        class Empty:
            name = "gemini_lite"

            async def stream(self, messages, **options):
                yield "usage", {"total": 10}   # only metadata, no words, no tool call

        class Good:
            name = "groq_backup"

            async def stream(self, messages, **options):
                yield "text", "hello"

        async def run():
            return [(kind, value) async for kind, value, _ in LLMRouter([Empty(), Good()]).stream([])]

        with self.assertLogs("companion.llm", "WARNING") as logs:
            self.assertIn(("text", "hello"), asyncio.run(run()))
        self.assertIn("empty reply", logs.output[0])


class MustUseToolTests(unittest.TestCase):
    """Data questions and commands must call a tool on the first LLM call; small talk must not."""

    def test_forced(self):
        from companion.router import must_use_tool
        for text in ("when is Riya's birthday?", "what's the internet code?", "how much did I pay Rahul?",
                     "what's planned for Friday?", "what do you know about Amit?", "where does Riya live now?",
                     "actually Riya moved to Bangalore, update that", "add milk to my shopping list", "kal ka plan kya hai?",
                     "what do I have to do today?"):
            self.assertTrue(must_use_tool(text), text)

    def test_not_forced(self):
        from companion.router import must_use_tool
        for text in ("hi, how are you?", "tell me a joke", "what is the capital of France?", "thanks yaar, aaj ka din kaafi busy tha",
                     "yes", "6 pm on Friday"):
            self.assertFalse(must_use_tool(text), text)
        # An answer to a question a tool asked must complete it with a tool; a casual offer does not count.
        self.assertTrue(must_use_tool("6 pm on Friday", "When should I remind you about 'go to the gym'?"))
        self.assertFalse(must_use_tool("no thanks", "Want me to set a reminder for that?"))


class SearchHelperTests(unittest.TestCase):
    def test_clean_words_drops_filler(self):
        from companion.search import clean_words
        self.assertEqual(clean_words("What is Riya ka birthday?"), ["riya", "birthday"])

    def test_pick_one_clear_winner_or_ask(self):
        from companion.search import pick
        one, ask = pick([{"id": 1, "s": 0.95}, {"id": 2, "s": 0.4}])
        self.assertEqual((one["id"], ask), (1, []))
        one, ask = pick([{"id": 1, "s": 0.8}, {"id": 2, "s": 0.7}, {"id": 3, "s": 0.65}])
        self.assertIsNone(one)
        self.assertEqual(len(ask), 3)
        self.assertEqual(pick([]), (None, []))



class TokenUsageTests(unittest.TestCase):
    def test_every_call_in_a_turn_is_counted(self):
        from companion.llm import usage

        class Fake:
            name, model = "fake", "m1"

            async def chat(self, messages, **options):
                return LLMResult(text="hi", provider="fake:m1", usage={"prompt": 100, "completion": 7, "reasoning": 2})

            async def stream(self, messages, **options):
                yield "text", "hi"
                yield "usage", {"prompt": 50, "completion": 3}

        async def turn():
            counted = usage.TurnUsage()
            token = usage.CURRENT.set(counted)
            try:
                router = LLMRouter([Fake()])
                await router.chat([])
                async for _ in router.stream([]):
                    pass
            finally:
                usage.CURRENT.reset(token)
            await LLMRouter([Fake()]).chat([])   # outside a turn: not counted
            return counted.summary()

        summary = asyncio.run(turn())
        self.assertEqual((summary["input"], summary["output"], summary["reasoning"], summary["total"]), (150, 10, 2, 160))
        self.assertEqual([c["provider"] for c in summary["calls"]], ["fake:m1", "fake:m1"])


if __name__ == "__main__":
    unittest.main()
