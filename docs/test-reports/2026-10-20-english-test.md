# English test, 20 Oct 2026

About 20 minutes of live chat with Buddy, English only, with new mixed scenarios that earlier tests did not cover.
A simulated user (another LLM, told to write only English) talked naturally. Groq answered the first turns. After
its daily quota ran out, Gemini 3.5 Flash Lite answered the rest. Bugs found in one round were fixed before the next.

| Round | Turns | Time | Result |
| --- | --- | --- | --- |
| 1 | 33 | 11 min | 32 intents covered, 4 bugs |
| 2 | 20 | 6 min | round-1 bugs fixed live, 2 new bugs |
| 3 | 8 fixed messages | 2 min | all fixes worked live, 1 more small bug (fixed, unit-tested) |

Tests: 102 pass (`.venv/bin/python -m unittest discover tests -t .`).

## What worked
- Saving and the "Where should I save?" ask: dentist appointment, then "as a reminder"; vegetarian (long-term);
  passport note.
- Goa packing list: create, show, tick charger, remove sunglasses. Movie watchlist: create, tick Dune.
- To-dos: three for tomorrow, show, done, move "call Amit" to Thursday.
- Reminders: weekly Monday reminder; "in 30 minutes"; cancel; 1 November at 10 am.
- Long-term facts: Amit's birthday saved, updated 12 → 14 June and read back; car saved, asked, forgotten.
- Questions and lookups: "whose birthday is in June", cafe wifi lookup, Thursday overview, Goa weather,
  days until Christmas.
- Deletes: forget one fact; delete the passport note.

## Bugs found and fixed
| # | What happened | Fix |
| --- | --- | --- |
| 1 | Gemini Lite wrote last year's date (2025-10-21 for "tomorrow") 5+ times per run. Each refusal cost a retry, and once the model got lost (3 web searches, nothing saved) | A last-year date that falls within 31 days before or 60 days after today is moved to this year before the tool runs (`tools/dates.py`). Other past-year dates are still refused |
| 2 | "Clear my to-dos for tomorrow" said "no to-dos for Wed 21 Oct" although "book a cab" was there. The model had sent 2025-10-21, and `clear_saved` had no year check | `clear_saved` checks the year like the other tools |
| 3 | "Add one for Saturday 6 pm to call mom" and "add to my calendar that I'm on leave" were offered only the read tools, so the model ran web searches | "add", "calendar", "diary" and "schedule it" with no place named now offer the six save tools |
| 4 | "Add pay rent to my to-dos for tomorrow" was refused 3 times: an earlier message's "Friday" was checked against "tomorrow" | The message before only counts when the new message names no day of its own |
| 5 | An empty LLM answer (after listing reminders) gave "Sorry yaar, kuch gadbad ho gayi" | One retry on an empty answer; an English user gets the fallback line in English |
| 6 | "I don't have details about your office saved" was replaced by "I haven't saved that yet. Where should I save it?" | A denial ("don't have", "nothing saved", "couldn't find") is not counted as claiming an action |

## Still to watch
- With Groq's daily quota used up, Gemini Flash Lite is slower (2–17 s per reply) and asks "Where should I save?"
  more often, because it gives itself lower confidence.
- The simulated user sometimes sends mixed-up messages ("add it as a reminder… actually cancel it"). Buddy's
  answers to those were reasonable.
- Not tested: live voice, push-to-talk, and reminders being spoken when due.

Test users (english-test, english-test2, english-test3) were removed. Your users (me, haha, consumer1,
consumer2) were not touched.
