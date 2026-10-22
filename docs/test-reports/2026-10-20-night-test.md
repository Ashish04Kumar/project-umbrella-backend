# Night test, 20 Oct 2026

About 20 minutes of live chat with Buddy (Groq first; Groq's daily quota ran out, so most replies came from
Gemini 3.5 Flash Lite). A simulated user (another LLM) talked naturally in English and Hinglish and covered
every feature. Three rounds; bugs found in a round were fixed before the next.

| Round | Turns | Time | Result |
| --- | --- | --- | --- |
| 1 | 33 | 11 min | 30/30 intents covered, 4 bugs |
| 2 | 22 | 6.5 min | 4 earlier bugs fixed live, 2 new bugs |
| 3 | 8 (fixed messages) | 2.5 min | both new bugs fixed live |

Tests: 99 pass (`.venv/bin/python -m unittest discover tests -t .`).

## What worked
Long-term save (name, city: two facts in one call), full-day reminder, duplicate reminder ("already set"),
reminder list, "internet password" finding the wifi note, list add / tick / remove / clear (asked yes/no first),
to-do add with priority, done, move to another day, reminder move, web search, maths, note delete,
Hinglish questions, the misspelt "tommorow", "Which reminder do you mean?" with dates, cancel all with yes/no.

## Bugs found and fixed
| # | What happened | Fix |
| --- | --- | --- |
| 1 | "Monday ko bank call… time abhi fix nahi hai" set Mon 12:00 AM | "abhi/now/later" are not a time; an unsaid 12:00 AM asks the time |
| 2 | "save it" sent 26 tools (~7k tokens): the fallback model refused (too big), reply failed | "save it" and the "Where should I save" answer offer only the 6 save tools (~3k) |
| 3 | "wifi password yaad kar le" was not taken as a save | "yaad kar le/lo/karo" = remember |
| 4 | "plant wala reminder cancel kar do" cancelled one of two plant reminders without asking | an id from a list no longer skips "which one?" when the words fit several |
| 5 | **"yes" to "Cancel all 2 of these?" cancelled 3** (also the bank reminder) | the yes cancels exactly the listed reminders; a different list is asked again |
| 6 | "Riya ab Bangalore me rehti hai, update it" asked where to save, then the update went to short-term | an update of a fact already in long-term memory stays there |

## Still to watch
- With Groq's daily quota used up, Gemini Flash Lite answers: slower (2–15 s) and sometimes makes its own
  choices (for example it saved the wifi password to short-term memory rather than long-term).
- The simulated user sometimes makes mistakes of its own (it once gave Riya's birthday as 15 Nov instead of
  3 May); those are not Buddy bugs.
- Not tested tonight: live voice and push-to-talk (no microphone here), the playground buttons (checked
  earlier in the browser), reminders being spoken when due.

Test users created for this run (night-test, night-test2, night-test3, typo-check) were removed from the
database; your users (me, haha, consumer1, consumer2) were not touched.
