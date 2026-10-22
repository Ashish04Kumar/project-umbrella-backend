# Personal data: short-term memory, long-term memory, reminders, to-dos, notes, lists

Status: **built (19 Oct 2026).** Tests: `tests/test_golden_search.py` (sample data below, tool calls from section 6), router and guard tests in `tests/test_units.py`. Normal conversation stays as it is. This spec makes the
"keep things for me" part solid, cheap in tokens and predictable.

Today in the examples: Mon 19 Oct 2026.

## 1. Principles

1. **Saving happens only when you ask** (keyword or clear intent). The background memory extractor that
   guesses facts after every message is switched off. This alone saves about 1,500 tokens per message.
2. **Routing uses both keyword and intent.** A keyword is a strong hint; the meaning decides.
   "Yaad rakhna, Riya ne mujhe reminder diya tha" has the word *reminder*, but it is a fact about Riya,
   not a request to be reminded, so it goes to memory.
3. **If it is really unclear, Umbrella asks one short question** instead of guessing
   ("Short term mein rakhun ya long term?", "Kis time yaad dilaun?").
4. **The LLM never writes SQL.** It picks a tool and fills its fields; the code runs fixed SQL.
   Counting, totals and date filters are done by SQL, so numbers are always right.
5. **Nothing personal is dumped into every prompt.** Only a tiny profile (name, city, at most 8 pinned
   facts) goes with each message; everything else is fetched by a tool when you ask.
6. **After a save/change, the confirmation is written by code** ("Reminder set: dawai, Tue 20 Oct, 9:00 AM"),
   so the second LLM call is skipped.
7. **Matching by words, not ids.** "Doodh wala kaam ho gaya" is enough: the tool finds the item.
   One match: done. Several: Umbrella asks which one. None: Umbrella says so.

## 2. How a message is routed (keyword + intent)

```
your message
   │
   ├─ 1. Keyword scan (code, free): which groups are hinted?
   │      "short term" → STM · "long term" → LTM · "reminder", "yaad dilana" → REMINDER
   │      "to-do", "todo", "kaam list" → TODO · "note", "likh lo" → NOTE · "list mein", "shopping list" → LIST
   │      "yaad rakhna" alone → STM or LTM (intent decides)
   │
   ├─ 2. Tools offered to the LLM = hinted groups + the small "read" set
   │      (no hint at all → only the read set: ~250 tokens instead of ~1,100 for all tools)
   │
   └─ 3. LLM decides by intent, with the rules below, and fills the fields
          • a save needs its required fields (a reminder needs a future time and "remind me" intent)
          • keyword without matching intent → not that action
          • missing field or two equally likely places → ask one question
```

| You say | Keyword hint | Intent → result |
|---|---|---|
| "Short term mein yaad rakhna, parson Rahul ko 2000 diye" | STM | STM save |
| "Long term mein daal, Riya ka birthday 3 May" | LTM | LTM save |
| "Kal 9 baje dawai ka reminder laga do" | REMINDER | reminder save |
| "Yaad rakhna, Riya ne mujhe reminder diya tha ki bill bharna hai" | REMINDER + yaad rakhna | **not** a reminder (no "remind me", no time): asks "Short term mein note kar loon?" |
| "Yaad rakhna Riya ka birthday 3 May hai" | yaad rakhna | birthday = lasting → LTM (or one question if you want STM/LTM always explicit) |
| "Aaj ke to-do mein doodh lana" | TODO | to-do save |
| "Riya ka birthday kab hai?" | none | read set → `ltm_find(subject="Riya")` |
| "Black hole kya hai?" | none | no tool |

## 2b. Human in the loop (essential)

Umbrella keeps you in the loop instead of guessing. It asks **one short question** when:

| Situation | Example | Umbrella asks |
|---|---|---|
| A required field is missing | "Bill ka reminder laga do" | "Kab yaad dilaun?" |
| Two places fit equally | "Yaad rakhna, Riya ne reminder diya tha bill ka" | "Short term mein note karun, ya bill ka reminder bhi lagaun?" |
| Several items match | "Call wala kaam ho gaya" (Amit ko call, Mom ko call) | "Kaunsa? 1) Amit ko call 2) Mom ko call" |
| Something will be deleted in bulk or for good | "Amit ke baare mein sab bhool jao", "shopping list khaali karo" | "Amit ki 3 baatein hata dun? (haan/nahi)" |
| The words are unclear (speech) | "Kal teen baje… ya tera baje?" | "3 baje ya 1 baje?" |

How it works:

```
"Bill ka reminder laga do"
  → tool: reminder_add(text="Bill") without a time → the tool answers "needs_time"
  → saved as a pending action on this thread: {action: reminder_add, text: "Bill", waiting_for: "time"}
  → Umbrella: "Kab yaad dilaun?"        (the playground shows quick buttons where it can: "Aaj shaam", "Kal subah")
"Kal 10 baje"
  → the pending action is filled: reminder_add("Bill", 2026-10-20T10:00) → "Reminder set: Bill, Tue 20 Oct, 10:00 AM"
```

Rules:
- At most one open question at a time; it expires after 10 minutes or when you change the topic.
- A clear request is never questioned ("kal 9 baje dawai yaad dilana" just happens).
- Deleting one item is done at once (and can be undone by saying it again); deleting many asks first.
- Every question offers the likely answers, so a one-word reply works.

## 2c. Asking: one search over short-term, long-term and notes

Both scopes live in one table (`memories`, column `scope`), so a question never has to pick one.
**Saving is separate** (you say short or long term); **asking uses one tool for both**:

`find_saved(query?, subject?, category?, from?, to?, month?, in?)` — one SQL over short-term memory,
long-term memory **and notes** (`expires_at` hides expired short-term items). Each result says where it
came from, so Umbrella can answer "Wifi password abc123 hai (12 Oct ke note se)" or "Riya ka birthday 3 May
hai; short term mein likha hai ki kal gift dena hai". `in` (`short` | `long` | `notes`) is set only when you
say it ("mere notes mein…", "short term mein kya hai?"). Nothing found anywhere → Umbrella says so and offers
to save it. To-dos, reminders and lists keep their own list tools (their questions name them, and they are
filtered by day, status or tick).

In the tables below, `stm_find(...)`, `ltm_find(...)` and `note_find(...)` are all this same `find_saved(...)`
(with the place shown only to make the examples clearer).

### Search inside `find_saved` (no vector search for now)

1. SQL filters first (subject, category, dates, month).
2. Full-text search on the text **and on keywords the LLM writes when saving** (`title`, `tags` in Hinglish and
   English: a wifi note gets `wifi, internet, password, code`), so "internet ka code" finds "wifi password".
3. `pg_trgm` fuzzy match for typos and spelling ("passwrd"); already available in Postgres.
4. Nothing found and little data: the newest ~30 titles go to the LLM to pick by meaning (~300 tokens).

Vector search (pgvector + embeddings) is not needed for day-to-day personal data; it is the planned upgrade for a
knowledge base, long documents or thousands of items. `find_saved` keeps the same tool shape when it is added.

## 2d. Clearing and manual override

Every store can be cleared by voice/text **or** from the playground panels (a "Clear" button per store):

| You say | Tool | Umbrella first asks |
|---|---|---|
| "Short term memory clear kar do" | `clear_saved(store=short)` | "Short term ki 6 baatein hata dun? (haan / nahi)" |
| "Long term saaf kar do" | `clear_saved(store=long)` | "Long term ki 14 baatein hamesha ke liye hata dun? Ye wapas nahi aayengi." |
| "Saare notes hata do" | `clear_saved(store=notes)` | "9 notes hata dun?" |
| "Aaj ke to-dos clear karo" / "saare to-dos" | `clear_saved(store=todos, day?)` | "Aaj ke 4 to-do hata dun?" |
| "Saare reminders cancel karo" | `clear_saved(store=reminders)` | "3 aane wale reminders cancel karun?" |
| "Shopping list khaali karo" | `list_clear(shopping)` | "Shopping list ke 7 items hata dun?" |

Bulk clearing is always two steps: the tool first returns the count and a short preview and stores a pending
action; only "haan" runs it. A panel button asks the same in a small confirm box. One item ("doodh wala hata do")
is removed at once.

## 2e. When a reminder is due: Umbrella says it out loud

```
due_at reached (server checks every 20 s)
  → playground: Umbrella speaks it (TTS), the robot looks at you, a chat bubble + popup appear
      "Ashish, 9 baj gaye: dawai lene ka time ho gaya."
  → Umbrella asks: "Ho gaya?"   [✅ Ho gaya]  [⏰ 10 min baad]
      ✅ / "ho gaya"           → status done
      ⏰ / "10 minute baad"     → due_at + 10 min (snoozed)
      no answer for 10 min     → said once more, then left as "fired"
  → repeating reminders (daily/weekly/monthly) move to their next time either way
```

Reminder statuses: `pending → fired → done | snoozed | cancelled`. The page must be open to speak (for now);
later the Android app (notifications) and the robot body speak it the same way. In a live call, Umbrella says it
between turns.

## 3. Data and tools per feature

### 3.1 Short-term memory (STM): things for the next few days

*Trigger:* "short term mein yaad rakhna", "short term mein daal", or a clearly temporary fact after "yaad rakhna".
*Lifetime:* 10 days by default (the date it is about + 1 day, at most 15 days). Then it disappears on its own.

Table `memories` with `scope = 'short'`:

| Column | Example |
|---|---|
| text | "Paid Rahul 2000" |
| about_date | 2026-10-17 (the day it happened / will happen; optional) |
| expires_at | 2026-10-29 |

Tools: `stm_add(text, about_date?)`, `stm_forget(match)`; asking uses `memory_find` (section 2c).

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Short term mein yaad rakhna, parson maine Rahul ko 2000 ka payment kiya | `stm_add("Paid Rahul 2000", about_date=2026-10-17)` |
| 2 | Short term: 5 din baad mujhe passport office jaana hai | `stm_add("Go to passport office", about_date=2026-10-24)` |
| 3 | Short term mein daal, aaj gym nahi gaya | `stm_add("Skipped gym", about_date=2026-10-19)` |
| 4 | Yaad rakhna short term, Amit ka parcel kal aayega | `stm_add("Amit's parcel arrives", about_date=2026-10-20)` |
| 5 | Short term mein likh lo, landlord ko rent baaki hai | `stm_add("Rent pending to landlord")` |
| 6 | Short term: is hafte diet pe hoon | `stm_add("On a diet this week", about_date=2026-10-19)` |
| 7 | Short term yaad rakh, car service Saturday ko di hai | `stm_add("Car given for service", about_date=2026-10-24)` |
| 8 | Rahul wala payment bhool jao | `stm_forget("Rahul payment")` |
| 9 | Passport wali baat hata do short term se | `stm_forget("passport")` |
| 10 | Short term: Rahul ko 2000 nahi, 2500 diye the | `stm_forget("Rahul")` + `stm_add("Paid Rahul 2500", 2026-10-17)` |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Short term mein kya-kya hai? | `stm_find()` |
| 2 | Maine Rahul ko kitne diye the? | `stm_find("Rahul")` |
| 3 | Is hafte kya-kya karna hai? | `stm_find(from=2026-10-19, to=2026-10-25)` |
| 4 | Kal kya hone wala hai? | `stm_find(from=2026-10-20, to=2026-10-20)` |
| 5 | Parson kya kiya tha maine? | `stm_find(from=2026-10-17, to=2026-10-17)` |
| 6 | Passport office kab jaana hai? | `stm_find("passport")` |
| 7 | Koi payment yaad hai pichhle dinon ka? | `stm_find("payment paid")` |
| 8 | Car kab service ko di thi? | `stm_find("car service")` |
| 9 | Short term memory clear hai? | `stm_find()` (empty → "kuch nahi hai") |
| 10 | Diet kab se hai? | `stm_find("diet")` |

### 3.2 Long-term memory (LTM): lasting facts

*Trigger:* "long term mein daal / yaad rakhna", or a clearly lasting fact after "yaad rakhna" (birthday, relation, job).
*Lifetime:* until changed or forgotten.

Table `memories` with `scope = 'long'`:

| Column | Example |
|---|---|
| category | `profile` · `person` · `date` · `preference` · `project` · `other` |
| subject | "Riya" · "me" · "Project Synq" |
| topic | "birthday" · "relation" · "job" · "food" (short) |
| text | "Sister, birthday 3 May" |
| date | 2026-05-03 (birthdays, anniversaries: lets SQL answer "is mahine kiska birthday") |
| key | `category:subject:topic` (same key = update, never a duplicate) |
| pinned | true for profile facts that go with every message (name, city; at most 8) |

Tools: `ltm_save(category, subject, topic, text, date?)`, `ltm_forget(match)`; asking uses `memory_find` (section 2c).

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Long term mein daal, Riya meri behen hai aur uska birthday 3 May hai | `ltm_save(person, Riya, relation, "Sister")` + `ltm_save(date, Riya, birthday, "Birthday", date=05-03)` |
| 2 | Long term: mera naam Ashish hai, Pune mein rehta hoon | `ltm_save(profile, me, name, "Ashish")` + `ltm_save(profile, me, city, "Pune")` |
| 3 | Long term yaad rakhna, mummy-papa ki anniversary 12 Dec | `ltm_save(date, Parents, anniversary, "Anniversary", date=12-12)` |
| 4 | Long term: main Project Synq pe kaam kar raha hoon, dating app | `ltm_save(project, Project Synq, about, "Dating app I am building")` |
| 5 | Long term mein daalo, mujhe paneer pasand hai, karela nahi | `ltm_save(preference, me, food, "Likes paneer, dislikes karela")` |
| 6 | Long term: Amit mera college friend hai, Delhi mein hai | `ltm_save(person, Amit, relation, "College friend, lives in Delhi")` |
| 7 | Long term: mera blood group B+ hai | `ltm_save(profile, me, blood_group, "B+")` |
| 8 | Riya ab Mumbai shift ho gayi, update kar do | `ltm_save(person, Riya, city, "Lives in Mumbai")` (same key → update) |
| 9 | Karela wali baat bhool jao | `ltm_forget("karela")` |
| 10 | Amit ke baare mein sab hata do | `ltm_forget("Amit")` (several matches → asks to confirm all) |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Riya ka birthday kab hai? | `ltm_find(subject="Riya")` |
| 2 | Is mahine kiska birthday hai? | `ltm_find(category=date, month=10)` (SQL on `date`) |
| 3 | Mere baare mein kya jaante ho? | `ltm_find(subject="me")` |
| 4 | Amit kaun hai? | `ltm_find(subject="Amit")` |
| 5 | Mujhe khane mein kya pasand hai? | `ltm_find(category=preference, query="food")` |
| 6 | Mere projects kaun se hain? | `ltm_find(category=project)` |
| 7 | Mummy-papa ki anniversary kab hai? | `ltm_find(query="anniversary")` |
| 8 | Mera blood group kya hai? | `ltm_find(subject="me", query="blood group")` |
| 9 | Delhi mein mera kaun rehta hai? | `ltm_find(query="Delhi")` |
| 10 | Long term mein kya-kya saved hai? | `ltm_find()` |

### 3.3 Reminders

*Trigger:* "reminder laga / reminder mein daal", "yaad dilana", "alarm". Needs a future time; if missing, asks "kis time?".

Table `reminders`: `text, due_at, repeat (none | daily | weekly | monthly), status (pending | fired | snoozed | done | cancelled), fired_at, created_at`.

Tools: `reminder_add(text, when, repeat?)`, `reminder_list(from?, to?, status?)`, `reminder_cancel(match)`, `reminder_move(match, when)`, `reminder_done(match)`, `reminder_snooze(match, minutes)`.

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Kal subah 9 baje dawai ka reminder laga do | `reminder_add("Dawai", 2026-10-20T09:00)` |
| 2 | Roz raat 10 baje paani peene ka reminder | `reminder_add("Paani piyo", 2026-10-19T22:00, daily)` |
| 3 | Friday shaam 6 baje gym yaad dilana | `reminder_add("Gym", 2026-10-23T18:00)` |
| 4 | Har Monday 10 baje team call ka reminder | `reminder_add("Team call", 2026-10-26T10:00, weekly)` |
| 5 | 2 ghante baad chai yaad dilana | `reminder_add("Chai", now+2h)` |
| 6 | Har mahine 5 tareekh ko rent ka reminder | `reminder_add("Rent", 2026-11-05T10:00, monthly)` |
| 7 | Bill bharne ka reminder laga do | missing time → asks "Kab yaad dilaun?" |
| 8 | Dawai wala reminder 10 baje kar do | `reminder_move("dawai", 2026-10-20T10:00)` |
| 9 | Gym wala reminder cancel kar do | `reminder_cancel("gym")` |
| 10 | Paani wala roz ka reminder band karo | `reminder_cancel("paani")` |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Mere reminders kya hain? | `reminder_list()` |
| 2 | Kal ke reminders? | `reminder_list(from=2026-10-20, to=2026-10-20)` |
| 3 | Aaj aur kya yaad dilana hai? | `reminder_list(from=now, to=2026-10-19T23:59)` |
| 4 | Is hafte kitne reminders hain? | `reminder_list(from=2026-10-19, to=2026-10-25)` → count from SQL |
| 5 | Roz wale reminders kaun se hain? | `reminder_list(repeat=daily)` |
| 6 | Dawai ka reminder kab ka hai? | `reminder_list(query="dawai")` |
| 7 | Friday ko kya-kya hai? | `reminder_list(from=2026-10-23, to=2026-10-23)` |
| 8 | Kaunse reminders ho gaye (beet gaye)? | `reminder_list(status=done)` |
| 9 | Agla reminder kab hai? | `reminder_list(limit=1)` |
| 10 | Maine kaunse reminders cancel kiye? | `reminder_list(status=cancelled)` |

### 3.4 To-dos (day-wise)

*Trigger:* "to-do mein daal", "aaj ka kaam", "kaam list". Every to-do belongs to a **day** (today if you don't say).
A pending to-do from an earlier day stays on its day and is shown as **overdue** ("pichhle dinon ka 1 baaki").

Table `todos`: `text, day, priority (low | normal | high), status (pending | done), done_at, created_at`.

Tools: `todo_add(items[], day?, priority?)`, `todo_list(day? | from/to, status?)` (returns items, counts and overdue),
`todo_done(match | items[])`, `todo_remove(match)`, `todo_move(match, day)`.

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Aaj ke to-do mein doodh lana aur bill bharna daal do | `todo_add(["Doodh lana", "Bill bharna"], day=2026-10-19)` |
| 2 | Kal ke to-do mein bank jaana | `todo_add(["Bank jaana"], day=2026-10-20)` |
| 3 | Friday ke liye presentation banana, high priority | `todo_add(["Presentation banana"], day=2026-10-23, high)` |
| 4 | To-do: gym, groceries, Amit ko call | `todo_add(["Gym", "Groceries", "Amit ko call"], day=today)` |
| 5 | Doodh wala kaam ho gaya | `todo_done("doodh")` |
| 6 | Aaj ke saare kaam ho gaye | `todo_done(day=today, all=true)` |
| 7 | Bank wala kal se Monday pe daal do | `todo_move("bank", 2026-10-26)` |
| 8 | Groceries wala to-do hata do | `todo_remove("groceries")` |
| 9 | Presentation ko high priority kar do | `todo_add` same text → update priority (or `todo_update`) |
| 10 | Kal jo bacha tha wo aaj ki list mein daal do | `todo_move(day=2026-10-18, status=pending, to=today)` |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Aaj ke to-do mein kya baaki hai? | `todo_list(day=today, status=pending)` (+ overdue) |
| 2 | Kal ke to-do mein kya tha? | `todo_list(day=2026-10-18, status=all)` |
| 3 | Friday ki list mein kya pending hai? | `todo_list(day=2026-10-23, status=pending)` |
| 4 | Is hafte kitne kaam poore kiye? | `todo_list(from=2026-10-19, to=2026-10-25, status=done)` → count from SQL |
| 5 | Koi high priority kaam baaki hai? | `todo_list(status=pending, priority=high)` |
| 6 | Pichhle dinon ka kya reh gaya? | `todo_list(to=2026-10-18, status=pending)` |
| 7 | Bank wala kaam kab ka hai? | `todo_list(query="bank")` |
| 8 | Aaj kitne kaam hain total? | `todo_list(day=today, status=all)` → counts |
| 9 | Kal ke liye kya plan hai? | `todo_list(day=2026-10-20)` |
| 10 | Pichhle hafte kya-kya kiya? | `todo_list(from=2026-10-12, to=2026-10-18, status=done)` |

### 3.5 Notes

*Trigger:* "note kar lo", "likh lo", "note mein daal". Exact text you want kept (passwords, codes, ideas, addresses).
Passwords and long text always go to notes, never to memory.

Table `notes`: `title, text, tags[], created_at` (+ full-text index; later hybrid search).

Tools: `note_save(text, title, tags)` (the LLM always fills a short title and 3–5 Hinglish/English keywords), `note_delete(match)`; asking uses `find_saved` (section 2c).

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Note kar lo, wifi password abc123 | `note_save("Wifi password abc123", title="Wifi")` |
| 2 | Likh lo: app ka idea, voice se kharcha add karna | `note_save("Voice se kharcha add karna", title="App idea", tags=[idea])` |
| 3 | Note: Amit ka address 21 MG Road, Delhi | `note_save("21 MG Road, Delhi", title="Amit's address")` |
| 4 | Is book ka naam note kar lo: Atomic Habits | `note_save("Atomic Habits", title="Book", tags=[book])` |
| 5 | Note mein daal, car insurance number XYZ123 | `note_save("XYZ123", title="Car insurance")` |
| 6 | Meeting notes: launch 1 Nov, Rahul design dekhega | `note_save("Launch 1 Nov; Rahul handles design", title="Meeting", tags=[work])` |
| 7 | Doctor ne kaha din mein 2 baar dawai | `note_save("Dawai din mein 2 baar", title="Doctor")` |
| 8 | Wifi wala note hata do | `note_delete("wifi")` |
| 9 | Wifi password badal gaya, ab xyz789 | `note_delete("wifi")` + `note_save("Wifi password xyz789", title="Wifi")` |
| 10 | Book wale note mein ek aur add karo: Deep Work | `note_save("Deep Work", title="Book", tags=[book])` |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Wifi password kya tha? | `note_find("wifi password")` |
| 2 | Amit ka address? | `note_find("Amit address")` |
| 3 | Mere app ideas kya the? | `note_find("idea")` |
| 4 | Kaunsi books note ki thi? | `note_find("book")` |
| 5 | Kal kya note kiya tha? | `note_find(from=2026-10-18, to=2026-10-18)` |
| 6 | Car insurance number? | `note_find("car insurance")` |
| 7 | Meeting mein kya decide hua tha? | `note_find("meeting")` |
| 8 | Doctor ne kya bola tha? | `note_find("doctor")` |
| 9 | Mere saare notes dikhao | `note_find()` |
| 10 | Is hafte ke work notes? | `note_find("work", from=2026-10-19)` |

### 3.6 Lists (shopping, watchlist, books, packing…)

*Trigger:* "list mein daal", "shopping list", "<name> list". A list has a name; items can be ticked.

Tables `lists(name)` and `list_items(text, checked, created_at)`.

Tools: `list_add(list, items[])`, `list_show(list?)` (no name → all lists with counts),
`list_check(list, items[])`, `list_remove(list, items[])`, `list_clear(list, checked_only?)`.

| # | Save / change (you say) | Tool call |
|---|---|---|
| 1 | Shopping list mein doodh, bread, ande daal do | `list_add(shopping, [doodh, bread, ande])` |
| 2 | Watchlist mein Dune daalo | `list_add(watchlist, [Dune])` |
| 3 | Goa trip ki packing list: charger, sunscreen | `list_add(goa packing, [charger, sunscreen])` |
| 4 | Doodh le liya | `list_check(shopping, [doodh])` |
| 5 | Bread hata do list se | `list_remove(shopping, [bread])` |
| 6 | Shopping list khaali kar do | `list_clear(shopping)` |
| 7 | Jo le liya wo list se hata do | `list_clear(shopping, checked_only=true)` |
| 8 | Books list mein Atomic Habits | `list_add(books, [Atomic Habits])` |
| 9 | Dune dekh li | `list_check(watchlist, [Dune])` |
| 10 | Shopping mein 2 kilo aloo bhi | `list_add(shopping, ["Aloo 2 kg"])` |

| # | Ask (you say) | Tool call |
|---|---|---|
| 1 | Shopping list mein kya hai? | `list_show(shopping)` |
| 2 | Kya-kya lena baaki hai? | `list_show(shopping)` → unchecked |
| 3 | Meri kitni lists hain? | `list_show()` |
| 4 | Watchlist mein kya bacha hai? | `list_show(watchlist)` |
| 5 | Goa ke liye kya pack karna hai? | `list_show(goa packing)` |
| 6 | Doodh liya tha? | `list_show(shopping)` → doodh: checked |
| 7 | Books list dikhao | `list_show(books)` |
| 8 | Shopping mein kitne items hain? | `list_show(shopping)` → count |
| 9 | Kaunsi movies dekh li? | `list_show(watchlist)` → checked |
| 10 | Packing list complete hai? | `list_show(goa packing)` → all checked? |

## 4. Token budget (estimates)

| Turn | Now | With this design |
|---|---|---|
| Normal chat | ~3,700 in (all memories + all 15 tools) + memory extractor ~1,500 | ~1,400 in (profile pins + read tools only), no extractor |
| Save/action ("reminder laga do") | ~8,000 (two LLM calls) + extractor | ~1,600 (one LLM call; code confirms) |
| Question ("aaj kya baaki hai?") | ~8,000 | ~3,000 (tool call + one answer) |

## 5. Decisions needed

1. STM lifetime: 10 days default (at most 15)? 
2. "Yaad rakhna" without "short/long term": let Umbrella decide by meaning (birthday → long, payment → short), or always ask?
3. Turn the background memory extractor off completely (save only when asked)?
4. Pinned profile facts (name, city) with every message: yes, at most 8?
5. To-dos with no day: today? Pending ones stay on their day and show as overdue (no automatic carry-over)?
6. Monthly repeat for reminders: needed?
7. Follow-up questions: quick-answer buttons in the playground (and later the app)?
8. Clear buttons in the playground panels (one per store, with a confirm box)?
9. Due reminders: spoken + "Ho gaya / 10 min baad" buttons; one repeat after 10 minutes if no answer?
