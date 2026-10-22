# Search and retrieval: how Buddy finds what you saved

Status: **built (19 Oct 2026).** Tests: `tests/test_golden_search.py` (sample data below, tool calls from section 6), router and guard tests in `tests/test_units.py`. Companion to [personal-data-spec.md](personal-data-spec.md)
(what is saved and how). This file is about **finding it again**: the tools, the search strategy for each
category, questions that span several categories, and 10 questions with answers per category.
Vector search is out of scope for now (planned later; the tools keep the same shape when it comes).

Today in the examples: **Mon 19 Oct 2026, 11:00**.

---

## 1. The retrieval pipeline (same for every category)

```
your question
  │
  ① Understand    LLM turns the question into a tool call with structured fields:
  │                who (subject/entities), when (exact dates from the date list in the prompt),
  │                what kind (category, status, store), and free words (query).
  │                Code validates every field (dates, enums, lengths) before anything runs.
  │
  ② Filter        SQL hard filters first: user, store/scope, not expired, status, date range, list name.
  │                This is where most questions are already answered exactly.
  │
  ③ Match         Inside the filtered rows, two matchers run on the free words:
  │                • full-text (Postgres tsvector, `simple` + `unaccent`), weighted:
  │                    A = title / subject / list name, B = keywords & entities, C = text
  │                • fuzzy (pg_trgm similarity) on title, subject and text: typos, Hinglish spellings
  │
  ④ Rank          score = fused rank of the two matchers (RRF)
  │                      + exact subject/entity match boost
  │                      + pinned boost (profile facts)
  │                      + recency (newer first when scores tie; for short-term, closest date first)
  │
  ⑤ Shape         Deduplicate (same key), keep the top k (8 by default), return compact rows:
  │                {store, id, title/subject, text snippet ≤ 160 chars, date, status}.
  │                Counts and totals are computed by SQL and returned as numbers.
  │
  ⑥ Fallback      0 results → relax (drop the free words, keep filters) → small data: newest ~30 titles
  │                for the LLM to pick by meaning → still nothing: "nahi mila" + offer to save (human in the loop)
  │
  ⑦ Answer        The LLM answers only from the returned rows and says where it came from
                   ("12 Oct ke note se", "long term se"). Never guesses a missing fact.
```

Why this is industry-grade without vectors: structured fields are captured when saving, so most questions
become exact SQL; lexical + fuzzy matching with weighted fields and rank fusion covers wording and spelling;
keywords written at save time cover paraphrases; every step is deterministic, testable and logged in the trace.

### Write-time help for search (done once, when saving)

| Field the LLM fills when saving | Used for |
|---|---|
| `title` (notes), `subject` + `topic` (memory), list `name` | weight-A matching ("wifi", "Riya") |
| `keywords` (3–5, Hinglish **and** English: `wifi, internet, password, code`) | paraphrases ("internet ka code") |
| `entities` (people/places/projects named: `["Riya"]`) on **every** store | cross-category questions ("Riya se related sab") |
| dates (`about_date`, `date`, `day`, `due_at`) as real dates | exact date filters ("parson", "is mahine") |

### Indexes (fast at any size)

GIN on each `search` tsvector; GIN `gin_trgm_ops` on title/subject/text; GIN on `entities`;
btree on `(user_id, day, status)` for to-dos, `(user_id, status, due_at)` for reminders,
`(user_id, scope, expires_at)` for memory. Target: every search under 50 ms.

---

## 2. Tool inventory: what exists, what changes, what is new

| Category | Tool | Status | Notes |
|---|---|---|---|
| Ask (memory + notes) | `find_saved(query?, subject?, category?, entities?, from?, to?, month?, in?, limit?)` | **New** | Replaces `recall` and `search_notes` for questions; one search over short-term, long-term and notes |
| Short-term | `stm_add(text, about_date?, entities?)` | **New** (today: `remember` with kind=event) | Lifetime 10 days |
| Short-term | `stm_forget(match)` | **Change** (today: `forget(id)`) | Matches by words |
| Long-term | `ltm_save(category, subject, topic, text, date?, entities?)` | **Change** (today: `remember`) | Key `category:subject:topic` |
| Long-term | `ltm_forget(match)` | **Change** (today: `forget(id)`) | |
| Reminders | `reminder_add(text, when, repeat?, entities?)` | **Change** (today: `create_reminder`) | + monthly repeat |
| Reminders | `reminder_list(from?, to?, status?, repeat?, query?, limit?)` | **Change** (today: `list_reminders`, no filters) | SQL filters + counts |
| Reminders | `reminder_cancel(match)`, `reminder_move(match, when)`, `reminder_done(match)`, `reminder_snooze(match, minutes)` | `cancel` **change** (today by id); others **new** | |
| To-dos | `todo_add(items[], day?, priority?, entities?)` | **Change** (today: `add_todo`, one item) | Many items at once, day-wise |
| To-dos | `todo_list(day? | from/to, status?, priority?, query?)` | **Change** (today: `list_todos(include_done)`) | Returns items + counts + overdue |
| To-dos | `todo_done(match | day+all)`, `todo_remove(match)`, `todo_move(match | day+status, to_day)` | `done` **change**; others **new** | |
| Notes | `note_save(text, title, keywords, entities?)` | **Change** (today: `save_note`) | Title and keywords always filled |
| Notes | `note_delete(match)` | **New** | |
| Lists | `list_add(list, items[])`, `list_show(list?)`, `list_check(list, items[])`, `list_remove(list, items[])`, `list_clear(list, checked_only?)` | `add`/`show`/`check` **change**; `remove`/`clear` **new** | Check by words, not ids |
| All stores | `clear_saved(store, day?)` | **New** | Two-step confirm (human in the loop) |
| Overview | `day_overview(date)` | **New** | One call: that day's to-dos, reminders and short-term items, with counts |
| Other | `calculate`, `web_search` | Exists | Unchanged |

**Match by words** (`match`): the tool searches the right store (pending items first) with the same
full-text + fuzzy matchers. 1 hit → act. Several → returns the candidates and Buddy asks "kaunsa?". 0 → says so.

---

## 3. Search strategy per category

| Category | Main filter (SQL) | Free-word matching | Ranking | Typical questions |
|---|---|---|---|---|
| **Short-term memory** | `scope='short'`, not expired, `about_date` range | full-text + fuzzy on text/keywords | closest `about_date` to today first | "parson kya kiya", "is hafte kya karna hai" |
| **Long-term memory** | `scope='long'`, `category`, `subject`, `month(date)` | weighted: subject/topic (A), keywords (B), text (C) | exact subject boost, pinned boost | "Riya ka birthday", "mere projects", "is mahine kiska birthday" |
| **Notes** | date range (`created_at`), tags | weighted: title (A), keywords/tags (B), text (C) + fuzzy | match score, then newest | "wifi password", "app ideas", "kal ka meeting note" |
| **Reminders** | `status`, `due_at` range, `repeat` | full-text on text (for `query`) | soonest `due_at` first | "kal ke reminders", "agla reminder", "roz wale" |
| **To-dos** | `day` / range, `status`, `priority`; overdue = `day < today AND pending` | full-text + fuzzy on text | priority, then day, then created | "aaj kya baaki", "Friday ka pending", "kitne kaam kiye" |
| **Lists** | `name` (fuzzy on the list name: "shoping" → shopping), `checked` | fuzzy on item text | unchecked first, then added order | "shopping mein kya hai", "doodh liya tha?" |

Free words that are only filler ("kya", "hai", "tha", "mera") are dropped before matching
(a small Hinglish stop-word list), so they never pull in wrong rows.

---

## 4. Questions that span several categories

The LLM may call **several tools in one round** (parallel) and chain at most 4 rounds. Shared `entities`
on every store let one search pull everything about a person, place or project.

| Pattern | Example | Tool calls |
|---|---|---|
| **Everything about X** (one search) | "Riya ke baare mein jo bhi saved hai" | `find_saved(entities=["Riya"], in=all)` + `todo_list(query="Riya")` + `reminder_list(query="Riya")` (parallel) |
| **Memory → to-do** (read, then act) | "Riya ke birthday ke liye ek to-do bana do us din ka" | `find_saved(subject="Riya", category=date)` → `todo_add(["Riya ka birthday gift"], day=2027-05-03)` |
| **Memory → list** | "Mummy ko jo pasand hai wo shopping list mein daal do" | `find_saved(subject="Mummy", category=preference)` → `list_add(shopping, ["Rasmalai"])` |
| **Notes → to-dos** | "Kal ki meeting ke action items aaj ke to-do mein daal do" | `find_saved(in=notes, query="meeting", from=2026-10-18, to=2026-10-18)` → `todo_add(["Logo final karna", "Pricing page"], day=today)` |
| **Notes → list** | "Books wale note ki saari books reading list mein daal do" | `find_saved(in=notes, query="book")` → `list_add(books, ["Atomic Habits", "Deep Work"])` |
| **Memory → reminder** | "Mummy-papa ki anniversary se ek din pehle reminder laga do" | `find_saved(query="anniversary")` → `reminder_add("Anniversary gift", 2026-12-11T10:00)` |
| **Day overview** (one call) | "Aaj ka kya scene hai?" | `day_overview(2026-10-19)` → to-dos + reminders + short-term for today, with counts |
| **Compare two stores** | "Shopping list mein se kya Mummy ko pasand hai?" | `list_show(shopping)` + `find_saved(subject="Mummy", category=preference)` (parallel) |
| **To-do ↔ reminder** | "Jo to-dos aaj ke hain unka shaam 6 baje reminder laga do" | `todo_list(day=today, status=pending)` → `reminder_add("Bill bharna, Amit ko call", 2026-10-19T18:00)` |
| **Short-term ↔ long-term** | "Rahul ko kitne diye the aur wo kaun hai?" | `find_saved(entities=["Rahul"], in=all)` (one call covers both scopes) |

Rules for combined questions:
- Read first, then write. A write that came from a lookup is confirmed in one line ("Rasmalai shopping list mein daal di").
- If the lookup finds nothing, stop and ask; never invent the missing piece (no birthday → "Riya ka birthday saved nahi hai, bata do?").
- If the lookup finds several candidates for a write, ask which (human in the loop).

---

## 5. Sample data used in the answers below

| Store | Rows |
|---|---|
| Short-term | Paid Rahul 2500 (17 Oct) · Car given for service (17 Oct) · Amit's parcel arrives (20 Oct) · On a diet this week (19 Oct) · Passport office visit (24 Oct) · Rent pending to landlord |
| Long-term | profile: name Ashish, city Pune, blood group B+ · Riya: sister, lives in Mumbai, birthday 3 May · Amit: college friend, Delhi · Rahul: colleague, designer · Mummy: likes rasmalai · Parents: anniversary 12 Dec · projects: Project Synq (dating app), Companion Umbrella (AI buddy) · me: likes paneer |
| Reminders | Dawai, daily 9:00 · Gym, Fri 23 Oct 18:00 · Team call, weekly Mon 10:00 (next 26 Oct) · Rent, monthly 5th 10:00 (next 5 Nov) · Chai, 19 Oct 10:30 (done) |
| To-dos | 18 Oct: Bank jaana (pending), Groceries (done) · 19 Oct: Doodh lana (done), Bill bharna (pending, high), Amit ko call (pending) · 20 Oct: Passport documents ready (pending) · 23 Oct: Presentation banana (pending, high), Riya ke liye gift order (pending) |
| Notes | Wifi: "Wifi password abc123" (12 Oct) · App idea: "Voice se kharcha add karna" (15 Oct) · Amit's address: "21 MG Road, Delhi" · Books: "Atomic Habits, Deep Work" · Car insurance: "XYZ123" · Meeting (18 Oct): "Launch 1 Nov; Rahul design; action: logo final, pricing page" · Doctor: "Dawai din mein 2 baar" |
| Lists | shopping: ande, bread, aloo 2 kg, ✓doodh · watchlist: ✓Dune, Oppenheimer · goa packing: charger, ✓sunscreen · books: Atomic Habits |

---

## 6. Ten questions per category, with answers and tool calls

### 6.1 Short-term memory

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Short term mein kya-kya hai? | `find_saved(in=short)` | "6 baatein: Rahul ko 2500 diye (17 Oct), car service ko di (17 Oct), Amit ka parcel kal aayega, is hafte diet, 24 Oct passport office, landlord ka rent baaki." |
| 2 | Maine Rahul ko kitne diye the? | `find_saved(entities=["Rahul"])` | "17 Oct ko 2500." |
| 3 | Parson kya kiya tha? | `find_saved(in=short, from=10-17, to=10-17)` | "Rahul ko 2500 diye aur car service ko di." |
| 4 | Kal kya hone wala hai? | `find_saved(in=short, from=10-20, to=10-20)` | "Amit ka parcel aayega." |
| 5 | Is hafte kya-kya karna hai? | `find_saved(in=short, from=10-19, to=10-25)` | "Diet chal raha hai, kal parcel, Saturday passport office." |
| 6 | Passport office kab jaana hai? | `find_saved(query="passport")` | "Saturday, 24 Oct." |
| 7 | Koi payment baaki hai kya? | `find_saved(in=short, query="payment rent baaki")` | "Landlord ka rent baaki hai." |
| 8 | Car service kab di thi? | `find_saved(query="car service")` | "17 Oct ko." |
| 9 | Diet kab tak hai? | `find_saved(query="diet")` | "Is hafte ke liye (19–25 Oct)." |
| 10 | Short term mein kuch bhi Amit wala? | `find_saved(in=short, entities=["Amit"])` | "Haan, uska parcel kal aayega." |

### 6.2 Long-term memory

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Riya ka birthday kab hai? | `find_saved(subject="Riya")` | "3 May." |
| 2 | Is mahine kiska birthday hai? | `find_saved(in=long, category=date, month=10)` | "Is mahine koi birthday saved nahi hai." |
| 3 | Mere baare mein kya jaante ho? | `find_saved(in=long, subject="me")` | "Aap Ashish ho, Pune mein rehte ho, blood group B+, paneer pasand hai." |
| 4 | Amit kaun hai? | `find_saved(subject="Amit")` | "Aapka college friend, Delhi mein rehta hai." |
| 5 | Mere projects kaun se hain? | `find_saved(in=long, category=project)` | "Project Synq (dating app) aur Companion Umbrella (AI buddy)." |
| 6 | Mummy ko kya pasand hai? | `find_saved(subject="Mummy", category=preference)` | "Rasmalai." |
| 7 | Anniversary kab hai mummy-papa ki? | `find_saved(query="anniversary")` | "12 Dec." |
| 8 | Mera blood group? | `find_saved(subject="me", query="blood group")` | "B+." |
| 9 | Riya abhi kahan rehti hai? | `find_saved(subject="Riya", query="lives city")` | "Mumbai." |
| 10 | Delhi mein mera kaun hai? | `find_saved(query="Delhi")` | "Amit (college friend). Note mein unka address bhi hai: 21 MG Road." |

### 6.3 Reminders

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Mere reminders kya hain? | `reminder_list()` | "4: dawai roz 9 baje, gym Friday 6 baje, team call har Monday 10 baje, rent har mahine 5 ko." |
| 2 | Kal ke reminders? | `reminder_list(from=10-20, to=10-20)` | "Kal subah 9 baje dawai." |
| 3 | Aaj aur kuch yaad dilana hai? | `reminder_list(from=now, to=10-19T23:59)` | "Aaj aur koi reminder nahi (dawai kal subah phir)." |
| 4 | Is hafte kitne reminders hain? | `reminder_list(from=10-19, to=10-25)` → `count` | "7: roz ki dawai (6 baar) aur Friday ka gym." |
| 5 | Roz wale reminders? | `reminder_list(repeat=daily)` | "Bas dawai, 9 baje." |
| 6 | Gym ka reminder kab hai? | `reminder_list(query="gym")` | "Friday, 23 Oct, shaam 6 baje." |
| 7 | Agla reminder kab hai? | `reminder_list(limit=1)` | "Kal subah 9 baje, dawai." |
| 8 | Rent ka reminder kab aayega? | `reminder_list(query="rent")` | "5 Nov, 10 baje (har mahine)." |
| 9 | Aaj ke kaunse ho gaye? | `reminder_list(status=done, from=10-19, to=10-19)` | "Chai wala, 10:30 pe." |
| 10 | Monday ko kya hai? | `reminder_list(from=10-26, to=10-26)` | "9 baje dawai, 10 baje team call." |

### 6.4 To-dos

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Aaj ke to-do mein kya baaki hai? | `todo_list(day=10-19, status=pending)` | "2 baaki: bill bharna (high), Amit ko call. Kal ka bank jaana bhi reh gaya hai." |
| 2 | Kal ke to-do mein kya tha? | `todo_list(day=10-18, status=all)` | "Bank jaana (baaki) aur groceries (ho gaya)." |
| 3 | Friday ki list mein kya pending hai? | `todo_list(day=10-23, status=pending)` | "Presentation banana (high) aur Riya ke liye gift order." |
| 4 | Aaj kitne kaam ho gaye? | `todo_list(day=10-19, status=done)` → `count` | "1: doodh lana." |
| 5 | Koi high priority kaam? | `todo_list(status=pending, priority=high)` | "Bill bharna (aaj) aur presentation (Friday)." |
| 6 | Pichhle dinon ka kya reh gaya? | `todo_list(to=10-18, status=pending)` | "18 Oct ka bank jaana." |
| 7 | Bank wala kaam kab ka hai? | `todo_list(query="bank")` | "18 Oct ka tha, abhi baaki hai." |
| 8 | Kal ke liye kya plan hai? | `todo_list(day=10-20)` | "Passport documents ready karna." |
| 9 | Is hafte total kitne kaam hain? | `todo_list(from=10-19, to=10-25, status=all)` → counts | "6 kaam: 1 ho gaya, 5 baaki." |
| 10 | Riya wala to-do kab ka hai? | `todo_list(query="Riya")` | "Friday, 23 Oct: gift order." |

### 6.5 Notes

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Wifi password kya hai? | `find_saved(query="wifi password")` | "abc123 (12 Oct ke note se)." |
| 2 | Internet ka code kya tha? | `find_saved(query="internet code")` (keywords hit) | "Wifi password abc123." |
| 3 | Mere app ideas? | `find_saved(query="app idea")` | "Note: voice se kharcha add karna. Projects: Project Synq, Companion Umbrella." |
| 4 | Amit ka address? | `find_saved(query="Amit address")` | "21 MG Road, Delhi." |
| 5 | Kaunsi books note ki thi? | `find_saved(in=notes, query="book")` | "Atomic Habits aur Deep Work." |
| 6 | Kal kya note kiya tha? | `find_saved(in=notes, from=10-18, to=10-18)` | "Meeting note: launch 1 Nov, Rahul design dekhega, action: logo final, pricing page." |
| 7 | Car insurance number? | `find_saved(query="car insurance")` | "XYZ123." |
| 8 | Doctor ne kya bola tha? | `find_saved(query="doctor")` | "Dawai din mein 2 baar." |
| 9 | Launch kab hai? | `find_saved(query="launch")` | "1 Nov (18 Oct ki meeting ke note se)." |
| 10 | Mere saare notes dikhao | `find_saved(in=notes)` | "7 notes: Wifi, App idea, Amit's address, Books, Car insurance, Meeting, Doctor." |

### 6.6 Lists

| # | You ask | Tool call | Buddy answers |
|---|---|---|---|
| 1 | Shopping list mein kya hai? | `list_show(shopping)` | "Ande, bread, aloo 2 kg (doodh le liya)." |
| 2 | Kya-kya lena baaki hai? | `list_show(shopping)` | "3 cheezein: ande, bread, aloo." |
| 3 | Meri kitni lists hain? | `list_show()` | "4: shopping, watchlist, goa packing, books." |
| 4 | Watchlist mein kya bacha hai? | `list_show(watchlist)` | "Oppenheimer (Dune dekh li)." |
| 5 | Goa ke liye kya pack karna hai? | `list_show(goa packing)` | "Charger (sunscreen rakh liya)." |
| 6 | Doodh liya tha? | `list_show(shopping)` | "Haan, le liya." |
| 7 | Books list dikhao | `list_show(books)` | "Atomic Habits." |
| 8 | Shoping list mein kitne items? (typo) | `list_show(shopping)` (fuzzy name) | "4 items, 3 baaki." |
| 9 | Kaunsi movies dekh li? | `list_show(watchlist)` | "Dune." |
| 10 | Packing poori hui? | `list_show(goa packing)` | "Nahi, charger baaki hai." |

### 6.7 Questions across categories

| # | You ask | Tool calls | Buddy answers |
|---|---|---|---|
| 1 | Riya ke baare mein jo bhi saved hai | `find_saved(entities=["Riya"], in=all)` + `todo_list(query="Riya")` + `reminder_list(query="Riya")` | "Behen, Mumbai mein, birthday 3 May. To-do: Friday ko gift order." |
| 2 | Aaj ka kya scene hai? | `day_overview(10-19)` | "To-do: bill bharna, Amit ko call (bank kal se baaki). Reminders: aaj aur nahi. Short term: diet chal raha hai." |
| 3 | Mummy ko jo pasand hai wo shopping list mein daal do | `find_saved(subject="Mummy", category=preference)` → `list_add(shopping, ["Rasmalai"])` | "Rasmalai shopping list mein daal di." |
| 4 | Kal ki meeting ke action items aaj ke to-do mein daal do | `find_saved(in=notes, query="meeting", from=10-18, to=10-18)` → `todo_add(["Logo final karna", "Pricing page banana"], day=10-19)` | "Aaj ke to-do mein daal diye: logo final, pricing page." |
| 5 | Books wale note ki books reading list mein daal do | `find_saved(in=notes, query="book")` → `list_add(books, ["Deep Work"])` | "Deep Work joda; Atomic Habits pehle se list mein hai." |
| 6 | Anniversary se ek din pehle reminder laga do | `find_saved(query="anniversary")` → asks "Kis time?" → `reminder_add("Anniversary gift", 2026-12-11T10:00)` | "11 Dec subah 10 baje ka reminder laga diya." |
| 7 | Shopping list mein se kya Mummy ko pasand hai? | `list_show(shopping)` + `find_saved(subject="Mummy", category=preference)` | "Abhi kuch nahi; unhe rasmalai pasand hai, daal dun?" |
| 8 | Aaj ke to-dos ka shaam 6 baje reminder laga do | `todo_list(day=10-19, status=pending)` → `reminder_add("Bill bharna, Amit ko call", 2026-10-19T18:00)` | "Shaam 6 baje ka reminder laga diya: bill bharna, Amit ko call." |
| 9 | Rahul ko kitne diye the aur wo kaun hai? | `find_saved(entities=["Rahul"], in=all)` | "17 Oct ko 2500 diye. Rahul aapka colleague hai, designer." |
| 10 | Friday ko kya-kya hai? | `day_overview(10-23)` | "To-do: presentation (high), Riya ka gift. Reminder: 6 baje gym." |

---

## 7. Measuring it (so "best in class" is a number, not a feeling)

1. **Golden set**: the 70 questions above (and more over time) with the expected tool calls and the rows that must come back.
2. **Search metrics** (tool layer, no LLM, runs in the test suite): recall@5 ≥ 95%, MRR ≥ 0.9, p95 latency < 50 ms.
3. **Routing metrics** (live LLM, run on demand): right tool chosen ≥ 95%, right fields (dates, subject) ≥ 95%,
   asked a question when it should ≥ 90%, never wrote when it should have asked.
4. **Answer check**: every fact in the answer appears in the returned rows (grounding).
5. **Trace**: each call's arguments and results are already visible in the playground (`{ }` button, Trace).

## 8. Safety and privacy

- Every query is scoped by `user_id`; values go as SQL parameters; the LLM never writes SQL.
- Bulk deletes always ask first (personal-data-spec, section 2d).
- Notes can hold secrets (passwords): they are returned only when asked, never placed in the prompt.
