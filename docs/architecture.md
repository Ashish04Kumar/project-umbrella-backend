# Companion Umbrella: architecture

The umbrella is a backend-only Python package (`companion`) that holds an AI companion's brain. Bodies (the mobile Buddy tab, a laptop UI, the Raspberry Pi pan-tilt robot) build their own frontends and talk to this brain. The brain never draws UI.

Status: milestone 1 (text brain) is built and tested; see section 14 for what comes next. How to run it: [../README.md](../README.md).

## 1. Two ways to use it

```
Embedded (same process, no network hop)        Server (any frontend over the network)

  robot.py / laptop_bot.py                        Phone (Buddy tab)   Laptop UI
        │ import companion                               │  WebSocket / HTTP  │
        ▼                                                ▼                    ▼
  ┌─────────────┐                                  ┌──────────────────────────────┐
  │  Companion  │                                  │ companion.server (FastAPI)   │
  │   (brain)   │                                  │      └─ Companion (brain)    │
  └─────────────┘                                  └──────────────────────────────┘
```

The same `Companion` object is used in both. The server is only a thin wrapper around it.

```python
from companion import Companion

bot = await Companion.create()                     # reads settings from .env
reply = await bot.chat(user_id="ashish", text="kal 9 baje dawai yaad dilana")
print(reply.text, reply.expression)                # "Done, kal subah 9 baje..." "happy"
# later: await bot.event(user_id="ashish", kind="person_appeared")   # robot camera
```

## 2. Technology

| Need | Choice | Why |
| --- | --- | --- |
| Language | Python 3.11+, typed | Same as the Synq backend |
| Agent loop | **LangGraph** (`StateGraph`) | Explicit graph of nodes; tool loop; already used by the Synq Conductor |
| Data shapes | Pydantic v2 (graph state is a TypedDict) | Validates tool arguments, extractor output, API messages |
| LLM | Own provider interface, OpenAI-compatible HTTP (stdlib urllib) | Replies: Groq `gpt-oss-120b`, then `qwen3.8-27b`. Memory/summary: Groq `gpt-oss-20b`, then qwen, then `gpt-oss-120b`, then laptop Ollama. No LangChain model wrappers |
| Tools | Own registry: a Python function + a Pydantic args model → JSON tool schema | The LLM sees the schema and decides which tool to call |
| Storage | PostgreSQL via asyncpg, own schema `companion` (inside the `synq` database for now) | Reminders, todos, notes, lists, memory, messages |
| Memory search | Postgres full-text search (built in) | Milestone 1; embeddings (pgvector) later, after approval |
| Web search | Tavily over plain HTTP (`TAVILY_API_KEY`) | Approved. The tool appears only when the key is set |
| Server mode | FastAPI + WebSocket, uvicorn | Same as the Synq backend |
| Time | `zoneinfo` (stdlib), default `Asia/Kolkata` | "kal", "parson", "next Sunday" need today's date in the prompt |
| Voice | STT: Groq Whisper `whisper-large-v3-turbo` (cloud), then laptop faster-whisper, then Sarvam. TTS: Piper (laptop), then Sarvam. Install extra `companion[voice]` | Same libraries as Synq Pod 4, used directly (no pipecat) so phone, playground and Pi share them |
| Tests | unittest (stdlib) against a throwaway Postgres schema, scripted fake LLM | No network in tests; a labelled Hinglish eval set comes in milestone 3 |

No library outside this list is added without asking the owner.

## 3. Folder structure

```
companion-umbrella/
  pyproject.toml
  .env.example
  companion/
    __init__.py            # Companion class (the public entry point)
    config.py              # settings from .env
    persona.py             # name, personality, rules, prompt builder, date list, script hint
    guard.py               # mood tag, junk and life-story filter, tool confirmation fallback
    graph.py               # LangGraph graph: nodes and edges
    state.py               # the graph's state (Pydantic)
    llm/                   # provider interface, OpenAI-compatible adapter (Groq, Ollama), fallback router
    tools/                 # registry + reminders, todos, notes, lists, memory, web_search, calculator
    memory/
      interface.py         # MemoryStore protocol: remember / recall / forget / update
      postgres.py          # local adapter (milestone 1)
      in_memory.py         # for tests and quick embedded runs
      extractor.py         # reads a turn, returns typed memory items
    store/                 # Postgres pool, migrations, reminders/todos/notes/lists/messages
    expressions.py         # shared expression list
    brain.py               # Companion class: chat(), due_reminders(), background memory work
    cli.py                 # terminal chat for testing
    voice/                 # audio helpers, STT/TTS providers with fallback, STT guard, speakable()
    realtime/              # live voice: pipecat services, LiveTurns (turn taking, barge-in, nudge), LiveKit sessions
    server/                # FastAPI app + playground.html (local testing UI)
    (later) events.py
  migrations/              # SQL files
  tests/                   # test_units.py (no database), test_postgres.py
  docs/architecture.md
```

## 4. One message, step by step (the LangGraph graph)

```
 user text
    │
    ▼
 [load_context] ── persona + rules + now + a list of the next 8 dates (kal, parson...)
    │               + last ~20 messages of this thread + its running summary
    │               + all long-term memories (up to 40, with id and key) + active events
    │               + next reminders + one unsure memory to ask about + reply-script hint
    ▼
 [agent] ◀──────────────┐   LLM writes a reply OR asks for tools
    │ tool calls?       │
    ├── yes ─▶ [tools] ─┘   code validates args, runs the tool, returns the result
    │                      (loop at most 4 rounds)
    ▼ no
 [guard]    mood tag parsed out (default "idle"); hidden reasoning and made-up life stories
    │       removed; junk output ("… … ...") dropped. If the LLM failed or produced junk after
    │       a tool ran, the reply is built from the tool results ("Reminder set: dawai, Tue 9:00 AM")
    ▼
 [respond]  reply text + expression back to the body; messages saved
    │
    ▼ (background, does not delay the reply)
 [extract_memory]  turn → typed items → save / ask / ignore
```

Graph state (`TurnState` in `graph.py`): `user_id`, `user_text`, `now`, `conversation_id`, `system`, `messages`, `new_messages`, `tool_rounds`, `tool_actions`, `tool_results`, `sources`, `reply`, `expression`, `done`.

Code, not the LLM, decides the dates ("parson" = the day after tomorrow) and the reply script (Roman when the user writes Roman), because the model got both wrong in testing.

Messages and memory are saved in our own store, not a LangGraph checkpointer, so the memory backend can be swapped for the owner's cloud memory without touching the graph.

## 5. Where an answer comes from

| The question is about | Source | Rule |
| --- | --- | --- |
| The user (name, likes, people) | Memory only | Not in memory → "tumne abhi tak bataya nahi" |
| Live or changing facts (weather, news, scores, prices) | `web_search` | Must search first; say the source |
| General knowledge, advice, fun | The LLM itself | Allowed; say when unsure |
| An action (remind, note, list) | Tools | Done in the database, then confirmed aloud |
| Medical, legal, money | LLM general info | Plus "kisi expert se confirm kar lena" |
| Self-harm or abuse | Safety reply | Gentle reply and a helpline; no jokes |

Umbrella has no life of its own (no home, family, trips). It never claims to have done things.

## 6. Tools

27 tools in groups (full list, arguments and examples: [personal-data-spec.md](personal-data-spec.md); search: [search-and-retrieval-spec.md](search-and-retrieval-spec.md)).

| Group | Tools | Offered when |
| --- | --- | --- |
| read | `find_saved` (memory + notes), `reminder_list`, `todo_list`, `list_show`, `day_overview`, `calculate`, `web_search` (with `TAVILY_API_KEY`) | Always (about 1,100 tokens of schemas) |
| short / long | `stm_add`, `stm_forget` / `ltm_save`, `ltm_forget` | "remember", "short term", "long term", "forget" ... |
| reminder | `reminder_add`, `reminder_cancel`, `reminder_move`, `reminder_done`, `reminder_snooze` | "remind", "reminder", times ... |
| todo / note / list | `todo_add`, `todo_done`, `todo_remove`, `todo_move` / `note_save`, `note_delete` / `list_add`, `list_check`, `list_remove`, `list_clear` | Their keywords |
| clear | `clear_saved` (count first, `confirm=true` only after a yes) | "clear", "delete all" ... |

`companion/router.py` picks the groups from keywords in the message (and in Umbrella's last question, for follow-ups like "9 baje"); the LLM then decides by meaning. Later: `read_call_insights`, `knowledge_base_search`, expenses/goals/habits, calendar, robot tools.

Every tool has a Pydantic args model. Invalid arguments are rejected in code and the LLM is told why. Write tools return `say` (what happened) or `ask` (one question back, e.g. "When should I remind you?"); when only write tools ran, that text is the reply, built by code, with no second LLM call. Matching by words (`search.pick`) acts on one clear hit and asks when several are close. Bulk deletes always ask first.

Where to save (owner's rule, 20 Oct): if the user names the place (memory / long term / short term / note / reminder / to-do / list) it is saved there, and a different tool is sent back. If not, every save tool carries `confidence` (0–1) and saves only at 0.8 or more; below that the tool answers "Where should I save “…”? (reminder / to-do / short-term memory / long-term memory / note / list)" and the playground shows one button per place. A reminder asks for whatever the user did not say, never guessing it: no day and no time → "Which day and what time…? (or say full day)"; only a day → "At what time on Mon 26 Oct…?"; only a time → "Which day…?". Answers build on the message before. **Full-day reminders** (`reminders.all_day`, migration `004`): only when the user says "full day / all day / pura din"; no time of their own, said out loud at 9:00 that morning ("Today, all day: …"), shown as "all day" in the panel and editable there. When the model says it saved something but no tool saved it, the reply is replaced by "I haven't saved that yet. Where should I save it? (…)".

**Picking one reminder** (cancel, done, move, snooze): one match is acted on; several give "Which reminder do you mean? 1) Study, Wed 21 Oct, 9:00 AM 2) …", answered by number (`number`), date (`day`) or, for cancel, "all" (`all_matching`, then `confirm` after a yes). Exact copies (same words and time) count as one and are cancelled together.

Night test fixes (20 Oct): "save it" and the answer to "Where should I save" offer only the six save tools (`tags=("save",)`, ~3k instead of ~5–7k prompt tokens, which had made requests too big for the fallback model); an update of a fact already in long-term memory stays there (no "where" question, and another place named in the same message does not apply); a "yes" to "Cancel all N of these?" cancels exactly the listed reminders (a different list is asked again); a reminder id taken by the model from a list does not skip "which one?" when the user's words fit several; a 12:00 AM the user did not say is asked; "abhi", "now", "later" are not a time; "yaad kar le" means remember.

English test fixes (20 Oct, `docs/test-reports/2026-10-20-english-test.md`): a last-year date that falls within 31 days before or 60 days after today is moved to this year before any tool runs (`tools/dates.py`), and other past-year dates are still refused; `clear_saved` checks the year too; "add", "calendar", "diary" and "schedule it" with no place named offer the six save tools; the message before only counts for the day check when the new message names no day; an empty LLM answer is retried once, and an English user gets the fallback line in English; a denial ("I don't have anything saved") is not treated as claiming an action.

**Every database change is recorded** (`store/audit.py`): one server log line per change, e.g. `🗄️ DB UPDATE reminders fd98c0f2 ("play football", Wed 21 Oct 4:00 PM): status pending → cancelled`; inside a turn it is also a trace event (`db_change`, shown in the Test tab) and part of the reply object (`db_changes`, shown on the chat's `{ } backend response`). Cancelling a reminder never deletes the row: its status becomes `cancelled`. A weekday the user names must match the date chosen.

Date and time checks in code (models get these wrong, live tests 19–20 Oct): a date in a past year is sent back with this week's dates; when the user said "today / tomorrow / day after" ("aaj / kal / parson") the date must match, and between midnight and 5 AM "tomorrow" is asked about (later today or the next day); a reminder at 1–5 o'clock asks "AM (night) or PM?" unless the user said AM or night; the same reminder (words and time) is never stored twice. Umbrella's previous message only offers its tool groups to the next message when it asked a question, so "hi" after "Reminder set…" cannot repeat the action.

## 7. Memory

### Four layers

| Layer | Holds | Lifetime | Goes into the prompt |
| --- | --- | --- | --- |
| Recent chat | Last ~20 messages of the thread | Always (a thread never resets) | All |
| Summary | Older messages of the thread, folded in as the chat grows | Always, updated in the background | The latest summary |
| Short-term memory | "short term ...", payments, plans, today's state | 10 days, then expires | No: fetched with `find_saved` |
| Long-term memory | Profile, people, dates, preferences, projects (`category`, `subject`, `topic`) | Until changed or deleted | Only the pinned profile (at most 8); the rest via `find_saved` |

### Threads

Each conversation belongs to a thread (`conversations.thread`, migration `002_threads.sql`). `main` is the user's own ongoing chat with the companion: it is never reset or replaced, it just keeps growing while older messages fold into its summary. **New conversation** (`POST /v1/users/{id}/threads`) opens a separate thread `t-xxxxxxxx` with its own short-term memory; long-term memory is shared by all threads. Chat and voice endpoints take `thread` (default `main`); live voice always uses `main`. Listing, naming, closing threads and whether a new thread should see the main thread's summary are future scope.

### Interface (what the cloud memory will plug into)

```python
class MemoryStore(Protocol):
    async def remember(self, user_id: str, item: MemoryItem) -> str: ...
    async def search(self, user_id: str, *, query=None, subject=None, category=None, entities=None,
                     date_from=None, date_to=None, month=None, scope=None, limit=8) -> list[MemoryItem]: ...
    async def recall(self, user_id: str, query: str, limit: int = 8) -> list[MemoryItem]: ...
    async def update(self, user_id: str, item_id: str, item: MemoryItem) -> None: ...
    async def forget(self, user_id: str, item_id: str) -> bool: ...
    async def list_all(self, user_id: str, kind=None, status="active", *, scope=None, pinned=None) -> list[MemoryItem]: ...
    async def clear(self, user_id: str, scope: str) -> int: ...
```

`MemoryItem`: `scope` (short/long), `category`, `subject`, `topic`, `text`, `event_date` + `yearly`, `keywords`, `entities`, `pinned`, `confidence`, `status`, `expires_at`. Long-term items with the same category, subject and topic update each other.

Adapters: `postgres`, `in_memory` (tests), `cloud` (the owner's memory project, later). Chosen with `COMPANION_MEMORY=postgres`. Search: weighted full text + trigram fuzzy, fused with RRF, then ranked across stores by how many of the question's words a hit covers ([search-and-retrieval-spec.md](search-and-retrieval-spec.md)). No vector search for now.

### Saving only when asked

Umbrella saves only when the user asks ("remember ...", "short term ...", "note down ...", "remind me ..."). The background extractor that guessed facts after every message is off (`COMPANION_AUTO_MEMORY=false`, saves about 1,500 tokens a message); its code stays in `memory/extractor.py` for later. The guard replaces any reply that claims something was saved or changed when no tool ran.

## 8. Database tables

| Table | Main columns |
| --- | --- |
| `companion_users` | id, display_name, timezone, settings (JSON) |
| `conversations` | id, user_id, thread (`main` or `t-…`), started_at, last_message_at, summary, summarized_count |
| `messages` | id, conversation_id, role, text, expression, tool_calls (JSON), created_at |
| `memories` | id, user_id, scope, category, subject, topic, key, text, event_date, yearly, keywords, entities, pinned, confidence, status, expires_at, weighted search vector |
| `reminders` | id, user_id, text, due_at, repeat (none/daily/weekly/monthly), status (pending/fired/snoozed/done/cancelled), fired_at, reminded_count, entities |
| `todos` | id, user_id, text, day, priority, done, done_at, entities |
| `notes` | id, user_id, title, text, keywords, entities, weighted search vector |
| `lists`, `list_items` | id, user_id, name / list_id, text, done |
| `tool_log` | id, user_id, tool, args, result, ok, created_at |

Migrations: `001_init.sql`, `002_threads.sql`, `003_personal_data.sql` (needs the `pg_trgm` extension, created in schema `public`).

**Reminders are spoken.** Bodies poll `GET /v1/users/{id}/reminders/due`; each due reminder comes back once with a `speak` line (the playground says it with TTS and shows ✅ Ho gaya / ⏰ 10 min baad, which call `POST .../reminders/{rid}/done` and `.../snooze`). No answer: it is said once more after 10 minutes. Repeating reminders move to their next time. The side panels' clear buttons call `POST /v1/users/{id}/clear` after an "are you sure?" box.

`user_id` is a plain string, so the brain does not depend on the Synq auth tables. The Synq backend passes its own user id.

## 8b. Voice (push-to-talk, step 1)

```
WAV clip ─▶ read + resample to 16 kHz ─▶ STT (groq → whisper → sarvam) ─▶ guard ─▶ chat(spoken=True) ─▶ speakable() ─▶ TTS (piper → sarvam) ─▶ WAV
```

- `Companion.voice_turn(user_id, wav)` returns what was heard, the reply and its audio. `POST /v1/voice?user_id=` (raw WAV body) and `POST /v1/tts` expose it.
- The guard (copied from Synq's `stt_guard.py`) drops Whisper's made-up "Thank you." / "Thanks for watching" from blips or near-silence, using speech length and loudness of the clip.
- Spoken replies get extra rules: Roman script (Piper reads English letters only), 1–3 sentences, no emojis/lists/links, never build on a misheard word. `speakable()` strips emojis, markdown and links before TTS.
- STT is Groq Whisper first (the owner approved sending audio to Groq). A short Roman-script Hinglish prompt keeps Hindi words in English letters. Laptop Whisper `base` is the fallback and is loaded at startup, so a Groq failure costs no model load.
- Measured (2026-10-19, Piper audio as input): Groq ~1.0–1.3 s per clip vs laptop Whisper `base` ~2.7 s (`small` ~7.5 s). Groq also hears Hinglish much better ("yaar aaj office mein bahut thak gaya" -> "Yar aj afas mein bahat-thak gaya", laptop -> "Yarr Aj-Office-Main-Bahat-Thak-Geya"). A push-to-talk turn is ~4.3 s: STT ~1.2 s + LLM ~1.1 s + TTS ~2 s (the whole reply is spoken at once here, no streaming).
- Groq free-tier speech limits apply (requests per day and audio seconds per hour, shared with the Groq organization).
- Known gap: Piper's voice is American English, so Hindi words sound foreign. Sarvam (Saarika STT, Bulbul TTS) is wired in but **untested: no `SARVAM_API_KEY` yet**.

## 8c. Live voice (real time, step 2, ported from Synq Pod 4)

```
Body mic ─LiveKit─▶ Silero VAD ─▶ VadTap ─▶ STT (companion.voice) ─▶ LiveTurns ─▶ TTS (companion.voice) ─LiveKit─▶ Body speaker
                                     │                                   │
                       speech timing + mic audio            chat_stream(): LangGraph streams the reply,
                       (turn taking, voice cues)            SentenceShaper cuts it into speakable sentences
```

| Feature | How |
| --- | --- |
| Transport | LiveKit Cloud (same project as Synq). `POST /v1/live/start {user_id}` starts the companion in a fresh room and returns `{url, token, room}`; `POST /v1/live/stop {room}`. One session per user, ends when the user leaves or after `COMPANION_LIVE_MAX_MINUTES` (with a spoken goodbye). |
| VAD | Pipecat's Silero VAD. `VadTap` watches it before STT so "is the user talking?" is never late. |
| Turn taking (smart) | Always on, no Send button. Pod 4's base rules (all Whisper pieces in, pieces joined) plus `realtime/turn_policy.py`: when the user stops, **Smart Turn v3** (pipecat's bundled ONNX model, ~0.2 s on CPU) hears the last 8 s of their voice and says how finished it sounds, and the **words** are checked ("…and", "aur", "ki", "umm", a comma = not finished). Wait: words trail off → hold up to 2.5 s; sounds finished → 0.3 s; a complete question → 0.5 s; voice unsure but words complete → 1.0 s; no verdict → 0.8 s; never more than 3 s. |
| Merge | If the user starts talking again after a turn was sent but before any of the reply is heard, that reply is cancelled (`Companion.discard_turn`: the half message and the unheard reply are not saved; tool steps that changed data are) and both parts go as one message. |
| STT guard | Each text piece is matched to its stretch of speech (`check_with_segments`); made-up "Thank you." from a blip is dropped. |
| Voice cues | `voice/cues.py` (copied from Pod 4): pause before answering, interruptions, pace, long pauses, loudness, intonation, trailing off, fillers -> a soft note in the prompt (tone and pacing only, never mentioned). |
| Streaming | The LLM streams (tool calls are collected from the stream); `SentenceShaper` hands each finished sentence to TTS at once, drops made-up life stories, silence talk in nudges and junk, max ~70 words. |
| Barge-in (smart) | Talking over the companion stops it after **0.5 s** of continuous speech (a cough or "hmm" does not); "wait", "stop", "ruko", "one sec" stop it as soon as they are heard; "ok", "haan", "hmm" said over it are not answered as a new turn. Its audio stops (`broadcast_interruption`) and the rest of the reply is dropped; only the sentences actually spoken are saved. |
| Live log | Every decision (Smart Turn verdict, waiting and why, sent and why, merge, barge-in, ignored sounds) is a `live_turn` trace event; the playground shows them under the Live bar. |
| Silence nudge | One gentle follow-up after `COMPANION_SILENCE_NUDGE_SECONDS` (10) of quiet, grounded in their last words, never twice without a new user turn, never in safety mode or during the goodbye. |
| Greeting | When the user joins: by name and with what's coming up, from memory. |
| Safety | Crisis words (English/Hinglish) -> `safety_resources` message (Tele-MANAS 14416, 112) and no more nudges. |
| Messages to the body | JSON over the LiveKit data channel: `caption_delta {role,text}` (Umbrella's words the moment the model writes them), `caption {role,text}` (the full turn), `expression {expression}`, `state {listening|thinking|speaking|idle}`, `safety_resources`. (Pipecat also sends its own `rtvi-ai` messages; bodies can ignore them.) |

Measured (headless LiveKit test, `scripts/live_e2e.py`): greeting, normal turn, barge-in, nudge (after ~12 s, once) all pass. From the end of speech to Umbrella's first audio: **~6.2 s with Groq STT** (7.3 s with laptop Whisper). Roughly: Silero VAD needs ~0.8 s of quiet to end the turn, Groq STT ~0.9–1.4 s per stretch of speech, Pod 4's turn-taking waits (0.6 s after the last piece, 0.7 s of silence), the LLM's first sentence ~1.0–1.5 s, Piper ~0.3 s, plus LiveKit transport and the test client's own buffering. Further cuts would mean shorter turn-taking waits (risk: answering while the user only paused), which Pod 4 tuned on purpose.

## 8d. Trace panel (seeing what the agent does)

The playground's **🧠 Trace** button opens three live columns (each chat bubble also has a `{ }` button: on your message the request object that went to the backend, incl. voice cues; on Umbrella's reply the whole response object; the streams carry them as `request`/`response` events, live turns as a `turn_objects` message):

| Column | Shows |
| --- | --- |
| Short-term memory | The message window (which messages go to the LLM whole, which are folded into the summary), the thread's running summary, working memory with expiry |
| Long-term memory | Saved items by category (pinned badge; badge when they were in the last prompt), unsure items waiting to be asked once (only with the extractor on) |
| Memory objects (DB) | `GET /v1/users/{id}/memory/raw`: the Postgres schema and this user's rows of `companion_users`, `memories`, `conversations`, `messages` |

The brain still emits its step events (context, model reasoning, tool calls, memory decisions...); the panel uses them only to refresh the memory columns. `request`/`response` events come from the server endpoints and from live turns.

How it works: `companion/trace.py` (`TraceHub`) collects events from the graph and the brain (`turn_start`, `stt`, `context`, `llm_call`, `llm_result`, `llm_error`, `tool_result`, `guard`, `speak`, `turn_end`, `memory_extract_start`, `memory_extract`, `memory_skip`, `summary`) and keeps the last 300 per user. `GET /v1/users/{id}/trace` streams them (server-sent events); `GET /v1/users/{id}/memory/state` returns both memories and their rules. Groq returns the gpt-oss reasoning (`include_reasoning`) and token usage, also when streaming. Events contain the user's words: local use only.

**Token counts.** Every reply object has `usage`: `{"input", "output", "reasoning", "total", "calls": [{provider, input, output, reasoning, reported}]}`, one entry per LLM call in that turn (agent rounds, retries and fallbacks included; background summaries and the test consumer are not counted). The brain sets a `TurnUsage` (`companion/llm/usage.py`) for the turn, and the LLM router adds each finished call to it. Streams ask for `stream_options.include_usage`, so Gemini sends counts too. The playground shows `🪙 in · out · calls` under each reply (hover for each call). The **🪙 Tokens** bar above the input has Start/Stop/Reset: while it runs, it adds up replies, LLM calls, input, output and total tokens with a timer; Stop freezes the total. It counts text, push-to-talk and Live replies; a Live reply the user talked over before it finished has no count. The counter lives only in that browser tab.

**There is no separate planner yet.** The agent is a ReAct-style tool loop: understanding, breaking a request down and planning happen inside the model's reasoning, which the panel shows.

**Future scope (plan 2): a planner node.** Before `agent`, a `plan` step would split a complex request into steps, the agent would complete them with tools, and a final check would confirm every step was done. It costs one extra LLM call (about 1 s and a few thousand tokens), so it should run only for long or multi-part requests.

## 8e. Streaming and input modes

Every reply streams: `POST /v1/chat/stream` (text) and `POST /v1/voice/stream` (push-to-talk) send server-sent events: `delta` (new text as it is written), `expression` (as soon as the mood tag is read, before any words), `sentence` (a checked sentence), `audio` (push-to-talk only: that sentence's voice, base64 WAV), `done` (the whole reply object). Live voice streams over LiveKit. Text replies keep their line breaks and have no word limit; spoken replies stop at ~70 words. Pressing the mic while Umbrella talks stops its voice and the stream (only the spoken part is saved).

**Push-to-talk with a confirm step.** While you hold the mic on, the playground sends the audio so far to `POST /v1/stt` about every 1.5 s and shows your words in the input box (not true streaming STT: the text updates in steps). Stopping the mic only stops: the final transcript of the whole clip goes into the box, you can fix it, and **Send** sends the clip to `/v1/voice/stream` with `text=` (the words you confirmed). Then STT is skipped and the guard trusts the text, but the audio is still used for voice cues and the reply is still spoken.

`companion/modes.py` holds one constant per input mode. Voice cues are measured and sent to the LLM only where `voice_cues=True`:

| Mode | Button | voice_cues |
| --- | --- | --- |
| `TEXT` | typing | False |
| `PUSH_TO_TALK` | 🎤 | False |
| `PUSH_TO_TALK_CUES` | 🎙️✨ | True: Silero VAD finds the speech in the clip; the body sends the gap since Umbrella's voice ended and whether it cut in |
| `LIVE` | 📞 | True |

The "usual" pace, loudness and pitch are kept per user and shared by all voice modes (in memory, reset when the server restarts).

## 9. Expressions and the robot plugin

Shared list in `expressions.py`: body states `idle`, `listening`, `thinking`, `speaking`, `error`, and reply moods `happy`, `curious`, `concerned`, `playful`, `surprised`, `excited`, `sad` (the last three are new). The LLM starts every reply with a mood tag, for example `[curious] Achha, phir kya hua?`; the guard strips it and returns it as `expression` (unknown or missing = `idle`). Starting with a tag (instead of JSON) keeps the reply streamable, and in streams `expression` is sent as soon as the tag is read, before any words.

**UI plugins (plug and play).** `COMPANION_UI_PLUGINS=robot` (comma list) switches front-end plugins on. Each lives in `companion/server/plugins/<name>/` with a `plugin.js` exporting `mount(container, bus)`; the server serves only switched-on plugins at `/plugins/<name>/...` and lists them in `/health`. The playground loads each one into the stage above the chat and gives it a small event bus: `state` (idle / listening / thinking / speaking / error), `expression` (the reply mood) and `level` (0..1: your mic while recording, Umbrella's voice while it plays, computed from each WAV or from the LiveKit track in live mode). A new body (another avatar, a status light) is a new folder with its own `mount()`.

**The robot plugin** is a port of the Synq app's Buddy tab robot (`mobile/src/features/buddy/robot/`) to plain three.js 0.186.1 (same version as the app, loaded from jsDelivr through an import map): the same `gg2-head.glb`, region colours, face shaders, saccade gaze, blinks, head float/sway/nods and mouth that follows loudness. On top of the five states it draws the seven moods (each with its own lids, pupils, mouth, head tilt, screen tint and liveliness; playful winks, surprised and excited open the mouth, sad looks down). A mood shows for 8 s after the reply and stays while Buddy speaks; listening, thinking and error always win. Open the playground with `?demo=robot` to cycle through every state and mood without the backend. It needs WebGL; without it the stage shows a short message.

## 10. Server protocol (WebSocket `/v1/chat`)

```jsonc
// body → brain
{"type": "message", "user_id": "u1", "text": "aaj mausam kaisa hai?"}
{"type": "event", "user_id": "u1", "kind": "person_appeared"}

// brain → body
{"type": "thinking"}
{"type": "reply", "text": "Pune mein abhi 28°C hai...", "expression": "happy", "sources": ["..."]}
{"type": "reminder_due", "id": "r1", "text": "Dawai le lo"}
```

A background task checks due reminders every 30 s and pushes `reminder_due` to connected bodies. Phone notifications while the app is closed need `expo-notifications` in the mobile app (not approved yet).

## 11. Proactive behaviour (milestone 4)

Triggers: app opened / person seen, due reminders, working-memory follow-ups ("interview kaisa gaya?"), occasional call-insight thoughts. Limits: at most N proactive messages a day, quiet hours, a cooldown between them, never twice about the same thing.

## 12. Settings (.env)

| Setting | Example | Meaning |
| --- | --- | --- |
| `COMPANION_LLM`, `COMPANION_LLM_FALLBACK` | `groq`, `groq_backup` | Reply provider order |
| `COMPANION_BACKGROUND_LLM` | `groq_background,groq_backup,groq,ollama` | Memory extraction and summary order |
| `GROQ_API_KEY`, `COMPANION_GROQ_MODEL`, `COMPANION_GROQ_BACKUP_MODEL`, `COMPANION_GROQ_BACKGROUND_MODEL` | `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-20b` | Groq key and models (all support tool calling) |
| `COMPANION_OLLAMA_MODEL` | `qwen3:4b` | Laptop fallback |
| `COMPANION_DATABASE_URL`, `COMPANION_DB_SCHEMA` | `postgresql://...`, `companion` | Store and memory |
| `COMPANION_MEMORY` | `postgres` | Memory adapter |
| `TAVILY_API_KEY` | | Web search; without it the tool is off and Umbrella says it cannot check live info |
| `COMPANION_TIMEZONE` | `Asia/Kolkata` | Default user timezone |
| `COMPANION_NAME` | `Umbrella` | Persona name |

Keys stay on the backend. With Groq, the user's messages and the selected memories go to Groq; with Tavily, search queries go to Tavily. Ollama keeps everything on the machine.

## 13. Hallucination guards (summary)

1. Personal facts only from memory; live facts only from search.
2. The prompt always has today's date and time.
3. Tool arguments validated in code; actions confirmed aloud.
4. Saving only when asked; a reply that claims a save with no tool call is replaced.
5. The guard node removes invented life stories and unknown expression tags.
6. The labelled Hinglish test set (`tests/eval/`) gives a score per category after every change.

It reduces hallucination a lot; it cannot make it zero for general knowledge.

## 14. Milestones

| # | Milestone | Done when |
| --- | --- | --- |
| 1 ✅ | Text brain | Terminal chat works; tools for reminders/todos/notes/lists/memory/calculator + web search; memory layers 1–4 with Postgres; expression tags; tests pass |
| 2 | Server + mobile Buddy chat | WebSocket server; Buddy tab shows the chat and plays the expression on the robot face |
| 3 | Memory hardening | Extractor eval set ≥ 85% per category; memory list screen; expense/goal/habit tools |
| 4 | Proactive | Greetings, follow-ups, reminders delivered on time, limits |
| 5a ✅ | Voice, push-to-talk | Whisper/Piper (+ Sarvam fallback), guard, spoken replies, playground mic |
| 5b ✅ | Voice, real time | Pod 4 parity over LiveKit: VAD, turn taking, STT guard, voice cues, streaming TTS, barge-in, silence nudge, greeting, safety (see 8c) |
| 6 | Call insights + knowledge base | After consent wording is approved |
| 7 | Robot body | Pi + YOLO events drive the same brain |

## 15. Known limits (milestone 1)

- **Groq free tier:** 8,000 tokens per minute and 200,000 tokens per day per model, shared by the whole Groq organization (Pod 4 uses the same key). A turn costs about 3,000–4,000 tokens, so roughly 50–60 turns a day per model; quick back-to-back turns hit the per-minute limit. Fix: a separate Groq key/organization for the companion, or Groq's paid Dev tier.
- **Laptop Ollama** (`qwen3:4b` on CPU) takes about 50 s for one short reply, so it is only in the background order, not for replies.
- Search is word-based (full text + fuzzy). Synonyms in another language ("ghar" for "address") are found only through the keywords saved with an item; embeddings (pgvector) come later.
- When the answer is already in the recent chat, the model may answer from it without calling a tool.
- The backup model (qwen) is weaker at tool calls than gpt-oss-120b; the guard catches made-up "saved" replies.

## 16. Open decisions

1. A separate Groq key or a paid tier for the companion.
2. Saving call insights per user (needs new consent wording).
3. Knowledge base content.
4. Extra expression names (`happy`, `curious`, `concerned`, `playful` proposed).
5. A SQLite adapter for the Pi so it runs without Postgres (later).
