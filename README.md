# Companion Umbrella

A backend-only Python package (`companion`) for an AI companion's brain: a LangGraph agent loop, persona, plug-and-play LLM providers, tools, memory, and later voice and perception events. Use it embedded in-process (`pip install`, no network hop) or, from milestone 2, as a server any frontend connects to. Frontends are not part of this project.

The first body is the Buddy tab of the Project Synq app. Later bodies include a laptop UI and a Raspberry Pi pan-tilt robot.

Design: [docs/architecture.md](docs/architecture.md). This folder is its own project. It lives inside the Synq checkout for now and can be moved out and given its own git repository.

## Run it (milestone 1: text brain)

```bash
cd companion-umbrella
python3 -m venv .venv
.venv/bin/pip install -e ".[server,voice,realtime]"
# Piper voice: copy en_US-amy-medium.onnx (+ .json) into models/ (gitignored)
cp .env.example .env          # then fill in GROQ_API_KEY and TAVILY_API_KEY
.venv/bin/python -m companion.cli --debug
```

Browser playground (chat on the left; memory, reminders, lists and to-dos on the right; due reminders pop up; 🎤 to talk, 🔊 to hear any reply, a voice speed slider):

```bash
.venv/bin/python -m companion.server      # then open http://127.0.0.1:8300
```

It listens on 127.0.0.1 only and has no login, so never expose it to the internet. The same server has the HTTP API the mobile Buddy will use (`POST /v1/chat`, `GET /v1/users/{id}/memory`, `.../reminders`, `.../reminders/due`, `.../lists`, `.../history`).

The same playground also exists as a separate Next.js app, `umbrella-d` (`/home/bs/projects/umbrella-d`, `npm run dev`, http://localhost:3200), which calls this API from the browser. `COMPANION_CORS_ORIGINS` in `.env` lists the addresses allowed to do that (default: that dashboard on port 3200).

In the terminal chat: `/memory` shows what Buddy remembers, `/reminders` the upcoming reminders, `/quit` exits. Due reminders show up in the terminal (checked every 20 s). `--user NAME` keeps a separate memory per user.

Embedded use:

```python
from companion import Companion

bot = await Companion.create()
reply = await bot.chat("ashish", "kal 9 baje dawai yaad dilana")
print(reply.text, reply.expression, reply.tools_used)
await bot.close()
```

Live voice (real time): in the playground press **📞 Live**, allow the mic, and talk normally; interrupt Buddy any time. Use earphones, otherwise Buddy hears itself and stops (barge-in). It needs `COMPANION_LIVEKIT_URL`, `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in `.env`. A headless check without a mic:

```bash
.venv/bin/python scripts/live_e2e.py
```

Agent trace: press **🧠 Trace** (next to Live) to see each turn's raw request and response objects live, short-term and long-term memory side by side, and the memory tables as stored in Postgres. Replies stream in every mode; 🎙️✨ is push-to-talk with voice cues (see `companion/modes.py`).

Robot: the 3D Buddy head sits above the chat and reacts (listening while you record, thinking while it waits, speaking with the voice, and the reply's mood). Open `http://127.0.0.1:8300/?demo=robot` to see all states and moods. Plugins are switched on with `COMPANION_UI_PLUGINS` in `.env`.

## Tests

```bash
.venv/bin/python -m unittest discover tests -t .
```

The Postgres tests use a throwaway schema `companion_test` and a scripted fake LLM (no network). They are skipped when `COMPANION_DATABASE_URL` is not set.

## Database

The tables live in their own schema (`COMPANION_DB_SCHEMA`, default `companion`). Migrations in `migrations/` run on startup. For now the schema sits inside the local `synq` database, because the database user cannot create new databases.
