"""Terminal chat for testing the brain without any frontend.

    .venv/bin/python -m companion.cli            # user id "me"
    .venv/bin/python -m companion.cli --user ashish --debug

Commands: /memory (what Umbrella remembers), /reminders, /quit
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from zoneinfo import ZoneInfo

from .brain import Companion


async def _watch_reminders(bot: Companion, user_id: str) -> None:
    while True:
        await asyncio.sleep(20)
        try:
            for reminder in await bot.due_reminders(user_id):
                print(f"\n⏰ Reminder: {reminder['text']}  ({reminder['due']})\nyou> ", end="", flush=True)
        except Exception as exc:  # keep chatting even if the database hiccups
            print(f"\n(reminder check failed: {exc})")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the companion in the terminal")
    parser.add_argument("--user", default="me", help="user id (memories are per user)")
    parser.add_argument("--debug", action="store_true", help="show tools, memory decisions and the provider")
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO if options.debug else logging.WARNING, format="  · %(name)s: %(message)s")

    bot = await Companion.create()
    print(f"{bot.settings.name} is here. Tools: {', '.join(bot.tools.names())}\nType /memory, /reminders or /quit.\n")
    print("you> ", end="", flush=True)
    watcher = asyncio.create_task(_watch_reminders(bot, options.user))
    try:
        while True:
            line = (await asyncio.to_thread(sys.stdin.readline))
            if not line:
                break
            text = line.strip()
            if text in ("/quit", "/exit"):
                break
            if text == "/memory":
                for item in await bot.memory.list_all(options.user):
                    print(f"  [{item.scope}/{item.category}] {item.subject} {item.topic}: {item.text}" + (f"  (until {item.expires_at:%d %b})" if item.expires_at else ""))
                print("you> ", end="", flush=True)
                continue
            if text == "/reminders":
                for row in (await bot.store.reminder_list(options.user))["reminders"]:
                    print(f"  {row['text']} at {row['due_at'].astimezone(ZoneInfo(bot.settings.timezone)):%d %b %I:%M %p} ({row['repeat']})")
                print("you> ", end="", flush=True)
                continue
            if not text:
                print("you> ", end="", flush=True)
                continue
            reply = await bot.chat(options.user, text)
            extra = ""
            if options.debug:
                extra = f"   ⟨{reply.provider}; tools: {', '.join(reply.tools_used) or '-'}⟩"
            print(f"{bot.settings.name} [{reply.expression}]> {reply.text}{extra}")
            for url in reply.sources[:3]:
                print(f"    source: {url}")
            print("you> ", end="", flush=True)
    finally:
        watcher.cancel()
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
