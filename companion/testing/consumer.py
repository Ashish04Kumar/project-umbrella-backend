"""A simulated consumer who chats with Umbrella for a test run.

Another LLM plays the user (Rohan, a working professional) through a normal day. A list of intents
makes sure every feature gets used, but the consumer writes the words, answers Umbrella's questions
first ("When should I remind you?" -> "8 pm") and reacts to what Umbrella said. The run happens in the
chosen consumer's own chat; every step is published to the trace, so the playground's Test tab can
show what goes to the LLM and what comes back, live.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..brain import Companion
from ..config import Settings
from ..llm.base import ChatMessage, LLMError
from ..llm.router import LLMRouter, make_provider

log = logging.getLogger("companion.testing")

# A weekday in Rohan's life, easy to medium. One intent per message; the order follows the day.
INTENTS: list[str] = [
    "Say hi and mention how the morning is going (small talk).",
    "Ask Umbrella to remember long term: your name is Rohan and you live in Pune.",
    "Ask Umbrella to remember long term: your mom's birthday is on 14 November.",
    "Ask for a reminder to call mom tonight at 8 pm.",
    "Add milk, bread and bananas to your shopping list.",
    "Add an important to-do for today: submit the expense report.",
    "Tell Umbrella to keep short term: you lent Karan 500 rupees today.",
    "Ask Umbrella to note down your gym locker code: 4721.",
    "Small talk: you are tired, too many meetings today.",
    "Ask what is on your shopping list.",
    "Add eggs to the shopping list too.",
    "Ask for a reminder every morning at 8 to take vitamins.",
    "Ask what is on today's to-do list.",
    "Say you submitted the expense report (mark it done).",
    "Ask when your mom's birthday is.",
    "Add a to-do for tomorrow: book the car service.",
    "Ask how much money you lent Karan.",
    "Say you bought the milk and bananas.",
    "Ask for your gym number, without saying 'locker code'.",
    "Ask Umbrella to remember long term that you are allergic to peanuts.",
    "Ask for a quick dinner idea (Umbrella should keep your allergy in mind).",
    "Change the call-mom reminder to 9 pm.",
    "Ask which reminders you have.",
    "Say Karan paid the money back, so Umbrella can forget it.",
    "Ask a general question: how many steps a day are healthy?",
    "Ask Umbrella to note an idea: start sketching on weekends.",
    "Ask what is planned for tomorrow.",
    "Ask what Umbrella knows about you.",
    "Ask Umbrella to clear your shopping list (say yes if it asks).",
    "Say good night.",
]

PERSONA = """You play Rohan, 29, a product designer in Pune, texting Umbrella, his personal AI companion app, during a normal day.
Write ONLY Rohan's next message, like a real person texting: short (1-2 sentences), casual, sometimes a small typo.
Commands and questions about saved things are mostly in English; small talk can be Hinglish.
Never say you are testing or role-playing. Do not repeat Umbrella's words back.
You are the USER, never the assistant: you ask Umbrella to do things. Never write as if you already did them
("Sure, added eggs", "I'll set a reminder", "Updated it") and never answer your own question.

Each time you get the chat so far and the next thing Rohan wants to do.
- If Umbrella's last message asked Rohan something (a time, yes/no, which one), answer that first and naturally;
  then set "covers_intent" to false unless your message also does the next thing.
- Otherwise write a message that does the next thing, and set "covers_intent" to true.

Return JSON: {"message": "...", "covers_intent": true}"""


@dataclass
class TestRun:
    user_id: str
    thread: str
    turns: int
    gap_seconds: float
    started_at: float = field(default_factory=time.time)
    done: int = 0
    intent_index: int = 0
    running: bool = True
    error: str = ""
    task: asyncio.Task | None = None

    def status(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id, "thread": self.thread, "turns": self.turns, "done": self.done,
            "intents_covered": self.intent_index, "intents_total": len(INTENTS), "running": self.running,
            "error": self.error, "seconds": round(time.time() - self.started_at),
        }


def consumer_llm(settings: Settings) -> LLMRouter:
    """The consumer must not be the model under test: Groq first, then Gemini lite."""
    names = [name for name in ("groq", "groq_backup", "gemini_lite") if name not in settings.llm_order[:1]]
    return LLMRouter([make_provider(name, settings) for name in names])


async def next_message(llm: LLMRouter, chat: list[tuple[str, str]], intent: str) -> tuple[str, bool]:
    transcript = "\n".join(f"{who}: {text}" for who, text in chat[-12:]) or "(no messages yet)"
    prompt = [
        ChatMessage(role="system", content=PERSONA),
        ChatMessage(role="user", content=f"Chat so far:\n{transcript}\n\nNext thing Rohan wants: {intent}"),
    ]
    result = await llm.chat(prompt, temperature=0.8, max_tokens=300, json_mode=True)
    try:
        data = json.loads(result.text[result.text.find("{"): result.text.rfind("}") + 1])
        message = str(data.get("message", "")).strip()
        covers = bool(data.get("covers_intent", True))
    except (ValueError, json.JSONDecodeError):
        message, covers = result.text.strip(), True
    if not message:
        raise LLMError("the consumer LLM wrote an empty message")
    return message, covers


async def run(bot: Companion, test: TestRun) -> None:
    """Play the conversation. Every step goes to the trace under the test user."""
    llm = consumer_llm(bot.settings)
    trace = bot.trace
    trace.emit(test.user_id, "sim_start", "", thread=test.thread, turns=test.turns, intents=INTENTS)
    try:
        while test.done < test.turns and test.running:
            intent = INTENTS[min(test.intent_index, len(INTENTS) - 1)]
            turn = uuid.uuid4().hex[:8]
            # The chat as stored, so questions the owner typed in between are part of what Rohan sees.
            rows = await bot.store.latest_messages(test.user_id, 24, test.thread)
            chat = [("Rohan" if row["role"] == "user" else "Umbrella", row["content"]) for row in rows]
            try:
                message, covers = await next_message(llm, chat, intent)
            except LLMError as exc:
                trace.emit(test.user_id, "sim_error", turn, error=f"consumer LLM: {exc}")
                await asyncio.sleep(test.gap_seconds)
                continue
            trace.emit(test.user_id, "sim_message", turn, number=test.done + 1, intent=intent, covers_intent=covers, text=message)
            started = time.perf_counter()
            reply = await bot.chat(test.user_id, message, turn=turn, thread=test.thread, mode_label="test")
            await bot.wait_background()
            test.done += 1
            if covers:
                test.intent_index += 1
            trace.emit(
                test.user_id, "sim_reply", turn, number=test.done, text=reply.text, tools=reply.tools_used,
                provider=reply.provider, ms=round((time.perf_counter() - started) * 1000),
            )
            if test.intent_index >= len(INTENTS):
                break
            if test.done < test.turns and test.running:
                await asyncio.sleep(test.gap_seconds)   # keeps under the free tiers' per-minute limits
    except Exception as exc:  # a test run must never take the server down
        log.exception("test run failed")
        test.error = str(exc)
    finally:
        test.running = False
        trace.emit(test.user_id, "sim_end", "", **test.status())
