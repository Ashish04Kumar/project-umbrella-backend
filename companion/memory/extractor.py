"""After each reply, read the turn and decide what (if anything) is worth remembering.

The LLM fills a fixed JSON form; code validates every item and then decides with `decide`:
save it, ask the user first (kept as "pending"), or ignore it.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..llm.base import ChatMessage
from ..persona import date_guide
from .interface import MemoryItem, MemoryStore, normalize_key

log = logging.getLogger("companion.extractor")

Decision = Literal["save", "ask", "ignore"]


class ExtractedItem(BaseModel):
    type: Literal["fact", "preference", "person", "routine", "event", "none"]
    key: str = ""
    text: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    explicit: bool = False        # the user said "remember this" / "note kar lo"
    hypothetical: bool = False    # "agar...", "kaash...", jokes, sarcasm
    sensitive: bool = False       # health, money trouble, relationship trouble, sexuality, religion
    until: date | None = None     # events: the last day it matters

    @field_validator("until", mode="before")
    @classmethod
    def _empty(cls, value: Any) -> Any:
        return value or None


def decide(item: ExtractedItem) -> Decision:
    """The save / ask / ignore rule from docs/architecture.md (section 7)."""
    if item.type == "none" or item.hypothetical or len(item.key) < 3 or len(item.text) < 3:
        return "ignore"
    if item.sensitive and not item.explicit:
        return "ignore"
    if item.explicit or item.confidence >= 0.8:
        return "save"
    if item.confidence >= 0.5:
        return "ask"
    return "ignore"


_INSTRUCTIONS = """You extract memories for a personal companion app. Read the latest user message (and the assistant reply for context) and list what is worth remembering about the USER long term or for the next few days.

Return JSON: {"items": [ ... ]} where each item is
{"type": "fact|preference|person|routine|event|none", "key": "short:slug", "text": "one plain English sentence",
 "confidence": 0.0-1.0, "explicit": bool, "hypothetical": bool, "sensitive": bool, "until": "YYYY-MM-DD" or null}

Types:
- fact: stable facts about the user (job, city, age, birthday, studies). key like fact:job, fact:city
- preference: likes or dislikes. key like food:paneer, music:arijit, movies:horror
- person: someone in the user's life. ONE item per person, key person:<first name> (e.g. person:riya), text with relation and details ("Sister Riya, birthday 3 May"). Facts about other people go here, never as the user's own preference.
- The user's own name: type fact, key fact:name.
- routine: what the user usually does. key like routine:gym, routine:wake_up
- event: something upcoming or a current temporary state (exam on Friday, unwell today). Set "until" to the last day it matters. key like event:interview
Rule of thumb: still true in a month? -> fact/preference/person/routine. Otherwise -> event.

Rules:
- Only what the user actually said about themselves or their people. Nothing from the assistant's words alone. Never invent details.
- Greetings, questions to the assistant, small talk, opinions about the world -> return {"items": []}.
- Requests for the assistant to do something (remind me, add to my list, note this down) are actions, not memories: skip them.
- "agar/kaash/if only", wishes, jokes and sarcasm -> hypothetical: true.
- explicit: true only if the user asked to remember/note it.
- sensitive: health conditions, money trouble, relationship trouble, sexuality, religion, politics.
- confidence: how sure you are this is a real, lasting truth the user would want remembered.
- If the same thing is already in "Known memories", reuse its key so it gets updated. If it changed ("ab chai chhod di"), reuse the key with the new text.
- Skip anything already saved by the tool actions listed.
- Write "text" in simple English, e.g. "Likes paneer a lot", "Sister Riya, birthday 3 May"."""


def _prompt(user_text: str, reply: str, tool_actions: list[str], known: list[MemoryItem], today: date) -> list[ChatMessage]:
    known_lines = "\n".join(f"- {item.key}: {item.text}" for item in known[:60]) or "- (none)"
    actions = "\n".join(f"- {line}" for line in tool_actions) or "- (none)"
    content = (
        f"Dates (use these, never count yourself):\n{date_guide(datetime.combine(today, time()))}\n\nKnown memories:\n{known_lines}\n\n"
        f"Tool actions this turn:\n{actions}\n\nUser: {user_text}\nAssistant: {reply}"
    )
    return [ChatMessage(role="system", content=_INSTRUCTIONS), ChatMessage(role="user", content=content)]


def parse_items(raw: str) -> list[ExtractedItem]:
    """Lenient parse: a broken item is skipped, not the whole answer."""
    try:
        payload = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return []
    items: list[ExtractedItem] = []
    for entry in payload.get("items", []) if isinstance(payload, dict) else []:
        try:
            items.append(ExtractedItem.model_validate(entry))
        except ValidationError:
            continue
    return items


async def extract(llm: Any, user_text: str, reply: str, tool_actions: list[str], known: list[MemoryItem], today: date) -> list[ExtractedItem]:
    result = await llm.chat(_prompt(user_text, reply, tool_actions, known, today), temperature=0.1, max_tokens=500, json_mode=True)
    return parse_items(result.text)


def explain(item: ExtractedItem, decision: Decision) -> str:
    """Why `decide` chose this, in plain words (for the Trace panel)."""
    if item.type == "none":
        return "nothing worth remembering"
    if item.hypothetical:
        return "hypothetical, a wish or a joke"
    if item.sensitive and not item.explicit:
        return "sensitive and not asked to remember"
    if item.explicit:
        return "the user asked to remember it"
    if decision == "save":
        return f"confident ({item.confidence:.2f} >= 0.8)"
    if decision == "ask":
        return f"unsure ({item.confidence:.2f}): ask the user once"
    return f"low confidence ({item.confidence:.2f} < 0.5)"


async def apply(memory: MemoryStore, user_id: str, items: list[ExtractedItem], tz: ZoneInfo, now: datetime) -> list[tuple[Decision, ExtractedItem, str]]:
    """Save, mark as pending (ask first) or ignore each item. Returns every item with its decision and the reason."""
    done: list[tuple[Decision, ExtractedItem, str]] = []
    # Never ask twice: skip an unsure item whose key is already known or already pending/asked.
    seen = {item.key for item in await memory.list_all(user_id)} | {item.key for item in await memory.list_all(user_id, status="pending")}
    for item in items:
        decision = decide(item)
        if decision == "ask" and normalize_key(item.key) in seen:
            done.append(("ignore", item, "already known or already asked"))
            continue
        if decision == "ignore":
            done.append(("ignore", item, explain(item, decision)))
            continue
        expires = None
        if item.type == "event":
            last_day = item.until if item.until and item.until >= now.date() else now.date() + timedelta(days=3)
            expires = datetime.combine(min(last_day, now.date() + timedelta(days=30)) + timedelta(days=1), time(), tzinfo=tz)
        if decision == "ask":
            expires = expires or now + timedelta(days=3)
        category = {"fact": "profile", "routine": "other"}.get(item.type, item.type)
        await memory.remember(
            user_id,
            MemoryItem(
                scope="short" if item.type == "event" else "long", category=category, key=item.key, text=item.text,
                confidence=item.confidence, sensitive=item.sensitive,
                status="active" if decision == "save" else "pending", expires_at=expires,
            ),
        )
        done.append((decision, item, explain(item, decision)))
    return done


_SUMMARY = """Summarize this chat between a user and their companion for the companion's own memory.
Keep: what the user talked about, how they felt, plans and promises, anything to follow up on. Max 120 words, plain English, no lists.
Start from the existing summary if there is one and merge the new part into it."""


async def summarize(llm: Any, existing: str, messages: list[ChatMessage]) -> str:
    chat = "\n".join(f"{m.role}: {m.content}" for m in messages)
    result = await llm.chat(
        [ChatMessage(role="system", content=_SUMMARY), ChatMessage(role="user", content=f"Existing summary:\n{existing or '(none)'}\n\nNew part:\n{chat}")],
        temperature=0.2, max_tokens=300,
    )
    return result.text.strip()
