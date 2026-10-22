"""The tools (docs/personal-data-spec.md and docs/search-and-retrieval-spec.md).

Each tool = a Pydantic model for its fields (what the LLM must send) + an async function.
Write tools return either "say" (a ready, exact confirmation: no second LLM call is needed) or
"ask" (one short question when something is missing, unclear or would delete a lot: human in the loop).
Groups decide which tools the router offers for a message (companion/router.py).
"""

from __future__ import annotations

import ast
import operator
import re
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import Settings
from ..memory.interface import MemoryItem, key_for
from ..search import clean_words, pick
from ..store.postgres import short
from .registry import Tool, ToolContext, ToolError, ToolRegistry
from .web_search import tavily_search

STM_DAYS, STM_MAX_DAYS = 10, 15


# --- helpers ---------------------------------------------------------------------------------------

def when_text(moment: datetime, tz: tzinfo) -> str:
    """'Tue 20 Oct, 9:00 AM' in the user's timezone."""
    local = moment.astimezone(tz)
    return local.strftime("%a %d %b, %I:%M %p").replace(" 0", " ")


def day_text(day: date) -> str:
    return day.strftime("%a %d %b").replace(" 0", " ")


def _empty_to_none(value: Any) -> Any:
    return None if value in ("", [], None) else value


def _check_year(day: date, ctx: ToolContext) -> None:
    """Some models (Gemini, live test 19 Oct) write last year's date ("Friday" -> 2025-10-31) although the prompt
    lists this week's dates. Do not save it: tell the model the right dates so it calls the tool again."""
    if day.year < ctx.now.year:
        from ..persona import date_guide
        raise ToolError(
            f"{day.isoformat()} is in a past year. Today is {ctx.now:%a %d %b %Y}. Use these dates and call the tool again:\n"
            + date_guide(ctx.now)
        )


def _check_dates(ctx: ToolContext, *days: date | None) -> None:
    """Questions too: "what's planned tomorrow?" read 2025-10-21 and answered "nothing" (live test, 20 Oct)."""
    for day in days:
        if day:
            _check_year(day, ctx)


def _confirmed(ctx: ToolContext, confirm: bool) -> bool:
    """A bulk delete runs only when the user answered our own "(yes/no)" question. A model sometimes sends
    confirm=true on the first call (it cleared a shopping list without asking: live test, 20 Oct)."""
    return confirm and "(yes/no)" in (ctx.last_assistant or "")


# "today", "tomorrow", "day after tomorrow" in the user's own words (English, Roman and Devanagari Hindi).
# "tomorrow" as people type it: tomorrow, tommorow, tomorow, tommorrow, tmrw, tmr.
TOMORROW_WORD = r"\bto?m{1,2}o?r{1,2}o?w\b|\btmrw?\b"
_DAY_AFTER = re.compile(r"\bday after (?:" + TOMORROW_WORD.replace("\\b", "") + r")\b|\bparso[n]?\b|परसों", re.IGNORECASE)
_TOMORROW = re.compile(TOMORROW_WORD + r"|\bkal\b|कल", re.IGNORECASE)
_TODAY = re.compile(r"\btoday\b|\btonight\b|\baaj\b|आज|\bthis (morning|afternoon|evening)\b", re.IGNORECASE)
# A date or weekday named outright: then the relative words are not checked.
_NAMED_DAY = re.compile(
    r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*day\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\b\d{1,2}(st|nd|rd|th)?\b\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|\b\d{1,2}[/-]\d{1,2}\b"
    r"|somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar",
    re.IGNORECASE,
)
DAY_QUESTION = "Which day do you mean"
# Full day names only ("sunscreen" is not Sunday).
_WEEKDAY = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar)\b",
                      re.IGNORECASE)
_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
             "som": 0, "man": 1, "bud": 2, "gur": 3, "shu": 4, "sha": 5, "rav": 6}
# Words for a time of day, a day, and "the whole day" in the user's own message. A reminder asks for
# whatever is missing (owner's rule, 20 Oct): no day -> which day; no time -> what time (or full day).
_TIME_WORDS = re.compile(
    r"\b\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.|baje|bje|o'?clock)\b|\b\d{1,2}:\d{2}\b|\b(at|by|around) \d{1,2}\b"
    r"|\b(noon|morning|afternoon|evening|night|tonight|midnight)\b|\bin \d+ (minutes?|mins?|hours?)\b"
    r"|baje|bje|subah|sham|shaam|raat|dopahar|सुबह|शाम|रात|बजे|दोपहर", re.IGNORECASE)
_DAY_WORDS = re.compile(
    r"\btoday\b|\btonight\b|" + TOMORROW_WORD + r"|\bday after\b|\bnext (week|month)\b|\bthis (week|weekend|evening|morning)\b|\bweekend\b"
    r"|\b(mon|tues?|wed(nes)?|thu(rs)?|fri|sat(ur)?|sun)(day)?\b|\b\d{1,2}(st|nd|rd|th)\b|\b\d{1,2}[/-]\d{1,2}\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b|\bevery\b|\bdaily\b|\bweekly\b|\bmonthly\b"
    r"|\baaj\b|\bkal\b|\bparso[n]?\b|\broz\b|har (din|hafte|mahine)|somvar|mangalvar|budhvar|guruvar|shukravar|shanivar|ravivar"
    r"|आज|कल|परसों", re.IGNORECASE)
_ALL_DAY = re.compile(r"\b(full|all|whole)[\s-]?day\b|\bpura din\b|\bpoora din\b|\bsaara din\b|पूरा दिन", re.IGNORECASE)
ALL_DAY_HOUR = 9   # a full-day reminder is said out loud at 9:00 that morning


def _answers_question(ctx: ToolContext) -> bool:
    """This message answers Buddy's question, so the message before it still counts ("my exam is on Friday"
    → "Save it as a reminder?" → "add in reminder": the day is Friday)."""
    from ..router import is_tool_question

    last = (ctx.last_assistant or "").lower()
    return "?" in last and (is_tool_question(last) or bool(re.search(r"remind|reminder|\bsave\b|memory|\bnote\b|to-?do", last)))


def _same_question(ask: str, ctx: ToolContext) -> bool:
    """Buddy asked this very question last time: asking again would loop (it did on a misspelt "tommorow")."""
    return ask.split("?")[0].strip().lower() == (ctx.last_assistant or "").split("?")[0].strip().lower()


def _check_relative_day(day: date, ctx: ToolContext, what: str) -> dict[str, Any] | None:
    """The user said "tomorrow" (or "aaj", "parson") but the model picked another day: send it back with the
    right date (Gemini set "tomorrow" two days ahead at 2:23 AM, 20 Oct). Between midnight and 5 AM "tomorrow"
    can mean later today or the next day, so ask once. Returns an "ask" answer, or None when the day is fine."""
    from ..router import is_tool_question

    text, today = ctx.user_text or "", ctx.now.date()
    if _answers_question(ctx) and not _DAY_WORDS.search(text):
        # "6 pm" answers "Which day…?" for the message before, so its day counts. A message that names its own
        # day ("…to-dos for tomorrow") stands alone: an earlier "Friday" refused it three times (English test, 20 Oct).
        text = f"{ctx.previous_user_text or ''} {text}"
    weekday = _WEEKDAY.search(text)
    if weekday and not re.search(r"\b(next|last|pichhle|agle|agla)\b", text, re.IGNORECASE):
        # "Wednesday" must be a Wednesday: a model once took it as Fri 23 Oct (20 Oct).
        wanted = _WEEKDAYS[weekday.group(0).lower()[:3]]
        ahead = (wanted - today.weekday()) % 7
        allowed = {today + timedelta(days=ahead)} | ({today + timedelta(days=7)} if ahead == 0 else set())
        if day not in allowed:
            right = min(allowed)
            raise ToolError(f"The user said {weekday.group(0)}, which is {day_text(right)} ({right.isoformat()}), not {day.isoformat()}. "
                            "Call the tool again with that date.")
        return None
    if not text.strip() or _NAMED_DAY.search(text) or DAY_QUESTION.lower() in (ctx.last_assistant or "").lower():
        return None
    if _DAY_AFTER.search(text):
        wanted = {today + timedelta(days=2)}
    elif _TOMORROW.search(text):
        if ctx.now.hour < 5:
            later, next_day = today, today + timedelta(days=1)
            return {"ask": f"{DAY_QUESTION} for '{what}': {day_text(later)} (later today, after the night) or {day_text(next_day)}?"}
        wanted = {today + timedelta(days=1)}
    elif _TODAY.search(text):
        wanted = {today}
    else:
        return None
    if day in wanted:
        return None
    right = next(iter(wanted))
    raise ToolError(f"The user said a relative day that means {day_text(right)} ({right.isoformat()}), not {day.isoformat()}. "
                    "Call the tool again with that date.")


def _check_night_hour(due: datetime, ctx: ToolContext, what: str) -> dict[str, Any] | None:
    """1-5 o'clock at night is rarely meant ("subah do baje" came out as 2:00 AM): ask AM or PM once,
    unless the user said AM / night themselves."""
    if not 1 <= due.hour <= 5 or re.search(r"\b(a\.?m\.?|raat|night|midnight|tonight)\b|रात", ctx.user_text or "", re.IGNORECASE):
        return None
    if "AM (night) or" in (ctx.last_assistant or ""):
        return None
    pm = due.replace(hour=due.hour + 12)
    return {"ask": f"For '{what}', do you mean {due.strftime('%-I:%M')} AM (night) or {pm.strftime('%-I:%M')} PM?"}


def _parse_moment(text: str, ctx: ToolContext) -> datetime:
    try:
        local = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ToolError("`when` must look like 2026-10-20T09:00.") from exc
    _check_year(local.date(), ctx)
    return local.replace(tzinfo=ctx.tz) if local.tzinfo is None else local.astimezone(ctx.tz)


def _start(day: date, ctx: ToolContext) -> datetime:
    return datetime.combine(day, time(), tzinfo=ctx.tz)


def _end(day: date, ctx: ToolContext) -> datetime:
    return datetime.combine(day, time(23, 59, 59), tzinfo=ctx.tz)


def _choices(candidates: list[dict[str, Any]], label: str = "text") -> str:
    return " ".join(f"{index}) {item[label]}" for index, item in enumerate(candidates, start=1))


class Fields(BaseModel):
    """Base for tool fields: trims text, turns "" into None for optional fields."""

    @field_validator("*", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value


class NoFields(BaseModel):
    pass


Entities = Field(default_factory=list, description="People/places/projects named in it, e.g. [\"Riya\"]")


# --- where to save: the user's words, else the model's confidence, else ask ------------------------
# 1. The user named the place ("save it in memory", "note down", "remind me", "add to my to-do") -> save there.
# 2. They did not -> save only if the model is at least SAVE_CONFIDENCE sure about the place.
# 3. Otherwise -> ask "Where should I save ...?" and save where they say.
SAVE_CONFIDENCE = 0.8
WHERE_QUESTION = "Where should I save"
PLACES = "reminder / to-do / short-term memory / long-term memory / note / list"
_NAMED_PLACES = {
    "short": r"short[\s-]?term",
    "long": r"long[\s-]?term",
    "memory": r"\bmemory\b|\bremember\b|yaad rakh|yaad kar",
    "note": r"\bnotes?\b|note (?:kar|karo|down)|likh (?:lo|do|le)|write (?:it |this |that )?down|\bjot\b",
    "reminder": r"remind|reminder|yaad dila|\balarm\b",
    "todo": r"to[\s-]?dos?\b|\btasks?\b",
    "list": r"\blists?\b|shopping|watchlist",
}
_DASH_CHARS = str.maketrans({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


class Placement(BaseModel):
    confidence: float | None = Field(
        default=None, ge=0, le=1,
        description="How sure you are (0-1) that this is the place the user wants it saved. If they named the place, 1. "
        f"Under {SAVE_CONFIDENCE} the user is asked where to save it, so be honest when it could fit elsewhere.",
    )


def named_places(text: str) -> set[str]:
    lowered = (text or "").translate(_DASH_CHARS).lower()
    found = {place for place, pattern in _NAMED_PLACES.items() if re.search(pattern, lowered)}
    if "memory" in found and not found & {"short", "long"}:
        found |= {"short", "long"}   # "memory" / "remember": short or long term, the model picks by meaning
    if "list" in found and not re.search(_NAMED_PLACES["list"], re.sub(r"to[\s-]?dos? list", "", lowered)):
        found.discard("list")        # only "to-do list" was said: that is the to-do place, not a named list
    return found


def _where(ctx: ToolContext, place: str, summary: str, confidence: float | None) -> dict[str, Any] | None:
    """None to save now, or an "ask" answer: Where should I save it?"""
    from ..router import is_tool_question

    last = (ctx.last_assistant or "").lower()
    if is_tool_question(last) and WHERE_QUESTION.lower() not in last:
        return None   # answering our own question ("When should I remind you?" -> "9 baje"): the place is settled
    named = named_places(ctx.user_text)
    if named and place in named:
        return None
    if named and place not in named and WHERE_QUESTION.lower() not in (ctx.last_assistant or "").lower():
        tools = {"short": "stm_add (short-term memory)", "long": "ltm_save (long-term memory)", "note": "note_save",
                 "reminder": "reminder_add", "todo": "todo_add", "list": "list_add"}
        names = " or ".join(tools[p] for p in sorted(named - {"memory"}))
        raise ToolError(f"The user said where to save it: use {names}, not this tool.")
    if (confidence or 0) >= SAVE_CONFIDENCE:
        return None
    return {"ask": f"{WHERE_QUESTION} “{summary}”? ({PLACES})"}


# --- ask: one search over short-term, long-term and notes ------------------------------------------

class FindSaved(Fields):
    query: str | None = Field(default=None, description="Key words of the question, e.g. 'wifi password'")
    subject: str | None = Field(default=None, description="Who/what it is about, e.g. 'Riya', 'me'")
    category: Literal["profile", "person", "date", "preference", "project", "other"] | None = None
    entities: list[str] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    month: int | None = Field(default=None, ge=1, le=12, description="For birthdays/anniversaries in a month")
    in_: list[Literal["short", "long", "notes"]] = Field(default_factory=list, alias="in", description="Only these places; empty = all")
    limit: int = Field(default=8, ge=1, le=20)

    _none = field_validator("query", "subject", "category", "date_from", "date_to", "month", mode="before")(lambda cls, v: _empty_to_none(v))


def _coverage(words: list[str], *fields: Any) -> float:
    """Share of the question's words found in a hit (prefix match, so 'pasword' does not count but 'books' finds 'book').
    Scores are ranked inside each store, so this is what decides between a memory and a note."""
    if not words:
        return 1.0
    haystack = " ".join(str(f) for f in fields if f).lower()
    return sum(1 for word in words if word[:4] in haystack) / len(words)


async def find_saved(ctx: ToolContext, args: FindSaved) -> dict[str, Any]:
    _check_dates(ctx, args.date_from, args.date_to)
    places = set(args.in_ or ["short", "long", "notes"])
    words = clean_words(args.query)
    results: list[dict[str, Any]] = []
    for scope in ("short", "long"):
        if scope not in places:
            continue
        items = await ctx.memory.search(
            ctx.user_id, query=args.query, subject=args.subject, category=args.category, entities=args.entities,
            date_from=args.date_from, date_to=args.date_to, month=args.month, scope=scope, limit=args.limit,
        )
        for item in items:
            results.append({
                "from": f"{scope} term", "id": short(item.id or ""), "subject": item.subject, "topic": item.topic,
                "text": item.text, "date": str(item.event_date or "") if not item.yearly else (item.event_date.strftime("%d %b") if item.event_date else ""),
                "score": round((item.score or 0) * (0.5 + 0.5 * _coverage(words, item.text, item.subject, item.topic, *item.keywords, *item.entities)), 4),
            })
    if "notes" in places and not args.month and args.category not in ("date", "profile") and not (args.subject and args.subject.lower() == "me"):
        notes = await ctx.store.note_search(
            ctx.user_id, query=args.query or args.subject, entities=args.entities or ([args.subject] if args.subject else None),
            date_from=args.date_from, date_to=args.date_to, limit=args.limit,
        )
        for note in notes:
            results.append({
                "from": "note", "id": short(note["id"]), "title": note["title"], "text": note["text"][:300],
                "saved": day_text(note["created_at"].astimezone(ctx.tz).date()),
                "score": round((note.get("score") or 0) * (0.5 + 0.5 * _coverage(words, note["text"], note["title"], *(note.get("keywords") or []), *(note.get("entities") or []))), 4),
            })
    results.sort(key=lambda row: row.get("score") or 0, reverse=True)
    answer: dict[str, Any] = {"results": results[: args.limit], "count": len(results)}
    if not results and "notes" in places and args.query:
        # Last fallback: the newest note titles, so the model can match by meaning.
        # Last fallback: the newest notes themselves, so the model can match by meaning ("internet code" -> "Wifi password").
        notes = await ctx.store.note_titles(ctx.user_id, limit=10)
        if notes:
            answer["no_word_match_but_these_notes_exist"] = [{"title": n["title"], "text": n["preview"]} for n in notes]
            answer["hint"] = "Nothing matched the words. If one of these notes answers the question by meaning, answer from it."
    if not results:
        # Lists, to-dos and reminders live elsewhere: never let "nothing found" read as "your list is empty".
        lists = [row["name"] for row in (await ctx.store.list_show(ctx.user_id, None))["lists"]]
        answer["not_searched_here"] = (
            "find_saved searches memory and notes only. For lists use list_show, for tasks todo_list, for reminders reminder_list."
            + (f" Lists that exist: {', '.join(lists)}." if lists else "")
        )
    return answer


# --- short-term memory ---------------------------------------------------------------------------

class StmAdd(Fields, Placement):
    text: str = Field(min_length=2, max_length=300, description="One plain sentence, e.g. 'Paid Rahul 2500'")
    about_date: date | None = Field(default=None, description="The day it happened or will happen (YYYY-MM-DD)")
    keywords: list[str] = Field(default_factory=list, max_length=8)
    entities: list[str] = Entities

    _none = field_validator("about_date", mode="before")(lambda cls, v: _empty_to_none(v))


async def stm_add(ctx: ToolContext, args: StmAdd) -> dict[str, Any]:
    if (question := _where(ctx, "short", args.text, args.confidence)):
        return question
    if args.about_date:
        _check_year(args.about_date, ctx)
    today = ctx.now.date()
    last_day = today + timedelta(days=STM_DAYS)
    if args.about_date:
        last_day = max(last_day, args.about_date + timedelta(days=1))
    last_day = min(last_day, today + timedelta(days=STM_MAX_DAYS))
    item = MemoryItem(
        scope="short", category="event", subject=(args.entities[0] if args.entities else ""), text=args.text,
        event_date=args.about_date, keywords=args.keywords, entities=args.entities,
        expires_at=datetime.combine(last_day + timedelta(days=1), time(), tzinfo=ctx.tz),
    )
    item_id = await ctx.memory.remember(ctx.user_id, item)
    return {"id": short(item_id), "say": f"Saved in short-term memory (until {day_text(last_day)}): {args.text}."}


class ForgetMatch(Fields):
    match: str = Field(min_length=2, max_length=120, description="Words that identify the item, e.g. 'Rahul payment'")
    all_matching: bool = Field(default=False, description="True only if the user clearly wants every match removed")
    confirm: bool = Field(default=False, description="True only after the user said yes to removing several")


async def _forget(ctx: ToolContext, args: ForgetMatch, scope: str) -> dict[str, Any]:
    label = "short-term" if scope == "short" else "long-term"
    items = await ctx.memory.search(ctx.user_id, query=args.match, scope=scope, limit=6)
    if not items:
        subject_items = await ctx.memory.search(ctx.user_id, subject=args.match, scope=scope, limit=6)
        items = subject_items
    if not items:
        return {"ask": f"I couldn't find '{args.match}' in {label} memory."}
    if args.all_matching and len(items) > 1:
        if not _confirmed(ctx, args.confirm):
            return {"ask": f"Remove these {len(items)} {label} items? " + _choices([{"text": i.text} for i in items]) + " (yes/no)"}
        for item in items:
            await ctx.memory.forget(ctx.user_id, item.id or "")
        return {"say": f"Removed {len(items)} {label} items."}
    candidates = [{"id": i.id, "text": i.text, "s": i.score or 1.0} for i in items]
    chosen, options = pick(candidates)
    if not chosen:
        return {"ask": "Which one should I remove? " + _choices(options)}
    await ctx.memory.forget(ctx.user_id, chosen["id"])
    return {"say": f"Removed from {label} memory: {chosen['text']}."}


async def stm_forget(ctx: ToolContext, args: ForgetMatch) -> dict[str, Any]:
    return await _forget(ctx, args, "short")


# --- long-term memory ----------------------------------------------------------------------------

class LtmFact(Fields):
    category: Literal["profile", "person", "date", "preference", "project", "other"] = Field(
        description="profile = about the user; person = someone in their life; date = birthday/anniversary; "
        "preference = likes/dislikes; project = their projects; other"
    )
    subject: str = Field(min_length=1, max_length=60, description="Who/what: 'me', 'Riya', 'Project Synq'")
    topic: str = Field(min_length=1, max_length=40, description="Short: name, city, job, relation, birthday, food...")
    text: str = Field(min_length=1, max_length=300, description="One plain sentence, e.g. 'Sister, lives in Mumbai'")
    fact_date: str | None = Field(default=None, alias="date", description="For dates: YYYY-MM-DD, or MM-DD when the year does not matter")
    keywords: list[str] = Field(default_factory=list, max_length=8)
    entities: list[str] = Entities

    _none = field_validator("fact_date", mode="before")(lambda cls, v: _empty_to_none(v))


class LtmSave(Fields, Placement):
    facts: list[LtmFact] = Field(min_length=1, max_length=8, description="One entry per fact: 'name Ashish, lives in Pune' is two facts")

    @model_validator(mode="before")
    @classmethod
    def _one_fact(cls, data: Any) -> Any:
        # Models sometimes send a single fact's fields directly; accept that too.
        if isinstance(data, dict) and "facts" not in data and "text" in data:
            return {"facts": [data]}
        return data


def _fact_date(text: str | None) -> tuple[date | None, bool]:
    if not text:
        return None, False
    try:
        if len(text) == 5:   # MM-DD: yearly (year 2000 is a leap year, so 02-29 works)
            return date.fromisoformat(f"2000-{text}"), True
        parsed = date.fromisoformat(text)
        return parsed, False
    except ValueError as exc:
        raise ToolError("`date` must be YYYY-MM-DD or MM-DD.") from exc


async def ltm_save(ctx: ToolContext, args: LtmSave) -> dict[str, Any]:
    # Updating a fact that is already in long-term memory ("Riya moved to Bangalore"): it stays where it is,
    # whatever else the message says ("…and short term: paid the electrician").
    saved = await ctx.memory.list_all(ctx.user_id, scope="long")
    known = {item.key for item in saved}
    subjects = {item.subject.strip().lower() for item in saved if item.subject}
    said_update = bool(re.search(r"\bupdate|\bchange|\bcorrect|\bmoved\b|\bnow\b|\bab\b|badal|shift ho", ctx.user_text or "", re.IGNORECASE))
    updates = all(
        key_for(MemoryItem(scope="long", category=f.category, subject="me" if f.category == "profile" else f.subject, topic=f.topic, text=f.text)) in known
        or (said_update and f.subject.strip().lower() in subjects)
        for f in args.facts
    )
    if not updates and (question := _where(ctx, "long", "; ".join(f.text for f in args.facts), args.confidence)):
        return question
    ids, said = [], []
    for fact in args.facts:
        event_date, yearly = _fact_date(fact.fact_date)
        if fact.category == "date" and event_date and not yearly:
            yearly = fact.topic.lower() in ("birthday", "anniversary", "bday")
        subject = "me" if fact.category == "profile" else fact.subject
        item = MemoryItem(
            scope="long", category=fact.category, subject=subject, topic=fact.topic, text=fact.text.rstrip(". "),
            event_date=event_date, yearly=yearly, keywords=fact.keywords,
            entities=list(dict.fromkeys([*fact.entities, *([subject] if subject.lower() != "me" else [])])),
            pinned=fact.category == "profile",
        )
        ids.append(short(await ctx.memory.remember(ctx.user_id, item)))
        said.append(f"{subject} · {fact.topic}: {item.text}")
    return {"ids": ids, "say": "Saved in long-term memory: " + "; ".join(said) + "."}


async def ltm_forget(ctx: ToolContext, args: ForgetMatch) -> dict[str, Any]:
    return await _forget(ctx, args, "long")


# --- reminders -----------------------------------------------------------------------------------

class ReminderAdd(Fields, Placement):
    text: str = Field(min_length=1, max_length=200, description="What to remind about, short")
    when: str | None = Field(default=None, description="Local date and time YYYY-MM-DDTHH:MM (or only YYYY-MM-DD for a full-day reminder). "
                             "Leave empty if the user gave no day or no time: never guess them")
    all_day: bool = Field(default=False, description="True when the user wants it for the whole day ('full day', 'all day', 'pura din'): no time")
    in_minutes: int | None = Field(default=None, ge=1, le=60 * 24 * 7, description="Instead of `when`, for 'in 10 minutes' / '2 ghante baad'")
    repeat: Literal["none", "daily", "weekly", "monthly"] = "none"
    entities: list[str] = Entities

    _none = field_validator("when", "in_minutes", mode="before")(lambda cls, v: _empty_to_none(v))


async def reminder_add(ctx: ToolContext, args: ReminderAdd) -> dict[str, Any]:
    if (question := _where(ctx, "reminder", args.text, args.confidence)):
        return question
    from ..router import is_tool_question

    # Their words this turn, plus the turn before when this one answers our question ("Monday" after
    # "Which day…?" for "remind me at 9 to call mom").
    words = ctx.user_text or ""
    last = (ctx.last_assistant or "").lower()
    about = re.search(r"remind you about '([^']+)'", last)
    same_reminder = not about or about.group(1).strip() == args.text.strip().lower()
    if _answers_question(ctx) and same_reminder:
        # Their answer builds on the message before, but only for this reminder (a new "gym" reminder must not
        # take the "full day" said for the exam).
        words = f"{ctx.previous_user_text or ''} {words}"
    # Full day only when the user said so ("full day", "pura din"): a model once made one on its own.
    all_day = bool(_ALL_DAY.search(words)) or (args.all_day and not words.strip())
    if not args.in_minutes and words.strip():
        day_said, time_said = bool(_DAY_WORDS.search(words)), bool(_TIME_WORDS.search(words)) or all_day
        if same_reminder and re.match(r"at what time on \w{3} \d{1,2} \w{3}", last):
            day_said = True   # our own question named the day ("At what time on Fri 23 Oct…?"): it is settled
        ask = None
        if not day_said and not time_said:
            ask = f"Which day and what time should I remind you about '{args.text}'? (or say full day)"
        elif not day_said:
            ask = f"Which day should I remind you about '{args.text}'?"
        elif not time_said:
            on = ""
            if args.when:
                try:
                    on = " on " + day_text(_parse_moment(args.when, ctx).date())
                except ToolError:
                    on = ""
            ask = f"At what time{on} should I remind you about '{args.text}'? (or say full day)"
        if ask and not (args.when and _same_question(ask, ctx)):
            return {"ask": ask}
        # (The same question again would loop, e.g. on words we don't recognise: then the model's reading of
        # their answer is taken.)
    if not args.when and not args.in_minutes:
        if words.strip():
            # They said the day and the time; the model left them out: it must fill them, not ask again.
            raise ToolError("The user gave the day and time: fill `when` (YYYY-MM-DDTHH:MM, or YYYY-MM-DD for full day).")
        return {"ask": f"Which day and what time should I remind you about '{args.text}'? (or say full day)"}
    if all_day:
        return await _add_all_day(ctx, args)
    due = ctx.now + timedelta(minutes=args.in_minutes) if args.in_minutes else _parse_moment(args.when or "", ctx)
    said_midnight = re.search(r"midnight|\b12\s*(am|a\.m)|raat (ke )?12|बारह", words, re.IGNORECASE)
    if not args.in_minutes and (due.hour, due.minute) == (0, 0) and not said_midnight:
        # 12:00 AM nobody said: the model filled a missing time with midnight ("time abhi fix nahi hai", 20 Oct).
        return {"ask": f"At what time on {day_text(due.date())} should I remind you about '{args.text}'? (or say full day)"}
    if not args.in_minutes:
        question = _check_relative_day(due.date(), ctx, args.text) or _check_night_hour(due, ctx, args.text)
        if question:
            return question
    if due < ctx.now - timedelta(minutes=1):
        return {"ask": f"{when_text(due, ctx.tz)} has already passed. When should I remind you instead?"}
    if due > ctx.now + timedelta(days=366 * 2):
        raise ToolError("That is more than two years away. Ask the user to confirm the date.")
    row = await ctx.store.reminder_add(ctx.user_id, args.text, due, args.repeat, args.entities)
    every = "" if args.repeat == "none" else f" (repeats {args.repeat})"
    if row.get("existing"):
        return {"id": short(row["id"]), "say": f"That reminder is already set: {row['text']}, {when_text(due, ctx.tz)}{every}."}
    return {"id": short(row["id"]), "say": f"Reminder set: {args.text}, {when_text(due, ctx.tz)}{every}."}


async def _add_all_day(ctx: ToolContext, args: ReminderAdd) -> dict[str, Any]:
    """A full-day reminder: no time of its own; it is said out loud at ALL_DAY_HOUR that morning
    (or right away when that hour has already passed today)."""
    day = _parse_moment(args.when or "", ctx).date()
    question = _check_relative_day(day, ctx, args.text)
    if question:
        return question
    if day < ctx.now.date():
        return {"ask": f"{day_text(day)} has already passed. Which day instead?"}
    due = max(datetime.combine(day, time(ALL_DAY_HOUR), tzinfo=ctx.tz), ctx.now + timedelta(minutes=1))
    row = await ctx.store.reminder_add(ctx.user_id, args.text, due, args.repeat, args.entities, all_day=True)
    every = "" if args.repeat == "none" else f" (repeats {args.repeat})"
    if row.get("existing"):
        return {"id": short(row["id"]), "say": f"That full-day reminder is already set: {row['text']}, {day_text(day)}{every}."}
    return {"id": short(row["id"]), "say": f"Full-day reminder set: {args.text}, {day_text(day)} (all day){every}."}


class ReminderList(Fields):
    date_from: date | None = None
    date_to: date | None = None
    status: Literal["upcoming", "done", "cancelled", "all"] = "upcoming"
    repeat: Literal["none", "daily", "weekly", "monthly"] | None = None
    query: str | None = None
    next_only: bool = Field(default=False, description="Only the very next reminder")

    _none = field_validator("date_from", "date_to", "repeat", "query", mode="before")(lambda cls, v: _empty_to_none(v))


async def reminder_list(ctx: ToolContext, args: ReminderList) -> dict[str, Any]:
    _check_dates(ctx, args.date_from, args.date_to)
    # A repeating reminder's stored time is its next one; for a window, look a little past its end too.
    start = _start(args.date_from, ctx) if args.date_from else None
    end = _end(args.date_to, ctx) if args.date_to else None
    status = "upcoming" if args.next_only else args.status
    found = await ctx.store.reminder_list(
        ctx.user_id, date_from=None if start and args.status == "upcoming" else start, date_to=None if start and args.status == "upcoming" else end,
        status=status, repeat=args.repeat, query=args.query, limit=1 if args.next_only else 50,
    )
    rows = found["reminders"]
    if start and end and args.status == "upcoming":
        # Keep reminders that fire inside the window (a daily one fires every day).
        def fires_in(row: dict[str, Any]) -> bool:
            if row["repeat"] == "none" or row["repeat"] == "monthly":
                return start <= row["due_at"] <= end
            return row["due_at"] <= end
        rows = [row for row in rows if fires_in(row)]
    items = [
        {"id": short(r["id"]), "text": r["text"], "repeat": r["repeat"], "status": r["status"],
         "when": f"{day_text(r['due_at'].astimezone(ctx.tz).date())} (all day)" if r.get("all_day") else when_text(r["due_at"], ctx.tz)}
        for r in rows
    ]
    answer: dict[str, Any] = {"reminders": items, "count": len(items)}
    if start and end:
        answer["times_in_window"] = await ctx.store.reminder_occurrences(rows, start, end)
    return answer


class ReminderPick(Fields):
    """How the user points at a reminder: words, and when several match, a date, a number from Buddy's
    list, or the id a list showed."""
    match: str | None = Field(default=None, max_length=120, description="Words that identify it, e.g. 'study'")
    day: date | None = Field(default=None, description="Its date, when the user names one ('the one on Wednesday'): YYYY-MM-DD")
    number: int | None = Field(default=None, ge=1, le=20, description="The user's pick from Buddy's numbered list ('the 2nd' = 2)")
    reminder_id: str | None = Field(default=None, max_length=36, description="The id from a reminder list, if you have it")

    _none = field_validator("match", "day", "number", "reminder_id", mode="before")(lambda cls, v: _empty_to_none(v))


class ReminderMatch(ReminderPick):
    pass


def _reminder_label(row: dict[str, Any], tz: tzinfo) -> str:
    when = f"{day_text(row['due_at'].astimezone(tz).date())} (all day)" if row.get("all_day") else when_text(row["due_at"], tz)
    return f"{row['text']}, {when}" + ("" if row.get("repeat", "none") == "none" else f" (repeats {row['repeat']})")


async def _reminder_groups(ctx: ToolContext, args: ReminderPick) -> list[list[dict[str, Any]]]:
    """Waiting reminders that fit what the user said, oldest first. Exact copies (same words, same time,
    left over from an old bug) form one group, so they are chosen and cancelled together."""
    if args.reminder_id:
        rows = await ctx.store.reminders_waiting(ctx.user_id, args.reminder_id)
    elif args.match:
        rows = await ctx.store.reminder_find(ctx.user_id, args.match)
        if any(row["s"] >= 1 for row in rows):
            rows = [row for row in rows if row["s"] >= 1]   # the words appear in it: drop the fuzzy look-alikes
        rows.sort(key=lambda row: (row["due_at"], row["text"].lower()))
    else:
        rows = await ctx.store.reminders_waiting(ctx.user_id)
    if args.day:
        rows = [row for row in rows if row["due_at"].astimezone(ctx.tz).date() == args.day]
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["text"].strip().lower(), row["due_at"]), []).append(row)
    return list(groups.values())


async def _one_reminder(ctx: ToolContext, args: ReminderPick, *, allow_all: bool = False) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """The reminder (with its exact copies) the user means, or a question: the numbered choices with their
    day and time, answerable by number, date or (for cancel) 'all'."""
    if args.reminder_id and not args.day and not args.number \
            and not re.search(r"\d|\b(first|second|third|last|pehla|doosra|teesra)\b", ctx.user_text or "", re.IGNORECASE) \
            and not _DAY_WORDS.search(ctx.user_text or ""):
        # An id the model took from a list, while the user's words fit several reminders ("plant wala reminder
        # cancel kar do" with two plant reminders): ask which, instead of picking one for them (20 Oct).
        rows = await ctx.store.reminders_waiting(ctx.user_id, args.reminder_id)
        words = args.match or (rows[0]["text"] if rows else "")
        if words:
            alike = await _reminder_groups(ctx, args.model_copy(update={"reminder_id": None, "match": words}))
            if len(alike) > 1:
                args = args.model_copy(update={"reminder_id": None, "match": words})
    groups = await _reminder_groups(ctx, args)
    what = f"'{args.match}'" if args.match else "that"
    on = f" on {day_text(args.day)}" if args.day else ""
    if not groups:
        return None, {"ask": f"I couldn't find a waiting reminder for {what}{on}."}
    if len(groups) == 1:
        return groups[0], None
    if args.number:
        if args.number <= len(groups):
            return groups[args.number - 1], None
    choices = " ".join(f"{i}) {_reminder_label(group[0], ctx.tz)}" for i, group in enumerate(groups, start=1))
    how = "Say the number or the date" + (", or 'all'." if allow_all else ".")
    return None, {"ask": f"Which reminder do you mean? {choices}. {how}"}


class ReminderCancel(ReminderPick):
    all_matching: bool = Field(default=False, description="True when the user wants every matching reminder cancelled ('all')")
    confirm: bool = Field(default=False, description="True only after the user said yes to Buddy's (yes/no) question")


async def reminder_cancel(ctx: ToolContext, args: ReminderCancel) -> dict[str, Any]:
    if args.all_matching:
        groups = await _reminder_groups(ctx, args)
        if not groups:
            return {"ask": f"I couldn't find any waiting reminders for '{args.match or 'that'}'."}
        rows = [row for group in groups for row in group]
        listed = " ".join(f"{i}) {_reminder_label(group[0], ctx.tz)}" for i, group in enumerate(groups, start=1))
        question = f"Cancel all {len(groups)} of these? {listed} (yes/no)" if len(groups) > 1 else f"Cancel this one? {listed} (yes/no)"
        if not _confirmed(ctx, args.confirm):
            return {"ask": question}
        if listed not in (ctx.last_assistant or ""):
            # The yes was for a different list (a model once dropped the words on the confirm call and every
            # reminder matched, 20 Oct): show exactly what would go and ask again.
            return {"ask": question}
        for row in rows:
            await ctx.store.reminder_set_status(ctx.user_id, row["id"], "cancelled")
        return {"say": f"Cancelled {len(groups)} reminder{'s' if len(groups) != 1 else ''}: " + "; ".join(_reminder_label(g[0], ctx.tz) for g in groups) + "."}
    chosen, problem = await _one_reminder(ctx, args, allow_all=True)
    if problem:
        return problem
    for row in chosen:
        await ctx.store.reminder_set_status(ctx.user_id, row["id"], "cancelled")
    copies = f" (and {len(chosen) - 1} duplicate cop{'ies' if len(chosen) > 2 else 'y'})" if len(chosen) > 1 else ""
    return {"say": f"Cancelled the reminder: {_reminder_label(chosen[0], ctx.tz)}{copies}."}


class ReminderMove(ReminderPick):
    when: str = Field(description="New local date and time YYYY-MM-DDTHH:MM")


async def reminder_move(ctx: ToolContext, args: ReminderMove) -> dict[str, Any]:
    group, problem = await _one_reminder(ctx, args)
    if problem:
        return problem
    chosen = group[0]
    due = _parse_moment(args.when, ctx)
    question = _check_relative_day(due.date(), ctx, chosen["text"]) or _check_night_hour(due, ctx, chosen["text"])
    if question:
        return question
    if due < ctx.now - timedelta(minutes=1):
        return {"ask": f"{when_text(due, ctx.tz)} has already passed. Which time instead?"}
    await ctx.store.reminder_reschedule(ctx.user_id, chosen["id"], due)
    for copy in group[1:]:   # exact duplicates of it go away
        await ctx.store.reminder_set_status(ctx.user_id, copy["id"], "cancelled")
    return {"say": f"Moved '{chosen['text']}' to {when_text(due, ctx.tz)}."}


async def reminder_done(ctx: ToolContext, args: ReminderMatch) -> dict[str, Any]:
    group, problem = await _one_reminder(ctx, args)
    if problem:
        return problem
    for row in group:
        await ctx.store.reminder_done(ctx.user_id, row["id"])
    return {"say": f"Marked done: {_reminder_label(group[0], ctx.tz)}."}


class ReminderSnooze(ReminderPick):
    minutes: int = Field(default=10, ge=1, le=24 * 60)


async def reminder_snooze(ctx: ToolContext, args: ReminderSnooze) -> dict[str, Any]:
    group, problem = await _one_reminder(ctx, args)
    if problem:
        return problem
    chosen = group[0]
    due = ctx.now + timedelta(minutes=args.minutes)
    await ctx.store.reminder_reschedule(ctx.user_id, chosen["id"], due, status="snoozed")
    return {"say": f"OK, I'll remind you about '{chosen['text']}' again at {when_text(due, ctx.tz)}."}


# --- to-dos --------------------------------------------------------------------------------------

class TodoAdd(Fields, Placement):
    items: list[str] = Field(min_length=1, max_length=20, description="One short task per item")
    day: date | None = Field(default=None, description="The day's list (YYYY-MM-DD). Set it whenever the user names a day ('Friday', 'kal'): take the date from the date list in the prompt. Empty only for today")
    priority: Literal["low", "normal", "high"] = "normal"
    entities: list[str] = Entities

    _none = field_validator("day", mode="before")(lambda cls, v: _empty_to_none(v))


async def todo_add(ctx: ToolContext, args: TodoAdd) -> dict[str, Any]:
    if (question := _where(ctx, "todo", ", ".join(args.items), args.confidence)):
        return question
    day = args.day or ctx.now.date()
    _check_year(day, ctx)
    question = _check_relative_day(day, ctx, ", ".join(args.items))
    if question:
        return question
    items = [item.strip() for item in args.items if item.strip()]
    if not items:
        return {"ask": "What should I add to the to-do list?"}
    added = await ctx.store.todo_add(ctx.user_id, items, day, args.priority, args.entities)
    which = "today's" if day == ctx.now.date() else f"{day_text(day)}'s"
    high = " (high priority)" if args.priority == "high" else ""
    return {"added": [short(a["id"]) for a in added], "say": f"Added to {which} to-do list{high}: {', '.join(items)}."}


class TodoList(Fields):
    day: date | None = None
    date_from: date | None = None
    date_to: date | None = None
    status: Literal["pending", "done", "all"] = "pending"
    priority: Literal["low", "normal", "high"] | None = None
    query: str | None = None

    _none = field_validator("day", "date_from", "date_to", "priority", "query", mode="before")(lambda cls, v: _empty_to_none(v))


async def todo_list(ctx: ToolContext, args: TodoList) -> dict[str, Any]:
    _check_dates(ctx, args.day, args.date_from, args.date_to)
    found = await ctx.store.todo_list(
        ctx.user_id, day=args.day, date_from=args.date_from, date_to=args.date_to, status=args.status,
        priority=args.priority, query=args.query, today=ctx.now.date(),
    )
    def shape(row: dict[str, Any]) -> dict[str, Any]:
        return {"text": row["text"], "day": day_text(row["day"]), "priority": row["priority"], "done": row.get("done", False)}
    answer = {"todos": [shape(r) for r in found["todos"]], "count": found["count"]}
    if found.get("overdue"):
        answer["overdue_from_earlier_days"] = [shape(r) for r in found["overdue"]]
    return answer


class TodoDone(Fields):
    match: str | None = Field(default=None, description="Words that identify the task, e.g. 'doodh'")
    day: date | None = Field(default=None, description="With all_of_day: that day's list")
    all_of_day: bool = False

    _none = field_validator("match", "day", mode="before")(lambda cls, v: _empty_to_none(v))


async def _one_todo(ctx: ToolContext, words: str, pending_only: bool = True) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = await ctx.store.todo_find(ctx.user_id, words, pending_only=pending_only)
    chosen, options = pick(candidates)
    if chosen:
        return chosen, None
    if options:
        return None, {"ask": "Which one? " + _choices([{"text": f"{o['text']} ({day_text(o['day'])})"} for o in options])}
    return None, {"ask": f"I couldn't find an open to-do for '{words}'."}


async def todo_done(ctx: ToolContext, args: TodoDone) -> dict[str, Any]:
    if args.all_of_day:
        day = args.day or ctx.now.date()
        rows = await ctx.store.todo_ids_for_day(ctx.user_id, day)
        if not rows:
            return {"say": f"Nothing open on {day_text(day)}'s list."}
        await ctx.store.todo_mark_done(ctx.user_id, [r["id"] for r in rows])
        return {"say": f"Marked {len(rows)} to-dos done for {day_text(day)}."}
    if not args.match:
        return {"ask": "Which task is done?"}
    chosen, problem = await _one_todo(ctx, args.match)
    if problem:
        return problem
    await ctx.store.todo_mark_done(ctx.user_id, [chosen["id"]])
    return {"say": f"Done: {chosen['text']}."}


class TodoMatch(Fields):
    match: str = Field(min_length=2, max_length=120)


async def todo_remove(ctx: ToolContext, args: TodoMatch) -> dict[str, Any]:
    chosen, problem = await _one_todo(ctx, args.match, pending_only=False)
    if problem:
        return problem
    await ctx.store.todo_delete(ctx.user_id, [chosen["id"]])
    return {"say": f"Removed the to-do: {chosen['text']}."}


class TodoMove(Fields):
    to_day: date = Field(description="The day to move to (YYYY-MM-DD)")
    match: str | None = Field(default=None, description="One task by words; or leave empty and give from_day")
    from_day: date | None = Field(default=None, description="Move every open task of this day")

    _none = field_validator("match", "from_day", mode="before")(lambda cls, v: _empty_to_none(v))


async def todo_move(ctx: ToolContext, args: TodoMove) -> dict[str, Any]:
    _check_year(args.to_day, ctx)
    question = _check_relative_day(args.to_day, ctx, args.match or "these tasks")
    if question:
        return question
    if args.from_day and not args.match:
        rows = await ctx.store.todo_ids_for_day(ctx.user_id, args.from_day)
        if not rows:
            return {"say": f"Nothing open on {day_text(args.from_day)}'s list."}
        await ctx.store.todo_set_day(ctx.user_id, [r["id"] for r in rows], args.to_day)
        return {"say": f"Moved {len(rows)} open to-dos to {day_text(args.to_day)}: {', '.join(r['text'] for r in rows)}."}
    if not args.match:
        return {"ask": "Which task should I move?"}
    chosen, problem = await _one_todo(ctx, args.match)
    if problem:
        return problem
    await ctx.store.todo_set_day(ctx.user_id, [chosen["id"]], args.to_day)
    return {"say": f"Moved '{chosen['text']}' to {day_text(args.to_day)}."}


# --- notes ---------------------------------------------------------------------------------------

class NoteSave(Fields, Placement):
    text: str = Field(min_length=1, max_length=4000, description="Exactly what the user wants kept")
    title: str = Field(min_length=1, max_length=60, description="Short title, e.g. 'Wifi'")
    keywords: list[str] = Field(min_length=1, max_length=8, description="3-5 search words in English and Hinglish, e.g. wifi, internet, password, code")
    entities: list[str] = Entities


async def note_save(ctx: ToolContext, args: NoteSave) -> dict[str, Any]:
    if (question := _where(ctx, "note", args.title or args.text[:60], args.confidence)):
        return question
    note_id = await ctx.store.note_save(ctx.user_id, args.text, args.title, args.keywords, args.entities)
    return {"id": short(note_id), "say": f"Noted ({args.title})."}


class NoteMatch(Fields):
    match: str = Field(min_length=2, max_length=120)


async def note_delete(ctx: ToolContext, args: NoteMatch) -> dict[str, Any]:
    candidates = await ctx.store.note_find(ctx.user_id, args.match)
    chosen, options = pick(candidates)
    if not chosen:
        if options:
            return {"ask": "Which note? " + _choices([{"text": o["title"] or o["text"][:40]} for o in options])}
        return {"ask": f"I couldn't find a note for '{args.match}'."}
    await ctx.store.note_delete(ctx.user_id, [chosen["id"]])
    return {"say": f"Deleted the note: {chosen['title'] or chosen['text'][:40]}."}


# --- lists ---------------------------------------------------------------------------------------

class ListItems(Fields, Placement):
    list_name: str = Field(alias="list", min_length=1, max_length=60, description="List name, e.g. shopping, watchlist, books")
    items: list[str] = Field(min_length=1, max_length=30)


async def list_add(ctx: ToolContext, args: ListItems) -> dict[str, Any]:
    if (question := _where(ctx, "list", f"{', '.join(args.items)} ({args.list_name} list)", args.confidence)):
        return question
    done = await ctx.store.list_add(ctx.user_id, args.list_name, [i.strip() for i in args.items if i.strip()])
    parts = []
    if done["added"]:
        parts.append(f"Added to {done['list']} list: {', '.join(done['added'])}.")
    if done["already_there"]:
        parts.append(f"Already there: {', '.join(done['already_there'])}.")
    return {"say": " ".join(parts) or "Nothing to add."}


class ListShow(Fields):
    list_name: str | None = Field(default=None, alias="list", description="List name; empty = all lists")

    _none = field_validator("list_name", mode="before")(lambda cls, v: _empty_to_none(v))


async def list_show(ctx: ToolContext, args: ListShow) -> dict[str, Any]:
    found = await ctx.store.list_show(ctx.user_id, args.list_name)
    if args.list_name and not found["found"]:
        return {"found": False, "note": f"There is no list called {args.list_name}."}
    if args.list_name:
        found["items"] = [{"text": i["text"], "checked": i["done"]} for i in found["items"]]
    return found


async def _list_change(ctx: ToolContext, args: ListItems, *, check: bool | None, delete: bool, verb: str) -> dict[str, Any]:
    changed, questions, missing = [], [], []
    for word in args.items:
        found_list, candidates = await ctx.store.list_items_match(ctx.user_id, args.list_name, word, unchecked_only=check is True)
        if found_list is None:
            return {"ask": f"I couldn't find a list called {args.list_name}."}
        chosen, options = pick(candidates)
        if chosen:
            await ctx.store.list_items_update([chosen["id"]], check=check, delete=delete)
            changed.append(chosen["text"])
        elif options:
            questions.append(f"'{word}': " + _choices(options))
        else:
            missing.append(word)
    parts = []
    if changed:
        parts.append(f"{verb}: {', '.join(changed)}.")
    if missing:
        parts.append(f"Not on the list: {', '.join(missing)}.")
    result: dict[str, Any] = {"say": " ".join(parts)} if parts else {}
    if questions:
        result["ask"] = "Which one? " + " ".join(questions)
    return result or {"say": "Nothing changed."}


async def list_check(ctx: ToolContext, args: ListItems) -> dict[str, Any]:
    return await _list_change(ctx, args, check=True, delete=False, verb="Ticked")


async def list_remove(ctx: ToolContext, args: ListItems) -> dict[str, Any]:
    return await _list_change(ctx, args, check=None, delete=True, verb="Removed")


class ListClear(Fields):
    list_name: str = Field(alias="list", min_length=1, max_length=60)
    checked_only: bool = Field(default=False, description="Only the ticked items")
    confirm: bool = Field(default=False, description="True only after the user said yes")


async def list_clear(ctx: ToolContext, args: ListClear) -> dict[str, Any]:
    shown = await ctx.store.list_show(ctx.user_id, args.list_name)
    if not shown.get("found"):
        return {"ask": f"I couldn't find a list called {args.list_name}."}
    count = shown["count"]["checked"] if args.checked_only else shown["count"]["total"]
    if count == 0:
        return {"say": f"The {shown['list']} list has nothing to clear."}
    if not _confirmed(ctx, args.confirm) and not args.checked_only:
        return {"ask": f"Clear all {count} items from the {shown['list']} list? (yes/no)"}
    done = await ctx.store.list_clear(ctx.user_id, args.list_name, args.checked_only)
    return {"say": f"Cleared {done['removed']} items from the {done['list']} list."}


# --- clearing a whole store, and the day overview ------------------------------------------------

class ClearSaved(Fields):
    store: Literal["short", "long", "notes", "todos", "reminders"]
    day: date | None = Field(default=None, description="To-dos only: that day's list; empty = all to-dos")
    confirm: bool = Field(default=False, description="True only after the user said yes to the count shown")

    _none = field_validator("day", mode="before")(lambda cls, v: _empty_to_none(v))


STORE_NAMES = {"short": "short-term memory items", "long": "long-term memory items", "notes": "notes",
               "todos": "to-dos", "reminders": "upcoming reminders"}


async def clear_saved(ctx: ToolContext, args: ClearSaved) -> dict[str, Any]:
    _check_dates(ctx, args.day)   # 2025-10-21 found no to-dos and said "none for Wed 21 Oct" (English test, 20 Oct)
    counted = await ctx.store.store_counts(ctx.user_id, args.store, args.day)
    name = STORE_NAMES[args.store] + (f" for {day_text(args.day)}" if args.day else "")
    if counted["count"] == 0:
        return {"say": f"There are no {name}."}
    if not _confirmed(ctx, args.confirm):
        preview = ", ".join(counted["preview"][:3])
        forever = " This can't be undone." if args.store == "long" else ""
        return {"ask": f"Delete all {counted['count']} {name} ({preview}{'…' if counted['count'] > 3 else ''})?{forever} (yes/no)"}
    removed = await ctx.store.store_clear(ctx.user_id, args.store, args.day)
    return {"say": f"Cleared {removed} {name}."}


class DayOverview(Fields):
    day: date | None = Field(default=None, description="YYYY-MM-DD; empty = today")

    _none = field_validator("day", mode="before")(lambda cls, v: _empty_to_none(v))


async def day_overview(ctx: ToolContext, args: DayOverview) -> dict[str, Any]:
    day = args.day or ctx.now.date()
    _check_year(day, ctx)
    todos = await todo_list(ctx, TodoList(day=day, status="all"))
    reminders = await reminder_list(ctx, ReminderList(date_from=day, date_to=day))
    short_items = await ctx.memory.search(ctx.user_id, scope="short", date_from=day, date_to=day, limit=10)
    return {
        "day": day_text(day), "todos": todos, "reminders": reminders["reminders"],
        "short_term": [i.text for i in short_items],
    }


# --- calculator and web search -------------------------------------------------------------------

class Calculate(Fields):
    expression: str = Field(min_length=1, max_length=200, description="Arithmetic only, e.g. (1200*12)/100")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate plain arithmetic without running any code."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ToolError("That power is too large.")
            return _OPS[type(node.op)](walk(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ToolError("Only numbers and + - * / % ** ( ) are allowed.")

    try:
        return walk(ast.parse(expression.replace("^", "**").replace("×", "*").replace(",", ""), mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ToolError("That is not a valid calculation.") from exc


async def calculate(ctx: ToolContext, args: Calculate) -> dict[str, Any]:
    value = safe_eval(args.expression)
    if isinstance(value, float):
        value = int(value) if value.is_integer() else round(value, 6)
    return {"result": value}


class WebSearch(Fields):
    query: str = Field(min_length=2, max_length=200, description="A clear search query, in English when possible")


def _web_search(api_key: str):
    async def run(ctx: ToolContext, args: WebSearch) -> dict[str, Any]:
        return await tavily_search(api_key, args.query)
    return run


# --- registry --------------------------------------------------------------------------------------

def build_registry(settings: Settings) -> ToolRegistry:
    W = True  # noqa: N806 - "writes"
    tools = [
        # read (always offered)
        Tool("find_saved", "Search what the user saved: short-term memory, long-term memory and notes, all at once. "
             "Use for any question about the user, their people, dates, projects, payments, passwords, ideas.", FindSaved, find_saved),
        Tool("reminder_list", "List reminders (upcoming by default), optionally in a date range, by repeat or words.", ReminderList, reminder_list),
        Tool("todo_list", "List to-dos of a day or range (pending by default); returns counts and overdue tasks.", TodoList, todo_list),
        Tool("list_show", "Show one named list (shopping, watchlist...) or, with no name, all lists with counts. Use this (not find_saved) for anything on a list.", ListShow, list_show),
        Tool("day_overview", "One day's to-dos, reminders and short-term items together ('aaj ka kya scene hai').", DayOverview, day_overview),
        Tool("calculate", "Do arithmetic exactly.", Calculate, calculate),
        # short-term memory
        Tool("stm_add", "Save to SHORT-TERM memory (next few days: payments, plans, today's situation). Only when the user asks to remember it.", StmAdd, stm_add, "short", W, ("save",)),
        Tool("stm_forget", "Remove something from short-term memory by words.", ForgetMatch, stm_forget, "short", W),
        # long-term memory
        Tool("ltm_save", "Save lasting facts to LONG-TERM memory (profile, people, birthdays, preferences, projects). Put EVERY fact the user gave "
             "in `facts`, one entry each (name and city are two entries; a birthday is its own entry with category=date and date). "
             "The user's own facts: category=profile, subject='me'. Same subject+topic replaces the old fact (use it to update: 'Riya moved to Bangalore').", LtmSave, ltm_save, "long", W, ("save",)),
        Tool("ltm_forget", "Remove something from long-term memory by words.", ForgetMatch, ltm_forget, "long", W),
        # reminders
        Tool("reminder_add", "Set a reminder ONLY when the user asks to be reminded. Leave `when` empty if they gave no time.", ReminderAdd, reminder_add, "reminder", W, ("save",)),
        Tool("reminder_cancel", "Cancel a reminder. Give its words; when Buddy listed numbered choices, pass the user's pick as "
             "`number` (or `day` for a date they name). 'All of them' -> all_matching=true, then confirm=true only after their yes.",
             ReminderCancel, reminder_cancel, "reminder", W),
        Tool("reminder_move", "Change one reminder's time.", ReminderMove, reminder_move, "reminder", W),
        Tool("reminder_done", "Mark a reminder done (the user did it).", ReminderMatch, reminder_done, "reminder", W),
        Tool("reminder_snooze", "Remind again after some minutes.", ReminderSnooze, reminder_snooze, "reminder", W),
        # to-dos
        Tool("todo_add", "Add tasks to a day's to-do list (today if no day). Several tasks at once; tasks with a different priority or day need their own call.", TodoAdd, todo_add, "todo", W, ("save",)),
        Tool("todo_done", "Mark a task done by words, or every task of a day.", TodoDone, todo_done, "todo", W),
        Tool("todo_remove", "Delete one task by words.", TodoMatch, todo_remove, "todo", W),
        Tool("todo_move", "Move a task (by words) or all open tasks of a day to another day.", TodoMove, todo_move, "todo", W),
        # notes
        Tool("note_save", "Save a note exactly as given (passwords, codes, addresses, ideas, meeting points). Always give a title and keywords.", NoteSave, note_save, "note", W, ("save",)),
        Tool("note_delete", "Delete one note by words.", NoteMatch, note_delete, "note", W),
        # lists
        Tool("list_add", "Add items to a named list (creates it if needed).", ListItems, list_add, "list", W, ("save",)),
        Tool("list_check", "Tick items on a list (bought, watched, packed).", ListItems, list_check, "list", W),
        Tool("list_remove", "Remove items from a list.", ListItems, list_remove, "list", W),
        Tool("list_clear", "Empty a list (or only its ticked items). Ask first; confirm=true only after a yes.", ListClear, list_clear, "list", W),
        # clearing a whole store
        Tool("clear_saved", "Delete everything in one store (short, long, notes, todos, reminders). First call shows the count; "
             "call again with confirm=true only after the user says yes.", ClearSaved, clear_saved, "clear", W),
    ]
    if settings.tavily_api_key:
        tools.append(Tool("web_search", "Search the web for live or changing facts: news, weather, scores, prices, releases.", WebSearch, _web_search(settings.tavily_api_key)))
    return ToolRegistry(tools)
