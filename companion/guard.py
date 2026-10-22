"""Checks a reply before it reaches the user: pulls out the mood tag, removes hidden reasoning,
and drops sentences where the AI claims a human life ("yesterday I went...")."""

from __future__ import annotations

import re
import unicodedata

from .expressions import ANY_TAG_PATTERN, DEFAULT_EXPRESSION, normalize

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"^\s*\[([a-zA-Z_]+)\]\s*")
_ANY_TAG = re.compile(ANY_TAG_PATTERN, re.IGNORECASE)
_SPEAKER = re.compile(r"^\s*(assistant|buddy)\s*:\s*", re.IGNORECASE)
# Our own instruction notes, if a model copies one into its reply (live test, 20 Oct).
_OUR_NOTES = re.compile(r"\((?:Their latest message is in English|No new words from them)[^)]*\)\s*", re.IGNORECASE)

# Copied from the Synq Pod 4 talker filter: first-person life stories an AI can't have.
_HUMAN_CLAIM = re.compile(
    r"\b(i|i've|i have|i had)\s+(just\s+|actually\s+|recently\s+|once\s+|also\s+)*"
    r"(saw|seen|went|gone|watched|ate|eaten|tried|visited|met|grew up|used to|remember when|was in|travel(l)?ed|"
    r"cooked|played|read)\b"
    r"|\b(yesterday|today|last (night|week|weekend|year)|this (morning|week|weekend)),?\s+i\b"
    r"|\bwhen i was (a kid|young|little|in|at|growing)\b"
    r"|\bmy (mom|mum|mother|dad|father|parents|family|ex|sister|brother|roommate|childhood|"
    r"boyfriend|girlfriend|partner)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?।])\s+")

FALLBACK_REPLY = "Sorry yaar, ek second ke liye kuch gadbad ho gayi. Phir se bologe?"
ENGLISH_FALLBACK_REPLY = "Sorry, something went wrong for a second. Could you say that again?"


def claims_human_life(sentence: str) -> bool:
    return bool(_HUMAN_CLAIM.search(sentence.replace("’", "'").replace("‘", "'")))


def clean_reply(raw: str, *, from_code: bool = False) -> tuple[str, str]:
    """Returns (text, expression). Never returns empty text. A reply built by code from tool results
    (`from_code`) skips the life-story filter: "My sister Riya's birthday" there is the user's own words."""
    text = _THINK.sub("", raw or "").strip()
    text = _OUR_NOTES.sub("", text).strip()
    text = _SPEAKER.sub("", text)
    match = _TAG.match(text)
    expression = normalize(match.group(1)) if match else DEFAULT_EXPRESSION
    if match:
        text = text[match.end():]
    text = _ANY_TAG.sub("", text).strip()
    if text.endswith("]") and "[" not in text:
        text = text[:-1].rstrip()   # a stray bracket left over from the tag
    if text.count('"') == 1:
        text = text.replace('"', "").strip()  # an unpaired quote mark the model left behind
    # Check sentence by sentence, but keep the original line breaks (lists, steps) when nothing is dropped.
    sentences = [s for s in _SENTENCE.split(text) if s.strip()]
    kept = sentences if from_code else [s for s in sentences if not claims_human_life(s)]
    if len(kept) != len(sentences):
        text = " ".join(kept).strip()
    if not text:
        return FALLBACK_REPLY, "concerned"
    return text, expression


# gpt-oss sometimes writes a burst of junk ("[curly ?]", "What ? ? ??", lines of "…") and even its
# own planning ("We need to respond with proper format") before the real answer (live test, 19 Oct).
_JUNK_MARK = re.compile(r"[\u00a0\u202f]\?|\?[\u00a0\u202f]|\[(?!(?:" + "|".join(
    ("happy", "curious", "concerned", "playful", "surprised", "excited", "sad", "idle")) + r")\])[a-z]*[\u00a0 ]*\?*\]", re.IGNORECASE)
_META = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*(?:we|i) (?:need|should|must) (?:to )?(?:respond|answer|reply|provide|list|start|use|correct)[^.\n]*[.\n]?"
    r"|(?:^|(?<=[.!?\n]))\s*(?:provide answer|need correct answer|proper format|use friendly tone)[^.\n]*[.\n]?",
    re.IGNORECASE,
)
_VALID_TAG = re.compile(ANY_TAG_PATTERN, re.IGNORECASE)


def looks_messy(raw: str) -> bool:
    """Junk marks or the model's own planning leaked into the reply: worth one more try."""
    text = _THINK.sub("", raw or "")
    return looks_broken(text) or bool(_JUNK_MARK.search(text)) or bool(_META.search(text))


def salvage(raw: str) -> str:
    """Keep the real answer from a messy reply: start at the last mood tag that is followed by a proper
    answer, drop junk lines and planning sentences. Clean replies come back unchanged."""
    text = _THINK.sub("", raw or "")
    if not looks_messy(text):
        return raw
    tags = list(_VALID_TAG.finditer(text))
    for tag in reversed(tags):
        tail = text[tag.start():]
        if sum(ch.isalpha() for ch in tail) >= 15 and not looks_broken(tail):
            text = tail
            break
    text = _META.sub(" ", text)
    kept = []
    for line in text.split("\n"):
        body = _JUNK_MARK.sub("", line).strip()
        letters = sum(ch.isalpha() for ch in body)
        if not line.strip():
            kept.append("")
        elif letters >= 3 and letters / max(len(body), 1) >= 0.5:
            kept.append(_JUNK_MARK.sub("", line).rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def looks_broken(raw: str) -> bool:
    """Junk output: mostly dots and symbols, or a run of '...' (the model lost the thread)."""
    text = _THINK.sub("", raw)
    # Combining marks (category M) count as letters, so Devanagari matras don't look like junk.
    letters = sum(1 for ch in text if ch.isalpha() or unicodedata.category(ch).startswith("M"))
    dots = text.count("…") + text.count("...")
    return dots >= 4 or (len(text) > 20 and letters / len(text) < 0.45)


# A reply that says something was saved or changed. Said when no tool ran, it is made up
# (a backup model once copied the "Saved to notes" style from earlier turns without saving anything).
_ACTION_CLAIM = re.compile(
    r"\b(saved (to|in|it|that|your)|added (to|it|that|your)|noted( it| that| down)?\b|reminder (is )?set|i'?ll remind you|"
    r"i('ve| have) (saved|added|noted|set|removed|deleted|cleared|updated|marked)|removed (from|it|that)|deleted (it|that|from)|"
    r"cleared (your|the|it)|marked (it |that )?(as )?done|ticked (off)?|snoozed)"
    # A sentence that starts like a code confirmation: "Updated: ...", "Moved 'gym' to ...", "Done: ..."
    r"|(^|[.!?]\s+)(updated|changed|moved|done|ticked|removed|cleared|deleted|saved|added|noted|cancelled|canceled)\b"
    # Hinglish: "tick off kar diye", "save kar diya", "hata diya", "add kar di", "note kar liya"
    r"|\b(kar|kr) (diya|diye|di|dia|liya|li|lia)\b|\bhata (diya|diye|di)\b|\bticked? off\b",
    re.IGNORECASE,
)

# Nothing was saved: say so and offer the places (the playground shows them as buttons).
UNSAVED_REPLY = ("I haven't saved that yet. Where should I save it? "
                 "(reminder / to-do / short-term memory / long-term memory / note / list)")
# Nothing was removed or changed, for a message that asked to cancel, delete or change something.
UNCHANGED_REPLY = "I haven't changed anything yet. Tell me exactly which one, and I'll do it."
_CHANGE_REQUEST = re.compile(
    r"\b(cancel|delete|remove|clear|forget|stop|move|change|shift|reschedule|done|mark)\b|hata|mita|band kar|badal|ho gaya", re.IGNORECASE)


def unsaved_reply(user_text: str) -> str:
    """What to say instead of a made-up "done": about changing, or about saving, as the user asked."""
    return UNCHANGED_REPLY if _CHANGE_REQUEST.search(user_text or "") else UNSAVED_REPLY


_DENIAL = re.compile(r"\b(?:don't|do not|didn't|did not) have\b|\b(?:haven't|have not|hasn't) got\b|\bnothing (?:is )?(?:saved|stored|noted)\b"
                     r"|\b(?:couldn't|could not|can't|cannot) find\b", re.IGNORECASE)


def claims_an_action(text: str) -> bool:
    """True if a statement says something was saved or changed. Questions are offers, not claims
    ("Reminder set kar doon?" asks first, which is right)."""
    statements = [s for s in re.split(r"(?<=[.!?])\s+", text.replace("’", "'")) if s.strip() and not s.strip().endswith("?")]
    # A denial is not a claim: "I don't have anything saved about your office" (English test, 20 Oct).
    return any(_ACTION_CLAIM.search(statement) and not _DENIAL.search(statement) for statement in statements)


def confirm_from_tools(results: list[dict]) -> str | None:
    """A plain reply built from what the tools actually returned (used when the LLM failed after tools ran)."""
    lines: list[str] = []
    for entry in results:
        result = entry["result"]
        if "say" in result or "ask" in result:
            lines += [result[key] for key in ("say", "ask") if key in result]
        elif not result.get("ok", True):
            lines.append(f"That didn't work: {result.get('error', 'something went wrong')}")
        elif entry["name"] == "calculate":
            lines.append(f"Jawab: {result['result']}.")
    if not lines:
        return None
    return " ".join(dict.fromkeys(lines))


# Copied from Synq Pod 4: a silence nudge must never point at the silence or the waiting.
_MENTIONS_SILENCE = re.compile(
    r"\b(silence|silent|went quiet|gone quiet|(you'?re|you are|bit|so) quiet|pause there|waiting for you|"
    r"i'?m (still )?(here|waiting)|no (rush|pressure)|take your time|whenever you'?re ready|pop in|"
    r"still there|lost you|didn'?t hear|cat got your tongue|chup|khamosh)\b",
    re.IGNORECASE,
)


def mentions_silence(sentence: str) -> bool:
    return bool(_MENTIONS_SILENCE.search(sentence.replace("’", "'").replace("‘", "'")))


# Words that mean the user may be in danger. When heard, the body shows support resources and
# the companion stops making small-talk nudges. The reply itself follows the persona's safety rule.
_CRISIS = re.compile(
    r"\b(suicid\w*|kill myself|end (it all|my life)|want to die|don'?t want to live|self[- ]?harm|cut myself|"
    r"marna chahta|marna chahti|mar jaana chahta|mar jana chahta|jeena nahi|jeene ka (mann|man) nahi|"
    r"khud ?kushi|aatmhatya|apne aap ko (maar|khatam))\b",
    re.IGNORECASE,
)

SUPPORT_RESOURCES = [
    {"name": "Tele-MANAS (India, 24x7, free)", "phone": "14416"},
    {"name": "Emergency (India)", "phone": "112"},
]


def sounds_like_crisis(text: str) -> bool:
    return bool(_CRISIS.search(text))
