"""Which tool groups to offer the LLM for a message (keyword hint; the LLM still decides by intent).

Offering only the groups a message hints at keeps the prompt small (all tools ≈ 1,800 tokens, the
read set ≈ 600). The read tools are always offered, so questions work without keywords. Buddy's own
last message is scanned too, so a short follow-up answer ("9 baje", "yes", "the first one") can still
complete the action Buddy asked about. See docs/personal-data-spec.md, section 2.
"""

from __future__ import annotations

import re

READ = "read"

_GROUPS = {
    "short": r"short[\s-]?term",
    "long": r"long[\s-]?term",
    "reminder": r"remind|reminder|yaad dila|alarm|snooze|wake me",
    "todo": r"to[\s-]?dos?\b|\btasks?\b|kaam ki list|kaam list",
    "note": r"\bnotes?\b|note (?:kar|karo|down)|likh (?:lo|do|le)|write (?:it |this |that )?down|jot",
    "list": r"\blists?\b|shopping|watchlist|packing|grocer",
    "clear": r"\bclear\b|\bclean up\b|saaf kar|empty|khaali|delete (?:all|every)|sab hata|wipe",
}
_REMEMBER = r"remember|yaad rakh|yaad kar|keep (?:it |this |that )?in mind|don'?t forget"
# "save it" / "store this" with no place named: every place is offered; the model picks with a confidence
# and the tool asks "Where should I save …?" when it is not sure.
_SAVE_ANYWHERE = r"\bsave (?:it|this|that|these)\b|\bstore (?:it|this|that)\b|save kar|\bsave\b.*\bsomewhere\b"
# "add one for Saturday 6 pm", "add to my calendar that I'm on leave": an add with no place named got only the
# read tools, so the model ran web searches (English test, 20 Oct). Offer the six save tools instead.
_ADD_ANYWHERE = r"\badd\b|\bcalendar\b|\bdiary\b|\bschedule (?:it|this|that|a|an)\b"
_FORGET = r"\bforget\b|bhool|\bremove\b|\bdelete\b|hata (?:do|de|dena)"
_DONE = r"\bdone\b|ho gaya|ho gayi|completed?|finished|kar liya|le liya|bought|\bpacked\b|watched|dekh li"
_CHANGE = r"\bmove\b|\bshift\b|postpone|reschedule|change|badal|cancel"
_UPDATE = r"\bupdate\b|\bcorrect (?:it|that|this)\b|\bmoved to\b|\bnow (?:lives|works|likes)\b|\bnot .* any ?more\b|ab (?:wo|woh|vo) "

# Questions about the user's own saved things (no tool keyword needed): "when is Riya's birthday?",
# "how much did I pay Rahul?", "what's the internet code?", "kal ka plan kya hai?".
_QUESTION = r"\?|^(?:what|when|where|who|which|how|did|do|does|is|are|kya|kab|kahan|kaun|kitn|show|list)\b"
_ABOUT_SAVED = (
    r"\b(?:my|mine|mera|meri|mere|maine|mujhe|hamara)\b|\b(?:did|have|do|should|can) i\b|\bi (?:paid|gave|told|said|saved)\b"
    r"|\b(?:saved|remembered|noted|overview|plan|planned|schedule|agenda)\b"
    r"|\b(?:password|passcode|wifi|wi-fi|pin|address|birthday|anniversary|blood group)\b|\bcode\b"
    r"|\babout [A-Z][a-z]+|[A-Z][a-z]+'s\b|\b(?:does|did|is) [A-Z][a-z]+ (?:live|work|like|stay|want)"
)


def asks_about_saved(message: str) -> bool:
    """A question that can only be answered from what the user saved. Such a turn must call a tool:
    answering from the chat window is how made-up answers happened (live test, 19 Oct)."""
    text = (message or "").translate(_DASHES).strip()
    return bool(re.search(_QUESTION, text.lower())) and bool(re.search(_ABOUT_SAVED, text) or re.search(_ABOUT_SAVED, text.lower()))


# Questions that write tools themselves ask back (code-built): the next message is the answer to them.
_TOOL_QUESTION = (r"when should i remind you|has already passed\. when|\(yes/no\)|which one did you mean|which (?:reminder|note|task|item)"
                  r"|which day do you mean|am \(night\) or|where should i save"
                  r"|which day and what time|which day should i remind|at what time")


def is_tool_question(text: str) -> bool:
    """Buddy's message is one of the questions the tools ask back (time, day, AM/PM, which one, yes/no, where)."""
    return bool(re.search(_TOOL_QUESTION, (text or "").lower()))


def must_use_tool(message: str, last_assistant: str = "") -> bool:
    """True when the first LLM call of this turn must call a tool: the message names a store or an action
    on one, asks about saved things, or answers a question a tool asked ("6 pm on Friday" after "When should
    I remind you ...?"). Buddy's own casual offers do not count: "no thanks" after "Want a reminder?" must not set one."""
    if re.search(_TOOL_QUESTION, (last_assistant or "").lower()):
        return True
    return bool(tool_groups(message) - {READ}) or asks_about_saved(message)


# Phones and models write "to‑do" with other dash characters (U+2010-U+2015, U+2212); a missed
# "to-do" once sent a task into a list called "to-do" (live test, 20 Oct).
_DASHES = str.maketrans({ch: "-" for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"})


def tool_groups(message: str, last_assistant: str = "") -> set[str]:
    """Groups hinted by the message (and by Buddy's last question, for short follow-up answers)."""
    groups = {READ}
    for text, follow_up in ((message, False), (last_assistant, True)):
        lowered = (text or "").translate(_DASHES).lower()
        if not lowered:
            continue
        if follow_up and "?" not in lowered:
            # Buddy's last message only counts when it asked something ("When should I remind you?").
            # A confirmation ("Reminder set: Study…") must not offer write tools to the next "hi": a model
            # once set the same reminder again on "hi" (20 Oct).
            continue
        if follow_up and re.search(r"where should i save", lowered):
            groups |= {"save"}   # only the six save tools: all 26 made the request too big for a fallback model
            continue
        found = {group for group, pattern in _GROUPS.items() if re.search(pattern, lowered)}
        groups |= found
        if follow_up:
            # Buddy asked something: also its confirm/choice questions ("Delete all 6 ...? (yes/no)").
            if re.search(r"\(yes/no\)|which one|which reminder|which note|when should i remind|which day do you mean|am \(night\) or"
                           r"|which day and what time|which day should i remind|at what time", lowered):
                groups |= found or {"short", "long", "reminder", "todo", "note", "list", "clear"}
            continue
        if re.search(_REMEMBER, lowered) and not found & {"short", "long", "note"}:
            groups |= {"short", "long"}           # "yaad rakhna": the meaning picks short or long term
        if re.search(_SAVE_ANYWHERE, lowered) and not found - {READ}:
            groups |= {"save"}            # "save it": the six save tools, not every tool of every place
        if re.search(_ADD_ANYWHERE, lowered) and not found - {READ}:
            groups |= {"save"}
        if re.search(_FORGET, lowered) and not found - {READ}:
            groups |= {"short", "long", "note", "todo", "list", "reminder"}
        if re.search(_DONE, lowered) and not found - {READ}:
            groups |= {"todo", "reminder", "list"}
        if re.search(_CHANGE, lowered) and not found - {READ}:
            groups |= {"todo", "reminder"}
        if re.search(_UPDATE, lowered):
            groups |= {"short", "long"}
    return groups
