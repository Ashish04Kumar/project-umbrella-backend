"""Quietly fix last year's dates in tool arguments.

Gemini Flash Lite often writes "tomorrow" as 2025-10-21 although the prompt says today is 20 Oct 2026
(English test, 20 Oct: five tool calls in one run). Refusing the call made it retry, and once it got lost
(three web searches for "add a reminder for Saturday"). When the same day and month this year is close
to today, it is clearly what was meant, so the year is changed before the tool runs. Other past-year
dates are left alone: the tool still refuses them and sends the model the right dates.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

log = logging.getLogger("companion.tools")

_ISO_START = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
DAYS_BACK = 31     # questions about last week or last month
DAYS_AHEAD = 60    # new reminders, to-dos and plans


def _fixed(text: str, today: date) -> str:
    match = _ISO_START.match(text)
    if not match or int(match.group(1)) >= today.year:
        return text
    try:
        moved = date(today.year, int(match.group(2)), int(match.group(3)))
    except ValueError:   # 29 Feb in a year without one
        return text
    if not today - timedelta(days=DAYS_BACK) <= moved <= today + timedelta(days=DAYS_AHEAD):
        return text
    return f"{today.year}{text[4:]}"


def fix_past_years(value: Any, today: date) -> Any:
    """The same arguments, with every close last-year date ("2025-10-21", "2025-10-21T18:00") set to this year."""
    if isinstance(value, dict):
        return {key: fix_past_years(item, today) for key, item in value.items()}
    if isinstance(value, list):
        return [fix_past_years(item, today) for item in value]
    if isinstance(value, str):
        fixed = _fixed(value, today)
        if fixed != value:
            log.info("📅 tool date %s looked like last year's slip; using %s", value, fixed)
        return fixed
    return value
