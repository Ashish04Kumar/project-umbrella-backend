"""A record of every change a turn makes to the database: what changed, in which table, before and after.

Each change is written as one line to the server log, for example

    🗄️ DB UPDATE reminders 9b43ad97 ("play football", Wed 21 Oct 4:00 PM): status pending → cancelled

and, while a tool runs inside a turn, also sent to the trace (event "db_change") and collected for the
reply object, so the playground can show exactly what a message did to the database.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

log = logging.getLogger("companion.db")


@dataclass
class TurnChanges:
    """Set by the graph while tools run: where a change goes besides the log."""

    trace: Any
    user_id: str
    turn: str
    changes: list[dict[str, Any]] = field(default_factory=list)


CURRENT: ContextVar[TurnChanges | None] = ContextVar("companion_db_changes", default=None)


def _show(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone().strftime("%a %d %b %-I:%M %p")
    if isinstance(value, date):
        return value.strftime("%a %d %b")
    return str(value)


def record(op: str, table: str, row_id: str | None, what: str, change: str = "") -> None:
    """One database change. `op`: INSERT / UPDATE / DELETE; `what`: the row in a few words;
    `change`: for an UPDATE, "field old → new"."""
    short_id = str(row_id or "")[:8]
    line = f"🗄️ DB {op} {table} {short_id} ({what})" + (f": {change}" if change else "")
    log.info(line)
    turn = CURRENT.get()
    if turn is not None:
        entry = {"op": op, "table": table, "id": short_id, "row": what, "change": change}
        turn.changes.append(entry)
        try:
            turn.trace.emit(turn.user_id, "db_change", turn.turn, **entry)
        except Exception:   # tracing never breaks a write
            pass


def describe(text: str, moment: Any = None) -> str:
    return f'"{text}"' + (f", {_show(moment)}" if moment is not None else "")


def arrow(field_name: str, old: Any, new: Any) -> str:
    return f"{field_name} {_show(old)} → {_show(new)}"
