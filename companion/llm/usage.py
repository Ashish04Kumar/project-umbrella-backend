"""Token counts of every LLM call a turn makes, so the playground can show input and output tokens per reply.

The brain sets a `TurnUsage` before running a turn; the LLM router adds one entry per finished call
(every agent round, retries and fallbacks included). Calls outside a turn (background summaries,
the test consumer) are not counted. Numbers are what the provider reports; a provider that
reports nothing adds a call with zeros and `reported=False`.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnUsage:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, provider: str, usage: dict[str, Any]) -> None:
        self.calls.append({
            "provider": provider,
            "input": int(usage.get("prompt") or 0),
            "output": int(usage.get("completion") or 0),
            "reasoning": int(usage.get("reasoning") or 0),   # thinking tokens, already inside "output"
            "reported": bool(usage),
        })

    def summary(self) -> dict[str, Any]:
        """{"input", "output", "reasoning", "total", "calls": [...]} for the reply object."""
        total_in = sum(call["input"] for call in self.calls)
        total_out = sum(call["output"] for call in self.calls)
        return {
            "input": total_in, "output": total_out, "reasoning": sum(call["reasoning"] for call in self.calls),
            "total": total_in + total_out, "calls": list(self.calls),
        }


CURRENT: ContextVar[TurnUsage | None] = ContextVar("companion_llm_usage", default=None)


def record(provider: str, usage: dict[str, Any]) -> None:
    turn = CURRENT.get()
    if turn is not None:
        turn.add(provider, usage)
