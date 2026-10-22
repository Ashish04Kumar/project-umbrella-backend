"""Tool registry: a tool is a Python function plus a Pydantic model for its arguments.

The LLM only sees each tool's name, description and JSON schema. When it asks for a tool,
`run_tool` parses and validates the arguments in code before anything is saved, and any
problem goes back to the LLM as an error message it can explain to the user.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..llm.base import ToolCall
from ..memory.interface import MemoryStore
from .dates import fix_past_years

log = logging.getLogger("companion.tools")


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace "$ref" with the definition itself and drop "$defs" (Gemini's tool schemas do not accept refs)."""
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = definitions[node["$ref"].rsplit("/", 1)[-1]]
                return resolve({key: value for key, value in target.items() if key != "title"})
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


class ToolError(Exception):
    """A clear reason the tool refused, shown to the LLM (e.g. 'that time is in the past')."""


@dataclass
class ToolContext:
    user_id: str
    tz: ZoneInfo
    now: datetime          # aware, in the user's timezone
    store: Any             # PostgresStore (typed loosely so tests can pass a fake)
    memory: MemoryStore
    settings: Settings
    last_assistant: str = ""   # Buddy's previous message: a "(yes/no)" question there is what a confirm answers
    user_text: str = ""        # the user's message this turn: tools check "tomorrow" / "aaj" against the date chosen
    previous_user_text: str = ""  # their message before, used when this one answers Buddy's question ("Monday" after "remind me at 9")


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], Awaitable[dict[str, Any]]]
    group: str = "read"     # the router offers a tool only when its group fits the message (companion/router.py)
    writes: bool = False    # changes data: its result carries "say" (a ready confirmation) or "ask" (a question)
    tags: tuple[str, ...] = ()  # extra groups: "save" = the six tools that save something new (offered for "save it")

    def schema(self) -> dict[str, Any]:
        parameters = _inline_refs(self.args.model_json_schema())
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": parameters}}


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self, groups: set[str] | None = None) -> list[dict[str, Any]]:
        """Tool schemas for the LLM; only the given groups when `groups` is set (saves prompt tokens)."""
        return [tool.schema() for tool in self.tools.values()
                if groups is None or tool.group in groups or groups & set(tool.tags)]

    def is_write(self, name: str) -> bool:
        tool = self.tools.get(name)
        return bool(tool and tool.writes)

    def names(self) -> list[str]:
        return list(self.tools)

    async def run_tool(self, call: ToolCall, ctx: ToolContext) -> dict[str, Any]:
        tool = self.tools.get(call.name)
        if tool is None:
            return {"ok": False, "error": f"There is no tool called {call.name}."}
        args: Any = None
        try:
            raw = fix_past_years(json.loads(call.arguments or "{}"), ctx.now.date())
            args = tool.args.model_validate(raw)
            result = {"ok": True, **(await tool.run(ctx, args))}
        except json.JSONDecodeError:
            result = {"ok": False, "error": "The tool arguments were not valid JSON."}
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in exc.errors())
            result = {"ok": False, "error": f"Invalid arguments: {problems}"}
        except ToolError as exc:
            result = {"ok": False, "error": str(exc)}
        except Exception:  # a bug or a database problem: never crash the chat
            log.exception("tool %s crashed", call.name)
            result = {"ok": False, "error": "The tool failed. Tell the user it did not work and suggest trying again."}
        try:
            logged_args = args.model_dump() if isinstance(args, BaseModel) else call.arguments
            await ctx.store.log_tool(ctx.user_id, call.name, logged_args, result, bool(result.get("ok")))
        except Exception:
            log.exception("could not write the tool log")
        return result
