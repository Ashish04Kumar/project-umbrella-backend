"""Provider-independent shapes for talking to any chat LLM."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    # Raw JSON text from the model. The tools node parses and validates it,
    # because models sometimes send broken JSON.
    arguments: str = "{}"
    # Provider data that must go back unchanged with this call in the next round. Gemini 3 puts its
    # "thought_signature" here and rejects the next request (HTTP 400) without it.
    extra: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages
    name: str | None = None          # tool name on role="tool" messages


class LLMResult(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider: str = ""
    reasoning: str = ""                                   # the model's own hidden thinking, when the provider returns it
    usage: dict[str, Any] = Field(default_factory=dict)   # prompt/completion tokens, when the provider returns them


class LLMProvider(Protocol):
    name: str

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 700,
        json_mode: bool = False,
        tool_choice: str = "auto",   # "required": must call a tool this round
    ) -> LLMResult: ...


class LLMError(RuntimeError):
    """A provider could not answer. Messages never include keys or user content.
    `retry_after` is set (seconds) when the provider hit its per-minute rate limit."""

    retry_after: float | None = None   # per-minute limit: wait this long and try again
    daily_reset: float | None = None   # daily limit: seconds until this model can be used again
    overloaded: bool = False           # the provider said it is busy (HTTP 5xx, e.g. Gemini "high demand")
