"""Tries providers in the order from .env until one answers."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..config import Settings
from . import usage
from .base import ChatMessage, LLMError, LLMProvider, LLMResult
from .openai_compat import OpenAICompatProvider

log = logging.getLogger("companion.llm")


def make_provider(name: str, settings: Settings) -> LLMProvider:
    if name == "groq":
        return OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", settings.groq_model, settings.groq_api_key, settings.llm_timeout_seconds, "GROQ_API_KEY")
    if name == "groq_backup":
        return OpenAICompatProvider("groq_backup", "https://api.groq.com/openai/v1", settings.groq_backup_model, settings.groq_api_key, settings.llm_timeout_seconds, "GROQ_API_KEY")
    if name == "groq_background":
        return OpenAICompatProvider("groq_background", "https://api.groq.com/openai/v1", settings.groq_background_model, settings.groq_api_key, settings.llm_timeout_seconds, "GROQ_API_KEY")
    if name in ("gemini", "gemini_backup", "gemini_lite"):
        # Google's OpenAI-compatible endpoint: same tool calling and streaming as Groq. gemini_lite: background work.
        model = {"gemini": settings.gemini_model, "gemini_backup": settings.gemini_backup_model, "gemini_lite": settings.gemini_lite_model}[name]
        # 20 s, not 40: when Gemini is busy it often sends nothing at all, and the next model should answer soon.
        return OpenAICompatProvider(name, "https://generativelanguage.googleapis.com/v1beta/openai", model, settings.gemini_api_key,
                                    min(settings.llm_timeout_seconds, 20.0), "GEMINI_API_KEY")
    if name == "ollama":
        return OpenAICompatProvider("ollama", f"{settings.ollama_url.rstrip('/')}/v1", settings.ollama_model, timeout=60.0)
    raise ValueError(f"Unknown LLM provider '{name}' (use gemini, gemini_backup, gemini_lite, groq, groq_backup, groq_background or ollama)")


# When every model is only rate limited per minute, waiting is better than a "something went wrong" reply
# (Groq's free tier allows 8k tokens a minute; one tool turn can use most of it).
MAX_RATE_WAIT = 30.0
OVERLOAD_PAUSE = 600.0   # a model that said "too busy" or timed out is skipped this long (each retry cost ~40 s)


def _wait_for(pairs: list[tuple[LLMProvider, LLMError]]) -> tuple[LLMProvider, float] | None:
    """The model to try again and how long to wait first: the one whose per-minute limit frees up soonest.
    None when no model failed only because of its per-minute limit, or the wait is too long."""
    limited = [(provider, error.retry_after) for provider, error in pairs if error.retry_after is not None]
    if not limited:
        return None
    provider, wait = min(limited, key=lambda pair: pair[1])
    return (provider, wait) if wait <= MAX_RATE_WAIT else None


def _counted(result: LLMResult, provider: LLMProvider) -> LLMResult:
    """Add this call's tokens to the running turn (llm/usage.py) and pass the result on."""
    usage.record(result.provider or f"{provider.name}:{getattr(provider, 'model', '')}", result.usage)
    return result


def _failed(provider: LLMProvider, error: LLMError, rest: list[LLMProvider]) -> None:
    """One clear terminal line per failure: which model, why (the provider's own words), and what comes next."""
    model = getattr(provider, "model", "")
    after = f"trying {rest[0].name} ({getattr(rest[0], 'model', '')})" if rest else "no other model left"
    paused = " · skipped for the next few minutes" if error.overloaded or error.daily_reset else ""
    log.warning("⚠️  LLM %s (%s) FAILED: %s%s → %s", provider.name, model, error, paused, after)


class LLMRouter:
    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("At least one LLM provider is needed")
        self.providers = providers
        self._daily_blocked: dict[str, float] = {}   # provider name -> monotonic time its daily limit resets

    @classmethod
    def from_settings(cls, settings: Settings, order: list[str] | None = None) -> "LLMRouter":
        unique = list(dict.fromkeys(order or settings.llm_order))
        return cls([make_provider(name, settings) for name in unique])

    def _usable(self) -> list[LLMProvider]:
        """Providers not blocked by their daily limit (all of them if every one is blocked)."""
        now = time.monotonic()
        usable = [p for p in self.providers if self._daily_blocked.get(p.name, 0) <= now]
        return usable or list(self.providers)

    def _note(self, provider: LLMProvider, error: LLMError) -> None:
        if error.daily_reset:
            # Skip this model until its daily window frees up, instead of asking it again every turn.
            self._daily_blocked[provider.name] = time.monotonic() + error.daily_reset
        elif error.overloaded and len(self.providers) > 1:
            # Busy right now: the next ones answer for a while (saves the wait on every turn).
            self._daily_blocked[provider.name] = time.monotonic() + OVERLOAD_PAUSE

    async def chat(self, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None, **options: Any) -> LLMResult:
        pairs: list[tuple[LLMProvider, LLMError]] = []
        usable = self._usable()
        for index, provider in enumerate(usable):
            try:
                return _counted(await provider.chat(messages, tools=tools, **options), provider)
            except LLMError as exc:
                pairs.append((provider, exc))
                self._note(provider, exc)
                _failed(provider, exc, usable[index + 1:])
        again = _wait_for(pairs)
        if again is not None:
            provider, wait = again
            log.warning("every LLM is rate limited; waiting %.0fs, then trying %s again", wait, provider.name)
            await asyncio.sleep(wait + 0.5)
            try:
                return _counted(await provider.chat(messages, tools=tools, **options), provider)
            except LLMError as exc:
                pairs.append((provider, exc))
                self._note(provider, exc)
        raise LLMError("All LLM providers failed: " + "; ".join(str(e) for _, e in pairs))

    async def stream(self, messages: list[ChatMessage], *, tools: list[dict[str, Any]] | None = None, **options: Any):
        """Yield (kind, value, provider). Falls back only if a provider fails before sending anything;
        once text has started, an error just ends the stream (what was said stays said)."""
        pairs: list[tuple[LLMProvider, LLMError]] = []
        retried = False
        providers = self._usable()
        while providers:
            provider = providers.pop(0)
            started = False      # text or a tool call has gone out: no switching to another model after that
            label = f"{provider.name}:{getattr(provider, 'model', '')}"
            reported: dict[str, Any] = {}
            try:
                try:
                    async for kind, value in provider.stream(messages, tools=tools, **options):
                        if kind in ("text", "tool_calls"):
                            started = True
                        elif kind == "usage":
                            reported = value
                        yield kind, value, label
                finally:
                    if started:
                        usage.record(label, reported)   # a stream that sent nothing is retried elsewhere: not counted
                if not started:
                    # Gemini sometimes ends a stream with only "stop": no words, no tool call (20 Oct).
                    raise LLMError(f"{provider.name} returned an empty reply")
                return
            except (LLMError, TimeoutError) as exc:
                if started:
                    log.warning("LLM %s stopped mid-reply (%s)", provider.name, exc)
                    return
                if isinstance(exc, LLMError):
                    error = exc
                else:
                    # No piece arrived in time: the model is busy. Skip it for a while instead of waiting
                    # again on every round (this made replies hang for a minute or more: live test, 20 Oct).
                    error = LLMError(f"{provider.name} sent nothing for {getattr(provider, 'timeout', 0):g}s ({type(exc).__name__})")
                    error.overloaded = True
                pairs.append((provider, error))
                self._note(provider, error)
                _failed(provider, error, providers)
            if not providers and not retried:
                again = _wait_for(pairs)
                if again is not None:
                    retried = True
                    log.warning("every LLM is rate limited; waiting %.0fs, then trying %s again", again[1], again[0].name)
                    await asyncio.sleep(again[1] + 0.5)
                    providers = [again[0]]
        raise LLMError("All LLM providers failed: " + "; ".join(str(e) for _, e in pairs))
