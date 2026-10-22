"""One adapter for every OpenAI-compatible chat API (Groq in the cloud, Ollama on the laptop).

Uses only the standard library (urllib), run in a thread so the async loop never blocks.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import ChatMessage, LLMError, LLMResult, ToolCall


# Gemini's documented stand-in for a tool call that has no signature (one made by another model earlier).
_NO_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _wire(message: ChatMessage, gemini: bool = False) -> dict[str, Any]:
    """Our message shape -> the OpenAI wire format. Only Gemini gets the extra signature field back;
    other providers may reject unknown fields."""
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = []
        for call in message.tool_calls:
            wired: dict[str, Any] = {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            if gemini:
                wired["extra_content"] = call.extra or _NO_SIGNATURE
            out["tool_calls"].append(wired)
    if message.role == "tool":
        out["tool_call_id"] = message.tool_call_id
        out["name"] = message.name
    return out


def _api_message(detail: str) -> str:
    """The provider's own error text (e.g. Gemini's "Function call is missing a thought_signature"), short.
    It says what went wrong with the request; it does not echo the user's messages or our key."""
    try:
        data = json.loads(detail)
        data = data[0] if isinstance(data, list) and data else data
        message = str((data.get("error") or {}).get("message", "")) if isinstance(data, dict) else ""
    except (ValueError, AttributeError):
        message = ""
    return (" - " + " ".join(message.split())[:180]) if message else ""


def _is_daily(detail: str) -> bool:
    """Groq says "tokens per day (TPD)"; Gemini names the quota, e.g. "GenerateRequestsPerDayPerProjectPerModel"."""
    lowered = detail.lower()
    return "per day" in lowered or "perday" in lowered


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, model: str, api_key: str = "", timeout: float = 40.0, key_name: str = "") -> None:
        self.name = name
        self.key_name = key_name   # the .env name of the key, for a clear error when it is missing
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _body(self, messages, tools, temperature, max_tokens, json_mode, tool_choice: str = "auto") -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_wire(m, gemini=self.name.startswith("gemini")) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice   # "required": the model must call a tool this round
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Keep hidden reasoning short: a chat reply should come back fast.
        if self.model.startswith("qwen/"):
            body["reasoning_effort"] = "none"
        elif self.model.startswith("gemini-"):
            body["reasoning_effort"] = "low"   # Gemini thinking: keep it short so replies stay fast
        elif self.model.startswith("openai/gpt-oss"):
            body["reasoning_effort"] = "low"
            body["include_reasoning"] = True  # return the (already generated) thinking so the Trace panel can show it
        return body

    def _request(self, body: dict[str, Any]) -> Request:
        headers = {"Content-Type": "application/json", "User-Agent": "companion-umbrella"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        request = self._request(body)
        for attempt in range(2):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                wait = float(exc.headers.get("retry-after") or 0) if exc.code == 429 else 0
                if exc.code == 429 and attempt == 0 and 0 < wait <= 10:
                    time.sleep(wait)  # the free tier resets tokens every few seconds; waiting beats switching
                    continue
                if exc.code in (500, 502, 503) and attempt == 0:
                    time.sleep(2)     # "high demand" (Gemini 503) often clears within seconds
                    continue
                hint = f" (retry after {wait:g}s)" if wait else ""
                detail = exc.read().decode("utf-8", "replace")
                daily_limit = False
                if exc.code == 429:
                    # Groq says which limit was hit; the daily one (TPD) does not reset for minutes or hours.
                    daily_limit = _is_daily(detail)
                    hint += " - daily token limit reached" if daily_limit else " - per-minute limit"
                else:
                    hint += _api_message(detail)
                error = LLMError(f"{self.name} returned HTTP {exc.code}{hint}")
                error.overloaded = exc.code in (500, 502, 503)
                if exc.code == 429 and not daily_limit:
                    error.retry_after = wait or 10.0
                elif exc.code == 429:
                    error.daily_reset = wait or 600.0
                raise error from exc
            except (URLError, OSError, json.JSONDecodeError) as exc:   # OSError: timeouts, broken pipe, reset
                if isinstance(exc, TimeoutError):
                    error = LLMError(f"{self.name} gave no answer in {self.timeout:g}s (TimeoutError)")
                else:
                    error = LLMError(f"{self.name} request failed: {type(exc).__name__}: {exc}"[:240])
                error.overloaded = isinstance(exc, TimeoutError)   # no answer in time: treat like "too busy"
                raise error from exc
        raise LLMError(f"{self.name} is rate limited")

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 700,
        json_mode: bool = False,
        tool_choice: str = "auto",
    ) -> LLMResult:
        if self.key_name and not self.api_key:
            raise LLMError(f"{self.key_name} is not set")
        payload = await asyncio.to_thread(self._post, self._body(messages, tools, temperature, max_tokens, json_mode, tool_choice))
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError(f"{self.name} returned no choices")
        message = choices[0].get("message") or {}
        calls = [
            ToolCall(
                id=str(call.get("id") or f"call_{index}"),
                name=str(call.get("function", {}).get("name", "")),
                arguments=str(call.get("function", {}).get("arguments") or "{}"),
                extra=call.get("extra_content") or None,
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        ]
        text = message.get("content") or ""
        if not calls and not str(text).strip():
            raise LLMError(f"{self.name} returned an empty reply")
        return LLMResult(
            text=str(text), tool_calls=calls, provider=f"{self.name}:{self.model}",
            reasoning=str(message.get("reasoning") or ""), usage=_usage(payload.get("usage")),
        )

    # --- streaming ------------------------------------------------------------------------------

    def _stream_events(self, body: dict[str, Any]) -> Iterator[StreamEvent]:
        """Blocking: yields ("text", piece) and ("reasoning", piece) as the model writes, then
        ("tool_calls", [...]) if it asked for tools, and ("usage", {...}) when the provider reports tokens."""
        calls: dict[int, dict[str, str]] = {}
        try:
            # include_usage: the last chunk carries the token counts (Gemini and Ollama send none without it).
            with urlopen(self._request({**body, "stream": True, "stream_options": {"include_usage": True}}), timeout=self.timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    usage = (chunk.get("x_groq") or {}).get("usage") or chunk.get("usage")
                    if usage:
                        yield ("usage", _usage(usage))
                    choices = chunk.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    if isinstance(delta.get("reasoning"), str) and delta["reasoning"]:
                        yield ("reasoning", delta["reasoning"])
                    if isinstance(delta.get("content"), str) and delta["content"]:
                        yield ("text", delta["content"])
                    # Tool calls arrive in pieces: the id and name first, then the arguments bit by bit.
                    for part in delta.get("tool_calls") or []:
                        index = int(part.get("index", 0))
                        known = calls.get(index)
                        if known and known["name"] and part.get("id") and part["id"] != known["id"]:
                            index = max(calls) + 1   # Gemini sends whole calls, all with index 0: a new id is a new call
                        entry = calls.setdefault(index, {"id": "", "name": "", "arguments": "", "extra": None})
                        entry["extra"] = part.get("extra_content") or entry["extra"]
                        entry["id"] = part.get("id") or entry["id"]
                        function = part.get("function") or {}
                        entry["name"] = function.get("name") or entry["name"]
                        entry["arguments"] += function.get("arguments") or ""
        except HTTPError as exc:
            hint, daily = "", False
            detail = exc.read().decode("utf-8", "replace")
            if exc.code == 429:
                daily = _is_daily(detail)
                hint = " - daily token limit reached" if daily else " - per-minute limit"
            else:
                hint = _api_message(detail)
            error = LLMError(f"{self.name} returned HTTP {exc.code}{hint}")
            error.overloaded = exc.code in (500, 502, 503)
            if exc.code == 429 and not daily:
                error.retry_after = float(exc.headers.get("retry-after") or 10)
            elif exc.code == 429:
                error.daily_reset = float(exc.headers.get("retry-after") or 600)
            raise error from exc
        except (URLError, OSError, json.JSONDecodeError) as exc:   # OSError: timeouts, broken pipe, reset
            # Say why ("Temporary failure in name resolution" = no internet): "URLError" alone hid it (20 Oct).
            error = LLMError(f"{self.name} stream failed: {type(exc).__name__}: {getattr(exc, 'reason', None) or exc}"[:240])
            error.overloaded = isinstance(exc, TimeoutError)
            raise error from exc
        if calls:
            yield ("tool_calls", [
                ToolCall(id=entry["id"] or f"call_{index}", name=entry["name"], arguments=entry["arguments"] or "{}", extra=entry["extra"])
                for index, entry in sorted(calls.items())
            ])

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.6,
        max_tokens: int = 700,
        tool_choice: str = "auto",
    ) -> AsyncIterator[StreamEvent]:
        if self.key_name and not self.api_key:
            raise LLMError(f"{self.key_name} is not set")
        body = self._body(messages, tools, temperature, max_tokens, False, tool_choice)
        async for event in pieces_from_thread(lambda: self._stream_events(body), self.timeout):
            yield event


def _usage(raw: Any) -> dict[str, Any]:
    """Keep only the token counts (Groq also sends timings)."""
    if not isinstance(raw, dict):
        return {}
    details = raw.get("completion_tokens_details") or {}
    return {
        "prompt": raw.get("prompt_tokens", 0), "completion": raw.get("completion_tokens", 0),
        "reasoning": details.get("reasoning_tokens", 0), "total": raw.get("total_tokens", 0),
    }


StreamEvent = tuple[str, Any]   # ("text" | "reasoning", "piece"), ("tool_calls", [ToolCall, ...]) or ("usage", {...})
_DONE = object()


async def pieces_from_thread(make_iterator, timeout: float) -> AsyncIterator[Any]:
    """Run a blocking iterator in a thread and hand its items to the event loop as they come.

    Same idea as Synq Pod 4's stream bridge: if the reader stops early (barge-in), the thread
    stops at its next item and is not waited for.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = False

    def run() -> None:
        try:
            for item in make_iterator():
                if stop:
                    return
                loop.call_soon_threadsafe(queue.put_nowait, item)
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        except Exception as error:  # handed to the loop and raised there
            loop.call_soon_threadsafe(queue.put_nowait, error)

    worker = loop.run_in_executor(None, run)
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), timeout)
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop = True
        worker.add_done_callback(lambda _: None)

