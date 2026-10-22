"""Web search through Tavily (https://tavily.com), plain HTTP with the standard library."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from .registry import ToolError


def _search(api_key: str, query: str, max_results: int) -> dict[str, Any]:
    body = {"query": query, "max_results": max_results, "include_answer": True, "search_depth": "basic"}
    request = Request(
        "https://api.tavily.com/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise ToolError(f"Web search failed (HTTP {exc.code}).") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ToolError("Web search is not reachable right now.") from exc


async def tavily_search(api_key: str, query: str, max_results: int = 5) -> dict[str, Any]:
    payload = await asyncio.to_thread(_search, api_key, query, max_results)
    results = [
        {
            "title": item.get("title", ""),
            "source": urlparse(item.get("url", "")).netloc.removeprefix("www."),
            "url": item.get("url", ""),
            "snippet": (item.get("content") or "")[:500],
        }
        for item in payload.get("results", [])[:max_results]
    ]
    return {"answer": payload.get("answer") or "", "results": results}
