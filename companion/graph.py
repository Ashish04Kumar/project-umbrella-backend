"""The LangGraph agent loop for one user message.

    START -> load_context -> agent --(tool calls?)--> tools -> agent ... -> guard -> END

- load_context: builds the system prompt from persona, time, memory, summaries and recent chat.
- agent: the LLM writes a reply or asks for tools (it decides which).
- tools: code validates and runs each tool call, results go back to the agent.
- guard: removes hidden reasoning and invented life stories, pulls out the mood tag.
Saving the messages and the background memory extraction happen in brain.py, after the graph.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .guard import ENGLISH_FALLBACK_REPLY, FALLBACK_REPLY, claims_an_action, unsaved_reply, clean_reply, confirm_from_tools, looks_broken, looks_messy, salvage
from .llm.base import ChatMessage, LLMError, LLMResult
from .memory.interface import MemoryItem, MemoryStore
from .persona import ENGLISH_NOTE, build_system_prompt, is_english
from .tools.builtin import when_text
from .tools.registry import ToolContext, ToolRegistry
from .store.audit import CURRENT, TurnChanges
from .store.postgres import data_owner
from .router import must_use_tool, tool_groups
from .trace import TraceHub, clip

log = logging.getLogger("companion.graph")
MEMORY_IN_PROMPT = 8   # pinned profile facts sent with every message; everything else is fetched by tools


class TurnState(TypedDict, total=False):
    turn: str                         # id of this turn, for the live trace
    thread: str                       # which chat: "main" (never reset) or a separate thread
    user_id: str
    user_text: str
    spoken: bool                      # the reply will be read aloud (voice turn)
    stream: bool                      # send the reply piece by piece through LangGraph's custom stream
    cue: str                          # a turn with no new user words (greeting, silence nudge): what to do
    voice_note: str                   # how the user sounded (voice cues), a soft hint
    timezone: str
    now: datetime
    conversation_id: str
    system: str
    messages: list[ChatMessage]       # recent chat + this turn (sent to the LLM)
    new_messages: list[ChatMessage]   # what this turn added (saved afterwards)
    tool_rounds: int
    tool_actions: list[str]           # short lines like "create_reminder ok: {...}" for the extractor and logs
    tool_results: list[dict[str, Any]]  # {"name": ..., "result": {...}} for the confirmation fallback
    code_reply: bool                    # the reply was built by code from write-tool results
    last_assistant: str                 # Umbrella's previous message (a tool's question makes the answer need a tool)
    previous_user: str                  # the user's message before this one (answers to Umbrella's questions build on it)
    db_changes: list[dict[str, Any]]    # what the tools changed in the database this turn (store/audit.py)
    groups: list[str]                 # tool groups the router offers for this message
    done: bool                        # the agent has its final answer (or every LLM failed)
    sources: list[str]
    raw_reply: str
    reply: str
    expression: str
    provider: str


def _merge(*groups: list[MemoryItem]) -> list[MemoryItem]:
    seen: set[str] = set()
    merged: list[MemoryItem] = []
    for group in groups:
        for item in group:
            if item.key not in seen:
                seen.add(item.key)
                merged.append(item)
    return merged


def build_graph(
    *, llm: Any, store: Any, memory: MemoryStore, tools: ToolRegistry, settings: Settings,
    trace: TraceHub | None = None,
):
    web_search = "web_search" in tools.names()
    hub = trace or TraceHub()

    def emit(state: TurnState, kind: str, **data: Any) -> None:
        hub.emit(state["user_id"], kind, state.get("turn", ""), **data)

    async def load_context(state: TurnState) -> TurnState:
        user_id = state["user_id"]
        tz = ZoneInfo(state["timezone"])
        thread = state.get("thread") or "main"
        conversation_id = await store.current_conversation(user_id, thread)
        history, summary, _ = await store.recent_messages(conversation_id, settings.recent_messages)
        # Only the pinned profile goes with every message; everything else is fetched by a tool when asked.
        long_term = (await memory.list_all(data_owner(user_id, thread), scope="long", pinned=True))[:MEMORY_IN_PROMPT]
        selection = f"{len(long_term)} pinned profile facts (others are fetched by tools)"
        events: list[MemoryItem] = []
        pending: list[MemoryItem] = []
        reminders: list[str] = []
        last_assistant = next((m.content for m in reversed(history) if m.role == "assistant"), "")
        previous_user = next((m.content for m in reversed(history) if m.role == "user"), "")
        extra = {"last_assistant": last_assistant, "previous_user": previous_user}
        groups = sorted(tool_groups(state["user_text"], last_assistant))
        system = build_system_prompt(
            name=settings.name, now=state["now"], web_search=web_search, memories=long_term, events=events,
            pending=pending, summary=summary, previous_summary="", upcoming_reminders=reminders,
            user_text=state["user_text"], spoken=state.get("spoken", False), voice_note=state.get("voice_note", ""),
        )
        history_chars = sum(len(m.content) for m in history)
        emit(
            state, "context",
            conversation_id=conversation_id, thread=thread, history=len(history), window=settings.recent_messages,
            summary=clip(summary, 400),
            long_term=[{"scope": i.scope, "category": i.category, "key": i.key, "text": i.text} for i in long_term], selection=selection,
            events=[{"key": i.key, "text": i.text} for i in events], asking=[i.text for i in pending],
            reminders=reminders, voice_note=state.get("voice_note", ""), spoken=state.get("spoken", False), groups=groups,
            cue=state.get("cue", ""), est_prompt_tokens=round((len(system) + history_chars) / 4),
        )
        if state.get("cue"):
            # No new words from the user: the instruction goes to the LLM but is not saved as their message.
            prompt_message = ChatMessage(role="user", content=f"(No new words from them. {state['cue']})")
            saved: list[ChatMessage] = []
        else:
            prompt_message = ChatMessage(role="user", content=state["user_text"])
            saved = [prompt_message]
        # Models copy the language of earlier replies, so an English message gets a note right next to it
        # (in the prompt only, never saved).
        language = [ChatMessage(role="system", content=ENGLISH_NOTE)] if not state.get("cue") and is_english(state["user_text"]) else []
        return {
            "conversation_id": conversation_id, "system": system, "messages": [*history, *language, prompt_message],
            "new_messages": saved, "tool_rounds": 0, "tool_actions": [], "tool_results": [], "sources": [], "groups": groups,
            "db_changes": [], **extra,
        }

    async def agent(state: TurnState) -> TurnState:
        use_tools = state["tool_rounds"] < settings.max_tool_rounds and not state.get("cue")
        prompt = [ChatMessage(role="system", content=state["system"]), *state["messages"]]
        round_number = state["tool_rounds"] + 1
        schemas = tools.schemas(set(state.get("groups") or [])) if use_tools else None
        # A data question or command: the first call must use a tool, so no model answers from the chat
        # window or says "saved" without saving (both seen with the backup model in the live test).
        required = bool(schemas) and round_number == 1 and must_use_tool(state["user_text"], state.get("last_assistant", ""))
        choice = "required" if required else "auto"
        emit(state, "llm_call", round=round_number, tools_offered=len(schemas or []), groups=state.get("groups"),
             messages=len(prompt), tool_choice=choice)
        # The whole request as the LLM gets it (for the playground's Test tab: "what goes to the LLM").
        emit(
            state, "llm_request", round=round_number, tool_choice=choice,
            tools=[schema["function"]["name"] for schema in schemas or []],
            messages=[
                {"role": m.role, "content": clip(m.content, 8000),
                 **({"tool_calls": [{"name": c.name, "arguments": clip(c.arguments, 600)} for c in m.tool_calls]} if m.tool_calls else {}),
                 **({"tool": m.name} if m.role == "tool" else {})}
                for m in prompt
            ],
        )
        started = time.perf_counter()
        try:
            if state.get("stream"):
                result = await streamed(prompt, schemas, choice)
            else:
                result = await llm.chat(prompt, tools=schemas, temperature=0.5, max_tokens=600, tool_choice=choice)
                if not result.tool_calls and (looks_messy(result.text) or not result.text.strip()):
                    # Junk or leaked planning: one more try usually comes back clean. (Streaming has already
                    # shown the text, so there the guard salvages it instead.)
                    # An empty answer too: Gemini Lite once sent nothing after listing reminders (English test, 20 Oct).
                    emit(state, "llm_retry", round=round_number, reason="messy output" if result.text.strip() else "empty answer",
                         text=clip(result.text, 400))
                    result = await llm.chat(prompt, tools=schemas, temperature=0.3, max_tokens=600, tool_choice=choice)
        except LLMError as exc:
            log.error("❌ EVERY LLM FAILED, sending the fallback reply: %s", exc)
            emit(state, "llm_error", round=round_number, error=str(exc), ms=round((time.perf_counter() - started) * 1000))
            return {"raw_reply": "", "provider": "fallback", "done": True}
        emit(
            state, "llm_result", round=round_number, provider=result.provider, ms=round((time.perf_counter() - started) * 1000),
            reasoning=clip(result.reasoning, 2000), text=clip(result.text, 800), usage=result.usage,
            tool_calls=[{"name": c.name, "arguments": clip(c.arguments, 400)} for c in result.tool_calls] if use_tools else [],
        )
        if result.tool_calls and use_tools:
            step = ChatMessage(role="assistant", content=result.text or "", tool_calls=result.tool_calls)
            return {"messages": [*state["messages"], step], "new_messages": [*state["new_messages"], step], "provider": result.provider}
        return {"raw_reply": result.text, "provider": result.provider, "done": True}

    async def streamed(prompt: list[ChatMessage], tool_schemas: list | None, choice: str = "auto") -> LLMResult:
        """Stream one LLM round: each text piece goes out through LangGraph's custom stream
        (brain.py turns it into spoken sentences) while the full result is collected here."""
        write = get_stream_writer()
        write({"type": "round_start"})
        text, reasoning, calls, provider, usage = "", "", [], "", {}
        async for kind, value, provider in llm.stream(prompt, tools=tool_schemas, temperature=0.5, max_tokens=600, tool_choice=choice):
            if kind == "text":
                text += value
                write({"type": "piece", "text": value})
            elif kind == "reasoning":
                reasoning += value
            elif kind == "tool_calls":
                calls = value
            elif kind == "usage":
                usage = value
        write({"type": "round_end", "tool_calls": bool(calls)})
        return LLMResult(text=text, tool_calls=calls, provider=provider, reasoning=reasoning, usage=usage)

    async def run_tools(state: TurnState) -> TurnState:
        tz = ZoneInfo(state["timezone"])
        # Tools read and write the thread's own data (each thread has its own memory, notes, reminders...).
        owner = data_owner(state["user_id"], state.get("thread") or "main")
        ctx = ToolContext(user_id=owner, tz=tz, now=datetime.now(tz), store=store, memory=memory, settings=settings,
                          last_assistant=state.get("last_assistant", ""), user_text=state.get("user_text", ""),
                          previous_user_text=state.get("previous_user", ""))
        last = state["messages"][-1]
        added: list[ChatMessage] = []
        actions = list(state["tool_actions"])
        results = list(state["tool_results"])
        sources = list(state["sources"])
        # Every database change a tool makes is recorded (log line, trace event, reply object).
        changes = TurnChanges(trace=hub, user_id=state["user_id"], turn=state.get("turn", ""), changes=list(state.get("db_changes") or []))
        for call in last.tool_calls:
            started = time.perf_counter()
            token = CURRENT.set(changes)
            try:
                result = await tools.run_tool(call, ctx)
            finally:
                CURRENT.reset(token)
            status = "ok" if result.get("ok") else "failed"
            emit(
                state, "tool_result", round=state["tool_rounds"] + 1, name=call.name, arguments=clip(call.arguments, 400),
                ok=bool(result.get("ok")), ms=round((time.perf_counter() - started) * 1000),
                result=clip(json.dumps(result, ensure_ascii=False, default=str), 900),
            )
            actions.append(f"{call.name} {status}: {json.dumps(result, ensure_ascii=False, default=str)[:300]}")
            if call.name == "web_search":
                sources += [r["url"] for r in result.get("results", []) if r.get("url")]
            added.append(ChatMessage(role="tool", tool_call_id=call.id, name=call.name, content=json.dumps(result, ensure_ascii=False, default=str)))
            results.append({"name": call.name, "result": result})
            log.info("tool %s %s", call.name, status)
        update: TurnState = {
            "messages": [*state["messages"], *added], "new_messages": [*state["new_messages"], *added],
            "tool_rounds": state["tool_rounds"] + 1, "tool_actions": actions, "tool_results": results, "sources": sources,
            "db_changes": changes.changes,
        }
        # Only data-changing tools ran and each one said what happened (or asked one question):
        # that text is the reply, written by code. No second LLM call.
        this_round = results[len(state["tool_results"]):]
        if this_round and all(tools.is_write(r["name"]) and ("say" in r["result"] or "ask" in r["result"]) for r in this_round):
            asks = [r["result"]["ask"] for r in this_round if "ask" in r["result"]]
            says = [r["result"]["say"] for r in this_round if "say" in r["result"]]
            tag = "[curious]" if asks else "[happy]"
            update["raw_reply"] = f"{tag} " + re.sub(r"(?<!\.)\.\.(?!\.)", ".", " ".join(says + asks))  # "hai.." -> "hai."
            update["done"] = True
            update["code_reply"] = True
            emit(state, "code_reply", reason="write tools answered for themselves (no second LLM call)")
        return update

    async def guard(state: TurnState) -> TurnState:
        raw = state.get("raw_reply", "")
        note = "reply checked: mood tag read, no made-up life story"
        if not state.get("code_reply") and looks_messy(raw):
            raw = salvage(raw)
            note = "messy LLM output: kept only the real answer"
        if raw.strip() and looks_broken(raw):
            log.warning("dropped a broken reply: %r", raw[:200])
        if not raw.strip() or looks_broken(raw):
            # The LLM failed or produced junk. If tools already did something, say exactly what.
            confirmation = confirm_from_tools(state["tool_results"])
            raw = f"[happy] {confirmation}" if confirmation else ""
            note = "LLM failed or wrote junk: reply built from the tool results" if confirmation else "LLM failed: fallback line"
        reply, expression = clean_reply(raw, from_code=bool(state.get("code_reply")))
        if reply == FALLBACK_REPLY and is_english(state.get("user_text", "")):
            reply = ENGLISH_FALLBACK_REPLY
        wrote = any(tools.is_write(r["name"]) and r["result"].get("ok", True) and "say" in r["result"] for r in state["tool_results"])
        if not wrote and claims_an_action(reply):
            # No tool changed anything (a read tool may have run), so nothing was saved or changed,
            # whatever the model says ("tick off kar diye" after only looking at the list: live test, 20 Oct).
            log.warning("reply claimed an action with no tool call: %r", reply[:200])
            reply, expression, note = unsaved_reply(state.get("user_text", "")), "concerned", "the LLM said it saved or changed something without a tool doing it: replaced"
        emit(state, "guard", expression=expression, reply=clip(reply, 800), note=note)
        final = ChatMessage(role="assistant", content=reply)
        return {"reply": reply, "expression": expression, "new_messages": [*state["new_messages"], final]}

    def after_agent(state: TurnState) -> str:
        return "guard" if state.get("done") else "tools"

    graph = StateGraph(TurnState)
    graph.add_node("load_context", load_context)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("guard", guard)
    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "guard": "guard"})
    graph.add_conditional_edges("tools", lambda state: "guard" if state.get("done") else "agent", {"guard": "guard", "agent": "agent"})
    graph.add_edge("guard", END)
    return graph.compile()

