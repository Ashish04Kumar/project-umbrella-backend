"""Settings read from companion-umbrella/.env (or the process environment)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def cors_origins() -> list[str]:
    """Browser pages on other addresses that may call this server: the umbrella-d dashboard (Next.js,
    port 3200) by default. COMPANION_CORS_ORIGINS in .env, comma separated; empty switches it off."""
    raw = os.getenv("COMPANION_CORS_ORIGINS", "http://localhost:3200,http://127.0.0.1:3200")
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def _list(name: str, default: str) -> list[str]:
    return [part.strip().lower() for part in _env(name, default).split(",") if part.strip()]


@dataclass
class Settings:
    llm_order: list[str] = field(default_factory=lambda: _list("COMPANION_LLM", "groq") + _list("COMPANION_LLM_FALLBACK", "groq_backup"))
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("COMPANION_GROQ_MODEL", "openai/gpt-oss-120b"))
    groq_backup_model: str = field(default_factory=lambda: _env("COMPANION_GROQ_BACKUP_MODEL", "qwen/qwen3.8-27b"))
    # Gemini (Google AI Studio key). Choose it with COMPANION_LLM=gemini; the user's messages then go to Google.
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    # 2.5 models are closed to new keys; 3.8-flash was overloaded (503) when this was set up (19 Oct 2026).
    gemini_model: str = field(default_factory=lambda: _env("COMPANION_GEMINI_MODEL", "gemini-3.7-flash"))
    gemini_backup_model: str = field(default_factory=lambda: _env("COMPANION_GEMINI_BACKUP_MODEL", "gemini-3.5-flash"))
    gemini_lite_model: str = field(default_factory=lambda: _env("COMPANION_GEMINI_LITE_MODEL", "gemini-3.5-flash-lite"))
    ollama_url: str = field(default_factory=lambda: _env("COMPANION_OLLAMA_URL", "http://127.0.0.1:11434"))
    ollama_model: str = field(default_factory=lambda: _env("COMPANION_OLLAMA_MODEL", "qwen3:4b"))
    # Memory extraction and summaries run after the reply, on their own order (a second Groq model
    # has its own rate limit, so the main model keeps its quota for replies).
    background_llm_order: list[str] = field(default_factory=lambda: _list("COMPANION_BACKGROUND_LLM", "groq_background,groq_backup,groq,ollama"))
    groq_background_model: str = field(default_factory=lambda: _env("COMPANION_GROQ_BACKGROUND_MODEL", "openai/gpt-oss-20b"))
    llm_timeout_seconds: float = 40.0

    database_url: str = field(default_factory=lambda: _env("COMPANION_DATABASE_URL"))
    db_schema: str = field(default_factory=lambda: _env("COMPANION_DB_SCHEMA", "companion"))
    memory_backend: str = field(default_factory=lambda: _env("COMPANION_MEMORY", "postgres").lower())

    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY"))

    # Voice: speech-to-text and text-to-speech, each with a fallback.
    stt_order: list[str] = field(default_factory=lambda: _list("COMPANION_STT", "groq") + _list("COMPANION_STT_FALLBACK", "whisper,sarvam"))
    tts_order: list[str] = field(default_factory=lambda: _list("COMPANION_TTS", "piper") + _list("COMPANION_TTS_FALLBACK", "sarvam"))
    groq_stt_model: str = field(default_factory=lambda: _env("COMPANION_GROQ_STT_MODEL", "whisper-large-v3-turbo"))
    whisper_model: str = field(default_factory=lambda: _env("COMPANION_WHISPER_MODEL", "base"))
    whisper_language: str = field(default_factory=lambda: _env("COMPANION_WHISPER_LANGUAGE"))  # empty: auto (Hindi or English)
    piper_voice: str = field(default_factory=lambda: _env("COMPANION_PIPER_VOICE", "models/en_US-amy-medium.onnx"))
    tts_speed: float = field(default_factory=lambda: float(_env("COMPANION_TTS_SPEED", "1.0") or 1.0))
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))
    sarvam_stt_model: str = field(default_factory=lambda: _env("COMPANION_SARVAM_STT_MODEL", "saarika:v2.5"))
    sarvam_tts_model: str = field(default_factory=lambda: _env("COMPANION_SARVAM_TTS_MODEL", "bulbul:v2"))
    sarvam_speaker: str = field(default_factory=lambda: _env("COMPANION_SARVAM_SPEAKER", "anushka"))
    sarvam_language: str = field(default_factory=lambda: _env("COMPANION_SARVAM_LANGUAGE", "hi-IN"))

    # Live voice over LiveKit (WebRTC).
    livekit_url: str = field(default_factory=lambda: _env("COMPANION_LIVEKIT_URL"))
    livekit_api_key: str = field(default_factory=lambda: _env("LIVEKIT_API_KEY"))
    livekit_api_secret: str = field(default_factory=lambda: _env("LIVEKIT_API_SECRET"))
    silence_nudge_seconds: float = field(default_factory=lambda: float(_env("COMPANION_SILENCE_NUDGE_SECONDS", "10") or 10))
    live_max_minutes: float = field(default_factory=lambda: float(_env("COMPANION_LIVE_MAX_MINUTES", "30") or 30))

    # Save memories only when the user asks (default). True brings back the background extractor that
    # guesses facts after every message (about 1,500 extra tokens per message).
    auto_memory: bool = field(default_factory=lambda: _env("COMPANION_AUTO_MEMORY", "false").lower() in ("1", "true", "yes"))

    # Front-end plugins the server offers to the playground (folders in companion/server/plugins).
    ui_plugins: list[str] = field(default_factory=lambda: _list("COMPANION_UI_PLUGINS", "robot"))

    name: str = field(default_factory=lambda: _env("COMPANION_NAME", "Umbrella"))
    timezone: str = field(default_factory=lambda: _env("COMPANION_TIMEZONE", "Asia/Kolkata"))

    # How much chat goes into each prompt; older messages are folded into a summary.
    recent_messages: int = 20
    summarize_after: int = 30
    max_tool_rounds: int = 4
