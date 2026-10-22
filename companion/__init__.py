"""Companion umbrella: a backend-only brain for AI companions."""

from .brain import Companion, Reply
from .config import Settings

__all__ = ["Companion", "Reply", "Settings"]
