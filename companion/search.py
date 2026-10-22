"""Search helpers shared by every store (docs/search-and-retrieval-spec.md, section 1).

- clean_words(): the words worth matching (filler like "kya", "hai", "the" is dropped)
- to_tsquery(): those words as a Postgres full-text query ("any word", prefix match)
- fuse(): Reciprocal Rank Fusion of the full-text ranking and the fuzzy (trigram) ranking, plus boosts
- pick(): the "match by words" rule for changing one item (one clear hit, or ask which)
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

# Words that never decide a search (English + Hinglish fillers and question words).
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on", "at", "for", "and", "or", "my", "me",
    "i", "you", "your", "what", "whats", "which", "who", "when", "where", "how", "do", "did", "does", "have", "has",
    "had", "it", "this", "that", "there", "any", "all", "show", "tell", "list", "please", "about", "from", "with",
    "kya", "kyaa", "hai", "hain", "tha", "thi", "the", "ka", "ki", "ke", "ko", "se", "me", "mein", "mai", "mera",
    "meri", "mere", "mujhe", "maine", "aur", "bhi", "koi", "kuch", "kaun", "kaunsa", "kaunsi", "kab", "kahan",
    "kitna", "kitne", "kitni", "batao", "bata", "dikhao", "wala", "wali", "wale", "yaar", "na", "toh", "to", "ye",
    "yeh", "woh", "wo", "us", "uska", "uski", "unka", "unki", "saved", "save", "find", "search",
}
_SPLIT = re.compile(r"[\s,.;:!?()\[\]{}\"'“”‘’/\\|&*<>=+\-]+")
_UNSAFE = re.compile(r"[^\w]", re.UNICODE)

RRF_K = 60  # standard Reciprocal Rank Fusion constant


def clean_words(query: str | None) -> list[str]:
    """Lowercase words of 2+ characters, fillers removed, order kept, no duplicates."""
    if not query:
        return []
    words = [_UNSAFE.sub("", word) for word in _SPLIT.split(query.lower())]
    return list(dict.fromkeys(word for word in words if len(word) >= 2 and word not in STOP_WORDS))[:12]


def to_tsquery(words: Sequence[str]) -> str:
    """Any word may match; "pass:*" also matches "password"."""
    return " | ".join(f"{word}:*" for word in words)


def fuse(rows: list[dict[str, Any]], *, boost_key: str = "boost") -> list[dict[str, Any]]:
    """Order rows by RRF over two rankings (row["fts"] and row["fuzzy"], higher = better) plus a boost.

    Each row gets "score" in 0..1 (1 = first in both lists with the full boost)."""
    if not rows:
        return []
    by_fts = sorted(rows, key=lambda row: row.get("fts") or 0, reverse=True)
    by_fuzzy = sorted(rows, key=lambda row: row.get("fuzzy") or 0, reverse=True)
    rank_fts = {id(row): index for index, row in enumerate(by_fts) if (row.get("fts") or 0) > 0}
    rank_fuzzy = {id(row): index for index, row in enumerate(by_fuzzy) if (row.get("fuzzy") or 0) > 0}
    best = 2 / (RRF_K + 1)
    for row in rows:
        rrf = sum(1 / (RRF_K + 1 + ranks[id(row)]) for ranks in (rank_fts, rank_fuzzy) if id(row) in ranks)
        row["score"] = round(min(1.0, (rrf / best) * 0.8 + min(row.get(boost_key) or 0, 1) * 0.2), 4)
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def pick(candidates: list[dict[str, Any]], *, score_key: str = "s") -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Choose the one item a change is about.

    Returns (item, []) when there is one clear match, (None, candidates) when Buddy must ask which,
    and (None, []) when nothing matches."""
    if not candidates:
        return None, []
    if len(candidates) == 1:
        return candidates[0], []
    top, second = candidates[0][score_key], candidates[1][score_key]
    if top - second >= 0.25:
        return candidates[0], []
    return None, candidates[:5]
