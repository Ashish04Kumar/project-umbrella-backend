"""Postgres pool and migrations, all inside one schema (default "companion")."""

from __future__ import annotations

import logging
import re

import asyncpg

from ..config import PROJECT_DIR

MIGRATIONS_DIR = PROJECT_DIR / "migrations"
log = logging.getLogger("companion.db")


def _check_schema(schema: str) -> str:
    # The schema name goes into SQL text, so allow only plain names.
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,40}", schema):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return schema


async def connect(database_url: str, schema: str) -> asyncpg.Pool:
    if not database_url:
        raise RuntimeError("COMPANION_DATABASE_URL is not set")
    schema = _check_schema(schema)
    setup = await asyncpg.connect(database_url)
    try:
        await setup.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    finally:
        await setup.close()
    pool = await asyncpg.create_pool(
        database_url, min_size=1, max_size=5, server_settings={"search_path": f"{schema},public"}
    )
    await migrate(pool)
    return pool


async def migrate(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as connection:
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {row["name"] for row in await connection.fetch("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            async with connection.transaction():
                await connection.execute(path.read_text(encoding="utf-8"))
                await connection.execute("INSERT INTO schema_migrations (name) VALUES ($1)", path.name)
            log.info("migration applied: %s", path.name)
