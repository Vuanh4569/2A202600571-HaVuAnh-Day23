"""Checkpointer adapter."""

from __future__ import annotations

from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer.

    TODO(student): implement SQLite support for the persistence extension track.
    The starter provides MemorySaver only — SQLite/Postgres are extension tasks.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        try:
            import sqlite3
            from langgraph_checkpoint_sqlite import SqliteSaver
        except Exception as exc:
            raise RuntimeError(
                "SQLite checkpointer requires langgraph-checkpoint-sqlite. Install with: pip install langgraph-checkpoint-sqlite"
            ) from exc

        # Use provided database_url or in-memory DB
        db = database_url or ":memory:"
        conn = sqlite3.connect(db)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        saver = SqliteSaver(conn=conn)
        return saver
    if kind == "postgres":
        raise NotImplementedError(
            "TODO(student): implement Postgres checkpointer (optional extension)"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
