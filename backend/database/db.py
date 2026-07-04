"""SQLite database — DDL and raw CRUD helpers.

All functions use context managers so connections are always closed.
No business logic lives here — only SQL.
"""

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

from backend.config import DB_PATH

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DDL — called once at startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't already exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                tts_voice     TEXT    NOT NULL DEFAULT 'Ezinne',
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token      TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consultations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                transcript TEXT    NOT NULL DEFAULT '',
                guidance   TEXT    NOT NULL DEFAULT '',
                escalate   INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );
        """)
        # Migrations for existing database instances
        for stmt in [
            "ALTER TABLE users ADD COLUMN tts_voice TEXT NOT NULL DEFAULT 'Ezinne'",
            "ALTER TABLE consultations ADD COLUMN conversation_id TEXT",
            "ALTER TABLE consultations ADD COLUMN triage_category TEXT",
            "ALTER TABLE consultations ADD COLUMN triage_department TEXT",
            "ALTER TABLE consultations ADD COLUMN triage_priority TEXT",
            "ALTER TABLE consultations ADD COLUMN triage_confidence REAL",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
    logger.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def db_create_user(username: str, email: str, password_hash: str, salt: str) -> int:
    """Insert a new user and return the new row id."""
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, salt),
        )
        return cur.lastrowid


def db_update_user_voice(user_id: int, voice: str) -> None:
    """Update user's preferred TTS voice choice."""
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET tts_voice = ? WHERE id = ?",
            (voice, user_id),
        )



def db_get_user_by_username(username: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def db_get_user_by_email(email: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


def db_get_user_by_id(user_id: int) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def db_create_session(token: str, user_id: int, expires_at: datetime) -> None:
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat()),
        )


def db_get_session(token: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        return dict(row) if row else None


def db_delete_session(token: str) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def db_purge_expired_sessions() -> int:
    """Remove expired sessions; returns count deleted."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (now,)
        )
        return cur.rowcount


# ---------------------------------------------------------------------------
# Consultations
# ---------------------------------------------------------------------------

def db_save_consultation(
    user_id: int, transcript: str, guidance: str, escalate: bool,
    conversation_id: str | None = None,
    triage: dict | None = None,
) -> int:
    t = triage or {}
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO consultations (user_id, transcript, guidance, escalate, conversation_id,"
            " triage_category, triage_department, triage_priority, triage_confidence)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id, transcript, guidance, int(escalate), conversation_id,
                t.get("category"), t.get("department"), t.get("priority"), t.get("confidence"),
            ),
        )
        return cur.lastrowid


def db_get_consultations(user_id: int) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM consultations WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def db_get_conversation(conversation_id: str) -> list[dict]:
    """Return all turns for a conversation, oldest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM consultations WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]


_PRIORITY_RANK = {"Emergency": 4, "High": 3, "Moderate": 2, "Low": 1}
_RANK_PRIORITY = {v: k for k, v in _PRIORITY_RANK.items()}


def db_get_conversations(user_id: int) -> list[dict]:
    """Return one summary row per distinct conversation, newest first.

    Each summary carries a *session-level* triage: the highest urgency
    reached in the conversation and the department/category from the most
    recent turn that produced one — so history cards show one verdict per
    session, not one per message.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.conversation_id,
                   MIN(c.created_at) AS started_at,
                   MAX(c.created_at) AS last_at,
                   COUNT(*)          AS turn_count,
                   (SELECT transcript FROM consultations c0
                     WHERE c0.conversation_id = c.conversation_id
                     ORDER BY c0.created_at ASC LIMIT 1) AS first_transcript,
                   MAX(CASE c.triage_priority
                         WHEN 'Emergency' THEN 4 WHEN 'High' THEN 3
                         WHEN 'Moderate' THEN 2 WHEN 'Low' THEN 1 ELSE 0 END) AS priority_rank,
                   (SELECT triage_department FROM consultations c1
                     WHERE c1.conversation_id = c.conversation_id
                       AND c1.triage_department IS NOT NULL
                     ORDER BY c1.created_at DESC LIMIT 1) AS department,
                   (SELECT triage_category FROM consultations c2
                     WHERE c2.conversation_id = c.conversation_id
                       AND c2.triage_category IS NOT NULL
                     ORDER BY c2.created_at DESC LIMIT 1) AS category
              FROM consultations c
             WHERE c.user_id = ? AND c.conversation_id IS NOT NULL
             GROUP BY c.conversation_id
             ORDER BY last_at DESC
            """,
            (user_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["priority"] = _RANK_PRIORITY.get(d.pop("priority_rank", 0) or 0)
            result.append(d)
        return result
