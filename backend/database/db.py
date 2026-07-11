"""SQLite & PostgreSQL database — DDL and raw CRUD helpers.

All functions use context managers so connections are always closed.
No business logic lives here — only SQL.
"""

import os
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

# Optional PostgreSQL imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

import sqlite3
from backend.config import DB_PATH

logger = logging.getLogger(__name__)


def _clean_postgres_url(url: str) -> str:
    """Extract a usable psycopg2 DSN from a possibly messy env value.

    Handles values where a whole Supabase/Prisma .env block was pasted in
    (DATABASE_URL="..." plus DIRECT_URL="..." lines), quotes around the URL,
    and query parameters psycopg2 doesn't understand (pgbouncer, direct_url).
    """
    if not url:
        return url
    import re
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    try:
        # Grab the FIRST postgres URL anywhere in the value, ignoring
        # surrounding quotes, KEY= prefixes, and any extra lines after it.
        m = re.search(r"postgres(?:ql)?://[^\s'\"]+", url)
        if m:
            url = m.group(0)
        else:
            url = url.strip().strip("'\"").split()[0]
        parsed = urlparse(url)
        if parsed.query:
            valid_options = {
                "sslmode", "sslrootcert", "sslcert", "sslkey",
                "connect_timeout", "application_name", "keepalives"
            }
            qs = parse_qs(parsed.query)
            filtered_qs = {k: v for k, v in qs.items() if k.lower() in valid_options}
            new_query = urlencode(filtered_qs, doseq=True)
            parsed = parsed._replace(query=new_query)
            return urlunparse(parsed)
        return url
    except Exception:
        return url


DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = _clean_postgres_url(DATABASE_URL)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn():
    if DATABASE_URL:
        if psycopg2 is None:
            raise RuntimeError(
                "DATABASE_URL is set for PostgreSQL but psycopg2 is not installed. "
                "Please run: pip install psycopg2-binary"
            )
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
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


def _row(row) -> dict:
    """Convert a DB row to a plain dict with JSON-friendly values.

    Postgres returns datetime objects for TIMESTAMP columns while SQLite
    returns ISO strings; the API schemas expect strings, so normalise here.
    """
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def _execute(conn, sql: str, params: tuple = ()):
    """Execute a SQL query using SQLite or PostgreSQL, translating placeholders."""
    if DATABASE_URL:
        sql = sql.replace("?", "%s")
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, params)
        return cur
    else:
        return conn.execute(sql, params)


def _insert(conn, sql: str, params: tuple = ()) -> int:
    """Execute an INSERT query and return the generated row ID."""
    if DATABASE_URL:
        sql = sql.replace("?", "%s") + " RETURNING id"
        cur = conn.cursor()
        cur.execute(sql, params)
        row_id = cur.fetchone()[0]
        cur.close()
        return row_id
    else:
        cur = conn.execute(sql, params)
        return cur.lastrowid


# ---------------------------------------------------------------------------
# DDL — called once at startup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't already exist."""
    if DATABASE_URL:
        with _get_conn() as conn:
            with conn.cursor() as cur:

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id            SERIAL PRIMARY KEY,
                        username      VARCHAR(255) NOT NULL UNIQUE,
                        email         VARCHAR(255) NOT NULL UNIQUE,
                        password_hash TEXT NOT NULL,
                        salt          TEXT NOT NULL,
                        first_name    VARCHAR(100),
                        middle_name   VARCHAR(100),
                        last_name     VARCHAR(100),
                        tts_voice     VARCHAR(50) NOT NULL DEFAULT 'Ezinne',
                        created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );

                    ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name  VARCHAR(100);
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS middle_name VARCHAR(100);
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name   VARCHAR(100);
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS birthdate   VARCHAR(10);
                    ALTER TABLE users ADD COLUMN IF NOT EXISTS gender      VARCHAR(20);

                    CREATE TABLE IF NOT EXISTS sessions (
                        token      VARCHAR(255) PRIMARY KEY,
                        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        expires_at TIMESTAMP NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS consultations (
                        id                SERIAL PRIMARY KEY,
                        user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        transcript        TEXT NOT NULL DEFAULT '',
                        guidance          TEXT NOT NULL DEFAULT '',
                        escalate          INTEGER NOT NULL DEFAULT 0,
                        conversation_id   VARCHAR(255),
                        triage_category   VARCHAR(255),
                        triage_department VARCHAR(255),
                        triage_priority   VARCHAR(255),
                        triage_confidence REAL,
                        created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );

                    ALTER TABLE consultations ADD COLUMN IF NOT EXISTS conversation_id   VARCHAR(255);
                    ALTER TABLE consultations ADD COLUMN IF NOT EXISTS triage_category   VARCHAR(255);
                    ALTER TABLE consultations ADD COLUMN IF NOT EXISTS triage_department VARCHAR(255);
                    ALTER TABLE consultations ADD COLUMN IF NOT EXISTS triage_priority   VARCHAR(255);
                    ALTER TABLE consultations ADD COLUMN IF NOT EXISTS triage_confidence REAL;
                """)
        logger.info("Database initialised on Supabase PostgreSQL")
    else:
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
            # Migrations for existing SQLite database instances
            for stmt in [
                "ALTER TABLE users ADD COLUMN tts_voice TEXT NOT NULL DEFAULT 'Ezinne'",
                "ALTER TABLE users ADD COLUMN first_name TEXT",
                "ALTER TABLE users ADD COLUMN middle_name TEXT",
                "ALTER TABLE users ADD COLUMN last_name TEXT",
                "ALTER TABLE users ADD COLUMN birthdate TEXT",
                "ALTER TABLE users ADD COLUMN gender TEXT",
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
        logger.info("Database initialised at SQLite %s", DB_PATH)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def db_create_user(
    username: str, email: str, password_hash: str, salt: str,
    first_name: str | None = None,
    middle_name: str | None = None,
    last_name: str | None = None,
    birthdate: str | None = None,   # ISO YYYY-MM-DD
    gender: str | None = None,      # Male | Female | Other
) -> int:
    """Insert a new user and return the new row id."""
    with _get_conn() as conn:
        return _insert(
            conn,
            "INSERT INTO users (username, email, password_hash, salt,"
            " first_name, middle_name, last_name, birthdate, gender)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (username, email, password_hash, salt, first_name, middle_name,
             last_name, birthdate, gender),
        )


def db_delete_user(user_id: int) -> None:
    """Erase a user; sessions and consultations cascade via FK ON DELETE."""
    with _get_conn() as conn:
        # SQLite needs both explicit deletes if FKs were created without
        # cascade in old databases; Postgres cascades but the extra deletes
        # are harmless no-ops there.
        _execute(conn, "DELETE FROM consultations WHERE user_id = ?", (user_id,))
        _execute(conn, "DELETE FROM sessions WHERE user_id = ?", (user_id,))
        _execute(conn, "DELETE FROM users WHERE id = ?", (user_id,))


def db_update_user_voice(user_id: int, voice: str) -> None:
    """Update user's preferred TTS voice choice."""
    with _get_conn() as conn:
        _execute(
            conn,
            "UPDATE users SET tts_voice = ? WHERE id = ?",
            (voice, user_id),
        )


def db_get_user_by_username(username: str) -> dict | None:
    with _get_conn() as conn:
        cur = _execute(
            conn, "SELECT * FROM users WHERE username = ?", (username,)
        )
        row = cur.fetchone()
        return _row(row) if row else None


def db_get_user_by_email(email: str) -> dict | None:
    with _get_conn() as conn:
        cur = _execute(
            conn, "SELECT * FROM users WHERE email = ?", (email,)
        )
        row = cur.fetchone()
        return _row(row) if row else None


def db_get_user_by_id(user_id: int) -> dict | None:
    with _get_conn() as conn:
        cur = _execute(
            conn, "SELECT * FROM users WHERE id = ?", (user_id,)
        )
        row = cur.fetchone()
        return _row(row) if row else None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def db_create_session(token: str, user_id: int, expires_at: datetime) -> None:
    with _get_conn() as conn:
        _execute(
            conn,
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat() if DATABASE_URL is None else expires_at),
        )


def db_get_session(token: str) -> dict | None:
    with _get_conn() as conn:
        cur = _execute(
            conn, "SELECT * FROM sessions WHERE token = ?", (token,)
        )
        row = cur.fetchone()
        return _row(row) if row else None


def db_delete_session(token: str) -> None:
    with _get_conn() as conn:
        _execute(conn, "DELETE FROM sessions WHERE token = ?", (token,))


def db_purge_expired_sessions() -> int:
    """Remove expired sessions; returns count deleted."""
    now = datetime.now(timezone.utc).isoformat() if DATABASE_URL is None else datetime.now(timezone.utc)
    with _get_conn() as conn:
        cur = _execute(
            conn, "DELETE FROM sessions WHERE expires_at < ?", (now,)
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
        return _insert(
            conn,
            "INSERT INTO consultations (user_id, transcript, guidance, escalate, conversation_id,"
            " triage_category, triage_department, triage_priority, triage_confidence)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id, transcript, guidance, int(escalate), conversation_id,
                t.get("category"), t.get("department"), t.get("priority"), t.get("confidence"),
            ),
        )


def db_get_consultations(user_id: int) -> list[dict]:
    with _get_conn() as conn:
        cur = _execute(
            conn,
            "SELECT * FROM consultations WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        )
        rows = cur.fetchall()
        return [_row(r) for r in rows]


def db_get_conversation(conversation_id: str) -> list[dict]:
    """Return all turns for a conversation, oldest first."""
    with _get_conn() as conn:
        cur = _execute(
            conn,
            "SELECT * FROM consultations WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()
        return [_row(r) for r in rows]


def db_delete_conversation(conversation_id: str, user_id: int) -> int:
    """Delete all turns in a conversation owned by user_id. Returns rows deleted."""
    with _get_conn() as conn:
        cur = _execute(
            conn,
            "DELETE FROM consultations WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        return cur.rowcount


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
        cur = _execute(
            conn,
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
        )
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = _row(r)
            d["priority"] = _RANK_PRIORITY.get(d.pop("priority_rank", 0) or 0)
            result.append(d)
        return result
