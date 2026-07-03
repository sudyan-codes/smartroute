# backend/conversation_store.py
"""
Conversation Store
------------------
Thread-safe, file-backed storage for chat sessions.

Public API:
  create_session()                      → str (new UUID session_id)
  append_message(sid, role, content)    → None
  get_transcript(sid)                   → List[dict]
  get_conversation(sid)                 → dict
  save_conversation(sid, conversation)  → None
  log_escalation(sid, level, trigger, label) → None
  list_conversation_ids()               → List[str]
"""

import json
import os
import uuid
import threading
from datetime import datetime, timezone

from config import CONVERSATIONS_FILE

# Ensure the parent directory for the conversations file exists
os.makedirs(os.path.dirname(CONVERSATIONS_FILE), exist_ok=True)

_lock = threading.Lock()

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _read_store() -> dict:
    """Read the entire conversations JSON file.
    Returns an empty dict if the file does not exist or is malformed.
    """
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_store(data: dict) -> None:
    """Write the entire conversations dict to disk."""
    with open(CONVERSATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _now_iso() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ─── Public API ───────────────────────────────────────────────────────────────

def create_session() -> str:
    """
    Create a new conversation session with a fresh UUID.
    Persists an empty session record to disk immediately.

    Returns:
        session_id (str): UUID string identifying the new session.
    """
    sid = str(uuid.uuid4())
    with _lock:
        store = _read_store()
        store[sid] = {
            "created_at": _now_iso(),
            "messages": [],
            "escalations": [],
        }
        _write_store(store)
    return sid


def append_message(session_id: str, role: str, content: str) -> None:
    """
    Append a single message to the session's message list.

    Parameters:
        session_id : UUID string of the session
        role       : "user" or "assistant"
        content    : The text of the message
    """
    with _lock:
        store = _read_store()
        session = store.setdefault(session_id, {
            "created_at": _now_iso(),
            "messages": [],
            "escalations": [],
        })
        session.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "timestamp": _now_iso(),
        })
        store[session_id] = session
        _write_store(store)


def get_transcript(session_id: str) -> list:
    """
    Return the list of all messages for a session.

    Returns:
        List of {role, content, timestamp} dicts.
        Empty list if session not found.
    """
    with _lock:
        store = _read_store()
        return store.get(session_id, {}).get("messages", [])


def log_escalation(
    session_id: str,
    level: int,
    trigger: str,
    label: str,
) -> None:
    """
    Append an escalation record to the session's escalation log.

    Parameters:
        session_id : UUID string of the session
        level      : Seriousness level 1–4
        trigger    : Human-readable reason for escalation
        label      : Name of the escalation target (e.g. "Support Manager")
    """
    with _lock:
        store = _read_store()
        session = store.setdefault(session_id, {
            "created_at": _now_iso(),
            "messages": [],
            "escalations": [],
        })
        session.setdefault("escalations", []).append({
            "timestamp": _now_iso(),
            "level": level,
            "label": label,
            "trigger": trigger,
        })
        store[session_id] = session
        _write_store(store)


def get_conversation(session_id: str) -> dict:
    """Return the full session dict for a given ID. Empty dict if not found."""
    with _lock:
        store = _read_store()
        return store.get(session_id, {})


def save_conversation(session_id: str, conversation: dict) -> None:
    """Persist an entire session dict, overwriting any existing entry."""
    with _lock:
        store = _read_store()
        store[session_id] = conversation
        _write_store(store)


def list_conversation_ids() -> list:
    """Return a list of all stored session IDs."""
    with _lock:
        store = _read_store()
        return list(store.keys())
