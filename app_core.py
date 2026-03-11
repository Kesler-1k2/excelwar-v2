from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

DB_PATH = Path("app_data.db")

# Keep lesson keys in one place for profile/progress tracking.
LESSON_NAMES = tuple(f"Lesson {index}" for index in range(1, 13))
XP_PER_LEVEL = 150
XP_PER_LESSON = 50
XP_PER_QUIZ = 20


def _default_learning_state() -> dict[str, Any]:
    return {
        "student_name": "Guest",
        "xp": 0,
        "level": 1,
        "completed": {lesson: False for lesson in LESSON_NAMES},
        "quiz_completion": {lesson: False for lesson in LESSON_NAMES},
        "badges": [],
        "activity_log": [],
    }


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_db() -> None:
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                email TEXT,
                name TEXT,
                avatar_url TEXT,
                avatar_bytes BLOB,
                avatar_mime TEXT,
                xp INTEGER,
                completed TEXT,
                quiz_completion TEXT,
                badges TEXT,
                activity_log TEXT,
                updated_at TEXT
            )
            """
        )


def _user_attr(user: Any, key: str) -> Any:
    if isinstance(user, dict):
        return user.get(key)
    return getattr(user, key, None)


def get_current_user() -> dict[str, Any] | None:
    user = None
    if hasattr(st, "user"):
        user = st.user
    elif hasattr(st, "experimental_user"):
        user = st.experimental_user

    if user is None:
        return None

    is_logged_in = _user_attr(user, "is_logged_in")
    if is_logged_in is False:
        return None

    email = _user_attr(user, "email")
    name = _user_attr(user, "name")
    picture = _user_attr(user, "picture") or _user_attr(user, "avatar")
    user_id = _user_attr(user, "sub") or _user_attr(user, "id") or email or name
    if not user_id:
        return None

    return {"id": str(user_id), "email": email, "name": name, "picture": picture}


def _load_user_state(user_id: str) -> dict[str, Any] | None:
    with _connect_db() as conn:
        row = conn.execute(
            "SELECT * FROM user_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None:
        return None

    completed = json.loads(row["completed"]) if row["completed"] else {}
    quiz_completion = json.loads(row["quiz_completion"]) if row["quiz_completion"] else {}
    badges = json.loads(row["badges"]) if row["badges"] else []
    activity_log = json.loads(row["activity_log"]) if row["activity_log"] else []

    for lesson in LESSON_NAMES:
        completed.setdefault(lesson, False)
        quiz_completion.setdefault(lesson, False)

    avatar_bytes = row["avatar_bytes"]
    if avatar_bytes in (None, b""):
        avatar_bytes = None

    return {
        "email": row["email"],
        "name": row["name"],
        "avatar_url": row["avatar_url"],
        "avatar_bytes": avatar_bytes,
        "avatar_mime": row["avatar_mime"],
        "xp": row["xp"] or 0,
        "completed": completed,
        "quiz_completion": quiz_completion,
        "badges": badges,
        "activity_log": activity_log,
    }


def init_app_state(user: dict[str, Any]) -> None:
    _ensure_db()
    defaults = _default_learning_state()
    user_id = user["id"]

    loaded = _load_user_state(user_id)

    name_from_user = (user.get("name") or "").strip() or "User"
    email_from_user = (user.get("email") or "").strip()

    state = defaults.copy()
    if loaded:
        state["student_name"] = loaded.get("name") or name_from_user
        state["xp"] = loaded.get("xp", 0)
        state["completed"] = loaded.get("completed", defaults["completed"]).copy()
        state["quiz_completion"] = loaded.get("quiz_completion", defaults["quiz_completion"]).copy()
        state["badges"] = loaded.get("badges", defaults["badges"])[:]
        state["activity_log"] = loaded.get("activity_log", defaults["activity_log"])[:]
        avatar_url = loaded.get("avatar_url")
        avatar_bytes = loaded.get("avatar_bytes")
        avatar_mime = loaded.get("avatar_mime")
        email = email_from_user or loaded.get("email")
        if not avatar_bytes and not avatar_url:
            avatar_url = user.get("picture")
    else:
        avatar_url = user.get("picture")
        avatar_bytes = None
        avatar_mime = None
        email = email_from_user

    st.session_state.user_id = user_id
    st.session_state.email = email
    st.session_state.avatar_url = avatar_url
    st.session_state.avatar_bytes = avatar_bytes
    st.session_state.avatar_mime = avatar_mime

    for key, value in state.items():
        if isinstance(value, dict):
            st.session_state[key] = value.copy()
        elif isinstance(value, list):
            st.session_state[key] = value[:]
        else:
            st.session_state[key] = value

    refresh_level()
    save_user_state()


def save_user_state() -> None:
    user_id = st.session_state.get("user_id")
    if not user_id:
        return

    avatar_bytes = st.session_state.get("avatar_bytes")
    avatar_blob = sqlite3.Binary(avatar_bytes) if avatar_bytes else None

    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO user_state (
                user_id,
                email,
                name,
                avatar_url,
                avatar_bytes,
                avatar_mime,
                xp,
                completed,
                quiz_completion,
                badges,
                activity_log,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                email = excluded.email,
                name = excluded.name,
                avatar_url = excluded.avatar_url,
                avatar_bytes = excluded.avatar_bytes,
                avatar_mime = excluded.avatar_mime,
                xp = excluded.xp,
                completed = excluded.completed,
                quiz_completion = excluded.quiz_completion,
                badges = excluded.badges,
                activity_log = excluded.activity_log,
                updated_at = excluded.updated_at
            """,
            (
                str(user_id),
                st.session_state.get("email"),
                st.session_state.get("student_name"),
                st.session_state.get("avatar_url"),
                avatar_blob,
                st.session_state.get("avatar_mime"),
                int(st.session_state.get("xp", 0)),
                json.dumps(st.session_state.get("completed", {}), ensure_ascii=False),
                json.dumps(st.session_state.get("quiz_completion", {}), ensure_ascii=False),
                json.dumps(st.session_state.get("badges", []), ensure_ascii=False),
                json.dumps(st.session_state.get("activity_log", []), ensure_ascii=False),
                datetime.utcnow().isoformat(),
            ),
        )


def refresh_level() -> None:
    st.session_state.level = 1 + (st.session_state.xp // XP_PER_LEVEL)


def log_activity(event: str) -> None:
    st.session_state.activity_log.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event,
        }
    )
    save_user_state()


def add_xp(amount: int, reason: str) -> None:
    st.session_state.xp += amount
    refresh_level()
    log_activity(f"+{amount} XP ({reason})")


def award_badge(badge_name: str) -> None:
    badges: list[str] = st.session_state.badges
    if badge_name not in badges:
        badges.append(badge_name)
        log_activity(f"Badge unlocked: {badge_name}")


def mark_lesson_completed(lesson_name: str) -> bool:
    completed = st.session_state.completed
    if completed.get(lesson_name):
        return False

    completed[lesson_name] = True
    save_user_state()
    return True


def mark_quiz_completed(lesson_name: str) -> bool:
    quiz_completion = st.session_state.quiz_completion
    if quiz_completion.get(lesson_name):
        return False

    quiz_completion[lesson_name] = True
    save_user_state()
    return True


def completed_lesson_count() -> tuple[int, int]:
    completed = sum(1 for is_complete in st.session_state.completed.values() if is_complete)
    return completed, len(LESSON_NAMES)


def completion_ratio() -> float:
    completed, total = completed_lesson_count()
    if total == 0:
        return 0.0
    return completed / total


def get_profile_name() -> str:
    name = st.session_state.get("student_name", "").strip()
    return name or "Guest"


def update_profile(name: str, avatar_bytes: bytes | None, avatar_mime: str | None) -> None:
    st.session_state.student_name = name or "User"
    if avatar_bytes is not None:
        st.session_state.avatar_bytes = avatar_bytes
        st.session_state.avatar_mime = avatar_mime
    save_user_state()


def navigate(page_key: str) -> None:
    st.session_state.active_page = page_key
    st.rerun()
