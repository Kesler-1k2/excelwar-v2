from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROFILE_DATA_FILE = Path("profile_data.json")

# Keep lesson keys in one place for profile/progress tracking.
LESSON_NAMES = tuple(f"Lesson {index}" for index in range(1, 13))
XP_PER_LEVEL = 150
XP_PER_LESSON = 50
XP_PER_QUIZ = 20


def _default_profile_data() -> dict[str, Any]:
    return {"name": ""}


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


def _ensure_lesson_keys() -> None:
    completed = st.session_state.get("completed")
    if not isinstance(completed, dict):
        completed = {}

    quiz_completion = st.session_state.get("quiz_completion")
    if not isinstance(quiz_completion, dict):
        quiz_completion = {}

    for lesson in LESSON_NAMES:
        completed.setdefault(lesson, False)
        quiz_completion.setdefault(lesson, False)

    st.session_state.completed = completed
    st.session_state.quiz_completion = quiz_completion


def load_profile_data() -> dict[str, Any]:
    if not PROFILE_DATA_FILE.exists():
        return _default_profile_data()

    try:
        with PROFILE_DATA_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return _default_profile_data()

    if not isinstance(data, dict):
        return _default_profile_data()

    return {"name": str(data.get("name", "") or "")}


def save_profile_data(profile_data: dict[str, Any]) -> None:
    safe_data = {"name": str(profile_data.get("name", "") or "")}
    with PROFILE_DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(safe_data, file, ensure_ascii=False, indent=2)


def init_app_state() -> None:
    defaults = _default_learning_state()

    for key, value in defaults.items():
        if key not in st.session_state:
            if isinstance(value, dict):
                st.session_state[key] = value.copy()
            elif isinstance(value, list):
                st.session_state[key] = value[:]
            else:
                st.session_state[key] = value

    _ensure_lesson_keys()

    if "profile_data" not in st.session_state:
        profile_data = load_profile_data()
        st.session_state.profile_data = profile_data

        saved_name = profile_data.get("name", "").strip()
        if saved_name:
            st.session_state.student_name = saved_name


def refresh_level() -> None:
    st.session_state.level = 1 + (st.session_state.xp // XP_PER_LEVEL)


def log_activity(event: str) -> None:
    st.session_state.activity_log.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "event": event,
        }
    )


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
    return True


def mark_quiz_completed(lesson_name: str) -> bool:
    quiz_completion = st.session_state.quiz_completion
    if quiz_completion.get(lesson_name):
        return False

    quiz_completion[lesson_name] = True
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


def navigate(page_key: str) -> None:
    st.session_state.active_page = page_key
    st.rerun()
