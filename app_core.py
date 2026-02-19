from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

PROFILE_DATA_FILE = Path("profile_data.json")
PROFILE_PIC_FILE = Path("profile_pic.png")

LESSON_NAMES = ("Lesson 1", "Lesson 2", "Lesson 3")
XP_PER_LEVEL = 150
XP_PER_LESSON = 50
XP_PER_QUIZ = 20


def _default_profile_data() -> dict[str, Any]:
    return {"name": "", "profile_pic": None}


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

    return {
        "name": data.get("name", "") or "",
        "profile_pic": data.get("profile_pic"),
    }


def save_profile_data(profile_data: dict[str, Any]) -> None:
    with PROFILE_DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(profile_data, file, ensure_ascii=False, indent=2)


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


def get_profile_name() -> str:
    name = st.session_state.get("student_name", "").strip()
    return name or "User"


def navigate(page_key: str) -> None:
    st.session_state.active_page = page_key
    st.rerun()
