from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from app_core import get_profile_name, init_app_state, navigate
from app_pages import chatbot, home, lessons, profile_progress, spreadsheet_lab


@dataclass(frozen=True)
class PageConfig:
    key: str
    title: str
    render: Callable[[], None]


PRIMARY_PAGES = [
    PageConfig("home", "Home", home.render),
    PageConfig("lessons", "Lessons", lessons.render),
    PageConfig("spreadsheet", "Spreadsheet Lab", spreadsheet_lab.render),
    PageConfig("chatbot", "AI Tutor", chatbot.render),
]

PROFILE_PAGE = PageConfig("profile", "Profile and Progress", profile_progress.render)

ALL_PAGES = {page.key: page for page in [*PRIMARY_PAGES, PROFILE_PAGE]}


st.set_page_config(page_title="ExcelWars", layout="wide")
init_app_state()

if "active_page" not in st.session_state:
    st.session_state.active_page = "home"

if st.session_state.active_page not in ALL_PAGES:
    st.session_state.active_page = "home"


def render_sidebar(active_page: str) -> None:
    st.sidebar.title("ExcelWars")
    st.sidebar.caption(f"Signed in as: {get_profile_name()}")

    for page in PRIMARY_PAGES:
        button_type = "primary" if page.key == active_page else "secondary"
        if st.sidebar.button(
            page.title,
            key=f"nav_{page.key}",
            use_container_width=True,
            type=button_type,
        ):
            navigate(page.key)

    st.sidebar.markdown("---")
    st.sidebar.caption("Quick Access")

    profile_button_type = "primary" if PROFILE_PAGE.key == active_page else "secondary"
    if st.sidebar.button(
        PROFILE_PAGE.title,
        key="nav_profile_quick",
        use_container_width=True,
        type=profile_button_type,
    ):
        navigate(PROFILE_PAGE.key)


active_page_key = st.session_state.active_page
render_sidebar(active_page_key)
ALL_PAGES[active_page_key].render()
