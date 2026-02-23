import streamlit as st

from app_core import get_profile_name, navigate


def render() -> None:
    st.title("ExcelWars")
    st.subheader(f"Welcome back, {get_profile_name()}!")
    st.caption("Learn spreadsheet skills with lessons, practice tasks, and AI support.")

    lesson_col, spreadsheet_col, profile_col = st.columns(3, gap="large")

    with lesson_col.container(border=True, height=250):
        st.markdown("### Lessons")
        st.write("Open lessons, follow guided steps, and submit quiz checks.")
        if st.button("Open Lessons", key="home_lessons", use_container_width=True):
            navigate("lessons")

    with spreadsheet_col.container(border=True, height=250):
        st.markdown("### Spreadsheet Lab")
        st.write("Use the spreadsheet editor, test formulas, and ask for suggestions.")
        if st.button("Open Spreadsheet", key="home_spreadsheet", use_container_width=True):
            navigate("spreadsheet")

    with profile_col.container(border=True, height=250):
        st.markdown("### Profile and Progress")
        st.write("Update your name and track lesson completion, XP, and badges.")
        if st.button("Open Profile", key="home_profile", use_container_width=True):
            navigate("profile")
