import streamlit as st

from app_core import get_profile_name, navigate


def render() -> None:
    st.title("🏠 ExcelWars")
    st.subheader(f"Welcome back, {get_profile_name()}!")
    st.caption("The perfect place to excel, in Excel.")

    lesson_col, spreadsheet_col, profile_col = st.columns(3, gap="large")

    with lesson_col.container(border=True, height=250):
        st.markdown("### 📚 Lessons")
        st.write("Study lesson content and complete quizzes.")
        if st.button("Open Lessons", key="home_lessons", use_container_width=True):
            navigate("lessons")

    with spreadsheet_col.container(border=True, height=250):
        st.markdown("### 🧮 Spreadsheet Lab")
        st.write("Use a single-grid Excel-style editor with formulas.")
        if st.button("Open Spreadsheet", key="home_spreadsheet", use_container_width=True):
            navigate("spreadsheet")

    with profile_col.container(border=True, height=250):
        st.markdown("### 👤 Profile & Progress")
        st.write("Manage your profile and track XP, levels, and completion.")
        if st.button("Open Profile", key="home_profile", use_container_width=True):
            navigate("profile")
