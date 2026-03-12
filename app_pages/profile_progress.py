import pandas as pd
import streamlit as st

from app_core import (
    XP_PER_LEVEL,
    completed_lesson_count,
    completion_ratio,
    get_profile_name,
    log_activity,
    navigate,
    save_profile_data,
)


def _render_profile_editor() -> None:
    st.subheader("Student Profile")

    profile_data = st.session_state.profile_data
    current_name = st.session_state.get("student_name", "")

    with st.form("profile_form"):
        new_name = st.text_input("Name", value=current_name)
        save_clicked = st.form_submit_button("Save Profile")

    if save_clicked:
        normalized_name = new_name.strip() or "Guest"
        profile_data["name"] = normalized_name
        st.session_state.student_name = normalized_name
        st.session_state.profile_data = profile_data
        save_profile_data(profile_data)
        log_activity("Profile updated")
        st.success("Profile saved.")


def _render_progress_summary() -> None:
    st.subheader("Progress Summary")

    xp = st.session_state.xp
    level = st.session_state.level
    completed, total = completed_lesson_count()

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("XP", xp)
    metric_col_2.metric("Level", level)
    metric_col_3.metric("Lessons", f"{completed}/{total}")

    xp_in_level = xp % XP_PER_LEVEL
    st.progress(xp_in_level / XP_PER_LEVEL)
    st.caption(f"{XP_PER_LEVEL - xp_in_level} XP to next level")

    st.progress(completion_ratio())
    st.caption("Overall lesson completion")

    completion_rows = [
        {"Lesson": lesson_name, "Completed": "Yes" if is_complete else "No"}
        for lesson_name, is_complete in st.session_state.completed.items()
    ]
    st.dataframe(pd.DataFrame(completion_rows), use_container_width=True, hide_index=True)


def _render_badges_and_log() -> None:
    badges = st.session_state.badges
    activity_log = st.session_state.activity_log

    st.subheader("Badges")
    if badges:
        for badge in badges:
            st.write(f"- {badge}")
    else:
        st.info("No badges earned yet.")

    st.subheader("Activity Log")
    if activity_log:
        log_df = pd.DataFrame(activity_log[::-1])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No activity recorded yet.")


def render() -> None:
    st.title("Profile and Progress")
    st.caption(f"Student: {get_profile_name()}")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        _render_profile_editor()

    with right_col:
        _render_progress_summary()

    _render_badges_and_log()

    if st.button("Go to Lessons", use_container_width=True):
        navigate("lessons")
