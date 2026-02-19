from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from app_core import (
    PROFILE_PIC_FILE,
    XP_PER_LEVEL,
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
        uploaded_picture = st.file_uploader("Profile Picture", type=["png", "jpg", "jpeg"])
        save_clicked = st.form_submit_button("Save Profile")

    if save_clicked:
        normalized_name = new_name.strip() or "Guest"

        profile_data["name"] = normalized_name
        st.session_state.student_name = normalized_name

        if uploaded_picture is not None:
            image = Image.open(uploaded_picture)
            image.save(PROFILE_PIC_FILE)
            profile_data["profile_pic"] = str(PROFILE_PIC_FILE)

        st.session_state.profile_data = profile_data
        save_profile_data(profile_data)
        log_activity("Profile updated")
        st.success("Profile saved.")

    saved_picture = profile_data.get("profile_pic")
    if saved_picture and Path(saved_picture).exists():
        st.image(saved_picture, width=160, caption="Current Profile Picture")


def _render_progress_summary() -> None:
    st.subheader("Progress Summary")

    xp = st.session_state.xp
    level = st.session_state.level

    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("XP", xp)
    metric_col_2.metric("Level", level)

    xp_in_level = xp % XP_PER_LEVEL
    progress_value = xp_in_level / XP_PER_LEVEL
    st.progress(progress_value)
    st.caption(f"{XP_PER_LEVEL - xp_in_level} XP to next level")

    completion_rows = []
    for lesson_name, is_complete in st.session_state.completed.items():
        completion_rows.append({"Lesson": lesson_name, "Completed": "Yes" if is_complete else "No"})

    st.dataframe(pd.DataFrame(completion_rows), use_container_width=True, hide_index=True)


def _render_badges_and_log() -> None:
    badges = st.session_state.badges
    activity_log = st.session_state.activity_log

    st.subheader("Badges")
    if badges:
        for badge in badges:
            st.write(f"- 🏅 {badge}")
    else:
        st.info("No badges earned yet.")

    st.subheader("Activity Log")
    if activity_log:
        log_df = pd.DataFrame(activity_log[::-1])
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No activity recorded yet.")


def render() -> None:
    st.title("👤 Profile & Progress")
    st.caption(f"Student: {get_profile_name()}")

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        _render_profile_editor()

    with right_col:
        _render_progress_summary()

    _render_badges_and_log()

    if st.button("Go to Lessons", use_container_width=True):
        navigate("lessons")
