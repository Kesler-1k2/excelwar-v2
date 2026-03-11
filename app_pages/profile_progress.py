from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from app_core import (
    XP_PER_LEVEL,
    completed_lesson_count,
    completion_ratio,
    get_profile_name,
    log_activity,
    navigate,
    update_profile,
)


def _build_circular_avatar_bytes(image_bytes: bytes, size: int = 220) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    cropped = image.crop((left, top, left + edge, top + edge)).resize((size, size))

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size - 1, size - 1), fill=255)

    avatar = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    avatar.paste(cropped, (0, 0), mask)
    return avatar


def _render_profile_card() -> None:
    st.subheader("Student Profile")

    avatar_bytes = st.session_state.get("avatar_bytes")
    avatar_url = st.session_state.get("avatar_url")

    if avatar_bytes:
        st.image(_build_circular_avatar_bytes(avatar_bytes), width=180)
    elif avatar_url:
        st.image(avatar_url, width=180)
    else:
        st.info("No profile picture yet.")

    st.markdown(f"### {st.session_state.get('student_name', 'Guest')}")
    email = st.session_state.get("email")
    if email:
        st.caption(email)


def _render_profile_editor_dropdown() -> None:
    with st.expander("Edit Profile", expanded=False):
        current_name = st.session_state.get("student_name", "")
        email = st.session_state.get("email", "")

        with st.form("profile_form"):
            new_name = st.text_input("Name", value=current_name)
            st.text_input("Email", value=email, disabled=True)
            uploaded_picture = st.file_uploader("Profile Picture", type=["png", "jpg", "jpeg"])
            save_clicked = st.form_submit_button("Save Profile")

        if save_clicked:
            normalized_name = new_name.strip() or "Guest"
            avatar_bytes = None
            avatar_mime = None

            if uploaded_picture is not None:
                avatar_bytes = uploaded_picture.getvalue()
                avatar_mime = uploaded_picture.type
                st.session_state.avatar_url = None

            update_profile(normalized_name, avatar_bytes, avatar_mime)
            log_activity("Profile updated")
            st.success("Profile saved.")
            st.rerun()


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
        _render_profile_card()
        _render_profile_editor_dropdown()

    with right_col:
        _render_progress_summary()

    _render_badges_and_log()

    if st.button("Go to Lessons", use_container_width=True):
        navigate("lessons")
