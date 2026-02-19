import streamlit as st

from app_core import (
    XP_PER_LESSON,
    XP_PER_QUIZ,
    add_xp,
    award_badge,
    mark_lesson_completed,
    mark_quiz_completed,
)

LESSONS = {
    "Lesson 1": {
        "title": "Basics of Excel",
        "summary": "Understand rows, columns, cells, and basic navigation.",
        "topics": ["Workbook structure", "Entering data", "Simple formatting"],
        "quiz": {
            "question": "Which function adds values together?",
            "options": ["SUM", "IF", "LEN"],
            "answer": "SUM",
        },
    },
    "Lesson 2": {
        "title": "Cell Formatting and Organization",
        "summary": "Present data clearly with formatting, sorting, and filtering.",
        "topics": ["Number formats", "Cell styling", "Sorting and filtering"],
        "quiz": {
            "question": "Filtering does what?",
            "options": ["Deletes data", "Shows selected data", "Changes numbers"],
            "answer": "Shows selected data",
        },
    },
    "Lesson 3": {
        "title": "Formulas, Functions, and Charts",
        "summary": "Use common formulas and visualize results with charts.",
        "topics": ["SUM/AVERAGE/MIN/MAX", "Cell references", "Basic chart types"],
        "quiz": {
            "question": "Which chart is best for trends?",
            "options": ["Line", "Pie", "Bar"],
            "answer": "Line",
        },
    },
}


def _render_quiz(lesson_name: str, quiz: dict[str, object]) -> None:
    question = quiz["question"]
    options = quiz["options"]
    answer = quiz["answer"]

    selection = st.radio(
        question,
        options,
        key=f"quiz_option_{lesson_name}",
    )

    if st.button("Submit Quiz", key=f"quiz_submit_{lesson_name}"):
        if selection == answer:
            if mark_quiz_completed(lesson_name):
                add_xp(XP_PER_QUIZ, f"{lesson_name} quiz")
                award_badge("Quiz Master")
                st.success("Correct. XP awarded.")
            else:
                st.success("Correct. Quiz already completed earlier.")
        else:
            st.error("Not quite. Try again.")


def _render_completion(lesson_name: str) -> None:
    if st.session_state.completed.get(lesson_name):
        st.success("Lesson completed.")
    else:
        st.info("Lesson not completed yet.")

    if st.button("Mark Lesson Complete", key=f"lesson_complete_{lesson_name}"):
        if mark_lesson_completed(lesson_name):
            add_xp(XP_PER_LESSON, f"{lesson_name} completion")
            award_badge("Lesson Finisher")
            st.success("Lesson marked complete. XP awarded.")
        else:
            st.info("This lesson is already complete.")


def render() -> None:
    st.title("📚 Lessons")
    st.write("Work through each lesson and complete its quiz.")

    tabs = st.tabs(list(LESSONS.keys()))

    for tab, lesson_name in zip(tabs, LESSONS, strict=False):
        lesson = LESSONS[lesson_name]

        with tab:
            st.header(f"{lesson_name}: {lesson['title']}")
            st.write(lesson["summary"])

            st.write("Key topics:")
            for topic in lesson["topics"]:
                st.write(f"- {topic}")

            st.subheader("Quiz")
            _render_quiz(lesson_name, lesson["quiz"])

            st.subheader("Completion")
            _render_completion(lesson_name)
