import streamlit as st

LESSON_CONTENT = {
    "Lesson 1": {
        "title": "Introduction to Excel and Basic Functions",
        "summary": (
            "Students learn the purpose of Excel and navigate the interface. "
            "They practice entering and formatting data and use formulas such as "
            "SUM, AVERAGE, MIN, and MAX."
        ),
        "topics": [
            "Excel interface (rows, columns, cells)",
            "Entering and formatting data",
            "Basic formulas",
        ],
    },
    "Lesson 2": {
        "title": "Data Organization and Formatting Tools",
        "summary": (
            "Students organize data using sorting and filtering, apply conditional formatting, "
            "and understand relative versus absolute cell references."
        ),
        "topics": [
            "Sorting and filtering",
            "Conditional formatting",
            "Cell references",
        ],
    },
    "Lesson 3": {
        "title": "Charts and Practical Application",
        "summary": (
            "Students create and customize charts, then complete a small project to organize, "
            "analyze, and present data."
        ),
        "topics": [
            "Creating charts",
            "Chart customization",
            "Real-world project",
        ],
    },
}


def render() -> None:
    st.title("🗂 Lesson Plan")
    st.write("Overview and summaries for the three Excel lessons.")

    tabs = st.tabs(list(LESSON_CONTENT.keys()))

    for tab, lesson_name in zip(tabs, LESSON_CONTENT, strict=False):
        lesson = LESSON_CONTENT[lesson_name]
        with tab:
            st.header(f"{lesson_name}: {lesson['title']}")
            st.subheader("Summary")
            st.write(lesson["summary"])
            st.subheader("Key Topics")
            for topic in lesson["topics"]:
                st.write(f"- {topic}")
