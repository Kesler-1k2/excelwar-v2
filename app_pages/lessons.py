import streamlit as st

from app_core import (
    XP_PER_LESSON,
    XP_PER_QUIZ,
    add_xp,
    award_badge,
    mark_lesson_completed,
    mark_quiz_completed,
)
from app_pages.spreadsheet_lab import render_lab

LESSONS = {
    "Lesson 1": {
        "title": "Basics of Excel",
        "summary": (
            "Build fluency with the worksheet grid, clean data entry habits, and first formulas. "
            "By the end, you can structure a small table and compute totals safely."
        ),
        "objectives": [
            "Identify rows, columns, and cell addresses without guessing.",
            "Enter consistent numeric/text data and avoid mixed formats.",
            "Use core formulas with confidence: SUM, AVERAGE, MIN, and MAX.",
        ],
        "mini_plan": [
            "Warm-up: map cell addresses and movement shortcuts.",
            "Guided practice: build a simple weekly expense table.",
            "Application: calculate totals, average spend, and highest/lowest values.",
        ],
        "topics": ["Workbook structure", "Entering data", "Simple formatting", "Core formulas"],
        "lab_prompt": (
            "Create columns for Category and Amount. Enter at least six expense rows, then use formulas "
            "to compute total, average, minimum, and maximum in summary cells."
        ),
        "coach_questions": [
            "Which value is your maximum spend, and in which category does it appear?",
            "What changed when you edited one Amount cell and recalculated?",
            "If a value is blank, how does your summary area respond?",
        ],
        "steps": [
            {
                "task": "Create headers `Category` and `Amount` in row 1.",
                "explain": "This gives your data a clear structure and makes formulas easier to read.",
            },
            {
                "task": "Enter at least 6 expense rows under those headers.",
                "explain": "A larger sample helps you practice meaningful totals and averages.",
            },
            {
                "task": "Add `SUM`, `AVERAGE`, `MIN`, and `MAX` formulas in summary cells.",
                "explain": "These are foundational functions you will reuse in every later lesson.",
                "formulas": [
                    "=SUM(B2:B7)",
                    "=AVERAGE(B2:B7)",
                    "=MIN(B2:B7)",
                    "=MAX(B2:B7)",
                ],
            },
            {
                "task": "Edit one amount value and observe formula updates.",
                "explain": "You learn that formulas are dynamic and tied to cell references.",
            },
            {
                "task": "Ask Gemini: `What changed?` and `Explain my formulas.`",
                "explain": "Gemini confirms exact edits and reinforces the concept behind each formula.",
            },
            {
                "task": "Submit the quiz and mark the lesson complete.",
                "explain": "This checks understanding and records XP progress.",
            },
        ],
        "quiz": {
            "question": "Which function adds values together?",
            "options": ["SUM", "IF", "LEN"],
            "answer": "SUM",
        },
    },
    "Lesson 2": {
        "title": "Cell Formatting and Organization",
        "summary": (
            "Turn raw tables into readable, decision-ready sheets. Focus on formatting intent, "
            "table organization, and trustworthy sorting/filtering habits."
        ),
        "objectives": [
            "Apply number and date formats that match the data type.",
            "Use visual hierarchy so headers and key fields are immediately clear.",
            "Organize records with sorting/filtering while preserving data integrity.",
        ],
        "mini_plan": [
            "Warm-up: compare poor vs strong spreadsheet formatting decisions.",
            "Guided practice: standardize a mixed-format sales table.",
            "Application: sort and filter to answer business questions quickly.",
        ],
        "topics": ["Number formats", "Cell styling", "Sorting and filtering", "Data hygiene"],
        "lab_prompt": (
            "Build a product sales table with Product, Region, Units, and Revenue. Format units as whole numbers "
            "and revenue as currency. Sort by Revenue and identify the top performer."
        ),
        "coach_questions": [
            "Which format choices improved readability the most?",
            "What is the risk of sorting only one column instead of the full table?",
            "How would you explain your filter logic to someone reviewing your file?",
        ],
        "steps": [
            {
                "task": "Create columns: `Product`, `Region`, `Units`, `Revenue`.",
                "explain": "This mirrors a realistic business dataset with mixed field types.",
            },
            {
                "task": "Enter sample rows and format `Units` as whole numbers and `Revenue` as currency.",
                "explain": "Correct formats improve accuracy and interpretation.",
            },
            {
                "task": "Style headers clearly (bold/alignment/spacing).",
                "explain": "Visual hierarchy makes your sheet easier to scan and review.",
            },
            {
                "task": "Sort by `Revenue` and identify top-performing rows.",
                "explain": "Sorting turns raw records into ranked insights.",
                "formulas": [
                    "=SUM(C2:C10)",
                    "=SUM(D2:D10)",
                    "=AVERAGE(D2:D10)",
                    "=MAX(D2:D10)",
                ],
            },
            {
                "task": "Ask Gemini how to improve your table clarity and what changed after edits.",
                "explain": "You get coaching on formatting and organization while verifying sheet changes.",
            },
            {
                "task": "Submit quiz and complete lesson.",
                "explain": "You validate the skill and lock in progress.",
            },
        ],
        "quiz": {
            "question": "Filtering does what?",
            "options": ["Deletes data", "Shows selected data", "Changes numbers"],
            "answer": "Shows selected data",
        },
    },
    "Lesson 3": {
        "title": "Formulas, Functions, and Charts",
        "summary": (
            "Connect calculations to storytelling. Build formula-driven analysis and prepare data "
            "that can be visualized into clear trends and comparisons."
        ),
        "objectives": [
            "Combine formulas to produce reusable analysis blocks.",
            "Use references intentionally so updates flow through the model.",
            "Prepare chart-ready ranges and defend which chart fits the question.",
        ],
        "mini_plan": [
            "Warm-up: evaluate chart choice based on question type.",
            "Guided practice: compute trend-supporting metrics from monthly data.",
            "Application: create a concise analysis section ready for charting.",
        ],
        "topics": ["SUM/AVERAGE/MIN/MAX", "Cell references", "Chart-ready data prep", "Insight framing"],
        "lab_prompt": (
            "Create monthly performance data (at least 8 periods), compute rolling summary metrics, "
            "and write 2-3 bullet insights that would pair with a line chart."
        ),
        "coach_questions": [
            "Which metric best captures trend direction over time?",
            "What changed in your summary after editing one monthly value?",
            "Why is a line chart better here than a pie chart?",
        ],
        "steps": [
            {
                "task": "Create monthly data for at least 8 periods (`Month`, `Value`).",
                "explain": "This builds a trend-ready dataset for analysis.",
            },
            {
                "task": "Add summary formulas (total, average, and other useful metrics).",
                "explain": "You convert raw values into interpretable performance indicators.",
                "formulas": [
                    "=SUM(B2:B9)",
                    "=AVERAGE(B2:B9)",
                    "=MIN(B2:B9)",
                    "=MAX(B2:B9)",
                ],
            },
            {
                "task": "Write 2-3 insight bullets based on your calculations.",
                "explain": "This practices communicating results, not only computing them.",
            },
            {
                "task": "Change one monthly value and observe how metrics shift.",
                "explain": "You see how source edits propagate through analytical outputs.",
                "formulas": [
                    "=B9-B8",
                    "=ROUND((B9-B8)/B8*100,2)",
                ],
            },
            {
                "task": "Ask Gemini to guide next improvements and explain chart choice.",
                "explain": "You align spreadsheet work with lesson goals on trend storytelling.",
            },
            {
                "task": "Submit quiz and mark lesson complete.",
                "explain": "You verify mastery and update your lesson progress.",
            },
        ],
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
    st.write("Each lesson now includes content, guided practice, and an embedded Spreadsheet + Gemini lab.")

    tabs = st.tabs(list(LESSONS.keys()))

    for tab, lesson_name in zip(tabs, LESSONS, strict=False):
        lesson = LESSONS[lesson_name]

        with tab:
            st.header(f"{lesson_name}: {lesson['title']}")
            st.write(lesson["summary"])

            st.subheader("Learning Objectives")
            for objective in lesson["objectives"]:
                st.write(f"- {objective}")

            st.subheader("Lesson Flow")
            for step in lesson["mini_plan"]:
                st.write(f"- {step}")

            st.subheader("Key Topics")
            for topic in lesson["topics"]:
                st.write(f"- {topic}")

            st.subheader("Guided Lab Task")
            st.write(lesson["lab_prompt"])
            st.caption("Use the embedded lab below. Gemini on the right can explain edits and formula outcomes.")

            st.subheader("Step-by-Step Instructions")
            for index, step in enumerate(lesson["steps"], start=1):
                st.markdown(f"**Step {index}:** {step['task']}")
                st.caption(f"What this does: {step['explain']}")
                formulas = step.get("formulas", [])
                if formulas:
                    st.code("\n".join(formulas), language="text")

            lab_namespace = lesson_name.lower().replace(" ", "_")
            render_lab(
                namespace=f"{lab_namespace}_lab",
                show_title=False,
                lesson_context={
                    "name": lesson_name,
                    "title": lesson["title"],
                    "summary": lesson["summary"],
                    "objectives": lesson["objectives"],
                    "topics": lesson["topics"],
                    "lab_prompt": lesson["lab_prompt"],
                },
            )

            st.subheader("Reflection Prompts")
            for prompt in lesson["coach_questions"]:
                st.write(f"- {prompt}")

            st.subheader("Quiz")
            _render_quiz(lesson_name, lesson["quiz"])

            st.subheader("Completion")
            _render_completion(lesson_name)
