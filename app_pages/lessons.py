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

LESSON_BLUEPRINTS = [
    {
        "title": "Totals with SUM",
        "summary": "Build a clean expense table and calculate totals with SUM.",
        "formula_focus": "SUM",
        "topics": ["Structured data entry", "Cell references", "SUM basics"],
        "objectives": [
            "Enter consistent numeric values in one column.",
            "Use SUM to total a selected range.",
            "Check how SUM updates after an edit.",
        ],
        "practice_task": "Create Amount values in B2:B10 and return the total in B11.",
        "quiz": {
            "question": "Which formula totals B2 through B10?",
            "options": ["=SUM(B2:B10)", "=AVG(B2:B10)", "=COUNT(B2:B10)"],
            "answer": "=SUM(B2:B10)",
        },
        "steps": [
            {"task": "Create headers Category and Amount."},
            {"task": "Enter at least 8 rows of amount data."},
            {
                "task": "Add a SUM formula for the total.",
                "explain": "SUM adds every numeric value in the selected range.",
                "formulas": ["=SUM(B2:B10)"],
            },
        ],
    },
    {
        "title": "Averages with AVERAGE",
        "summary": "Compute a typical value and explain when average is useful.",
        "formula_focus": "AVERAGE",
        "topics": ["Descriptive statistics", "Range selection", "Data review"],
        "objectives": [
            "Calculate mean values from a numeric range.",
            "Compare average and total results.",
            "Interpret average in plain language.",
        ],
        "practice_task": "Enter scores in C2:C11 and calculate the mean in C12.",
        "quiz": {
            "question": "Which function returns the mean?",
            "options": ["AVERAGE", "MAX", "ROUND"],
            "answer": "AVERAGE",
        },
        "steps": [
            {"task": "Enter 10 numeric scores in column C."},
            {
                "task": "Compute the mean score.",
                "explain": "AVERAGE divides the total by the count of numbers.",
                "formulas": ["=AVERAGE(C2:C11)"],
            },
        ],
    },
    {
        "title": "Lowest Value with MIN",
        "summary": "Identify the smallest value in a dataset quickly.",
        "formula_focus": "MIN",
        "topics": ["Comparing values", "Range scanning", "Outlier checks"],
        "objectives": [
            "Use MIN to detect the smallest numeric value.",
            "Validate a result by checking source cells.",
            "Use MIN in quality checks.",
        ],
        "practice_task": "Enter monthly costs in D2:D9 and return the lowest value in D10.",
        "quiz": {
            "question": "What does MIN return?",
            "options": ["Largest value", "Smallest value", "Number of cells"],
            "answer": "Smallest value",
        },
        "steps": [
            {"task": "Enter cost values in D2:D9."},
            {
                "task": "Calculate the smallest value.",
                "explain": "MIN returns the lowest numeric value in the range.",
                "formulas": ["=MIN(D2:D9)"],
            },
        ],
    },
    {
        "title": "Highest Value with MAX",
        "summary": "Find top values for ranking and reporting.",
        "formula_focus": "MAX",
        "topics": ["Ranking", "Performance metrics", "Comparative analysis"],
        "objectives": [
            "Use MAX to identify highest performance values.",
            "Check whether your max value matches the table.",
            "Use MAX in summary sections.",
        ],
        "practice_task": "Enter weekly sales in E2:E9 and return the highest value in E10.",
        "quiz": {
            "question": "Which function finds the largest number?",
            "options": ["MAX", "MIN", "IF"],
            "answer": "MAX",
        },
        "steps": [
            {"task": "Enter numeric sales data in E2:E9."},
            {
                "task": "Calculate the top value.",
                "explain": "MAX returns the highest numeric value in the range.",
                "formulas": ["=MAX(E2:E9)"],
            },
        ],
    },
    {
        "title": "Counts with COUNT",
        "summary": "Count how many numeric values exist in a range.",
        "formula_focus": "COUNT",
        "topics": ["Completeness checks", "Numeric detection", "Data auditing"],
        "objectives": [
            "Use COUNT to count numeric cells only.",
            "Understand why text cells are ignored by COUNT.",
            "Use COUNT to check data completeness.",
        ],
        "practice_task": "Mix text and numbers in F2:F12 and count numeric entries in F13.",
        "quiz": {
            "question": "COUNT includes which cells?",
            "options": ["Only numeric cells", "All non-empty cells", "Only text cells"],
            "answer": "Only numeric cells",
        },
        "steps": [
            {"task": "Enter a mix of numbers and text in F2:F12."},
            {
                "task": "Count numeric values.",
                "explain": "COUNT ignores text and blank cells.",
                "formulas": ["=COUNT(F2:F12)"],
            },
        ],
    },
    {
        "title": "Precision with ROUND",
        "summary": "Format results for reporting by controlling decimal places.",
        "formula_focus": "ROUND",
        "topics": ["Decimal formatting", "Readable reports", "Numeric precision"],
        "objectives": [
            "Round decimal values to required precision.",
            "Compare raw and rounded values.",
            "Apply rounding to percentage or currency outputs.",
        ],
        "practice_task": "Place 12.9876 in G2 and round to 2 decimals in G3.",
        "quiz": {
            "question": "What does ROUND(value, 2) do?",
            "options": ["Rounds to 2 decimal places", "Rounds to nearest 10", "Counts decimals"],
            "answer": "Rounds to 2 decimal places",
        },
        "steps": [
            {"task": "Enter decimal values in G2:G6."},
            {
                "task": "Create a rounded result column.",
                "explain": "ROUND controls how many digits remain after the decimal.",
                "formulas": ["=ROUND(G2,2)"],
            },
        ],
    },
    {
        "title": "Decisions with IF",
        "summary": "Create pass/fail and threshold checks with IF logic.",
        "formula_focus": "IF",
        "topics": ["Logical tests", "Conditional outputs", "Rule-based labeling"],
        "objectives": [
            "Write IF tests with clear true/false outputs.",
            "Check threshold logic for simple grading.",
            "Use IF to label table rows.",
        ],
        "practice_task": "If H2 is 75, return Pass for >= 50, otherwise Fail.",
        "quiz": {
            "question": "Which function creates conditional outcomes?",
            "options": ["IF", "SUM", "CONCAT"],
            "answer": "IF",
        },
        "steps": [
            {"task": "Enter sample scores in H2:H10."},
            {
                "task": "Label each row Pass or Fail.",
                "explain": "IF evaluates one condition and returns one of two values.",
                "formulas": ['=IF(H2>=50,"Pass","Fail")'],
            },
        ],
    },
    {
        "title": "Multiple Conditions with AND",
        "summary": "Require multiple conditions to be true before approving a result.",
        "formula_focus": "AND",
        "topics": ["Compound logic", "Validation rules", "Boolean results"],
        "objectives": [
            "Combine two or more checks into one decision.",
            "Recognize when AND returns TRUE.",
            "Use AND inside IF formulas.",
        ],
        "practice_task": "Return Eligible if Score >= 60 and Attendance >= 80.",
        "quiz": {
            "question": "When does AND return TRUE?",
            "options": ["When all conditions are true", "When any condition is true", "When no condition is true"],
            "answer": "When all conditions are true",
        },
        "steps": [
            {"task": "Enter Score in I and Attendance in J columns."},
            {
                "task": "Create eligibility logic.",
                "explain": "AND returns TRUE only when every test is TRUE.",
                "formulas": ['=IF(AND(I2>=60,J2>=80),"Eligible","Not Eligible")'],
            },
        ],
    },
    {
        "title": "Alternative Conditions with OR",
        "summary": "Accept any of several valid conditions with OR.",
        "formula_focus": "OR",
        "topics": ["Fallback rules", "Flexible criteria", "Boolean logic"],
        "objectives": [
            "Build formulas where any valid condition passes.",
            "Compare OR behavior against AND.",
            "Apply OR to qualification rules.",
        ],
        "practice_task": "Mark Approved if either K2 >= 90 or L2 = \"Yes\".",
        "quiz": {
            "question": "OR returns TRUE when:",
            "options": ["Any condition is true", "All conditions are true", "No conditions are true"],
            "answer": "Any condition is true",
        },
        "steps": [
            {"task": "Create score and override columns in K and L."},
            {
                "task": "Apply OR-based approval logic.",
                "explain": "OR returns TRUE if at least one condition is TRUE.",
                "formulas": ['=IF(OR(K2>=90,L2="Yes"),"Approved","Review")'],
            },
        ],
    },
    {
        "title": "Reverse Logic with NOT",
        "summary": "Invert logic tests to simplify rejection and exception rules.",
        "formula_focus": "NOT",
        "topics": ["Logic inversion", "Exception checks", "Rule clarity"],
        "objectives": [
            "Flip logical outcomes with NOT.",
            "Use NOT for exception reporting.",
            "Combine NOT with IF for clear messaging.",
        ],
        "practice_task": "Return Missing if M2 is blank; otherwise return Entered.",
        "quiz": {
            "question": "What does NOT(TRUE) return?",
            "options": ["FALSE", "TRUE", "0"],
            "answer": "FALSE",
        },
        "steps": [
            {"task": "Prepare cells where some entries are blank."},
            {
                "task": "Label blank-value exceptions.",
                "explain": "NOT reverses TRUE/FALSE output from another logical test.",
                "formulas": ['=IF(NOT(M2<>""),"Missing","Entered")'],
            },
        ],
    },
    {
        "title": "Text Length with LEN",
        "summary": "Measure text length for data validation and formatting checks.",
        "formula_focus": "LEN",
        "topics": ["Text validation", "Character counts", "Input quality"],
        "objectives": [
            "Count characters in text entries.",
            "Validate ID length requirements.",
            "Use LEN in basic quality checks.",
        ],
        "practice_task": "Place an ID in N2 and return its character count in N3.",
        "quiz": {
            "question": "LEN(\"Excel\") returns:",
            "options": ["5", "4", "6"],
            "answer": "5",
        },
        "steps": [
            {"task": "Enter sample IDs or codes in N2:N8."},
            {
                "task": "Count character lengths.",
                "explain": "LEN returns the number of characters in a cell value.",
                "formulas": ["=LEN(N2)"],
            },
        ],
    },
    {
        "title": "Text Joining with CONCAT",
        "summary": "Combine multiple text fields into a final display string.",
        "formula_focus": "CONCAT",
        "topics": ["Text assembly", "Readable output", "Concatenation patterns"],
        "objectives": [
            "Join first and last name fields.",
            "Add separators for readability.",
            "Build labels from multiple source columns.",
        ],
        "practice_task": "Join first name in O2 and last name in P2 into Q2.",
        "quiz": {
            "question": "Which function combines text values?",
            "options": ["CONCAT", "COUNT", "MAX"],
            "answer": "CONCAT",
        },
        "steps": [
            {"task": "Enter first names in O and last names in P."},
            {
                "task": "Create a full-name output column.",
                "explain": "CONCAT joins text values in the order provided.",
                "formulas": ['=CONCAT(O2," ",P2)'],
            },
        ],
    },
]

LESSONS = {f"Lesson {index}": lesson for index, lesson in enumerate(LESSON_BLUEPRINTS, start=1)}


def _render_bullets(items: list[str]) -> None:
    bullet_text = "\n".join(f"- {item}" for item in items)
    st.markdown(bullet_text)


def _render_quiz(lesson_name: str, quiz: dict[str, object]) -> None:
    selection = st.radio(
        quiz["question"],
        quiz["options"],
        key=f"quiz_option_{lesson_name}",
    )

    if st.button("Submit Quiz", key=f"quiz_submit_{lesson_name}"):
        if selection == quiz["answer"]:
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


def _lesson_examples(steps: list[dict[str, object]]) -> list[str]:
    examples: list[str] = []
    seen: set[str] = set()

    for step in steps:
        formulas = step.get("formulas", [])
        if not isinstance(formulas, list):
            continue

        for formula in formulas:
            formula_text = str(formula).strip()
            if formula_text and formula_text not in seen:
                seen.add(formula_text)
                examples.append(formula_text)

    return examples


def render() -> None:
    st.title("Lessons")
    st.markdown(
        "Each lesson focuses on one core Excel function with a spreadsheet practice task and quiz."
    )

    tabs = st.tabs(list(LESSONS.keys()))

    for tab, lesson_name in zip(tabs, LESSONS, strict=False):
        lesson = LESSONS[lesson_name]

        with tab:
            st.header(f"{lesson_name}: {lesson['title']}")
            st.markdown(lesson["summary"])
            st.caption(f"Function focus: {lesson['formula_focus']}")

            st.subheader("Learning Objectives")
            _render_bullets(lesson["objectives"])

            st.subheader("Key Topics")
            _render_bullets(lesson["topics"])

            st.subheader("Spreadsheet Practice Task")
            st.markdown(lesson["practice_task"])
            st.caption("Use the embedded spreadsheet to complete this task and review suggestions.")

            st.subheader("Step-by-Step Practice")
            for index, step in enumerate(lesson["steps"], start=1):
                st.markdown(f"**Step {index}:** {step['task']}")
                explain = step.get("explain")
                if explain:
                    st.caption(f"Why this matters: {explain}")

                formulas = step.get("formulas", [])
                if formulas:
                    st.code("\n".join(str(formula) for formula in formulas), language="text")

            lesson_examples = _lesson_examples(lesson["steps"])
            lab_namespace = lesson_name.lower().replace(" ", "_")
            render_lab(
                namespace=f"{lab_namespace}_lab",
                show_title=False,
                show_new_sheet_button=False,
                lesson_context={
                    "name": lesson_name,
                    "title": lesson["title"],
                    "summary": lesson["summary"],
                    "objectives": lesson["objectives"],
                    "topics": lesson["topics"],
                    "lab_prompt": lesson["practice_task"],
                    "examples": lesson_examples[:8],
                },
            )

            st.subheader("Quiz")
            _render_quiz(lesson_name, lesson["quiz"])

            st.subheader("Completion")
            _render_completion(lesson_name)


def formula_coverage() -> tuple[int, int]:
    # Helper used for quick curriculum sanity checks.
    formulas = {str(lesson.get("formula_focus", "")).strip().upper() for lesson in LESSONS.values()}
    formulas.discard("")

    return len(formulas), len(LESSONS)
