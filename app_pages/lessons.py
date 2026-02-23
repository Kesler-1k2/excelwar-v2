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
        "title": "Totals and Counting",
        "summary": "Build a clean table and compute totals and numeric counts.",
        "formula_focus": ["SUM", "COUNT"],
        "topics": ["Structured data entry", "Totals", "Numeric counting"],
        "objectives": [
            "Use SUM for range totals.",
            "Use COUNT for numeric-cell counts.",
            "Validate totals after editing values.",
        ],
        "practice_task": "Fill B2:B10 with numbers, then calculate total in B11 and count in B12.",
        "quiz": {
            "question": "Which formula returns a total?",
            "options": ["=SUM(B2:B10)", "=COUNT(B2:B10)", "=LEN(B2)"],
            "answer": "=SUM(B2:B10)",
        },
        "steps": [
            {"task": "Enter at least 8 numeric values in B2:B10."},
            {"task": "Calculate a total in B11.", "formulas": ["=SUM(B2:B10)"]},
            {"task": "Count numeric cells in B12.", "formulas": ["=COUNT(B2:B10)"]},
        ],
    },
    {
        "title": "Extremes with MIN and MAX",
        "summary": "Find the smallest and largest values in a dataset.",
        "formula_focus": ["MIN", "MAX"],
        "topics": ["Range scanning", "Extremes", "Quality checks"],
        "objectives": [
            "Return the lowest value with MIN.",
            "Return the highest value with MAX.",
            "Use both to build summary boxes.",
        ],
        "practice_task": "Enter costs in C2:C11 and return MIN in C12 and MAX in C13.",
        "quiz": {
            "question": "Which function returns the highest value?",
            "options": ["MIN", "MAX", "MEDIAN"],
            "answer": "MAX",
        },
        "steps": [
            {"task": "Enter numeric values in C2:C11."},
            {"task": "Calculate minimum and maximum values.", "formulas": ["=MIN(C2:C11)", "=MAX(C2:C11)"]},
        ],
    },
    {
        "title": "Averages and Median",
        "summary": "Compare mean and median to understand data center and spread.",
        "formula_focus": ["AVERAGE", "MEDIAN"],
        "topics": ["Central tendency", "Distribution checks", "Interpretation"],
        "objectives": [
            "Calculate average values with AVERAGE.",
            "Calculate midpoint values with MEDIAN.",
            "Compare how outliers affect both metrics.",
        ],
        "practice_task": "Enter scores in D2:D12 and calculate AVERAGE in D13 and MEDIAN in D14.",
        "quiz": {
            "question": "Which function returns the middle value?",
            "options": ["AVERAGE", "MEDIAN", "ROUND"],
            "answer": "MEDIAN",
        },
        "steps": [
            {"task": "Enter at least 10 score values in D2:D12."},
            {"task": "Calculate mean and median results.", "formulas": ["=AVERAGE(D2:D12)", "=MEDIAN(D2:D12)"]},
        ],
    },
    {
        "title": "Precision and Number Transformations",
        "summary": "Round values and apply exponent and root operations.",
        "formula_focus": ["ROUND", "POWER", "SQRT"],
        "topics": ["Decimal precision", "Exponents", "Square roots"],
        "objectives": [
            "Round decimal outputs for reporting.",
            "Raise values with POWER.",
            "Compute square roots with SQRT.",
        ],
        "practice_task": "Use E2=12.9876 and E3=25. Compute rounded, power, and square-root outputs.",
        "quiz": {
            "question": "Which formula rounds to two decimals?",
            "options": ["=ROUND(E2,2)", "=POWER(E2,2)", "=SQRT(E2)"],
            "answer": "=ROUND(E2,2)",
        },
        "steps": [
            {"task": "Enter decimals and whole numbers in E2:E6."},
            {"task": "Build transformed outputs.", "formulas": ["=ROUND(E2,2)", "=POWER(E3,2)", "=SQRT(E3)"]},
        ],
    },
    {
        "title": "Counting Data States",
        "summary": "Count non-empty cells, blank cells, and condition-matching rows.",
        "formula_focus": ["COUNTA", "COUNTBLANK", "COUNTIF"],
        "topics": ["Data completeness", "Blank checks", "Conditional counts"],
        "objectives": [
            "Count non-empty entries with COUNTA.",
            "Count blanks with COUNTBLANK.",
            "Count matching values with COUNTIF.",
        ],
        "practice_task": "Use F2:F15 with mixed values and blanks. Calculate COUNTA, COUNTBLANK, and COUNTIF.",
        "quiz": {
            "question": "Which function counts blank cells?",
            "options": ["COUNTA", "COUNTBLANK", "COUNTIF"],
            "answer": "COUNTBLANK",
        },
        "steps": [
            {"task": "Create mixed text/number/blank data in F2:F15."},
            {
                "task": "Count non-empty, blank, and conditional values.",
                "formulas": ['=COUNTA(F2:F15)', '=COUNTBLANK(F2:F15)', '=COUNTIF(F2:F15,">=50")'],
            },
        ],
    },
    {
        "title": "Conditional Aggregation",
        "summary": "Aggregate only rows that match criteria.",
        "formula_focus": ["SUMIF", "AVERAGEIF"],
        "topics": ["Criteria filters", "Conditional totals", "Conditional means"],
        "objectives": [
            "Use SUMIF to total matching rows.",
            "Use AVERAGEIF to average matching rows.",
            "Write criteria with relational operators.",
        ],
        "practice_task": "Use category in G2:G12 and values in H2:H12. Aggregate values where category is \"Food\".",
        "quiz": {
            "question": "Which formula returns conditional totals?",
            "options": ["=SUMIF(G2:G12,\"Food\",H2:H12)", "=SUM(H2:H12)", "=COUNTIF(G2:G12,\"Food\")"],
            "answer": '=SUMIF(G2:G12,"Food",H2:H12)',
        },
        "steps": [
            {"task": "Enter categories in G and amounts in H."},
            {
                "task": "Calculate conditional totals and averages.",
                "formulas": ['=SUMIF(G2:G12,"Food",H2:H12)', '=AVERAGEIF(H2:H12,">=50")'],
            },
        ],
    },
    {
        "title": "IF Logic with Relational Operators",
        "summary": "Build rule-based outputs using IF with operator comparisons.",
        "formula_focus": ["IF"],
        "operator_focus": [">", ">=", "<", "<=", "=", "<>"],
        "topics": ["Decision logic", "Comparisons", "Rule labeling"],
        "objectives": [
            "Use IF for true/false outcomes.",
            "Apply relational operators in tests.",
            "Use <> for not-equal checks.",
        ],
        "practice_task": "Create pass/fail and status labels in I2:I12 using IF and operator conditions.",
        "quiz": {
            "question": "Which operator means not equal in Excel?",
            "options": ["!=", "<>", "><"],
            "answer": "<>",
        },
        "steps": [
            {"task": "Enter scores and status values in columns I and J."},
            {
                "task": "Create IF checks with different operators.",
                "formulas": [
                    '=IF(I2>=50,"Pass","Fail")',
                    '=IF(I2<>J2,"Different","Same")',
                    '=IF(I2<=40,"Low","OK")',
                ],
            },
        ],
    },
    {
        "title": "Combined Logic with AND, OR, and NOT",
        "summary": "Combine and invert logic for robust validation rules.",
        "formula_focus": ["AND", "OR", "NOT"],
        "topics": ["Compound logic", "Fallback logic", "Inverse logic"],
        "objectives": [
            "Require multiple conditions with AND.",
            "Allow alternate conditions with OR.",
            "Invert test outcomes with NOT.",
        ],
        "practice_task": "Create eligibility and exception labels from two columns of conditions.",
        "quiz": {
            "question": "When does AND return TRUE?",
            "options": ["When all conditions are true", "When any condition is true", "Always"],
            "answer": "When all conditions are true",
        },
        "steps": [
            {"task": "Enter score and attendance columns in K and L."},
            {
                "task": "Build combined logical checks.",
                "formulas": [
                    '=IF(AND(K2>=60,L2>=80),"Eligible","Not Eligible")',
                    '=IF(OR(K2>=90,L2>=95),"Approved","Review")',
                    '=IF(NOT(K2>=50),"Retest","Pass")',
                ],
            },
        ],
    },
    {
        "title": "Top-N and Ranking",
        "summary": "Extract ranked values and compute rank positions.",
        "formula_focus": ["LARGE", "SMALL", "RANK.EQ"],
        "topics": ["Top-k analysis", "Bottom-k analysis", "Ranking"],
        "objectives": [
            "Return the k-th largest value with LARGE.",
            "Return the k-th smallest value with SMALL.",
            "Rank entries with RANK.EQ.",
        ],
        "practice_task": "Use values in M2:M12. Return 2nd largest, 3rd smallest, and each row rank.",
        "quiz": {
            "question": "Which formula returns the 2nd largest value?",
            "options": ["=LARGE(M2:M12,2)", "=SMALL(M2:M12,2)", "=RANK.EQ(M2,M2:M12)"],
            "answer": "=LARGE(M2:M12,2)",
        },
        "steps": [
            {"task": "Enter at least 10 numeric values in M2:M12."},
            {
                "task": "Compute top-k, bottom-k, and rank outputs.",
                "formulas": ['=LARGE(M2:M12,2)', '=SMALL(M2:M12,3)', '=RANK.EQ(M2,M2:M12,0)'],
            },
        ],
    },
    {
        "title": "Randomized Data Generation",
        "summary": "Generate random test values for simulations and stress tests.",
        "formula_focus": ["RAND", "RANDBETWEEN"],
        "topics": ["Simulation setup", "Random sampling", "Test data creation"],
        "objectives": [
            "Generate decimal random values with RAND.",
            "Generate bounded integers with RANDBETWEEN.",
            "Use random values to test formulas quickly.",
        ],
        "practice_task": "Fill N2:N11 with random decimals and O2:O11 with random integers between 1 and 100.",
        "quiz": {
            "question": "Which function returns random integers in a range?",
            "options": ["RAND", "RANDBETWEEN", "ROUND"],
            "answer": "RANDBETWEEN",
        },
        "steps": [
            {"task": "Create a random decimal column."},
            {"task": "Create a random integer column.", "formulas": ["=RAND()", "=RANDBETWEEN(1,100)"]},
        ],
    },
    {
        "title": "Text Length and Extraction",
        "summary": "Measure and slice text values for validation and formatting workflows.",
        "formula_focus": ["LEN", "LEFT", "RIGHT", "MID"],
        "topics": ["Character counts", "Prefix/suffix extraction", "Substring extraction"],
        "objectives": [
            "Use LEN to count characters.",
            "Extract leading and trailing text.",
            "Extract middle substrings with MID.",
        ],
        "practice_task": "Use IDs in P2:P10 and extract prefix, suffix, and mid-segments.",
        "quiz": {
            "question": "Which function extracts from a start position and length?",
            "options": ["LEFT", "RIGHT", "MID"],
            "answer": "MID",
        },
        "steps": [
            {"task": "Enter text IDs in P2:P10."},
            {
                "task": "Build text-shaping outputs.",
                "formulas": ['=LEN(P2)', '=LEFT(P2,3)', '=RIGHT(P2,2)', '=MID(P2,2,4)'],
            },
        ],
    },
    {
        "title": "Text Matching and Joining",
        "summary": "Search text patterns and build output strings from multiple fields.",
        "formula_focus": ["CONCAT", "FIND", "SEARCH"],
        "topics": ["Text search", "Case sensitivity", "String assembly"],
        "objectives": [
            "Join text values with CONCAT.",
            "Locate case-sensitive matches with FIND.",
            "Locate case-insensitive matches with SEARCH.",
        ],
        "practice_task": "Combine first/last names and find keyword positions in text cells.",
        "quiz": {
            "question": "Which function is case-insensitive?",
            "options": ["FIND", "SEARCH", "LEFT"],
            "answer": "SEARCH",
        },
        "steps": [
            {"task": "Enter first names in Q and last names in R, plus sentence text in S."},
            {
                "task": "Join names and locate keyword positions.",
                "formulas": ['=CONCAT(Q2," ",R2)', '=FIND("Excel",S2)', '=SEARCH("excel",S2)'],
            },
        ],
    },
]

LESSON_PLAN_ORDER = [
    "Totals and Counting",
    "Extremes with MIN and MAX",
    "Averages and Median",
    "Precision and Number Transformations",
    "IF Logic with Relational Operators",
    "Combined Logic with AND, OR, and NOT",
    "Counting Data States",
    "Conditional Aggregation",
    "Top-N and Ranking",
    "Randomized Data Generation",
    "Text Length and Extraction",
    "Text Matching and Joining",
]

LESSON_BY_TITLE = {lesson["title"]: lesson for lesson in LESSON_BLUEPRINTS}
ORDERED_LESSONS = [LESSON_BY_TITLE[title] for title in LESSON_PLAN_ORDER]
LESSONS = {f"Lesson {index}": lesson for index, lesson in enumerate(ORDERED_LESSONS, start=1)}


def _render_bullets(items: list[str]) -> None:
    st.markdown("\n".join(f"- {item}" for item in items))


def _render_quiz(lesson_name: str, quiz: dict[str, object]) -> None:
    selection = st.radio(quiz["question"], quiz["options"], key=f"quiz_option_{lesson_name}")

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


def _formula_focus_text(lesson: dict[str, object]) -> str:
    focus = lesson.get("formula_focus", [])
    if isinstance(focus, list):
        focus_text = ", ".join(str(item) for item in focus)
    else:
        focus_text = str(focus)

    operators = lesson.get("operator_focus", [])
    if isinstance(operators, list) and operators:
        operator_text = ", ".join(str(item) for item in operators)
        return f"Formula focus: {focus_text} | Operator focus: {operator_text}"

    return f"Formula focus: {focus_text}"


def render() -> None:
    st.title("Lessons")
    st.markdown("Each lesson combines explanation, spreadsheet practice, and a quick quiz.")

    tabs = st.tabs(list(LESSONS.keys()))

    for tab, lesson_name in zip(tabs, LESSONS, strict=False):
        lesson = LESSONS[lesson_name]
        with tab:
            st.header(f"{lesson_name}: {lesson['title']}")
            st.markdown(lesson["summary"])
            st.caption(_formula_focus_text(lesson))

            st.subheader("Learning Objectives")
            _render_bullets(lesson["objectives"])

            st.subheader("Key Topics")
            _render_bullets(lesson["topics"])

            st.subheader("Spreadsheet Practice Task")
            st.markdown(lesson["practice_task"])
            st.caption("Complete the task in the embedded spreadsheet and use Gemini for feedback.")

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
                    "examples": lesson_examples[:12],
                },
            )

            st.subheader("Quiz")
            _render_quiz(lesson_name, lesson["quiz"])

            st.subheader("Completion")
            _render_completion(lesson_name)


def formula_coverage() -> tuple[int, int]:
    formulas: set[str] = set()
    for lesson in LESSONS.values():
        focus = lesson.get("formula_focus", [])
        if isinstance(focus, list):
            formulas.update(str(item).strip().upper() for item in focus if str(item).strip())
        else:
            focus_text = str(focus).strip().upper()
            if focus_text:
                formulas.add(focus_text)

    return len(formulas), len(LESSONS)
