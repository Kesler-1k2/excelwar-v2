from __future__ import annotations

import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import google.generativeai as genai
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

DEFAULT_ROWS = 12
DEFAULT_COLS = 8
MODEL_NAME = "gemini-2.5-flash"

RANGE_PATTERN = re.compile(r"\b([A-Z]+)([1-9]\d*):([A-Z]+)([1-9]\d*)\b")
CELL_PATTERN = re.compile(r"\b([A-Z]+)([1-9]\d*)\b")


def _column_name(index: int) -> str:
    label = ""
    current = index + 1

    while current:
        current, remainder = divmod(current - 1, 26)
        label = chr(65 + remainder) + label

    return label


def _column_index(column_name: str) -> int:
    index = 0
    for character in column_name:
        index = index * 26 + (ord(character) - 64)
    return index - 1


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value)


def _build_blank_sheet(rows: int, cols: int) -> pd.DataFrame:
    return pd.DataFrame(
        "",
        index=range(rows),
        columns=[_column_name(i) for i in range(cols)],
    )


def _normalize_sheet(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.fillna("").copy()
    normalized.columns = [_column_name(i) for i in range(len(normalized.columns))]

    for column in normalized.columns:
        normalized[column] = normalized[column].map(_normalize_cell)

    return normalized.reset_index(drop=True)


def _init_gemini() -> str | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)

    return api_key


def _resize_sheet(dataframe: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    resized = _build_blank_sheet(rows, cols)

    max_rows = min(rows, len(dataframe))
    max_cols = min(cols, len(dataframe.columns))

    for row_index in range(max_rows):
        for col_index in range(max_cols):
            resized.iat[row_index, col_index] = _normalize_cell(dataframe.iat[row_index, col_index])

    return resized


def _diff_sheets(before: pd.DataFrame, after: pd.DataFrame) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    max_rows = max(len(before), len(after))
    max_cols = max(len(before.columns), len(after.columns))

    for row_index in range(max_rows):
        for col_index in range(max_cols):
            before_value = ""
            after_value = ""

            if row_index < len(before) and col_index < len(before.columns):
                before_value = _normalize_cell(before.iat[row_index, col_index])

            if row_index < len(after) and col_index < len(after.columns):
                after_value = _normalize_cell(after.iat[row_index, col_index])

            if before_value != after_value:
                changes.append(
                    {
                        "cell": f"{_column_name(col_index)}{row_index + 1}",
                        "old": before_value,
                        "new": after_value,
                    }
                )

    return changes


def _read_uploaded_sheet(uploaded_file) -> pd.DataFrame:
    extension = Path(uploaded_file.name).suffix.lower()

    if extension in {".xlsx", ".xls"}:
        imported = pd.read_excel(uploaded_file, header=None, dtype=str)
    else:
        imported = pd.read_csv(uploaded_file, header=None, dtype=str)

    if imported.empty:
        return _build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)

    imported.columns = [_column_name(i) for i in range(imported.shape[1])]
    return _normalize_sheet(imported)


def _flatten(values):
    for value in values:
        if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
            yield from _flatten(value)
        else:
            yield value


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(int(value))

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value)

    if value in (None, ""):
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _clean_numeric(value: float | int) -> float | int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _numeric_values(args) -> list[float]:
    numbers: list[float] = []

    for value in _flatten(args):
        number = _to_number(value)
        if number is not None:
            numbers.append(number)

    return numbers


def fx_sum(*args):
    numbers = _numeric_values(args)
    return _clean_numeric(sum(numbers)) if numbers else 0


def fx_average(*args):
    numbers = _numeric_values(args)
    if not numbers:
        return 0
    return _clean_numeric(sum(numbers) / len(numbers))


def fx_min(*args):
    numbers = _numeric_values(args)
    return _clean_numeric(min(numbers)) if numbers else 0


def fx_max(*args):
    numbers = _numeric_values(args)
    return _clean_numeric(max(numbers)) if numbers else 0


def fx_count(*args):
    return len(_numeric_values(args))


def fx_round(value, digits=0):
    number = _to_number(value)
    if number is None:
        return 0

    digits_number = _to_number(digits) or 0
    return _clean_numeric(round(number, int(digits_number)))


def fx_if(condition, true_value, false_value):
    return true_value if condition else false_value


def fx_and(*args):
    return all(bool(item) for item in _flatten(args))


def fx_or(*args):
    return any(bool(item) for item in _flatten(args))


def fx_not(value):
    return not bool(value)


def fx_len(value):
    return len(str(value))


def fx_concat(*args):
    return "".join(str(item) for item in _flatten(args) if item is not None)


FUNCTION_MAP = {
    "SUM": fx_sum,
    "AVERAGE": fx_average,
    "MIN": fx_min,
    "MAX": fx_max,
    "COUNT": fx_count,
    "ROUND": fx_round,
    "IF": fx_if,
    "AND": fx_and,
    "OR": fx_or,
    "NOT": fx_not,
    "LEN": fx_len,
    "CONCAT": fx_concat,
}

FUNCTION_PATTERN = re.compile(
    r"\b(" + "|".join(FUNCTION_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def _parse_literal(value: Any):
    text = _normalize_cell(value)
    stripped = text.strip()

    if stripped == "":
        return ""

    upper_value = stripped.upper()
    if upper_value == "TRUE":
        return True
    if upper_value == "FALSE":
        return False

    number = _to_number(stripped)
    if number is not None:
        return _clean_numeric(number)

    return text


def _to_python_literal(value: Any) -> str:
    if value in (None, ""):
        return "0"

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, (int, float)):
        return repr(value)

    return repr(str(value))


def _evaluate_formula(
    expression: str,
    raw_df: pd.DataFrame,
    memo: dict[tuple[int, int], Any],
    stack: set[tuple[int, int]],
):
    range_placeholders: dict[str, str] = {}

    def replace_range(match: re.Match[str]) -> str:
        start_col_name, start_row_str, end_col_name, end_row_str = match.groups()

        start_col = _column_index(start_col_name)
        end_col = _column_index(end_col_name)
        start_row = int(start_row_str) - 1
        end_row = int(end_row_str) - 1

        col_from, col_to = sorted((start_col, end_col))
        row_from, row_to = sorted((start_row, end_row))

        values = []
        for row_index in range(row_from, row_to + 1):
            for col_index in range(col_from, col_to + 1):
                values.append(_evaluate_cell(raw_df, row_index, col_index, memo, stack))

        token = f"__RANGE_TOKEN_{len(range_placeholders)}__"
        range_placeholders[token] = repr(values)
        return token

    def replace_cell(match: re.Match[str]) -> str:
        col_name, row_str = match.groups()
        col_index = _column_index(col_name)
        row_index = int(row_str) - 1

        value = _evaluate_cell(raw_df, row_index, col_index, memo, stack)
        return _to_python_literal(value)

    expression = expression.replace("^", "**")
    expression = RANGE_PATTERN.sub(replace_range, expression)
    expression = CELL_PATTERN.sub(replace_cell, expression)

    for token, replacement in range_placeholders.items():
        expression = expression.replace(token, replacement)

    expression = FUNCTION_PATTERN.sub(lambda m: f"FUNCTION_MAP['{m.group(1).upper()}']", expression)

    try:
        result = eval(expression, {"__builtins__": {}}, {"FUNCTION_MAP": FUNCTION_MAP})
    except Exception:  # noqa: BLE001
        return "ERR"

    if isinstance(result, float) and result.is_integer():
        return int(result)

    if isinstance(result, list):
        return ", ".join(str(item) for item in result)

    return result


def _evaluate_cell(
    raw_df: pd.DataFrame,
    row_index: int,
    col_index: int,
    memo: dict[tuple[int, int], Any],
    stack: set[tuple[int, int]],
):
    if row_index < 0 or col_index < 0:
        return 0

    if row_index >= len(raw_df) or col_index >= len(raw_df.columns):
        return 0

    cell_key = (row_index, col_index)
    if cell_key in memo:
        return memo[cell_key]

    if cell_key in stack:
        return "#CYCLE!"

    stack.add(cell_key)

    raw_value = _normalize_cell(raw_df.iat[row_index, col_index])

    if raw_value.startswith("="):
        value = _evaluate_formula(raw_value[1:].strip(), raw_df, memo, stack)
    else:
        value = _parse_literal(raw_value)

    stack.remove(cell_key)
    memo[cell_key] = value
    return value


def _calculate_preview(raw_df: pd.DataFrame) -> pd.DataFrame:
    preview_df = raw_df.copy()
    memo: dict[tuple[int, int], Any] = {}

    for row_index in range(len(raw_df)):
        for col_index in range(len(raw_df.columns)):
            preview_df.iat[row_index, col_index] = _evaluate_cell(raw_df, row_index, col_index, memo, set())

    return preview_df


def _session_key(namespace: str, suffix: str) -> str:
    return f"{namespace}_{suffix}"


def _init_sheet_state(namespace: str) -> dict[str, str]:
    keys = {
        "sheet": _session_key(namespace, "raw_data"),
        "changes": _session_key(namespace, "change_log"),
        "version": _session_key(namespace, "version"),
        "messages": _session_key(namespace, "chat_messages"),
        "cache": _session_key(namespace, "chat_cache"),
        "lesson_signature": _session_key(namespace, "lesson_signature"),
        "import_button": _session_key(namespace, "import_button"),
        "new_sheet_button": _session_key(namespace, "new_sheet_button"),
        "apply_size_button": _session_key(namespace, "apply_size_button"),
        "mode_radio": _session_key(namespace, "mode_radio"),
        "editor": _session_key(namespace, "editor"),
        "preview": _session_key(namespace, "preview"),
        "export_mode": _session_key(namespace, "export_mode"),
        "download_xlsx": _session_key(namespace, "download_xlsx"),
        "download_csv": _session_key(namespace, "download_csv"),
        "chat_form": _session_key(namespace, "chat_form"),
        "chat_input": _session_key(namespace, "chat_input"),
        "chat_clear": _session_key(namespace, "chat_clear"),
    }

    if keys["sheet"] not in st.session_state:
        st.session_state[keys["sheet"]] = _build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)
    if keys["changes"] not in st.session_state:
        st.session_state[keys["changes"]] = []
    if keys["version"] not in st.session_state:
        st.session_state[keys["version"]] = 0
    if keys["messages"] not in st.session_state:
        st.session_state[keys["messages"]] = []
    if keys["cache"] not in st.session_state:
        st.session_state[keys["cache"]] = {}
    if keys["lesson_signature"] not in st.session_state:
        st.session_state[keys["lesson_signature"]] = ""

    return keys


def _build_change_context(change_log_key: str, limit: int = 20) -> str:
    changes = st.session_state.get(change_log_key, [])

    if not changes:
        return "No spreadsheet changes have been tracked yet."

    recent = changes[-limit:]
    lines = [f"{item['cell']}: '{item['old']}' -> '{item['new']}'" for item in recent]
    return "Recent spreadsheet changes:\n" + "\n".join(lines)


def _build_lesson_context(lesson_context: dict[str, Any] | None) -> str:
    if not lesson_context:
        return "No specific lesson context provided."

    lesson_name = str(lesson_context.get("name", "")).strip()
    lesson_title = str(lesson_context.get("title", "")).strip()
    summary = str(lesson_context.get("summary", "")).strip()
    lab_task = str(lesson_context.get("lab_prompt", "")).strip()
    objectives = lesson_context.get("objectives", [])
    topics = lesson_context.get("topics", [])

    objective_lines = []
    if isinstance(objectives, list):
        objective_lines = [f"- {str(item)}" for item in objectives[:5]]

    topic_lines = []
    if isinstance(topics, list):
        topic_lines = [f"- {str(item)}" for item in topics[:6]]

    return "\n".join(
        [
            f"Lesson: {lesson_name} - {lesson_title}",
            f"Lesson summary: {summary}",
            "Lesson objectives:",
            *(objective_lines or ["- No objectives listed."]),
            "Lesson topics:",
            *(topic_lines or ["- No topics listed."]),
            f"Guided lab task: {lab_task}",
        ]
    )


def _lesson_signature(lesson_context: dict[str, Any] | None) -> str:
    if not lesson_context:
        return "no-lesson-context"
    try:
        return json.dumps(lesson_context, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return str(lesson_context)


def _ask_gemini(
    user_input: str,
    api_key: str | None,
    *,
    change_log_key: str,
    version_key: str,
    cache_state_key: str,
    lesson_context: dict[str, Any] | None = None,
) -> str:
    if not api_key:
        return "GOOGLE_API_KEY is missing. Add it to `.env` to enable Gemini responses."

    sheet_version = st.session_state.get(version_key, 0)
    lesson_signature = _lesson_signature(lesson_context)
    request_cache_key = f"{lesson_signature}:{sheet_version}:{user_input.strip()}"
    cache = st.session_state[cache_state_key]

    if request_cache_key in cache:
        return cache[request_cache_key]

    prompt = (
        "You are an Excel lesson coach helping with a live spreadsheet.\n"
        f"{_build_lesson_context(lesson_context)}\n\n"
        f"{_build_change_context(change_log_key)}\n\n"
        "Guidance rules:\n"
        "- Teach based on this lesson's goals and task.\n"
        "- Give actionable next steps the learner can do in this sheet now.\n"
        "- When asked what changed, answer with exact cell references and old/new values.\n"
        "- Keep explanations short, clear, and student-friendly.\n"
        f"User question: {user_input}"
    )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        reply = response.text or "I could not generate a response."
    except Exception as error:  # noqa: BLE001
        reply = f"API error: {error}"

    cache[request_cache_key] = reply
    return reply


def render_lab(
    namespace: str = "spreadsheet",
    show_title: bool = True,
    lesson_context: dict[str, Any] | None = None,
    show_new_sheet_button: bool = True,
) -> None:
    if show_title:
        st.title("🧮 Spreadsheet Lab")
        st.write("Single-grid spreadsheet editor with Excel-style formulas.")

    keys = _init_sheet_state(namespace)
    api_key = _init_gemini()
    current_lesson_signature = _lesson_signature(lesson_context)
    if st.session_state[keys["lesson_signature"]] != current_lesson_signature:
        st.session_state[keys["lesson_signature"]] = current_lesson_signature
        st.session_state[keys["cache"]] = {}

    left_col, right_col = st.columns([2.2, 1], gap="large")

    with left_col:
        if show_new_sheet_button:
            controls_col, new_sheet_col, help_col = st.columns([2, 1, 1], gap="large")
        else:
            controls_col, help_col = st.columns([3, 1], gap="large")
            new_sheet_col = None

        with controls_col:
            uploaded_file = st.file_uploader(
                "Import CSV/XLSX",
                type=["csv", "xlsx", "xls"],
                key=_session_key(namespace, "uploader"),
            )
            if uploaded_file is not None and st.button("Import File", key=keys["import_button"]):
                try:
                    imported = _read_uploaded_sheet(uploaded_file)
                    st.session_state[keys["sheet"]] = imported
                    st.session_state[keys["changes"]] = []
                    st.session_state[keys["version"]] += 1
                    st.success("Sheet imported.")
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(f"Could not import file: {error}")

        if new_sheet_col is not None:
            with new_sheet_col:
                if st.button("New Blank Sheet", use_container_width=True, key=keys["new_sheet_button"]):
                    st.session_state[keys["sheet"]] = _build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)
                    st.session_state[keys["changes"]] = []
                    st.session_state[keys["version"]] += 1
                    st.rerun()

        with help_col:
            st.caption("Examples: `=SUM(A1:A5)`, `=A1+B1`, `=IF(A1>50,\"Pass\",\"Fail\")`")

        raw_df = st.session_state[keys["sheet"]]

        lesson_examples = []
        if lesson_context and isinstance(lesson_context.get("examples"), list):
            lesson_examples = [str(item).strip() for item in lesson_context["examples"] if str(item).strip()]
        if lesson_examples:
            st.caption("Lesson formula examples:")
            st.code("\n".join(lesson_examples), language="text")

        with st.expander("Sheet Size", expanded=False):
            current_rows = len(raw_df)
            current_cols = len(raw_df.columns)

            resize_rows = st.number_input(
                "Rows",
                min_value=1,
                max_value=300,
                value=current_rows,
                key=_session_key(namespace, "rows_input"),
            )
            resize_cols = st.number_input(
                "Columns",
                min_value=1,
                max_value=52,
                value=current_cols,
                key=_session_key(namespace, "cols_input"),
            )

            if st.button("Apply Size", key=keys["apply_size_button"]):
                resized = _resize_sheet(raw_df, int(resize_rows), int(resize_cols))
                st.session_state[keys["sheet"]] = resized
                st.session_state[keys["changes"]] = []
                st.session_state[keys["version"]] += 1
                st.rerun()

        mode = st.radio(
            "Grid Mode",
            ["Edit Values/Formulas", "Calculated Preview"],
            horizontal=True,
            key=keys["mode_radio"],
        )

        raw_df = st.session_state[keys["sheet"]]

        if mode == "Edit Values/Formulas":
            before_df = raw_df.copy()
            edit_display_df = raw_df.copy()
            edit_display_df.index = range(1, len(edit_display_df) + 1)
            edited_df = st.data_editor(
                edit_display_df,
                num_rows="dynamic",
                use_container_width=True,
                key=keys["editor"],
                hide_index=False,
                column_config={
                    "_index": st.column_config.NumberColumn("Row", disabled=True),
                },
            )
            normalized_edited_df = _normalize_sheet(edited_df.reset_index(drop=True))
            changes = _diff_sheets(before_df, normalized_edited_df)
            st.session_state[keys["sheet"]] = normalized_edited_df
            if changes:
                st.session_state[keys["changes"]] = changes
                st.session_state[keys["version"]] += 1
                formula_entered = any(
                    item["new"].strip().startswith("=") and item["new"] != item["old"]
                    for item in changes
                )
                if formula_entered:
                    st.session_state[keys["mode_radio"]] = "Calculated Preview"
                    st.rerun()
            preview_df = _calculate_preview(st.session_state[keys["sheet"]])
        else:
            preview_df = _calculate_preview(raw_df)
            preview_display_df = preview_df.copy()
            preview_display_df.index = range(1, len(preview_display_df) + 1)
            st.data_editor(
                preview_display_df,
                disabled=True,
                use_container_width=True,
                key=keys["preview"],
                hide_index=False,
                column_config={
                    "_index": st.column_config.NumberColumn("Row", disabled=True),
                },
            )

        export_mode = st.selectbox(
            "Export Content",
            ["Calculated values", "Raw values/formulas"],
            key=keys["export_mode"],
        )

        export_df = preview_df if export_mode == "Calculated values" else st.session_state[keys["sheet"]]

        excel_buffer = BytesIO()
        export_df.to_excel(excel_buffer, index=False, header=False)

        st.download_button(
            "Download XLSX",
            data=excel_buffer.getvalue(),
            file_name="excelwars_sheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=keys["download_xlsx"],
        )

        csv_data = export_df.to_csv(index=False, header=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="excelwars_sheet.csv",
            mime="text/csv",
            key=keys["download_csv"],
        )

    with right_col:
        st.subheader("🤖 Gemini Spreadsheet Chat")
        if lesson_context:
            st.caption(
                f"Lesson coach mode: {lesson_context.get('name', '')} - {lesson_context.get('title', '')}"
            )
        st.caption("Ask things like: what changed? what changed in A3?")
        st.caption(f"Tracked sheet version: {st.session_state[keys['version']]}")

        for message in st.session_state[keys["messages"]]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form(keys["chat_form"], clear_on_submit=True):
            question = st.text_input("Ask Gemini about this sheet", key=keys["chat_input"])
            send_clicked = st.form_submit_button("Send", use_container_width=True)

        if st.button("Clear Chat", use_container_width=True, key=keys["chat_clear"]):
            st.session_state[keys["messages"]] = []
            st.rerun()

        if send_clicked and question.strip():
            user_input = question.strip()
            st.session_state[keys["messages"]].append({"role": "user", "content": user_input})
            bot_reply = _ask_gemini(
                user_input,
                api_key,
                change_log_key=keys["changes"],
                version_key=keys["version"],
                cache_state_key=keys["cache"],
                lesson_context=lesson_context,
            )
            st.session_state[keys["messages"]].append({"role": "assistant", "content": bot_reply})
            st.rerun()


def render() -> None:
    render_lab(namespace="spreadsheet", show_title=True)
