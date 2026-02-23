from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Any

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app_pages.spreadsheet_engine import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    build_blank_sheet,
    calculate_preview,
    diff_sheets,
    normalize_sheet,
    read_uploaded_sheet,
    resize_sheet,
)

MODEL_NAME = "gemini-2.5-flash"
CHANGE_HISTORY_LIMIT = 300
SHEET_CONTEXT_MAX_ROWS = 30
SHEET_CONTEXT_MAX_COLS = 12


def _init_gemini() -> str | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    return api_key


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
        st.session_state[keys["sheet"]] = build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)
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


def _build_sheet_context(sheet_key: str) -> str:
    # Give Gemini a full-sheet snapshot instead of only last edited cells.
    sheet = st.session_state.get(sheet_key)
    if not isinstance(sheet, pd.DataFrame) or sheet.empty:
        return "Spreadsheet state: empty sheet."

    raw_df = normalize_sheet(sheet)
    safe_raw = raw_df.fillna("").astype(str)

    formula_cells = int(safe_raw.apply(lambda column: column.str.startswith("=").sum()).sum())
    non_empty_cells = int((safe_raw != "").sum().sum())
    numeric_cells = int(pd.to_numeric(safe_raw.stack(), errors="coerce").notna().sum())

    row_count, col_count = safe_raw.shape
    max_rows = min(SHEET_CONTEXT_MAX_ROWS, row_count)
    max_cols = min(SHEET_CONTEXT_MAX_COLS, col_count)

    raw_preview = safe_raw.iloc[:max_rows, :max_cols]
    calculated_preview = calculate_preview(raw_df).iloc[:max_rows, :max_cols].fillna("")
    calculated_preview = calculated_preview.astype(str)

    row_note = ""
    if row_count > max_rows:
        row_note = f" (showing first {max_rows} of {row_count} rows)"

    col_note = ""
    if col_count > max_cols:
        col_note = f" (showing first {max_cols} of {col_count} columns)"

    return "\n".join(
        [
            "Spreadsheet snapshot:",
            f"- Rows: {row_count}{row_note}",
            f"- Columns: {col_count}{col_note}",
            f"- Non-empty cells: {non_empty_cells}",
            f"- Numeric cells: {numeric_cells}",
            f"- Formula cells: {formula_cells}",
            "Raw values/formulas preview:",
            raw_preview.to_string(index=False),
            "Calculated values preview:",
            calculated_preview.to_string(index=False),
        ]
    )


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
    sheet_state_key: str,
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
        f"{_build_sheet_context(sheet_state_key)}\n\n"
        f"{_build_change_context(change_log_key)}\n\n"
        "Guidance rules:\n"
        "- Teach based on this lesson's goals and task.\n"
        "- Use the full sheet snapshot for analysis, not only recent edits.\n"
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
        st.title("Spreadsheet Lab")
        st.write("Single-grid spreadsheet editor with formula support.")

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
                    imported = read_uploaded_sheet(uploaded_file)
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
                    st.session_state[keys["sheet"]] = build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)
                    st.session_state[keys["changes"]] = []
                    st.session_state[keys["version"]] += 1
                    st.rerun()

        with help_col:
            st.caption(
                "Examples: `=SUM(A1:A5)`, `=IF(A1<=50,\"Low\",\"High\")`, `=IF(A1<>B1,\"Different\",\"Same\")`"
            )

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
                resized = resize_sheet(raw_df, int(resize_rows), int(resize_cols))
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
            editor_widget_key = f"{keys['editor']}_{st.session_state[keys['version']]}"

            edited_df = st.data_editor(
                edit_display_df,
                num_rows="dynamic",
                use_container_width=True,
                key=editor_widget_key,
                hide_index=False,
                column_config={"_index": st.column_config.NumberColumn("Row", disabled=True)},
            )

            normalized_edited_df = normalize_sheet(edited_df.reset_index(drop=True))
            changes = diff_sheets(before_df, normalized_edited_df)
            if changes:
                st.session_state[keys["sheet"]] = normalized_edited_df
                history = st.session_state[keys["changes"]]
                history.extend(changes)
                st.session_state[keys["changes"]] = history[-CHANGE_HISTORY_LIMIT:]
                st.session_state[keys["version"]] += 1

                formula_entered = any(
                    item["new"].strip().startswith("=") and item["new"] != item["old"] for item in changes
                )
                if formula_entered:
                    st.session_state[keys["mode_radio"]] = "Calculated Preview"
                st.rerun()

            preview_df = calculate_preview(st.session_state[keys["sheet"]])
        else:
            preview_df = calculate_preview(raw_df)
            preview_display_df = preview_df.copy()
            preview_display_df.index = range(1, len(preview_display_df) + 1)
            st.data_editor(
                preview_display_df,
                disabled=True,
                use_container_width=True,
                key=keys["preview"],
                hide_index=False,
                column_config={"_index": st.column_config.NumberColumn("Row", disabled=True)},
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
        st.subheader("Gemini Spreadsheet Chat")
        if lesson_context:
            st.caption(f"Lesson coach mode: {lesson_context.get('name', '')} - {lesson_context.get('title', '')}")

        with st.form(keys["chat_form"], clear_on_submit=False):
            st.text_input("Ask Gemini about this sheet", key=keys["chat_input"])
            send_clicked = st.form_submit_button("Send", use_container_width=True)

        if st.button("Clear Chat", use_container_width=True, key=keys["chat_clear"]):
            st.session_state[keys["messages"]] = []
            st.session_state[keys["chat_input"]] = ""
            st.rerun()

        if send_clicked:
            user_input = st.session_state.get(keys["chat_input"], "").strip()
            if user_input:
                st.session_state[keys["messages"]].append({"role": "user", "content": user_input})
                bot_reply = _ask_gemini(
                    user_input,
                    api_key,
                    sheet_state_key=keys["sheet"],
                    change_log_key=keys["changes"],
                    version_key=keys["version"],
                    cache_state_key=keys["cache"],
                    lesson_context=lesson_context,
                )
                st.session_state[keys["messages"]].append({"role": "assistant", "content": bot_reply})
            st.session_state[keys["chat_input"]] = ""

        for message in st.session_state[keys["messages"]]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def render() -> None:
    render_lab(namespace="spreadsheet", show_title=True)
