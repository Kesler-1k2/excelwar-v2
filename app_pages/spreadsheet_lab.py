from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_ROWS = 12
DEFAULT_COLS = 8
SHEET_STATE_KEY = "spreadsheet_raw_data"

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


def _resize_sheet(dataframe: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    resized = _build_blank_sheet(rows, cols)

    max_rows = min(rows, len(dataframe))
    max_cols = min(cols, len(dataframe.columns))

    for row_index in range(max_rows):
        for col_index in range(max_cols):
            resized.iat[row_index, col_index] = _normalize_cell(dataframe.iat[row_index, col_index])

    return resized


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


def _init_sheet_state() -> None:
    if SHEET_STATE_KEY not in st.session_state:
        st.session_state[SHEET_STATE_KEY] = _build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)


def render() -> None:
    st.title("🧮 Spreadsheet Lab")
    st.write("Single-grid spreadsheet editor with Excel-style formulas.")

    _init_sheet_state()

    controls_col, new_sheet_col, help_col = st.columns([2, 1, 1], gap="large")

    with controls_col:
        uploaded_file = st.file_uploader("Import CSV/XLSX", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None and st.button("Import File", key="sheet_import"):
            try:
                st.session_state[SHEET_STATE_KEY] = _read_uploaded_sheet(uploaded_file)
                st.success("Sheet imported.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(f"Could not import file: {error}")

    with new_sheet_col:
        if st.button("New Blank Sheet", use_container_width=True):
            st.session_state[SHEET_STATE_KEY] = _build_blank_sheet(DEFAULT_ROWS, DEFAULT_COLS)
            st.rerun()

    with help_col:
        st.caption("Examples: `=SUM(A1:A5)`, `=A1+B1`, `=IF(A1>50,\"Pass\",\"Fail\")`")

    raw_df = st.session_state[SHEET_STATE_KEY]

    with st.expander("Sheet Size"):
        current_rows = len(raw_df)
        current_cols = len(raw_df.columns)

        resize_rows = st.number_input("Rows", min_value=1, max_value=300, value=current_rows)
        resize_cols = st.number_input("Columns", min_value=1, max_value=52, value=current_cols)

        if st.button("Apply Size", key="apply_sheet_size"):
            st.session_state[SHEET_STATE_KEY] = _resize_sheet(raw_df, int(resize_rows), int(resize_cols))
            st.rerun()

    mode = st.radio(
        "Grid Mode",
        ["Edit Values/Formulas", "Calculated Preview"],
        horizontal=True,
    )

    raw_df = st.session_state[SHEET_STATE_KEY]

    if mode == "Edit Values/Formulas":
        edited_df = st.data_editor(
            raw_df,
            num_rows="dynamic",
            use_container_width=True,
            key="spreadsheet_editor",
        )
        st.session_state[SHEET_STATE_KEY] = _normalize_sheet(edited_df)
        preview_df = _calculate_preview(st.session_state[SHEET_STATE_KEY])
    else:
        preview_df = _calculate_preview(raw_df)
        st.data_editor(
            preview_df,
            disabled=True,
            use_container_width=True,
            key="spreadsheet_preview",
        )

    export_mode = st.selectbox(
        "Export Content",
        ["Calculated values", "Raw values/formulas"],
    )

    export_df = preview_df if export_mode == "Calculated values" else st.session_state[SHEET_STATE_KEY]

    excel_buffer = BytesIO()
    export_df.to_excel(excel_buffer, index=False, header=False)

    st.download_button(
        "Download XLSX",
        data=excel_buffer.getvalue(),
        file_name="excelwars_sheet.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    csv_data = export_df.to_csv(index=False, header=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_data,
        file_name="excelwars_sheet.csv",
        mime="text/csv",
    )
