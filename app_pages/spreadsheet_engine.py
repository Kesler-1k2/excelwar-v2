from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ROWS = 12
DEFAULT_COLS = 8

RANGE_PATTERN = re.compile(r"\b([A-Z]+)([1-9]\d*):([A-Z]+)([1-9]\d*)\b")
CELL_PATTERN = re.compile(r"\b([A-Z]+)([1-9]\d*)\b")


def column_name(index: int) -> str:
    label = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        label = chr(65 + remainder) + label
    return label


def column_index(column_label: str) -> int:
    index = 0
    for character in column_label:
        index = index * 26 + (ord(character) - 64)
    return index - 1


def normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def build_blank_sheet(rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> pd.DataFrame:
    return pd.DataFrame(
        "",
        index=range(rows),
        columns=[column_name(i) for i in range(cols)],
    )


def normalize_sheet(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.fillna("").copy()
    normalized.columns = [column_name(i) for i in range(len(normalized.columns))]
    for column in normalized.columns:
        normalized[column] = normalized[column].map(normalize_cell)
    return normalized.reset_index(drop=True)


def resize_sheet(dataframe: pd.DataFrame, rows: int, cols: int) -> pd.DataFrame:
    resized = build_blank_sheet(rows, cols)
    max_rows = min(rows, len(dataframe))
    max_cols = min(cols, len(dataframe.columns))
    for row_index in range(max_rows):
        for col_index in range(max_cols):
            resized.iat[row_index, col_index] = normalize_cell(dataframe.iat[row_index, col_index])
    return resized


def diff_sheets(before: pd.DataFrame, after: pd.DataFrame) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    max_rows = max(len(before), len(after))
    max_cols = max(len(before.columns), len(after.columns))

    for row_index in range(max_rows):
        for col_index in range(max_cols):
            before_value = ""
            after_value = ""

            if row_index < len(before) and col_index < len(before.columns):
                before_value = normalize_cell(before.iat[row_index, col_index])

            if row_index < len(after) and col_index < len(after.columns):
                after_value = normalize_cell(after.iat[row_index, col_index])

            if before_value != after_value:
                changes.append(
                    {
                        "cell": f"{column_name(col_index)}{row_index + 1}",
                        "old": before_value,
                        "new": after_value,
                    }
                )

    return changes


def read_uploaded_sheet(uploaded_file) -> pd.DataFrame:
    extension = Path(uploaded_file.name).suffix.lower()

    if extension in {".xlsx", ".xls"}:
        imported = pd.read_excel(uploaded_file, header=None, dtype=str)
    else:
        imported = pd.read_csv(uploaded_file, header=None, dtype=str)

    if imported.empty:
        return build_blank_sheet()

    imported.columns = [column_name(i) for i in range(imported.shape[1])]
    return normalize_sheet(imported)


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


def _is_blank(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        return list(_flatten(value))
    return [value]


def _criteria_parts(criteria: Any) -> tuple[str, Any]:
    if isinstance(criteria, bool):
        return "=", criteria

    if isinstance(criteria, (int, float)) and not pd.isna(criteria):
        return "=", float(criteria)

    text = str(criteria).strip()
    for operator in (">=", "<=", "<>", ">", "<", "="):
        if text.startswith(operator):
            return operator, text[len(operator) :].strip()

    return "=", text


def _matches_criteria(value: Any, criteria: Any) -> bool:
    operator, operand = _criteria_parts(criteria)

    value_num = _to_number(value)
    operand_num = _to_number(operand)
    numeric_comparison = value_num is not None and operand_num is not None

    if numeric_comparison:
        left = float(value_num)
        right = float(operand_num)
    else:
        left = str(normalize_cell(value)).strip().lower()
        right = str(normalize_cell(operand)).strip().lower()

    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == "<>":
        return left != right

    return left == right


def _paired_values(criteria_range: Any, target_range: Any | None = None) -> list[tuple[Any, Any]]:
    criteria_values = _sequence(criteria_range)
    if target_range is None:
        return [(value, value) for value in criteria_values]

    target_values = _sequence(target_range)
    pair_count = min(len(criteria_values), len(target_values))
    return list(zip(criteria_values[:pair_count], target_values[:pair_count], strict=False))


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


def fx_counta(*args):
    return sum(1 for value in _flatten(args) if not _is_blank(value))


def fx_countblank(*args):
    return sum(1 for value in _flatten(args) if _is_blank(value))


def fx_countif(criteria_range, criteria):
    return sum(1 for value in _sequence(criteria_range) if _matches_criteria(value, criteria))


def fx_round(value, digits=0):
    number = _to_number(value)
    if number is None:
        return 0
    digits_number = _to_number(digits) or 0
    return _clean_numeric(round(number, int(digits_number)))


def fx_power(number, power):
    number_value = _to_number(number)
    power_value = _to_number(power)
    if number_value is None or power_value is None:
        return "ERR"
    return _clean_numeric(number_value**power_value)


def fx_sqrt(number):
    number_value = _to_number(number)
    if number_value is None or number_value < 0:
        return "ERR"
    return _clean_numeric(math.sqrt(number_value))


def fx_rand():
    return random.random()


def fx_randbetween(low, high):
    low_value = _to_number(low)
    high_value = _to_number(high)
    if low_value is None or high_value is None:
        return "ERR"

    low_int = int(low_value)
    high_int = int(high_value)
    if low_int > high_int:
        return "ERR"

    return random.randint(low_int, high_int)


def fx_if(condition, true_value, false_value):
    return true_value if condition else false_value


def fx_and(*args):
    return all(bool(item) for item in _flatten(args))


def fx_or(*args):
    return any(bool(item) for item in _flatten(args))


def fx_not(value):
    return not bool(value)


def fx_sumif(criteria_range, criteria, sum_range=None):
    pairs = _paired_values(criteria_range, sum_range if sum_range is not None else criteria_range)
    total = 0.0
    for check_value, target_value in pairs:
        if _matches_criteria(check_value, criteria):
            number = _to_number(target_value)
            if number is not None:
                total += number
    return _clean_numeric(total)


def fx_averageif(criteria_range, criteria, average_range=None):
    pairs = _paired_values(criteria_range, average_range if average_range is not None else criteria_range)
    selected_values: list[float] = []
    for check_value, target_value in pairs:
        if _matches_criteria(check_value, criteria):
            number = _to_number(target_value)
            if number is not None:
                selected_values.append(number)

    if not selected_values:
        return 0

    return _clean_numeric(sum(selected_values) / len(selected_values))


def fx_median(*args):
    numbers = _numeric_values(args)
    if not numbers:
        return 0
    return _clean_numeric(float(np.median(numbers)))


def fx_large(values, k):
    numbers = sorted(_numeric_values((values,)), reverse=True)
    k_number = _to_number(k)
    if not numbers or k_number is None:
        return "ERR"

    k_index = int(k_number)
    if k_index < 1 or k_index > len(numbers):
        return "ERR"

    return _clean_numeric(numbers[k_index - 1])


def fx_small(values, k):
    numbers = sorted(_numeric_values((values,)))
    k_number = _to_number(k)
    if not numbers or k_number is None:
        return "ERR"

    k_index = int(k_number)
    if k_index < 1 or k_index > len(numbers):
        return "ERR"

    return _clean_numeric(numbers[k_index - 1])


def fx_rank_eq(number, ref, order=0):
    number_value = _to_number(number)
    numbers = _numeric_values((ref,))
    order_value = int(_to_number(order) or 0)

    if number_value is None or not numbers:
        return "ERR"

    if order_value == 1:
        return 1 + sum(1 for item in numbers if item < number_value)

    return 1 + sum(1 for item in numbers if item > number_value)


def fx_len(value):
    return len(str(value))


def fx_left(text, length=1):
    length_value = int(_to_number(length) or 1)
    if length_value < 0:
        return "ERR"
    return str(text)[:length_value]


def fx_right(text, length=1):
    length_value = int(_to_number(length) or 1)
    if length_value < 0:
        return "ERR"
    if length_value == 0:
        return ""
    return str(text)[-length_value:]


def fx_mid(text, start, length):
    start_value = int(_to_number(start) or 0)
    length_value = int(_to_number(length) or 0)
    if start_value < 1 or length_value < 0:
        return "ERR"
    begin = start_value - 1
    return str(text)[begin : begin + length_value]


def fx_concat(*args):
    return "".join(str(item) for item in _flatten(args) if item is not None)


def fx_find(find_text, within_text, start=1):
    start_value = int(_to_number(start) or 1)
    if start_value < 1:
        return "ERR"

    index = str(within_text).find(str(find_text), start_value - 1)
    if index == -1:
        return "ERR"
    return index + 1


def fx_search(find_text, within_text, start=1):
    start_value = int(_to_number(start) or 1)
    if start_value < 1:
        return "ERR"

    index = str(within_text).lower().find(str(find_text).lower(), start_value - 1)
    if index == -1:
        return "ERR"
    return index + 1


FUNCTION_MAP = {
    "AND": fx_and,
    "AVERAGE": fx_average,
    "AVERAGEIF": fx_averageif,
    "CONCAT": fx_concat,
    "COUNT": fx_count,
    "COUNTA": fx_counta,
    "COUNTBLANK": fx_countblank,
    "COUNTIF": fx_countif,
    "FIND": fx_find,
    "IF": fx_if,
    "LARGE": fx_large,
    "LEFT": fx_left,
    "LEN": fx_len,
    "MAX": fx_max,
    "MEDIAN": fx_median,
    "MID": fx_mid,
    "MIN": fx_min,
    "NOT": fx_not,
    "OR": fx_or,
    "POWER": fx_power,
    "RAND": fx_rand,
    "RANDBETWEEN": fx_randbetween,
    "RANK_EQ": fx_rank_eq,
    "RIGHT": fx_right,
    "ROUND": fx_round,
    "SEARCH": fx_search,
    "SMALL": fx_small,
    "SQRT": fx_sqrt,
    "SUM": fx_sum,
    "SUMIF": fx_sumif,
}

FUNCTION_PATTERN = re.compile(
    r"\b(" + "|".join(FUNCTION_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def _parse_literal(value: Any):
    text = normalize_cell(value)
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

        start_col = column_index(start_col_name)
        end_col = column_index(end_col_name)
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
        col_idx = column_index(col_name)
        row_idx = int(row_str) - 1
        value = _evaluate_cell(raw_df, row_idx, col_idx, memo, stack)
        return _to_python_literal(value)

    expression = expression.replace("^", "**")
    expression = re.sub(r"(?i)\bRANK\.EQ\b", "RANK_EQ", expression)
    expression = expression.replace("<>", "!=")
    expression = re.sub(r"(?<![<>=!])=(?!=)", "==", expression)
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
    raw_value = normalize_cell(raw_df.iat[row_index, col_index])

    if raw_value.startswith("="):
        value = _evaluate_formula(raw_value[1:].strip(), raw_df, memo, stack)
    else:
        value = _parse_literal(raw_value)

    stack.remove(cell_key)
    memo[cell_key] = value
    return value


def calculate_preview(raw_df: pd.DataFrame) -> pd.DataFrame:
    preview_df = raw_df.copy()
    memo: dict[tuple[int, int], Any] = {}

    for row_index in range(len(raw_df)):
        for col_index in range(len(raw_df.columns)):
            preview_df.iat[row_index, col_index] = _evaluate_cell(raw_df, row_index, col_index, memo, set())

    return preview_df


__all__ = [
    "DEFAULT_COLS",
    "DEFAULT_ROWS",
    "build_blank_sheet",
    "calculate_preview",
    "diff_sheets",
    "normalize_sheet",
    "read_uploaded_sheet",
    "resize_sheet",
]
