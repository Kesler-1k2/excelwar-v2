import os
from io import StringIO

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

MODEL_NAME = "gemini-2.5-flash"
EXCEL_TUTOR_INSTRUCTIONS = (
    "You are an Excel-only tutor. You may answer only questions about Excel, spreadsheets, "
    "formulas, functions, tables, charts, data cleaning, formatting, and workbook workflows. "
    "If a user asks about anything outside Excel/spreadsheets, refuse briefly and redirect them "
    "to ask an Excel-related question."
)


def _init_gemini() -> str | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)

    return api_key


def markdown_to_df(markdown_text: str) -> pd.DataFrame | None:
    """
    Detects and converts a Markdown table into a Pandas DataFrame.
    Returns None if no valid table is found.
    """
    lines = markdown_text.strip().split("\n")

    # Keep only lines that look like table rows
    table_lines = [line for line in lines if "|" in line]

    if len(table_lines) < 2:
        return None

    cleaned_lines = []

    for line in table_lines:
        # Remove alignment rows like |---|---|
        stripped = line.replace("|", "").replace("-", "").replace(":", "").strip()
        if stripped == "":
            continue

        cleaned_lines.append(line.strip().strip("|"))

    if len(cleaned_lines) < 2:
        return None

    # Convert to CSV-like format
    csv_like = "\n".join(
        [",".join(cell.strip() for cell in row.split("|")) for row in cleaned_lines]
    )

    try:
        df = pd.read_csv(StringIO(csv_like))
        return df
    except Exception:
        return None


def render() -> None:
    st.title("🤖 Excel Tutor")
    st.write(
        "AI Excel Assistant powered by Google Gemini. "
        "Ask me anything about Excel formulas, functions, and more!"
    )

    api_key = _init_gemini()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chat_cache" not in st.session_state:
        st.session_state.chat_cache = {}

    # Display previous messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                df = markdown_to_df(message["content"])
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")

    if not user_input:
        return

    # Show user message
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot reply
    if user_input in st.session_state.chat_cache:
        bot_reply = st.session_state.chat_cache[user_input]
    elif not api_key:
        bot_reply = (
            "GOOGLE_API_KEY is missing. "
            "Add it to `.env` to enable Gemini responses."
        )
    else:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            prompt = (
                f"{EXCEL_TUTOR_INSTRUCTIONS}\n\n"
                f"User question: {user_input}\n\n"
                "If the question is in-scope, provide a practical Excel-focused answer."
            )
            response = model.generate_content(prompt)
            bot_reply = response.text
            st.session_state.chat_cache[user_input] = bot_reply
        except Exception as error:
            bot_reply = f"API error: {error}"

    # Storing assistant reply...
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    # Render assistant reply
    with st.chat_message("assistant"):
        df = markdown_to_df(bot_reply)
        if df is not None:
            st.dataframe(df, use_container_width=True)
        else:
            st.markdown(bot_reply)
