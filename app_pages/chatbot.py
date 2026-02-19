import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

MODEL_NAME = "gemini-2.5-flash"


def _init_gemini() -> str | None:
    load_dotenv(dotenv_path=".env")
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        genai.configure(api_key=api_key)

    return api_key


def render() -> None:
    st.title("🤖 Gemini Chatbot")

    api_key = _init_gemini()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    if "chat_cache" not in st.session_state:
        st.session_state.chat_cache = {}

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Type your message...")

    if not user_input:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if user_input in st.session_state.chat_cache:
        bot_reply = st.session_state.chat_cache[user_input]
    elif not api_key:
        bot_reply = "GOOGLE_API_KEY is missing. Add it to `.env` to enable Gemini responses."
    else:
        try:
            model = genai.GenerativeModel(MODEL_NAME)
            response = model.generate_content(user_input)
            bot_reply = response.text
            st.session_state.chat_cache[user_input] = bot_reply
        except Exception as error:  # noqa: BLE001
            bot_reply = f"API error: {error}"

    st.session_state.chat_messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
