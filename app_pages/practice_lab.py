import pandas as pd
import streamlit as st

from app_core import navigate


def render() -> None:
    st.title("🧪 Practice Lab")
    st.write("Track your practice tasks and notes.")

    default_df = pd.DataFrame(
        {
            "Task": [
                "Enter a weekly budget table",
                "Use SUM for totals",
                "Use AVERAGE for mean values",
                "Create a chart from your data",
            ],
            "Status": ["Not started", "Not started", "Not started", "Not started"],
            "Notes": ["", "", "", ""],
        }
    )

    if "practice_table" not in st.session_state:
        st.session_state.practice_table = default_df

    st.session_state.practice_table = st.data_editor(
        st.session_state.practice_table,
        num_rows="dynamic",
        use_container_width=True,
        key="practice_editor",
    )

    if st.button("Open Spreadsheet Lab", use_container_width=True):
        navigate("spreadsheet")
