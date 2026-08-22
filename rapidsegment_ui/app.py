"""
RapidSegment — No-Code Segmentation Platform (multipage entry point)
===================================================================
Run with:  streamlit run app.py

Pages (shared session state + shared DuckDB at .rapidsegment_suite/):
    pages/1_Data_Loader.py         — Module 1: Data Source & Profiling
    pages/2_Workbench.py           — Module 2: The Workbench
    pages/3_Execution_Console.py   — Module 3: Real-Time Execution & Artifact Console
"""
import streamlit as st

st.set_page_config(page_title="RapidSegment", layout="wide")
st.title("RapidSegment — No-Code Segmentation Platform")
st.caption("Choose a module from the sidebar.")
st.page_link("pages/1_Data_Loader.py", label="1 · Data Loader & Profiling", icon="📥")
st.page_link("pages/2_Workbench.py", label="2 · Workbench", icon="⚙️")
st.page_link("pages/3_Execution_Console.py", label="3 · Execution & Artifacts", icon="🚀")
st.page_link("pages/4_Results_Dashboard.py", label="4 · Results Dashboard", icon="📊")
st.page_link("pages/5_Leaderboard.py", label="5 · Leaderboard", icon="🏆")
st.page_link("pages/6_Arena.py", label="6 · Arena", icon="⚔️")