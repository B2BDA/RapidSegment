"""
RapidSegment — No-Code Segmentation Platform (multipage entry point)
===================================================================
Run with:  streamlit run app.py

Pages (shared session state + shared DuckDB at .rapidsegment_suite/):
    pages/1_Data_Loader.py   — Module 1: Data Source & Profiling
    pages/2_Workbench.py     — Module 2: The Workbench
"""
import streamlit as st

st.set_page_config(page_title="RapidSegment", layout="wide")
st.title("RapidSegment — No-Code Segmentation Platform")
st.caption("Choose a module from the sidebar.")
st.page_link("pages/1_Data_Loader.py", label="1 · Data Loader & Profiling", icon="📥")
st.page_link("pages/2_Workbench.py", label="2 · Workbench", icon="⚙️")