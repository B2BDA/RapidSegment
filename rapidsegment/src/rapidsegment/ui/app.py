"""
RapidSegment — No-Code Segmentation Platform
============================================
Launch with:
    rapidsegment-ui
or:
    python -c "from rapidsegment.ui import run_ui; run_ui()"
"""

from pathlib import Path
import sys
import streamlit as st


def run_ui():
    """Launch the RapidSegment Streamlit multipage app."""
    import streamlit.web.cli as stcli

    app_path = str(Path(__file__).resolve())

    # Tell Streamlit to run this file (pages/ will be discovered automatically)
    sys.argv = ["streamlit", "run", app_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())


def _home():
    st.title("RapidSegment — No-Code Segmentation Platform")
    st.caption("Choose a module from the sidebar.")
    st.page_link("pages/1_Data_Loader.py", label="1 · Data Loader & Profiling", icon="📥")
    st.page_link("pages/2_Workbench.py", label="2 · Workbench", icon="⚙️")
    st.page_link("pages/3_Execution_Console.py", label="3 · Execution & Artifacts", icon="🚀")
    st.page_link("pages/4_Results_Dashboard.py", label="4 · Results Dashboard", icon="📊")
    st.page_link("pages/5_Leaderboard.py", label="5 · Leaderboard", icon="🏆")
    st.page_link("pages/6_Arena.py", label="6 · Arena", icon="⚔️")


# Explicit navigation so the entry page can be named (automatic detection would
# label it by the filename, i.e. "app"). Edit the `title=` values to rename.
if __name__ == "__main__":
    st.set_page_config(page_title="RapidSegment", layout="wide")
    pg = st.navigation({
        "Modules": [
            st.Page(_home, title="Home", icon="🏠"),
            st.Page("pages/1_Data_Loader.py", title="Data Loader", icon="📥"),
            st.Page("pages/2_Workbench.py", title="Workbench", icon="⚙️"),
            st.Page("pages/3_Execution_Console.py", title="Execution", icon="🚀"),
            st.Page("pages/4_Results_Dashboard.py", title="Results", icon="📊"),
            st.Page("pages/5_Leaderboard.py", title="Leaderboard", icon="🏆"),
            st.Page("pages/6_Arena.py", title="Arena", icon="⚔️"),
        ]
    })
    pg.run()