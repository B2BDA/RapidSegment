"""
RapidSegment — No-Code Segmentation Platform
============================================
Launch with:
    rapidsegment-ui
or:
    python -c "from rapidsegment.ui import run_ui; run_ui()"
"""

import os
import shutil
from pathlib import Path
import sys
import streamlit as st

from rapidsegment.ui._exit import render_exit_button, write_server_pid
from rapidsegment.ui._theme import apply_cyberpunk_theme
from rapidsegment.ui._state import SUITE_DIR

try:
    import rapidsegment as _rs
    _RS_VERSION = getattr(_rs, "__version__", "unknown")
except Exception:
    _RS_VERSION = "unknown"


def run_ui():
    """Launch the RapidSegment Streamlit multipage app."""
    import subprocess

    app_path = str(Path(__file__).resolve())

    # Launch Streamlit in a child process so we can record its PID and later
    # terminate the whole server from the UI's "Exit UI" button.
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--global.developmentMode=false"],
    )
    try:
        write_server_pid(proc.pid)
    except Exception:
        pass

    try:
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.terminate()
        except Exception:
            pass


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
    apply_cyberpunk_theme()
    st.sidebar.caption(f"RapidSegment {_RS_VERSION} | {SUITE_DIR}")
    st.sidebar.divider()
    render_exit_button()

    # Danger zone: clear suite data
    with st.sidebar.expander("Danger Zone", expanded=False):
        if st.button("Clear all suite data", type="primary",
                     help="Wipe .rapidsegment_suite (datasets, experiments, artifacts)."):
            st.session_state["m_clear_confirm"] = True
        if st.session_state.get("m_clear_confirm"):
            st.warning("This will delete ALL datasets, experiments, and artifacts. Cannot be undone.")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes, delete everything", type="primary"):
                    try:
                        shutil.rmtree(SUITE_DIR, ignore_errors=True)
                        os.makedirs(SUITE_DIR, exist_ok=True)
                        st.session_state.pop("m_clear_confirm", None)
                        st.success("Suite data cleared. Reloading...")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
            with cc2:
                if st.button("Cancel"):
                    st.session_state.pop("m_clear_confirm", None)

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