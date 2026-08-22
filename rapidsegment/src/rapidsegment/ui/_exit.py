"""Exit the RapidSegment UI and stop the Streamlit server.

Streamlit has no built-in stop-server button, so when a user clicks Exit we
terminate the Streamlit server process (and its children). run_ui() records the
server PID when it launches the app; if the app was launched some other way
(e.g. `streamlit run app.py` directly) we fall back to walking the process tree
to find the Streamlit server. psutil is a hard dependency of the package, so the
process-tree walk is always available.
"""

import os
import tempfile

import streamlit as st

# Where run_ui() records the Streamlit server PID (single-user local app).
_PID_FILE = os.path.join(tempfile.gettempdir(), "rapidsegment_ui_server.pid")


def write_server_pid(pid):
    try:
        with open(_PID_FILE, "w") as fh:
            fh.write(str(pid))
    except Exception:
        pass


def _read_server_pid():
    try:
        if os.path.exists(_PID_FILE):
            with open(_PID_FILE) as fh:
                return int(fh.read().strip())
    except Exception:
        pass
    return None


def _kill_tree(proc):
    """Kill a psutil process and all of its descendants."""
    try:
        for child in proc.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
        proc.kill()
        return True
    except Exception:
        return False


def exit_ui():
    """Terminate the Streamlit server (and this app run)."""
    targets = []

    try:
        import psutil

        pid = _read_server_pid()
        if pid is not None:
            try:
                targets.append(psutil.Process(pid))
            except Exception:
                pass

        # Fallback / supplement: walk up the process tree to find the
        # Streamlit server (its command line contains "streamlit").
        me = psutil.Process(os.getpid())
        anc = me
        for _ in range(8):
            anc = anc.parent()
            if anc is None:
                break
            cmd = " ".join(anc.cmdline()).lower()
            if "streamlit" in cmd:
                targets.append(anc)
    except Exception:
        pass

    seen = set()
    for t in targets:
        if t.pid in seen:
            continue
        seen.add(t.pid)
        _kill_tree(t)

    if not targets:
        # Last resort: kill our own process tree.
        try:
            import psutil
            me = psutil.Process(os.getpid())
            for child in me.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            me.kill()
        except Exception:
            try:
                os.kill(os.getpid(), 9)
            except Exception:
                pass

    # Always halt this script run.
    try:
        st.stop()
    except Exception:
        pass
    # If st.stop() did not end it (already torn down), force exit.
    try:
        os._exit(0)
    except Exception:
        pass


def render_exit_button(label="Exit UI", key="exit_ui_global"):
    if st.sidebar.button(
        label,
        key=key,
        width="content",
        help="Stop the Streamlit server and close the app.",
    ):
        exit_ui()

