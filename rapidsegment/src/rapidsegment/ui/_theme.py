"""RapidSegment "hacker-terminal" (emerald-on-black) Streamlit theme helpers.

Apply once per page (after `st.set_page_config`) via `apply_cyberpunk_theme()`
to give the whole RapidSegment UI an emerald system-console neon look (the
visual language is ported from an external HTML reference). The base palette is
also declared in `.streamlit/config.toml`; this injected CSS guarantees the look
regardless of the working directory Streamlit is launched from.
"""

import streamlit as st

_CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --background-color: #000000 !important;
    --secondary-background-color: #03150E !important;
    --primary-color: #34D399 !important;
    --text-color: #6EE7B7 !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #000000;
    background-image:
        linear-gradient(90deg, rgba(255,0,0,0.04), rgba(0,255,0,0.02), rgba(0,0,255,0.04)),
        linear-gradient(to bottom, rgba(0,0,0,0), rgba(16,185,129,0.05) 50%, rgba(0,0,0,0)),
        linear-gradient(to right, rgba(52,211,153,0.08) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(52,211,153,0.08) 1px, transparent 1px);
    background-size: 3px 100%, 100% 4px, 48px 48px, 48px 48px;
    background-attachment: fixed;
}

[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stToolbar"] {
    background: rgba(0,0,0,0.6) !important;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(52,211,153,0.4) !important;
    box-shadow: 1px 0 15px rgba(52,211,153,0.6);
}
[data-testid="stHeader"] {
    border-bottom: 1px solid rgba(52,211,153,0.4) !important;
    box-shadow: 0 1px 15px rgba(52,211,153,0.6);
}

h1, h2, h3 {
    color: #6EE7B7 !important;
    text-shadow: 0 0 8px rgba(52,211,153,0.7);
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    letter-spacing: -0.02em;
}
.stMarkdown, .stCaption, label, .stTextInput label, .stSelectbox label {
    font-family: 'JetBrains Mono', monospace;
}

.stButton > button {
    border: 1px solid #34D399;
    border-radius: 2px;
    color: #6EE7B7;
    background-color: rgba(52,211,153,0.06);
    box-shadow: 0 0 6px rgba(52,211,153,0.45);
    font-family: 'JetBrains Mono', monospace;
}
.stButton > button:hover {
    background-color: rgba(52,211,153,0.16);
    box-shadow: 0 0 10px rgba(52,211,153,0.7);
}
.stButton > button[kind="primary"] {
    background-color: #34D399 !important;
    color: #001014 !important;
    border-color: #34D399 !important;
    font-weight: 700;
}
.stButton > button[kind="primary"]:hover {
    background-color: #6EE7B7 !important;
    border-color: #6EE7B7 !important;
}

.stDataFrame, .stTable {
    border: 1px solid rgba(52,211,153,0.5) !important;
    box-shadow: 0 0 20px rgba(16,185,129,0.10);
}
.stDataFrame table thead th, .stTable thead th {
    background-color: rgba(2,44,34,0.6) !important;
    color: #34D399 !important;
}
.stDataFrame table tbody tr:hover, .stTable tbody tr:hover {
    background-color: rgba(16,185,129,0.10) !important;
}

.stTextInput input, .stTextArea textarea, .stSelectbox > div {
    border: 1px solid rgba(52,211,153,0.5) !important;
    border-radius: 2px !important;
    background-color: rgba(0,0,0,0.6) !important;
    color: #6EE7B7 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #6EE7B7 !important;
    box-shadow: 0 0 12px rgba(52,211,153,0.5), inset 0 0 8px rgba(52,211,153,0.1) !important;
}

/* Selected items in a multiselect. Streamlit renders each selected value as a
   chip carrying a `data-tag` attribute, with an emerald fill + WHITE text by
   default — force dark text for contrast. (NOTE: not `[data-baseweb="tag"]` —
   that BaseWeb attribute is not emitted; the chip uses `data-tag`.) */
[data-testid="stMultiSelectTagsContainer"] [data-tag],
[data-testid="stMultiSelectTagsContainer"] [data-tag] * {
    color: #001014 !important;
}

/* Success alerts (e.g. Leaderboard "Best performer"): bright emerald fill with
   black text for high contrast. Streamlit keys the content by kind via
   `data-testid="stAlertContentSuccess"` (ErrorElement.js — `i` is the
   capitalised kind, so success -> stAlertContentSuccess), letting us target
   success without touching error/warning. `.stSuccess` class names are NOT
   emitted, and the `kind` prop is a transient `$kind` (not in the DOM). */
[data-testid="stAlertContentSuccess"],
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]),
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) {
    background-color: #34D399 !important;
}
[data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentSuccess"] * {
    color: #001014 !important;
}

input[type="checkbox"], input[type="radio"] { accent-color: #34D399; }
.stCheckbox label, .stRadio label { font-family: 'JetBrains Mono', monospace; }

a { color: #6EE7B7 !important; text-shadow: 0 0 6px rgba(52,211,153,0.5); }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #000000; border-left: 1px solid rgba(16,185,129,0.3); box-shadow: inset 0 0 10px rgba(0,0,0,0.8); }
::-webkit-scrollbar-thumb { background: rgba(52,211,153,0.5); border-radius: 0px; box-shadow: 0 0 10px rgba(52,211,153,0.8); }
::-webkit-scrollbar-thumb:hover { background: rgba(52,211,153,0.9); box-shadow: 0 0 15px rgba(52,211,153,1); }

@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
.stSpinner { animation: pulse 1.6s ease-in-out infinite; }
</style>
"""


def apply_cyberpunk_theme():
    """Inject the cyberpunk/AMOLED stylesheet into the current Streamlit page."""
    st.markdown(_CYBERPUNK_CSS, unsafe_allow_html=True)
