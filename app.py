from __future__ import annotations

from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SITE_HTML_PATH = APP_DIR / "static" / "site" / "index.html"

st.set_page_config(
    page_title="HelmetNet",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Streamlit chrome removal + true edge-to-edge canvas ---
st.markdown(
    """
    <style>
      /* Hide built-in Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Remove Streamlit default padding so the site can be full-bleed */
      .stApp { padding: 0 !important; }
      [data-testid="stAppViewContainer"] { padding: 0 !important; }
      [data-testid="stMain"] { padding: 0 !important; }
      [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }

      /* Ensure the embedded HTML can fully control its own layout */
      .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

if not SITE_HTML_PATH.exists():
    st.error(f"Missing site HTML at: {SITE_HTML_PATH}")
    st.stop()

# Read HTML content directly and render inline.
html = SITE_HTML_PATH.read_text(encoding="utf-8")

# Streamlit's modern API (preferred). Falls back gracefully if not available.
if hasattr(st, "html"):
    # st.html can render without an iframe and (optionally) run JS.
    st.html(html, unsafe_allow_javascript=True, width="stretch")
else:
    # Legacy fallback: renders in an iframe.
    import streamlit.components.v1 as components

    components.html(html, height=1200, scrolling=False)
