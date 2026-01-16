import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
SITE_HTML_PATH = APP_DIR / "static" / "site" / "index.html"

st.set_page_config(page_title="HelmetNet", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

html = SITE_HTML_PATH.read_text(encoding="utf-8")

# Force iframe rendering so JS works (React-like navigation will function)
components.html(html, height=1400, scrolling=False)
