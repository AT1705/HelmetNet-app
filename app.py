from __future__ import annotations

from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
SITE_DIR = APP_DIR / "static" / "site"

HOME_HTML = SITE_DIR / "index.html"
ABOUT_HTML = SITE_DIR / "about.html"
DEMO_HTML = SITE_DIR / "demo.html"


def inject_edge_to_edge_css():
    st.markdown(
        """
        <style>
          /* Hide Streamlit chrome */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}

          /* Full-bleed canvas */
          .stApp { padding: 0 !important; }
          [data-testid="stAppViewContainer"] { padding: 0 !important; }
          [data-testid="stMain"] { padding: 0 !important; }
          [data-testid="stMainBlockContainer"] {
            padding: 0 !important;
            max-width: 100% !important;
          }
          .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_page() -> str:
    # Newer Streamlit
    try:
        return st.query_params.get("page", "home")
    except Exception:
        # Older Streamlit
        qp = st.experimental_get_query_params()
        return qp.get("page", ["home"])[0]


def render_html(path: Path):
    if not path.exists():
        st.error(f"Missing HTML file: {path}")
        st.stop()

    html = path.read_text(encoding="utf-8")

    # Prefer st.html when available
    if hasattr(st, "html"):
        st.html(html, unsafe_allow_javascript=False, width="stretch")
    else:
        import streamlit.components.v1 as components
        components.html(html, height=1600, scrolling=False)


st.set_page_config(
    page_title="HelmetNet",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_edge_to_edge_css()

page = get_page()

if page == "about":
    render_html(ABOUT_HTML)
elif page == "demo":
    render_html(DEMO_HTML)
else:
    render_html(HOME_HTML)
