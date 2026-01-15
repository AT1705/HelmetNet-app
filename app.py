import streamlit as st

# ============================================================
# PAGE CONFIG (Landing page)
# ============================================================
st.set_page_config(
    page_title="HelmetNet | Portal Rasmi",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# NAV HELPERS (robust + back-stack)
# ============================================================
def _init_nav():
    if "hn_nav_stack" not in st.session_state:
        st.session_state.hn_nav_stack = []

def nav_to(page_path: str, current_page: str = "app.py"):
    """
    Push current page into a stack, then navigate to the target page.
    Uses st.switch_page when available; otherwise shows a link fallback (no crash).
    """
    _init_nav()
    st.session_state.hn_nav_stack.append(current_page)

    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    else:
        st.warning("Your Streamlit version does not support automatic navigation.")
        if hasattr(st, "page_link"):
            st.page_link(page_path, label="Open page", icon="➡️")

def nav_back(default_page: str = "app.py"):
    """
    Pop the last page from stack and navigate back.
    If empty, go to default_page (landing).
    """
    _init_nav()
    target = st.session_state.hn_nav_stack.pop() if st.session_state.hn_nav_stack else default_page

    if hasattr(st, "switch_page"):
        # If stack stores "app.py", switch_page expects file path.
        # Streamlit supports switching to "app.py" in most setups; if not, fallback gracefully.
        try:
            st.
