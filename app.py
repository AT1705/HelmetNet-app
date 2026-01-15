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
            st.switch_page(target if target != "app.py" else "app.py")
        except Exception:
            st.switch_page("app.py")
    else:
        st.info("Use the sidebar to navigate back.")
        if hasattr(st, "page_link"):
            st.page_link("app.py", label="Back to Home", icon="⬅️")


# ============================================================
# LANDING CSS (Clean SaaS hero + two CTA buttons like sample)
# ============================================================
LANDING_CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Hide sidebar on landing */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Page container */
.block-container{
  padding-top: 1.2rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1180px;
}

/* FontAwesome for clean icons */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

:root{
  --navy:#002d62;
  --navy2:#001a3a;
  --gold:#d4af37;
  --ink:#0b1220;
  --muted: rgba(11,18,32,0.72);
  --line: rgba(2, 45, 98, 0.16);
  --bg: #ffffff;
}

.hn-topbar{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 10px;
  padding: 0.55rem 0.75rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(0,45,98,0.03);
}
.hn-brand{
  display:flex;
  align-items:center;
  gap: 10px;
}
.hn-logo{
  width: 42px; height: 42px;
  border-radius: 12px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight: 950;
  color: var(--navy);
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(255,255,255,0.45));
  border: 1px solid rgba(0,0,0,0.08);
}
.hn-brand-title{ font-weight: 950; color: var(--ink); line-height: 1.1; }
.hn-brand-sub{ font-size: 0.86rem; color: var(--muted); margin-top: 2px; }

.hn-chip{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(0,45,98,0.03);
  color: rgba(0,45,98,0.95);
  font-weight: 850;
  font-size: 0.86rem;
  white-space: nowrap;
}

/* Hero */
.hn-hero{
  margin-top: 1.35rem;
  display:grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 22px;
  align-items:center;
}
.hn-hero h1{
  font-size: 2.35rem;
  font-weight: 950;
  margin: 0 0 0.55rem 0;
  color: var(--ink);
  letter-spacing: -0.2px;
}
.hn-lead{
  margin: 0 0 1.1rem 0;
  color: var(--muted);
  line-height: 1.6;
  max-width: 62ch;
}
.hn-micro{
  display:flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 0.65rem;
}
.hn-micro span{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(0,0,0,0.02);
  color: rgba(11,18,32,0.80);
  font-weight: 800;
  font-size: 0.88rem;
}

/* CTA row */
.hn-cta-row{
  display:flex;
  gap: 12px;
  align-items:center;
  margin-top: 0.25rem;
}

/* Button styling (Streamlit buttons inside wrappers) */
.stButton > button{
  border-radius: 8px !important;
  padding: 0.55rem 1.05rem !important;
  font-weight: 900 !important;
}

/* Primary (filled) */
.hn-btn-primary .stButton > button{
  background: var(--navy) !important;
  color: #ffffff !important;
  border: 1px solid var(--navy) !important;
}
.hn-btn-primary .stButton > button:hover{
  background: var(--navy2) !important;
  border: 1px solid var(--navy2) !important;
}

/* Secondary (outlined) */
.hn-btn-secondary .stButton > button{
  background: #ffffff !important;
  color: var(--navy) !important;
  border: 1px solid rgba(0,45,98,0.45) !important;
}
.hn-btn-secondary .stButton > button:hover{
  background: rgba(0,45,98,0.05) !important;
}

/* Right illustration panel */
.hn-visual{
  border-radius: 18px;
  border: 1px solid rgba(0,0,0,0.08);
  background:
    radial-gradient(700px 360px at 75% 25%, rgba(0,45,98,0.12), rgba(0,45,98,0) 60%),
    radial-gradient(520px 240px at 30% 70%, rgba(212,175,55,0.18), rgba(212,175,55,0) 55%),
    linear-gradient(135deg, rgba(0,0,0,0.02), rgba(0,0,0,0.00));
  padding: 18px 18px;
  min-height: 280px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.hn-visual .title{
  font-weight: 950;
  color: var(--ink);
  font-size: 1.05rem;
}
.hn-visual .desc{
  margin-top: 6px;
  color: var(--muted);
  line-height: 1.55;
  font-size: 0.95rem;
}
.hn-visual .grid{
  margin-top: 14px;
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.hn-tile{
  border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(255,255,255,0.82);
  padding: 10px 10px;
}
.hn-tile .k{ font-weight: 950; color: var(--ink); margin: 0; }
.hn-tile .v{ margin: 3px 0 0 0; color: rgba(11,18,32,0.70); font-size: 0.88rem; line-height: 1.45; }

/* Responsive */
@media (max-width: 900px){
  .hn-hero{ grid-template-columns: 1fr; }
  .hn-cta-row{ flex-direction: column; align-items: stretch; }
  .block-container{ padding-left: 1rem !important; padding-right: 1rem !important; }
  .hn-chip{ display:none; }
}
</style>
"""
st.markdown(LANDING_CSS, unsafe_allow_html=True)

# ============================================================
# TOP BAR (simple portal identity)
# ============================================================
st.markdown(
    """<div class="hn-topbar">
  <div class="hn-brand">
    <div class="hn-logo">HN</div>
    <div>
      <div class="hn-brand-title">HelmetNet</div>
      <div class="hn-brand-sub">JPJ-inspired Computer Vision Portal · CSC738</div>
    </div>
  </div>
  <div class="hn-chip"><i class="fa-solid fa-shield-halved"></i>&nbsp;Operational Demo</div>
</div>""",
    unsafe_allow_html=True,
)

# ============================================================
# HERO SECTION (Left copy + Right visual + Two Buttons)
# ============================================================
st.markdown('<div class="hn-hero">', unsafe_allow_html=True)

# Left side
st.markdown(
    """<div>
  <h1>AI Helmet Compliance Detection</h1>
  <p class="hn-lead">
    HelmetNet provides automated detection for motorcycle helmet usage across images, videos, and real-time webcam streams.
    The interface is designed for a clean, credible government-portal experience with guided user flow and compliance insights.
  </p>
</div>""",
    unsafe_allow_html=True,
)

# Right side visual (CSS-based; no external images required)
st.markdown(
    """<div class="hn-visual">
  <div>
    <div class="title"><i class="fa-solid fa-chart-line"></i>&nbsp;Compliance Dashboard</div>
    <div class="desc">
      Run Image, Video, and Live detection. Generate compliance insights referencing Section 119(2) Road Transport Act 1987
      and SIRIM MS 1:2011.
    </div>
  </div>
  <div class="grid">
    <div class="hn-tile">
      <p class="k"><i class="fa-solid fa-image"></i>&nbsp;Image</p>
      <p class="v">Fast inference with downloadable annotated output.</p>
    </div>
    <div class="hn-tile">
      <p class="k"><i class="fa-solid fa-film"></i>&nbsp;Video</p>
      <p class="v">Frame skipping for smooth processing and preview.</p>
    </div>
    <div class="hn-tile">
      <p class="k"><i class="fa-solid fa-video"></i>&nbsp;Live</p>
      <p class="v">WebRTC mode for desktop and mobile usage.</p>
    </div>
    <div class="hn-tile">
      <p class="k"><i class="fa-solid fa-gavel"></i>&nbsp;Policy</p>
      <p class="v">Prescriptive guidance panel for outcomes.</p>
    </div>
  </div>
</div>""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)  # close hn-hero

# ============================================================
# CTA BUTTONS (Same row like your sample)
# ============================================================
st.write("")
cta1, cta2, cta3 = st.columns([1, 1, 2], gap="small")

with cta1:
    st.markdown('<div class="hn-btn-secondary">', unsafe_allow_html=True)
    if st.button("About HelmetNet", use_container_width=True):
        nav_to("pages/1_About_HelmetNet.py", current_page="app.py")
    st.markdown("</div>", unsafe_allow_html=True)

with cta2:
    st.markdown('<div class="hn-btn-primary">', unsafe_allow_html=True)
    if st.button("Start Demo", use_container_width=True):
        nav_to("pages/2_Detection.py", current_page="app.py")
    st.markdown("</div>", unsafe_allow_html=True)

with cta3:
    # Optional: micro trust line like in your sample (customers trust…)
    st.markdown(
        '<div style="display:flex; align-items:center; height:100%; color: rgba(11,18,32,0.62); font-weight: 800; font-size: 0.88rem;">'
        '<i class="fa-solid fa-circle-check"></i>&nbsp;Professional Demo Portal · Clean UI/UX · Research-backed Models'
        "</div>",
        unsafe_allow_html=True,
    )

st.write("")
st.caption("HelmetNet (CSC738) · Landing Page · © 2025–2026")



