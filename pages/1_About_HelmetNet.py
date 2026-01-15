import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="About HelmetNet",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# NAV HELPERS (same idea as landing; supports Back UX)
# ============================================================
def _init_nav():
    if "hn_nav_stack" not in st.session_state:
        st.session_state.hn_nav_stack = []

def nav_to(page_path: str, current_page: str):
    _init_nav()
    st.session_state.hn_nav_stack.append(current_page)

    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    else:
        st.warning("Your Streamlit version does not support automatic navigation.")
        if hasattr(st, "page_link"):
            st.page_link(page_path, label="Open page", icon="➡️")

def nav_back(default_page: str = "app.py"):
    _init_nav()
    target = st.session_state.hn_nav_stack.pop() if st.session_state.hn_nav_stack else default_page

    if hasattr(st, "switch_page"):
        try:
            st.switch_page(target if target != "app.py" else "app.py")
        except Exception:
            st.switch_page("app.py")
    else:
        st.info("Use the sidebar to return Home.")
        if hasattr(st, "page_link"):
            st.page_link("app.py", label="Back to Home", icon="⬅️")

# ============================================================
# HERO IMAGE (replace anytime)
# (Helmet safety / testing lab related image)
# ============================================================
HERO_IMAGE_URL = "https://www.caberg.it/mondo-caberg/wp-content/uploads/2024/12/test-sicurezza-caschi-caberg-1.jpg"

# ============================================================
# CSS (Zebra-like hero: big image + dark overlay + text)
# Also: sidebar menu with clean buttons; hide default multipage nav if desired
# ============================================================
ABOUT_CSS = f"""
<style>
/* Hide Streamlit chrome */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

/* Container width */
.block-container {{
  padding-top: 1rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1250px;
}}

/* FontAwesome */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

:root {{
  --navy:#002d62;
  --navy2:#001a3a;
  --gold:#d4af37;
  --ink:#0b1220;
  --muted: rgba(11,18,32,0.72);
  --panel: rgba(255,255,255,0.94);
  --border: rgba(0,0,0,0.10);
}}

/* Hide default multipage nav (so only your custom sidebar menu shows) */
[data-testid="stSidebarNav"] {{
  display: none !important;
}}

/* Sidebar look */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(0,45,98,0.98) 0%, rgba(0,26,58,0.98) 70%, rgba(0,13,31,0.98) 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.10) !important;
}}
[data-testid="stSidebar"] * {{
  color: rgba(255,255,255,0.92) !important;
}}
.hn-side-title {{
  font-weight: 950;
  font-size: 1.05rem;
  margin: 0.25rem 0 0.25rem 0;
}}
.hn-side-sub {{
  color: rgba(255,255,255,0.80);
  font-size: 0.86rem;
  margin-bottom: 0.65rem;
}}

/* Sidebar button styling */
.stButton > button {{
  border-radius: 10px !important;
  padding: 0.62rem 0.95rem !important;
  font-weight: 900 !important;
  width: 100%;
}}
.hn-side-primary .stButton > button {{
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(242,208,107,0.85)) !important;
  color: #002d62 !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
}}
.hn-side-secondary .stButton > button {{
  background: rgba(255,255,255,0.08) !important;
  color: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}}
.hn-side-ghost .stButton > button {{
  background: transparent !important;
  color: rgba(255,255,255,0.92) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}}

/* HERO (image background + overlay) */
.hn-hero {{
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(0,0,0,0.10);
  min-height: 320px;
  background-image:
    linear-gradient(90deg, rgba(0,0,0,0.72) 0%, rgba(0,0,0,0.35) 55%, rgba(0,0,0,0.10) 100%),
    url("{HERO_IMAGE_URL}");
  background-size: cover;
  background-position: center;
  box-shadow: 0 18px 44px rgba(0,0,0,0.18);
}}
.hn-hero-inner {{
  padding: 28px 26px;
  max-width: 920px;
}}
.hn-hero-kicker {{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.10);
  border: 1px solid rgba(255,255,255,0.18);
  color: rgba(255,255,255,0.92);
  font-weight: 850;
  font-size: 0.86rem;
}}
.hn-hero-title {{
  margin: 14px 0 8px 0;
  color: white;
  font-weight: 950;
  font-size: 2.2rem;
  letter-spacing: -0.2px;
}}
.hn-hero-text {{
  margin: 0;
  color: rgba(255,255,255,0.90);
  line-height: 1.6;
  max-width: 70ch;
  font-size: 1.02rem;
}}

/* Content cards */
.hn-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 12px 24px rgba(0,0,0,0.07);
}}
.hn-card h3 {{
  margin: 0 0 10px 0;
  font-size: 1.05rem;
  font-weight: 950;
  color: var(--ink);
}}
.hn-muted {{
  color: var(--muted);
  line-height: 1.6;
}}

.hn-tag {{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  border-radius: 999px;
  padding: 6px 10px;
  background: rgba(0,45,98,0.06);
  border: 1px solid rgba(0,45,98,0.14);
  color: rgba(0,45,98,0.95);
  font-weight: 850;
  font-size: 0.86rem;
}}

/* Responsive */
@media (max-width: 900px) {{
  .block-container{{ padding-left: 1rem !important; padding-right: 1rem !important; }}
  .hn-hero{{ min-height: 360px; }}
  .hn-hero-title{{ font-size: 1.85rem; }}
}}
</style>
"""
st.markdown(ABOUT_CSS, unsafe_allow_html=True)

# ============================================================
# SIDEBAR MENU (Home, Demo, Back)
# ============================================================
with st.sidebar:
    st.markdown('<div class="hn-side-title">HelmetNet</div>', unsafe_allow_html=True)
    st.markdown('<div class="hn-side-sub">Navigation</div>', unsafe_allow_html=True)

    st.markdown('<div class="hn-side-ghost">', unsafe_allow_html=True)
    if st.button("⬅ Back", use_container_width=True):
        nav_back(default_page="app.py")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hn-side-secondary">', unsafe_allow_html=True)
    if st.button("Home (Landing)", use_container_width=True):
        nav_to("app.py", current_page="pages/1_About_HelmetNet.py")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="hn-side-primary">', unsafe_allow_html=True)
    if st.button("Start Demo", use_container_width=True):
        nav_to("pages/2_Detection.py", current_page="pages/1_About_HelmetNet.py")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**References**")
    st.write("• Section 119(2) Road Transport Act 1987")
    st.write("• SIRIM MS 1:2011")

# ============================================================
# HERO SECTION (Zebra-like)
# ============================================================
st.markdown(
    """<div class="hn-hero">
  <div class="hn-hero-inner">
    <div class="hn-hero-kicker"><i class="fa-solid fa-flask"></i>&nbsp;Research Journey · CSC738</div>
    <div class="hn-hero-title">Our Path to HelmetNet</div>
    <p class="hn-hero-text">
      HelmetNet evolved through four structured experiments focused on dataset quality, labeling discipline,
      and compliance clarity. This page documents the journey from early misclassification issues to an optimized
      model suitable for demonstration-grade compliance insights.
    </p>
  </div>
</div>""",
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# CONTENT (Experiments narrative)
# ============================================================
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown(
        """<div class="hn-card">
  <h3><i class="fa-solid fa-diagram-project"></i> Project Overview</h3>
  <div class="hn-muted">
    HelmetNet is an AI-powered helmet compliance detector supporting <strong>Image</strong>, <strong>Video</strong>,
    and <strong>Real-time (Webcam)</strong> inference. The primary focus of this research was not only model training,
    but also <strong>label quality</strong>, class definitions, and edge-case handling.
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """<div class="hn-card">
  <h3><i class="fa-solid fa-flask"></i> Model Evolution (Experiment 1 → 4)</h3>
  <div class="hn-muted">
    Each iteration corrected a specific failure mode and improved reliability under practical camera conditions.
  </div>
  <div style="margin-top: 12px; display:flex; flex-wrap:wrap; gap:10px;">
    <span class="hn-tag"><i class="fa-solid fa-1"></i>&nbsp;Experiment 1</span>
    <span class="hn-tag"><i class="fa-solid fa-2"></i>&nbsp;Experiment 2</span>
    <span class="hn-tag"><i class="fa-solid fa-3"></i>&nbsp;Experiment 3</span>
    <span class="hn-tag"><i class="fa-solid fa-4"></i>&nbsp;Experiment 4</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("Experiment 1 — Baseline (Poor cap detection)", expanded=True):
        st.markdown(
            """
- **Problem:** Caps / head coverings often misclassified as helmets.
- **Cause:** Inconsistent labeling rules and insufficient negative samples.
- **Impact:** High false compliance rate.
"""
        )

    with st.expander("Experiment 2 — Class refinement + initial relabeling"):
        st.markdown(
            """
- **Change:** Cleaner separation of compliant vs non-compliant examples.
- **Improvement:** Added more negative examples (caps, hoodies, blur).
- **Outcome:** Reduced confusion but edge cases persisted.
"""
        )

    with st.expander("Experiment 3 — Hard-case mining + annotation consistency"):
        st.markdown(
            """
- **Change:** Focused on motion blur, side angles, small rider scale, occlusions.
- **Outcome:** Better stability, fewer spurious detections.
"""
        )

    with st.expander("Experiment 4 — Optimized labeling (Best model)"):
        st.markdown(
            """
- **Change:** Finalized strict labeling rubric aligned to compliance outcomes.
- **Outcome:** Best balance for demo-grade reliability and clearer compliance signaling.
"""
        )

with right:
    st.markdown(
        """<div class="hn-card">
  <h3><i class="fa-solid fa-landmark"></i> Compliance Orientation</h3>
  <div class="hn-muted">
    HelmetNet is positioned as a compliance-support tool. In the detection console, results trigger a Safety Action Panel
    referencing:
    <ul>
      <li><strong>Section 119(2) Road Transport Act 1987</strong> for non-compliance outcomes</li>
      <li><strong>SIRIM MS 1:2011</strong> for compliance reference</li>
    </ul>
    This is designed for academic demonstration and prescriptive analytics narratives.
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """<div class="hn-card">
  <h3><i class="fa-solid fa-list-check"></i> What You Can Do Next</h3>
  <div class="hn-muted">
    <ol>
      <li>Proceed to <strong>Start Demo</strong> to run inference (Image/Video/Live).</li>
      <li>Compare results across <strong>Model 1–4</strong> to see the impact of dataset quality.</li>
      <li>Review <strong>Safety Protocols</strong> for prescriptive guidance and governance notes.</li>
    </ol>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

st.write("")
st.caption("HelmetNet (CSC738) · About Page · Hero-style layout with sidebar navigation")
