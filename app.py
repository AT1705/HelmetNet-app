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
# NAVIGATION HELPER (works across Streamlit versions)
# ============================================================
def nav_to(page_path: str):
    """
    Prefer st.switch_page if available; fallback to a visible page_link.
    This avoids hard errors on older Streamlit builds.
    """
    if hasattr(st, "switch_page"):
        st.switch_page(page_path)
    else:
        # Older Streamlit: no programmatic switch; show the link as fallback
        st.warning("Your Streamlit version does not support automatic navigation. Use the link below.")
        if hasattr(st, "page_link"):
            st.page_link(page_path, label="Open page", icon="➡️")
        else:
            st.info(f"Open from sidebar: {page_path}")

# ============================================================
# JPJ-INSPIRED LANDING CSS (NO sidebar + portal look)
# ============================================================
LANDING_CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Hide sidebar + multipage nav on landing */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Layout */
.block-container {
  padding-top: 0.75rem !important;
  padding-bottom: 2.5rem !important;
  max-width: 1200px;
}

/* FontAwesome for professional icons */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

:root{
  --navy:#002d62;
  --navy2:#001a3a;
  --gold:#d4af37;
  --gold2:#f2d06b;
  --ink:#0b1220;
  --muted: rgba(11,18,32,0.72);
  --panel: rgba(255,255,255,0.94);
  --border: rgba(0,0,0,0.10);
}

/* Top utility bar (JPJ-like) */
.hn-utility {
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(0,45,98,0.06);
  border: 1px solid rgba(0,45,98,0.12);
}
.hn-utility .left, .hn-utility .right{
  display:flex;
  align-items:center;
  gap: 10px;
  flex-wrap: wrap;
}
.hn-chip{
  display:inline-flex;
  align-items:center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(0,45,98,0.06);
  border: 1px solid rgba(0,45,98,0.12);
  color: rgba(0,45,98,0.95);
  font-weight: 800;
  font-size: 0.86rem;
  text-decoration:none;
}

/* Hero */
.hn-hero{
  margin-top: 12px;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.22);
  background:
    radial-gradient(1000px 420px at 15% 15%, rgba(212,175,55,0.26), rgba(0,45,98,0) 60%),
    linear-gradient(135deg, var(--navy) 0%, var(--navy2) 60%, #000d1f 100%);
  box-shadow: 0 18px 44px rgba(0,0,0,0.18);
  padding: 22px 22px;
  color: white;
}
.hn-header{
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
}
.hn-brand{
  display:flex;
  align-items:center;
  gap: 12px;
}
.hn-badge{
  width: 46px; height: 46px;
  border-radius: 14px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(255,255,255,0.35));
  color: var(--navy);
  font-weight: 900;
  box-shadow: 0 12px 20px rgba(0,0,0,0.20);
}
.hn-brand-title{
  font-weight: 900;
  letter-spacing: 0.2px;
  font-size: 1.05rem;
  line-height: 1.1;
}
.hn-brand-sub{
  opacity: 0.92;
  font-size: 0.86rem;
  margin-top: 2px;
}
.hn-status{
  display:flex;
  align-items:center;
  gap: 10px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.18);
  font-size: 0.86rem;
  white-space: nowrap;
}

.hn-grid{
  margin-top: 16px;
  display:grid;
  grid-template-columns: 1.15fr 0.85fr;
  gap: 16px;
}
.hn-title{
  font-size: 2.25rem;
  font-weight: 950;
  margin: 14px 0 10px 0;
}
.hn-lead{
  margin: 0 0 14px 0;
  color: rgba(255,255,255,0.92);
  line-height: 1.6;
  max-width: 65ch;
}

.hn-panel{
  background: var(--panel);
  color: var(--ink);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 14px 14px;
  box-shadow: 0 14px 28px rgba(0,0,0,0.12);
}
.hn-panel h3{
  margin: 0 0 10px 0;
  font-size: 1.05rem;
  font-weight: 950;
}
.hn-row{
  display:flex;
  gap: 10px;
  align-items:flex-start;
  padding: 10px 10px;
  border-radius: 12px;
  background: rgba(0,45,98,0.04);
  border: 1px solid rgba(0,45,98,0.10);
  margin-bottom: 10px;
}
.hn-ic{
  width: 34px; height: 34px;
  border-radius: 10px;
  display:flex;
  align-items:center;
  justify-content:center;
  background: rgba(0,45,98,0.10);
  color: var(--navy);
  flex: 0 0 auto;
}
.hn-k{ font-weight: 900; margin: 0; }
.hn-v{ margin: 3px 0 0 0; color: rgba(11,18,32,0.78); font-size: 0.92rem; line-height: 1.45; }

/* Announcement bar */
.hn-announce{
  margin-top: 12px;
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(212,175,55,0.18);
  border: 1px solid rgba(212,175,55,0.35);
  color: rgba(255,255,255,0.96);
}

/* Streamlit buttons styled as portal CTA */
.stButton > button{
  width: 100%;
  border-radius: 12px !important;
  padding: 0.72rem 1.05rem !important;
  font-weight: 950 !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}
.hn-btn-primary .stButton > button{
  background: linear-gradient(135deg, var(--gold) 0%, var(--gold2) 60%, var(--gold) 100%) !important;
  color: var(--navy) !important;
  box-shadow: 0 14px 26px rgba(0,0,0,0.18) !important;
}
.hn-btn-secondary .stButton > button{
  background: rgba(255,255,255,0.10) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
}

/* Responsive */
@media (max-width: 880px){
  .hn-grid{ grid-template-columns: 1fr; }
  .hn-title{ font-size: 1.85rem; }
  .hn-status{ display:none; }
  .block-container{ padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
"""
st.markdown(LANDING_CSS, unsafe_allow_html=True)

# ============================================================
# TOP UTILITY BAR (JPJ-style)
# JPJ site shows language + portal/links style navigation. :contentReference[oaicite:1]{index=1}
# ============================================================
utility_html = """<div class="hn-utility">
  <div class="left">
    <span class="hn-chip"><i class="fa-solid fa-landmark"></i>&nbsp;Portal Rasmi HelmetNet</span>
    <span class="hn-chip"><i class="fa-solid fa-language"></i>&nbsp;BM</span>
    <span class="hn-chip">ENG</span>
  </div>
  <div class="right">
    <span class="hn-chip"><i class="fa-solid fa-circle-info"></i>&nbsp;CSC738</span>
    <span class="hn-chip"><i class="fa-solid fa-shield-halved"></i>&nbsp;Safety Analytics</span>
  </div>
</div>"""
st.markdown(utility_html, unsafe_allow_html=True)

# ============================================================
# HERO (NO indentation that triggers Markdown code blocks)
# ============================================================
hero_html = """<div class="hn-hero">
  <div class="hn-header">
    <div class="hn-brand">
      <div class="hn-badge">HN</div>
      <div>
        <div class="hn-brand-title">HelmetNet</div>
        <div class="hn-brand-sub">JPJ-Inspired Computer Vision Portal · CSC738</div>
      </div>
    </div>
    <div class="hn-status"><i class="fa-solid fa-signal"></i>&nbsp;Operational Demo · Analytics Enabled</div>
  </div>

  <div class="hn-grid">
    <div>
      <div class="hn-title">AI Helmet Compliance Detection</div>
      <p class="hn-lead">
        HelmetNet provides automated detection for motorcycle helmet usage across images, videos, and real-time webcam streams.
        The portal design emphasizes clean configuration, guided user flow, and compliance-oriented insights.
      </p>

      <div class="hn-announce">
        <i class="fa-solid fa-bullhorn"></i>&nbsp;
        Demo Mode: Results are for academic demonstration and decision-support only.
      </div>

      <div style="height: 14px;"></div>

      <div class="hn-panel">
        <h3>Quick Access</h3>

        <div class="hn-row">
          <div class="hn-ic"><i class="fa-solid fa-book-open"></i></div>
          <div>
            <p class="hn-k">About HelmetNet</p>
            <p class="hn-v">Research journey and the 4 experiments (Model 1 → Model 4), including labeling optimizations.</p>
          </div>
        </div>

        <div class="hn-row">
          <div class="hn-ic"><i class="fa-solid fa-eye"></i></div>
          <div>
            <p class="hn-k">Start Detection Demo</p>
            <p class="hn-v">Run inference on Image, Video, and Real-time (Webcam) modes with a clean console layout.</p>
          </div>
        </div>

        <div class="hn-row" style="margin-bottom:0;">
          <div class="hn-ic"><i class="fa-solid fa-gavel"></i></div>
          <div>
            <p class="hn-k">Safety Protocols</p>
            <p class="hn-v">Prescriptive guidance referencing Section 119(2) Road Transport Act 1987 and SIRIM MS 1:2011.</p>
          </div>
        </div>
      </div>
    </div>

    <div class="hn-panel">
      <h3>Portal Actions</h3>
      <div style="color: rgba(11,18,32,0.72); line-height: 1.55; margin-bottom: 10px;">
        Use the official portal actions below to proceed.
      </div>
      <div style="height: 6px;"></div>
      <!-- Buttons are rendered by Streamlit below (more reliable than HTML buttons). -->
      <div style="color: rgba(11,18,32,0.72); font-size: 0.92rem; line-height: 1.55;">
        Recommended: Review “About HelmetNet” first, then start the demo.
      </div>
    </div>
  </div>
