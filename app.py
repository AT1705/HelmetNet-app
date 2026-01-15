import streamlit as st
import textwrap

# ============================================================
# PAGE CONFIG (Landing page hides sidebar initially)
# ============================================================
st.set_page_config(
    page_title="HelmetNet | JPJ-Inspired Portal",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# GLOBAL GOV PORTAL CSS (Landing variant: hide sidebar/nav)
# ============================================================
GOV_CSS_LANDING = """
<style>
/* --- Base Layout --- */
.block-container { padding-top: 0.75rem !important; padding-bottom: 2.5rem !important; max-width: 1200px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* --- Hide sidebar on landing --- */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* --- Typography --- */
:root{
  --jpj-navy: #002d62;
  --jpj-gold: #d4af37;
  --jpj-white: #ffffff;
  --jpj-ink: #0b1220;
  --panel: rgba(255,255,255,0.92);
  --panel-border: rgba(0,0,0,0.08);
}

/* --- FontAwesome (icons) --- */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

/* --- Hero wrapper --- */
.hn-hero {
  position: relative;
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,0.22);
  background: radial-gradient(1200px 500px at 15% 20%, rgba(212,175,55,0.28), rgba(0,45,98,0.0) 55%),
              linear-gradient(135deg, #002d62 0%, #001a3a 55%, #000d1f 100%);
  padding: 34px 28px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}

.hn-topbar {
  display:flex; align-items:center; justify-content:space-between;
  gap: 14px; padding: 10px 14px;
  border-radius: 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.16);
}

.hn-brand {
  display:flex; align-items:center; gap: 12px;
  color: white;
}
.hn-brand-badge{
  width: 44px; height: 44px;
  border-radius: 12px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(255,255,255,0.40));
  color: #002d62;
  font-weight: 900;
  box-shadow: 0 10px 18px rgba(0,0,0,0.20);
}
.hn-brand-title { font-size: 1.05rem; font-weight: 800; letter-spacing: 0.2px; line-height: 1.1; }
.hn-brand-sub { font-size: 0.85rem; opacity: 0.92; margin-top: 2px; }

.hn-pill {
  display:flex; align-items:center; gap: 10px;
  padding: 8px 12px;
  border-radius: 999px;
  color: rgba(255,255,255,0.95);
  background: rgba(255,255,255,0.07);
  border: 1px solid rgba(255,255,255,0.16);
  font-size: 0.86rem;
  white-space: nowrap;
}

.hn-hero-grid{
  display:grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 18px;
  margin-top: 18px;
}

.hn-title{
  color:white;
  font-size: 2.35rem;
  font-weight: 900;
  letter-spacing: 0.2px;
  margin: 12px 0 10px 0;
}
.hn-lead{
  color: rgba(255,255,255,0.92);
  font-size: 1.02rem;
  line-height: 1.55;
  margin: 0 0 16px 0;
  max-width: 60ch;
}

.hn-panel{
  background: rgba(255,255,255,0.92);
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 12px 26px rgba(0,0,0,0.12);
}

.hn-panel h3{
  margin: 0 0 10px 0;
  font-size: 1.0rem;
  font-weight: 850;
  color: #0b1220;
}

.hn-kv{
  display:grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin-top: 8px;
}
.hn-kv .row{
  display:flex; gap: 10px; align-items:flex-start;
  padding: 10px 10px;
  border-radius: 12px;
  background: rgba(0,45,98,0.04);
  border: 1px solid rgba(0,45,98,0.10);
}
.hn-kv .ic{
  width: 34px; height: 34px; border-radius: 10px;
  display:flex; align-items:center; justify-content:center;
  background: rgba(0,45,98,0.10);
  color: #002d62;
  flex: 0 0 auto;
}
.hn-kv .txt .k{ font-weight: 850; color: #0b1220; margin: 0; }
.hn-kv .txt .v{ margin: 2px 0 0 0; color: rgba(11,18,32,0.82); font-size: 0.92rem; }

.hn-divider{
  height: 1px;
  background: rgba(255,255,255,0.18);
  margin: 16px 0 14px 0;
}

/* --- Buttons (Streamlit) --- */
.stButton > button{
  width: 100%;
  border-radius: 12px !important;
  padding: 0.72rem 1.1rem !important;
  font-weight: 850 !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}

.hn-btn-primary .stButton > button{
  background: linear-gradient(135deg, #d4af37 0%, #f2d06b 60%, #d4af37 100%) !important;
  color: #002d62 !important;
  box-shadow: 0 12px 24px rgba(0,0,0,0.18) !important;
}
.hn-btn-secondary .stButton > button{
  background: rgba(255,255,255,0.10) !important;
  color: white !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
}

/* --- Responsive --- */
@media (max-width: 880px){
  .hn-hero-grid{ grid-template-columns: 1fr; }
  .hn-title{ font-size: 1.95rem; }
  .block-container{ padding-left: 1rem !important; padding-right: 1rem !important; }
  .hn-pill{ display:none; }
}
</style>
"""

st.markdown(textwrap.dedent(hero_html), unsafe_allow_html=True)

hero_html = """
<div class="hn-hero">
  <div class="hn-topbar">
    <div class="hn-brand">
      <div class="hn-brand-badge">HN</div>
      <div>
        <div class="hn-brand-title">HelmetNet</div>
        <div class="hn-brand-sub">JPJ-Inspired Computer Vision Portal · CSC738</div>
      </div>
    </div>
    <div class="hn-pill"><i class="fa-solid fa-shield-halved"></i>&nbsp;Operational Demo · Safety Analytics Enabled</div>
  </div>

  <div class="hn-hero-grid">
    <div>
      <div class="hn-title">AI Helmet Compliance Detection</div>
      <p class="hn-lead">
        HelmetNet provides automated detection for motorcycle helmet usage across images, videos,
        and real-time webcam streams. The interface is redesigned to resemble a Malaysian Government Portal
        experience with clean configuration, guided user flow, and compliance insights.
      </p>

      <div class="hn-divider"></div>

      <div class="hn-panel">
        <h3>Quick Start</h3>
        <div class="hn-kv">
          <div class="row">
            <div class="ic"><i class="fa-solid fa-circle-info"></i></div>
            <div class="txt">
              <p class="k">Research Narrative</p>
              <p class="v">Review the 4 experiments and model evolution from initial labeling issues to optimized compliance detection.</p>
            </div>
          </div>
          <div class="row">
            <div class="ic"><i class="fa-solid fa-play"></i></div>
            <div class="txt">
              <p class="k">Interactive Demo</p>
              <p class="v">Run inference on Image, Video, and Real-time detection with a professional dashboard layout.</p>
            </div>
          </div>
          <div class="row">
            <div class="ic"><i class="fa-solid fa-gavel"></i></div>
            <div class="txt">
              <p class="k">Safety Protocols</p>
              <p class="v">Prescriptive guidance referencing Section 119(2) Road Transport Act 1987 and SIRIM MS 1:2011.</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="hn-panel">
      <h3>Portal Actions</h3>
      <div style="margin-top: 10px; color: rgba(11,18,32,0.78); font-size: 0.90rem; line-height: 1.55;">
        Use the buttons below to proceed.
      </div>
    </div>
  </div>
</div>
"""

st.markdown(textwrap.dedent(hero_html), unsafe_allow_html=True)

# Buttons rendered normally (not inside HTML)
c1, c2 = st.columns([1, 1], gap="medium")

with c1:
    st.markdown('<div class="hn-btn-secondary">', unsafe_allow_html=True)
    if st.button("About HelmetNet", use_container_width=True):
        st.switch_page("pages/1_About_HelmetNet.py")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="hn-btn-primary">', unsafe_allow_html=True)
    if st.button("Start Demo", use_container_width=True):
        st.switch_page("pages/2_Detection.py")
    st.markdown("</div>", unsafe_allow_html=True)



