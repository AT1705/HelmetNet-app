import streamlit as st

st.set_page_config(
    page_title="HelmetNet | Safety Protocols",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GOV PORTAL CSS (Same as other sub-pages)
# ============================================================
GOV_CSS_SUBPAGES = """
<style>
.block-container { padding-top: 1.1rem !important; padding-bottom: 2.5rem !important; max-width: 1200px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

:root{
  --jpj-navy:#002d62;
  --jpj-gold:#d4af37;
  --ink:#0b1220;
  --muted: rgba(11,18,32,0.70);
  --panel: rgba(255,255,255,0.92);
  --panel-border: rgba(0,0,0,0.08);
}

/* FontAwesome */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css');

/* Sidebar styling */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(0,45,98,0.98) 0%, rgba(0,26,58,0.98) 70%, rgba(0,13,31,0.98) 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.10) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.92) !important; }

/* Sidebar nav (multipage) */
[data-testid="stSidebarNav"]{
  padding-top: 0.25rem !important;
}
[data-testid="stSidebarNav"] ul{
  padding: 0.25rem 0.35rem !important;
}
[data-testid="stSidebarNav"] li a{
  border-radius: 12px !important;
  padding: 0.55rem 0.75rem !important;
  margin: 0.20rem 0 !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  font-weight: 750 !important;
}
[data-testid="stSidebarNav"] li a:hover{
  background: rgba(212,175,55,0.18) !important;
  border: 1px solid rgba(212,175,55,0.35) !important;
}
[data-testid="stSidebarNav"] li a[aria-current="page"]{
  background: linear-gradient(135deg, rgba(212,175,55,0.95), rgba(242,208,107,0.80)) !important;
  color: #002d62 !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.18) !important;
}

/* Add icons to the 3 pages via nth-of-type (About=1, Detection=2, Safety=3) */
[data-testid="stSidebarNav"] li:nth-of-type(1) a::before{
  font-family: "Font Awesome 6 Free"; font-weight: 900; content: "\\f02d";
  margin-right: 10px;
}
[data-testid="stSidebarNav"] li:nth-of-type(2) a::before{
  font-family: "Font Awesome 6 Free"; font-weight: 900; content: "\\f06e";
  margin-right: 10px;
}
[data-testid="stSidebarNav"] li:nth-of-type(3) a::before{
  font-family: "Font Awesome 6 Free"; font-weight: 900; content: "\\f0e3";
  margin-right: 10px;
}

/* Page header */
.hn-pagehead{
  border-radius: 16px;
  padding: 18px 18px;
  background: radial-gradient(1000px 260px at 10% 10%, rgba(212,175,55,0.22), rgba(0,45,98,0.0) 55%),
              linear-gradient(135deg, rgba(0,45,98,0.98) 0%, rgba(0,26,58,0.98) 70%, rgba(0,13,31,0.98) 100%);
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 14px 34px rgba(0,0,0,0.14);
  color: white;
}
.hn-title{ font-size: 1.75rem; font-weight: 900; margin: 0; }
.hn-sub{ margin: 6px 0 0 0; opacity: 0.92; font-size: 0.98rem; }

.hn-card{
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 12px 24px rgba(0,0,0,0.08);
}
.hn-card h3{ margin: 0 0 10px 0; font-size: 1.05rem; font-weight: 900; color: var(--ink); }
.hn-muted{ color: var(--muted); line-height: 1.6; }

.hn-callout{
  border-radius: 14px;
  padding: 14px 14px;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(0,45,98,0.05);
}

@media (max-width: 880px){
  .block-container{ padding-left: 1rem !important; padding-right: 1rem !important; }
}
</style>
"""
st.markdown(GOV_CSS_SUBPAGES, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="hn-pagehead">
  <div class="hn-title">Safety Protocols & Prescriptive Analytics</div>
  <div class="hn-sub">Operational guidance, compliance references, and recommended response workflow</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# CONTENT
# ============================================================
a, b = st.columns([1.1, 0.9], gap="large")

with a:
    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-gavel"></i> Regulatory & Standards References</h3>
  <div class="hn-muted">
    HelmetNet’s prescriptive outputs are structured around two compliance anchors:
    <ul>
      <li><strong>Section 119(2) Road Transport Act 1987</strong> — used to frame non-compliance scenarios (no helmet detected).</li>
      <li><strong>SIRIM MS 1:2011</strong> — used as a compliance reference when a helmet is detected.</li>
    </ul>
    This portal does not issue legal determinations. It provides <strong>decision support</strong> and compliance-oriented messaging
    suitable for demonstrations, analytics, and SOP-aligned workflows.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-sitemap"></i> Prescriptive Analytics Logic (Operational Rules)</h3>
  <div class="hn-muted">
    The system converts detections into recommended actions using a simple rule layer:
    <ol>
      <li><strong>If “no helmet” detected</strong> → classify as non-compliance event and trigger safety action guidance.</li>
      <li><strong>If helmet detected and no violations</strong> → classify as compliant event and recommend positive reinforcement / KPI logging.</li>
      <li><strong>Always log</strong> the model variant, confidence threshold, timestamp, and counts for auditability.</li>
    </ol>
  </div>
  <div class="hn-callout">
    <div class="hn-muted">
      <strong>Recommended KPI Set:</strong> Compliance rate (%), violations per hour, hotspot locations, confidence distribution,
      repeat occurrences by time window, and false-positive review rate (human QA sampling).
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with b:
    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-clipboard-check"></i> Suggested SOP Response Workflow</h3>
  <div class="hn-muted">
    A practical response workflow for a pilot deployment:
    <ol>
      <li><strong>Detect</strong> event (image/video/live) and compute compliance counts.</li>
      <li><strong>Flag</strong> non-compliance segments for review (avoid acting on single-frame anomalies).</li>
      <li><strong>Review</strong> with an operator for edge cases (occlusion, blur, partial rider visibility).</li>
      <li><strong>Record</strong> the event summary to an audit log (model version, threshold, timestamp).</li>
      <li><strong>Respond</strong> using SOP: education messaging, warnings, or escalation as applicable.</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-user-shield"></i> Governance & Deployment Notes</h3>
  <div class="hn-muted">
    For real-world usage (beyond academic demo), implement:
    <ul>
      <li><strong>Privacy controls:</strong> data minimization, retention limits, access control.</li>
      <li><strong>Model governance:</strong> versioning, bias checks, threshold management.</li>
      <li><strong>Auditability:</strong> event logs and QA sampling for performance drift.</li>
      <li><strong>Security:</strong> secure secrets management (TURN credentials, access keys).</li>
    </ul>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")
st.caption("HelmetNet (CSC738) · Safety Protocols Page · Prescriptive analytics guidance")
