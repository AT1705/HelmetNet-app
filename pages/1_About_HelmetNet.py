import streamlit as st

st.set_page_config(
    page_title="About HelmetNet",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GOV PORTAL CSS (Sub-pages: modern sidebar + icons + active highlight)
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

.hn-tag{
  display:inline-flex; align-items:center; gap: 8px;
  border-radius: 999px; padding: 6px 10px;
  background: rgba(0,45,98,0.06);
  border: 1px solid rgba(0,45,98,0.14);
  color: rgba(0,45,98,0.95);
  font-weight: 800;
  font-size: 0.86rem;
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
  <div class="hn-title">About HelmetNet</div>
  <div class="hn-sub">Research Journey · Model Iterations (Experiment 1 → 4) · CSC738</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# CONTENT
# ============================================================
c1, c2 = st.columns([1.15, 0.85], gap="large")

with c1:
    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-diagram-project"></i> Project Overview</h3>
  <div class="hn-muted">
    HelmetNet is a Computer Vision pipeline designed to detect helmet compliance for motorcycle riders.
    This portal demonstrates how model quality evolves through iterative dataset labeling, class definitions,
    and annotation discipline. The system supports inference across <strong>Images</strong>, <strong>Videos</strong>,
    and <strong>Real-time streams</strong>.
    <br><br>
    The redesign you are viewing emphasizes a professional “government portal” experience: guided navigation,
    clean configuration, and compliance-oriented insights.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-flask"></i> The 4 Experiments (Model Evolution)</h3>
  <div class="hn-muted">
    Below is the research progression captured as four experiments. Each experiment addresses a concrete failure mode
    and refines the dataset and labeling policy to reduce false positives/negatives and improve generalization.
  </div>
  <div style="margin-top: 12px; display:flex; flex-wrap:wrap; gap:10px;">
    <span class="hn-tag"><i class="fa-solid fa-1"></i>&nbsp;Experiment 1</span>
    <span class="hn-tag"><i class="fa-solid fa-2"></i>&nbsp;Experiment 2</span>
    <span class="hn-tag"><i class="fa-solid fa-3"></i>&nbsp;Experiment 3</span>
    <span class="hn-tag"><i class="fa-solid fa-4"></i>&nbsp;Experiment 4</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("Experiment 1 — Poor cap detection (baseline limitations)", expanded=True):
        st.markdown(
            """
- **Observed issue:** The model frequently confused **caps / head coverings** with helmets, producing poor discrimination.
- **Root cause:** Early dataset labeling lacked consistent rules for “helmet vs non-helmet headwear”, resulting in noisy supervision.
- **Impact:** High false positives for helmet compliance, lowering enforcement reliability.
- **Learning:** Model performance is bottlenecked by labeling policy quality more than raw architecture.
"""
        )

    with st.expander("Experiment 2 — Refined classes and initial relabeling"):
        st.markdown(
            """
- **Adjustment:** Introduced stricter separation of compliant vs non-compliant cases and began cleaning ambiguous samples.
- **Better negative examples:** Added more “non-helmet” variants (caps, hoodies, blurred heads) to teach the decision boundary.
- **Result:** Reduced cap-as-helmet confusion, but edge cases remained (angles, partial occlusion, low light).
"""
        )

    with st.expander("Experiment 3 — Hard-case mining and annotation consistency"):
        st.markdown(
            """
- **Adjustment:** Focused on *hard cases* (side profiles, motion blur, group scenes, small riders).
- **Policy improvement:** More consistent bounding box placement and class naming normalization.
- **Result:** Noticeable stability improvement, fewer spurious detections on background objects.
"""
        )

    with st.expander("Experiment 4 — Optimized labeling and compliance focus (best model)"):
        st.markdown(
            """
- **Adjustment:** Finalized a strict labeling rubric aligned to compliance outcomes (helmet vs no helmet) and reduced ambiguity.
- **Balanced dataset:** Better distribution of compliant/non-compliant, lighting, camera distance, and viewpoints.
- **Result:** Best overall operational behavior: improved precision/recall tradeoff and clearer compliance signaling.
"""
        )

with c2:
    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-landmark"></i> Compliance Orientation</h3>
  <div class="hn-muted">
    HelmetNet is positioned as a compliance-support tool rather than a purely technical demo.
    The portal integrates prescriptive guidance referencing:
    <ul>
      <li><strong>Section 119(2) Road Transport Act 1987</strong> (non-compliance signaling)</li>
      <li><strong>SIRIM MS 1:2011</strong> (helmet compliance standard reference)</li>
    </ul>
    In the detection demo page, the Safety Action Panel changes automatically based on detection outcomes.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
<div class="hn-card">
  <h3><i class="fa-solid fa-list-check"></i> What This Portal Demonstrates</h3>
  <div class="hn-muted">
    <ol>
      <li><strong>Detection Modes:</strong> Image, Video, and Real-time (Webcam)</li>
      <li><strong>Operational UI:</strong> Clean settings at the top, minimal clutter, clear result signaling</li>
      <li><strong>Prescriptive Analytics:</strong> Actionable guidance for enforcement and safety awareness</li>
    </ol>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")
st.caption("HelmetNet (CSC738) · About Page · JPJ-inspired portal styling")
