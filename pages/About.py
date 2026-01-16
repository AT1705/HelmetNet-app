import streamlit as st

from shared.style import inject_global_css
from shared.components import navbar


st.set_page_config(
    page_title="About | HelmetNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_css()
navbar(active="about")

# Hero
st.markdown(
    """
<section class="hn-cta" style="padding: 72px 0;">
  <div class="hn-container" style="text-align:left; max-width: 80rem;">
    <h1 style="font-size: 3rem; font-weight: 900; margin: 0 0 1rem 0;">About HelmetNet</h1>
    <p style="font-size: 1.125rem; color: rgba(226,232,240,0.85); max-width: 56rem; margin:0;">
      A Computer Vision pipeline designed to detect helmet compliance for motorcycle riders through iterative research and development.
    </p>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# About container (two column)
st.markdown(
    """
<section class="hn-section hn-section-white" style="padding: 64px 0;">
  <div class="hn-container">
    <div class="hn-card" style="overflow:hidden;">
      <div style="padding: 40px;">
        <h2 class="hn-section-title" style="margin-bottom: 24px;">About HelmetNet</h2>

        <div class="hn-about-grid">
          <div class="hn-about-copy">
            <p class="hn-about-p">
              HelmetNet is a Computer Vision pipeline designed to detect helmet compliance for motorcycle riders.
              This portal demonstrates how model quality evolves through iterative dataset labeling, class definitions,
              and annotation discipline.
            </p>
            <p class="hn-about-p">
              The system supports inference across <span class="hn-strong">Images</span>,
              <span class="hn-strong">Videos</span>, and <span class="hn-strong">Real-time streams</span>.
            </p>

            <div class="hn-about-callout">
              The redesign you are viewing emphasizes a professional "government portal" experience: guided navigation,
              clean configuration, and compliance-oriented insights.
            </div>
          </div>

          <div class="hn-about-media">
            <div class="hn-about-image"></div>
          </div>
        </div>

        <div style="margin-top: 40px;">
          <h3 style="font-size: 1.5rem; font-weight: 800; margin: 0 0 18px 0;">Compliance Orientation</h3>
          <p class="hn-muted" style="margin: 0 0 20px 0; font-size: 1rem;">
            HelmetNet is positioned as a compliance-support tool rather than a purely technical demo. The portal integrates prescriptive guidance referencing:
          </p>

          <div class="hn-ref-grid">
            <div class="hn-ref">
              <div class="hn-ref-title">Section 119(2) Road Transport Act 1987</div>
              <div class="hn-muted">Non-compliance signaling and legal framework for helmet enforcement</div>
            </div>
            <div class="hn-ref">
              <div class="hn-ref-title">SIRIM MS 1:2011</div>
              <div class="hn-muted">Helmet compliance standard reference for safety certification</div>
            </div>
          </div>
        </div>

        <div style="margin-top: 40px;">
          <h3 style="font-size: 1.5rem; font-weight: 800; margin: 0 0 18px 0;">Technology Stack</h3>
          <div class="hn-tech-grid">
            <div class="hn-tech">
              <div class="hn-tech-title">Deep Learning</div>
              <div class="hn-muted" style="font-size: 0.95rem;">CNN-based models trained for helmet detection with 99.2% accuracy</div>
            </div>
            <div class="hn-tech">
              <div class="hn-tech-title">Computer Vision</div>
              <div class="hn-muted" style="font-size: 0.95rem;">Multi-angle detection with real-time image processing capabilities</div>
            </div>
            <div class="hn-tech">
              <div class="hn-tech-title">Edge Computing</div>
              <div class="hn-muted" style="font-size: 0.95rem;">Sub-second detection time with local processing for privacy</div>
            </div>
          </div>
        </div>

      </div>
    </div>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# Experiments (requested as dropdowns)
experiments = [
    {
        "number": 1,
        "title": "Poor cap detection (baseline limitations)",
        "issue": "The model frequently confused caps / head coverings with helmets, producing poor discrimination.",
        "learning": "Model performance is bottlenecked by labeling policy quality more than raw architecture.",
    },
    {
        "number": 2,
        "title": "Helmet type classification refinement",
        "issue": "System struggled to distinguish between different helmet types and safety standards.",
        "learning": "Dataset diversity across helmet types and viewing angles is critical for robust detection.",
    },
    {
        "number": 3,
        "title": "Real-time stream optimization",
        "issue": "Processing latency exceeded acceptable thresholds for live traffic monitoring.",
        "learning": "Architecture optimization and edge computing integration essential for real-time applications.",
    },
    {
        "number": 4,
        "title": "Multi-angle detection enhancement",
        "issue": "Detection accuracy dropped significantly for side and rear viewing angles.",
        "learning": "Comprehensive multi-angle dataset coverage ensures consistent performance across deployment scenarios.",
    },
]

st.markdown(
    """
<section class="hn-section hn-section-slate" style="padding: 64px 0;">
  <div class="hn-container" style="text-align:center;">
    <h2 class="hn-section-title">The 4 Experiments</h2>
    <p class="hn-section-subtitle" style="max-width: 56rem;">
      Research progression addressing concrete failure modes and refining the dataset
    </p>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# Grid of expanders styled via global CSS
wrap = st.container()
with wrap:
    cols = st.columns(2, gap="large")
    for i, exp in enumerate(experiments):
        with cols[i % 2]:
            with st.expander(f"E{exp['number']} — {exp['title']}", expanded=False):
                st.markdown(
                    f"""
<div class="hn-expander-body">
  <div style="margin-bottom: 14px;">
    <div class="hn-expander-label">Issue</div>
    <div class="hn-muted">{exp['issue']}</div>
  </div>
  <div>
    <div class="hn-expander-label">Learning</div>
    <div style="color: #374151;">{exp['learning']}</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

