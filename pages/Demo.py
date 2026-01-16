import time

import streamlit as st

from shared.style import inject_global_css
from shared.components import navbar, html_table


st.set_page_config(
    page_title="Demo | HelmetNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
navbar(active="demo")

# ------------ Session state ------------
if "detections" not in st.session_state:
    st.session_state.detections = []
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None
if "mode" not in st.session_state:
    st.session_state.mode = "Image Detection"

# ------------ Header ------------
st.markdown(
    """
<section class="hn-cta" style="padding: 48px 0;">
  <div class="hn-container" style="text-align:left;">
    <h1 style="font-size: 2.25rem; font-weight: 900; margin:0 0 0.6rem 0;">HelmetNet Detection System</h1>
    <p style="margin:0; color: rgba(226,232,240,0.85); font-size: 1.05rem;">AI-powered helmet compliance detection</p>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

# ------------ Sidebar (Configuration) ------------
with st.sidebar:
    st.markdown(
        """
<div style="padding: 8px 6px 2px 6px;">
  <div style="font-weight: 900; font-size: 1.05rem; color: var(--hn-slate-900);">Configuration</div>
  <div class="hn-muted" style="font-size: 0.9rem; margin-top: 2px;">Model + session settings</div>
</div>
<hr style="border:none; border-top: 1px solid var(--hn-border); margin: 14px 0 18px 0;" />
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='font-weight: 800; color: var(--hn-slate-700); font-size: 0.85rem; margin-bottom: 8px;'>Model Settings</div>", unsafe_allow_html=True)

    model = st.selectbox(
        "Model Path",
        [
            "YOLOv8 v3.2 (Recommended)",
            "Faster R-CNN",
            "EfficientDet-D4",
        ],
        index=0,
        label_visibility="visible",
    )

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Minimum confidence for detections to be included in the results table.",
    )

    st.markdown("<hr style='border:none; border-top: 1px solid var(--hn-border); margin: 18px 0 14px 0;' />", unsafe_allow_html=True)
    st.markdown("<div style='font-weight: 800; color: var(--hn-slate-700); font-size: 0.85rem; margin-bottom: 10px;'>Session Stats</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div style="display:flex; flex-direction:column; gap: 10px; font-size: 0.95rem;">
  <div style="display:flex; justify-content:space-between;">
    <span class="hn-muted">Total Detections</span>
    <span style="font-weight: 800; color: var(--hn-slate-900);">{len(st.session_state.detections)}</span>
  </div>
  <div style="display:flex; justify-content:space-between;">
    <span class="hn-muted">Model Status</span>
    <span style="font-weight: 800; color: var(--hn-green-500);">
      <span style="display:inline-block; width: 8px; height: 8px; border-radius: 999px; background: var(--hn-green-500); margin-right: 8px;"></span>
      Loaded
    </span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ------------ Main content ------------
# Mode tabs (Streamlit widgets; styled globally)
modes = ["Image Detection", "Video Detection", "Real Time Detection"]
st.session_state.mode = st.radio(
    "",
    options=modes,
    index=modes.index(st.session_state.mode) if st.session_state.mode in modes else 0,
    horizontal=True,
    label_visibility="collapsed",
)

mode = st.session_state.mode

if mode == "Image Detection":
    st.markdown(
        """
<div class="hn-container" style="margin-top: 18px;">
  <div class="hn-card" style="padding: 24px;">
    <div style="display:flex; justify-content:space-between; gap: 16px; flex-wrap: wrap;">
      <div>
        <div style="font-size: 1.25rem; font-weight: 900; color: var(--hn-slate-900); margin-bottom: 6px;">Upload an Image</div>
        <div class="hn-muted" style="font-size: 0.92rem;">Supported formats: JPG, PNG, BMP</div>
      </div>
      <div style="text-align:right; background: var(--hn-slate-50); padding: 12px 14px; border-radius: 10px; border: 1px solid var(--hn-border);">
        <div style="font-size: 0.72rem; color: var(--hn-slate-500); font-weight: 800; margin-bottom: 4px;">Quick Tips</div>
        <div style="font-size: 0.78rem; color: var(--hn-slate-700);">Clear, well-lit images</div>
        <div style="font-size: 0.78rem; color: var(--hn-slate-700);">Max size: 10MB</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    uploader_container = st.container()
    with uploader_container:
        st.markdown('<div class="hn-container" style="margin-top: 14px;">', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        st.session_state.uploaded_bytes = uploaded.getvalue()

    # Action row
    if st.session_state.uploaded_bytes:
        st.markdown('<div class="hn-container" style="margin-top: 10px;">', unsafe_allow_html=True)
        run = st.button("Run Detection", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run:
            with st.spinner("Analyzing image with AI algorithms..."):
                time.sleep(1.6)

            # Mock detections (same as Figma prototype)
            st.session_state.detections = [
                {"id": 1, "label": "Helmet", "confidence": 96.8, "compliance": "COMPLIANT", "bbox": "245, 120, 180, 160"},
                {"id": 2, "label": "Motorcycle", "confidence": 98.5, "compliance": "N/A", "bbox": "150, 200, 400, 350"},
                {"id": 3, "label": "Person", "confidence": 97.2, "compliance": "N/A", "bbox": "220, 100, 200, 380"},
            ]

    # Uploaded image preview
    if st.session_state.uploaded_bytes:
        st.markdown(
            """
<div class="hn-container" style="margin-top: 18px;">
  <div class="hn-card" style="padding: 24px;">
    <div style="font-size: 1.05rem; font-weight: 900; color: var(--hn-slate-900); margin-bottom: 14px;">Uploaded Image</div>
    <div style="background: var(--hn-slate-50); border: 1px solid var(--hn-border); border-radius: var(--hn-radius-lg); padding: 10px;">
""",
            unsafe_allow_html=True,
        )
        st.image(st.session_state.uploaded_bytes, use_container_width=True)
        st.markdown("</div></div></div>", unsafe_allow_html=True)

    # Results
    st.markdown(
        """
<div class="hn-container" style="margin-top: 18px;">
  <div class="hn-card" style="overflow:hidden;">
    <div style="padding: 18px 24px; border-bottom: 1px solid var(--hn-border); display:flex; justify-content:space-between; align-items:center; gap: 12px;">
      <div style="font-size: 1.2rem; font-weight: 900; color: var(--hn-slate-900);">Results</div>
      <div class="hn-badge">Model: HelmetNet</div>
    </div>
    <div style="padding: 22px 24px;">
      <div style="font-weight: 900; color: var(--hn-slate-900); margin-bottom: 12px;">
        Detections Table <span class="hn-muted" style="font-weight: 600; font-size: 0.82rem;">(sorted by confidence)</span>
      </div>
""",
        unsafe_allow_html=True,
    )

    # Sort by confidence descending and filter by threshold
    rows = [
        d for d in st.session_state.detections
        if float(d.get("confidence", 0)) >= float(threshold)
    ]
    rows = sorted(rows, key=lambda x: float(x.get("confidence", 0)), reverse=True)

    st.markdown(html_table(rows), unsafe_allow_html=True)

    st.markdown(
        """
      <div style="margin-top: 18px; background: var(--hn-amber-50); border: 1px solid var(--hn-amber-200); padding: 16px; border-radius: var(--hn-radius-lg); color: #334155;">
        <div style="font-size: 0.92rem; margin-bottom: 6px;">
          <span style="font-weight: 900; color: var(--hn-slate-900);">Integration approach:</span>
          Replace mock generation with a backend endpoint returning
          <code style="background:#fff; border:1px solid var(--hn-amber-300); padding: 2px 8px; border-radius: 8px; color: #1f2937;">{ detections: [{ label, conf, x, y, w, h }] }</code>.
        </div>
        <div class="hn-muted" style="font-size: 0.9rem;">
          Tip: For your final demo, add "export report" (model version, threshold, timestamp, detections).
        </div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

elif mode == "Video Detection":
    st.markdown(
        """
<div class="hn-container" style="margin-top: 18px;">
  <div class="hn-card" style="padding: 64px 24px; text-align:center;">
    <div style="font-size: 2rem; font-weight: 900; margin-bottom: 8px; color: var(--hn-slate-900);">Video Detection</div>
    <div class="hn-muted" style="max-width: 36rem; margin: 0 auto 18px auto;">
      Upload a video file to process frame-by-frame helmet detection
    </div>
    <div style="max-width: 240px; margin: 0 auto;">
""",
        unsafe_allow_html=True,
    )
    st.button("Coming Soon", use_container_width=True)
    st.markdown("</div></div></div>", unsafe_allow_html=True)

else:  # Real Time Detection
    st.markdown(
        """
<div class="hn-container" style="margin-top: 18px;">
  <div class="hn-card" style="padding: 64px 24px; text-align:center;">
    <div style="font-size: 2rem; font-weight: 900; margin-bottom: 8px; color: var(--hn-slate-900);">Real Time Detection</div>
    <div class="hn-muted" style="max-width: 36rem; margin: 0 auto 18px auto;">
      Connect to a webcam or RTSP stream for live helmet detection
    </div>
    <div style="max-width: 240px; margin: 0 auto;">
""",
        unsafe_allow_html=True,
    )
    st.button("Coming Soon", use_container_width=True)
    st.markdown("</div></div></div>", unsafe_allow_html=True)
