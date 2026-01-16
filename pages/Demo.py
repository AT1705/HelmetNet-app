"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Modern Safety Theme UI
"""

import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Demo | HelmetNet",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SHARED UI / BRANDING (match app.py / About.py)
# ============================================================
BRAND = {
    "bg": "#F8FAFC",
    "card": "rgba(255,255,255,0.90)",
    "text": "#0F172A",
    "muted": "#475569",
    "border": "rgba(148,163,184,0.35)",
    "slate700": "#334155",
    "slate800": "#1F2937",
    "slate900": "#0F172A",
    "amber": "#F59E0B",
    "amberHover": "#FBBF24",
}

def inject_global_css(active_page: str) -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{ background: {BRAND["bg"]}; color: {BRAND["text"]}; }}
          .block-container {{
            padding-top: 5.2rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
          }}
          #MainMenu, footer, header {{ visibility: hidden; }}
          [data-testid="stStatusWidget"] {{ display: none; }}

          .hn-nav {{
            position: fixed; top: 0; left: 0; right: 0;
            z-index: 9999;
            background: rgba(255,255,255,0.92);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(226,232,240,1);
            box-shadow: 0 1px 8px rgba(15,23,42,0.06);
          }}
          .hn-nav-inner {{
            max-width: 1200px; margin: 0 auto;
            padding: 0.8rem 1rem;
            display:flex; align-items:center; justify-content:space-between;
          }}
          .hn-brand {{
            display:flex; align-items:center; gap:0.6rem;
            font-weight: 800; font-size: 1.15rem; letter-spacing:-0.02em;
            color: {BRAND["text"]};
          }}
          .hn-brand-badge {{
            width: 36px; height: 36px; border-radius: 12px;
            display:flex; align-items:center; justify-content:center;
            background: linear-gradient(135deg, {BRAND["slate700"]}, {BRAND["slate900"]});
            color:white; box-shadow: 0 10px 25px rgba(15,23,42,0.18);
          }}
          .hn-links {{ display:flex; align-items:center; gap:1.1rem; }}
          .hn-link {{
            font-weight: 600; color: {BRAND["muted"]};
            text-decoration:none; padding: 0.35rem 0.2rem;
          }}
          .hn-link:hover {{ color: {BRAND["text"]}; }}
          .hn-link.active {{ color: {BRAND["text"]}; }}

          .hn-cta {{
            display:inline-flex; align-items:center; justify-content:center;
            padding: 0.55rem 1rem; border-radius: 14px;
            font-weight: 800; text-decoration:none;
            background: {BRAND["amber"]}; color: {BRAND["slate900"]};
            box-shadow: 0 10px 25px rgba(245,158,11,0.25);
            border: 1px solid rgba(245,158,11,0.35);
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
          }}
          .hn-cta:hover {{
            background: {BRAND["amberHover"]};
            transform: translateY(-1px);
            box-shadow: 0 14px 35px rgba(245,158,11,0.28);
            color: {BRAND["slate900"]};
          }}

          /* Sidebar: soften into “portal” config */
          section[data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.86);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(226,232,240,1);
          }}
          .sidebar-card {{
            background: rgba(248,250,252,0.80);
            border: 1px solid rgba(148,163,184,0.28);
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 18px 40px rgba(15,23,42,0.05);
          }}

          /* Tabs: modern pill list */
          .stTabs [data-baseweb="tab-list"] {{
            background: rgba(255,255,255,0.92);
            padding: 0.45rem;
            border-radius: 16px;
            border: 1px solid rgba(148,163,184,0.25);
            box-shadow: 0 18px 40px rgba(15,23,42,0.05);
            gap: 0.25rem;
          }}
          .stTabs [data-baseweb="tab"] {{
            height: 46px;
            border-radius: 14px;
            font-weight: 900;
            color: {BRAND["muted"]};
            padding: 0 1.2rem;
          }}
          .stTabs [aria-selected="true"] {{
            background: {BRAND["amber"]} !important;
            color: {BRAND["slate900"]} !important;
          }}

          /* Buttons: amber primary */
          .stButton > button {{
            background: {BRAND["amber"]};
            color: {BRAND["slate900"]};
            border: 1px solid rgba(245,158,11,0.35);
            border-radius: 14px;
            padding: 0.6rem 1.1rem;
            font-weight: 900;
            box-shadow: 0 14px 35px rgba(245,158,11,0.18);
            transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
          }}
          .stButton > button:hover {{
            background: {BRAND["amberHover"]};
            transform: translateY(-1px);
            box-shadow: 0 18px 45px rgba(245,158,11,0.22);
            color: {BRAND["slate900"]};
          }}
          .stDownloadButton > button {{
            background: linear-gradient(135deg, {BRAND["slate700"]}, {BRAND["slate900"]});
            color: white;
            border: none;
            border-radius: 14px;
            font-weight: 900;
          }}

          /* Metrics: portal tiles */
          [data-testid="metric-container"] {{
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(148,163,184,0.25);
            padding: 1rem;
            border-radius: 16px;
            box-shadow: 0 18px 40px rgba(15,23,42,0.05);
            border-left: 4px solid {BRAND["amber"]};
          }}
          [data-testid="stMetricValue"] {{
            font-weight: 900;
            color: {BRAND["text"]};
          }}

          /* Alert blocks (from your original app.py) */
          .alert-danger {{
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
            color: white;
            padding: 18px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.15rem;
            font-weight: 900;
            animation: pulse 2s infinite;
            margin: 16px 0;
            box-shadow: 0 18px 40px rgba(239,68,68,0.20);
            border: 2px solid rgba(252,165,165,0.9);
          }}
          .alert-success {{
            background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
            color: white;
            padding: 18px;
            border-radius: 16px;
            text-align: center;
            font-size: 1.15rem;
            font-weight: 900;
            margin: 16px 0;
            box-shadow: 0 18px 40px rgba(34,197,94,0.18);
            border: 2px solid rgba(134,239,172,0.9);
          }}
          @keyframes pulse {{
            0%, 100% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.88; transform: scale(1.01); }}
          }}

          /* Info box */
          .info-box {{
            background: rgba(59, 130, 246, 0.08);
            padding: 1rem;
            border-radius: 16px;
            border-left: 4px solid {BRAND["slate700"]};
            margin: 0.8rem 0;
            color: {BRAND["text"]};
            font-weight: 600;
          }}

          /* File uploader */
          [data-testid="stFileUploader"] {{
            background: rgba(255,255,255,0.92);
            padding: 1.25rem;
            border-radius: 16px;
            border: 2px dashed rgba(245,158,11,0.75);
            box-shadow: 0 18px 40px rgba(15,23,42,0.05);
          }}

          audio {{ display: none; }}
        </style>

        <div class="hn-nav">
          <div class="hn-nav-inner">
            <div class="hn-brand">
              <div class="hn-brand-badge">🛡️</div>
              HelmetNet
            </div>
            <div class="hn-links">
              <a class="hn-link" href="/">Home</a>
              <a class="hn-link" href="/About">About</a>
              <a class="hn-cta" href="/Demo">Start Demo</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

inject_global_css(active_page="demo")

# ============================================================
# CONFIGURATION (from your original app.py)
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 3
DEFAULT_MODEL_PATH = "best.pt"

# ============================================================
# UTILS & LOGIC (FROM YOUR ORIGINAL app.py)
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        if Path(path).exists():
            model = YOLO(path)
            st.sidebar.success("✅ Model loaded")
            return model
        st.sidebar.warning("⚠️ Model not found")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None

def play_alarm():
    # Logic from app_2.py (Rate limited)
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0
    if time.time() - st.session_state.last_alarm > 3:
        if Path("alert.mp3").exists():
            st.audio("alert.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()

def draw_boxes(frame, detections):
    """
    Manually draws bounding boxes (Logic from app_2.py).
    """
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        # Red for no helmet, Green for helmet
        color = (0, 0, 139) if det["class"] in NO_HELMET_LABELS else (0, 100, 0)
        label = f'{det["class"]} {det["confidence"]:.2f}'

        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw Label Background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return img

def detect_frame(frame, model, conf_threshold):
    # Logic from app_2.py
    results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False, device="cpu")

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()

        detections.append({"class": cls_name, "confidence": conf, "bbox": bbox})

        if cls_name in NO_HELMET_LABELS:
            no_helmet_count += 1
        else:
            helmet_count += 1

    return detections, {
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "alert": no_helmet_count > 0,
    }

# ============================================================
# WEBRTC CLASS (FROM YOUR ORIGINAL app.py)
# ============================================================
class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.helmet = 0
        self.no_helmet = 0
        self.frame_cnt = 0
        self.last_dets = []  # Cache detections
        self.alert = False

    def set_model(self, model, conf):
        self.model = model
        self.conf = conf

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return img

        self.frame_cnt += 1

        # OPTIMIZATION: Run AI only every 3rd frame
        if self.frame_cnt % FRAME_SKIP == 0:
            try:
                detections, stats = detect_frame(img, self.model, self.conf)
                self.last_dets = detections
                self.helmet = stats["helmet_count"]
                self.no_helmet = stats["no_helmet_count"]
                self.alert = stats["alert"]
            except Exception:
                pass

        # Draw the (new or cached) boxes
        return draw_boxes(img, self.last_dets)

# ============================================================
# SIDEBAR (as requested: model selection + confidence in sidebar)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🤖 Model Settings**")

    # 3 model options (requested)
    MODEL_OPTIONS = {
        "HelmetNet (best.pt)": "best.pt",
        "YOLOv8 Nano (yolov8n.pt)": "yolov8n.pt",
        "YOLOv8 Small (yolov8s.pt)": "yolov8s.pt",
    }
    model_choice = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_choice]

    confidence_threshold = st.slider("🎯 Confidence Threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total Detections", st.session_state.total_detections)

    st.markdown("</div>", unsafe_allow_html=True)

# LOAD MODEL (preserving original behavior; now uses chosen model_path)
model = load_model(model_path)
if not model:
    st.sidebar.warning(f"⚠️ Could not load {model_path}, using default YOLOv8n")
    model = YOLO("yolov8n.pt")

# ============================================================
# MAIN APP UI
# ============================================================
st.markdown(
    """
    <div style="margin-top:0.3rem;">
      <div style="font-size:2.4rem;font-weight:900;letter-spacing:-0.03em;color:#0F172A;">🛵 HelmetNet</div>
      <div style="color:#475569;font-weight:700;margin-top:0.2rem;">AI Helmet Detection System</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real-Time Detection"])

# ============================================================
# TAB 1: IMAGE DETECTION
# ============================================================
with tab1:
    st.markdown("### 📸 Upload an Image")

    col_u, col_tip = st.columns([2, 1], gap="large")
    with col_tip:
        st.markdown(
            """
            <div class="info-box">
              <strong>Tips</strong><br/>
              • Clear, well-lit images<br/>
              • JPG, PNG, BMP
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_u:
        img_file = st.file_uploader(
            "Choose image",
            ["jpg", "jpeg", "png", "bmp"],
            key="img",
            label_visibility="collapsed",
        )

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analyzing..."):
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Side-by-side: output image + results table (requested)
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.markdown("**🎯 Result**")
            st.image(annotated_rgb, use_container_width=True)

        with right:
            st.markdown("**📋 Detections**")
            if dets:
                rows = []
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    rows.append(
                        {
                            "class": d["class"],
                            "confidence": round(float(d["confidence"]), 3),
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No objects detected above the confidence threshold.")

        # Original image (optional, but still useful)
        st.markdown("**📷 Original**")
        st.image(frame[:, :, ::-1], use_container_width=True)

        # Alerts (from your original UI logic)
        if stats["alert"]:
            st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)

        st.markdown("### 📊 Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Helmets", stats["helmet_count"])
        m2.metric("🔴 No Helmets", stats["no_helmet_count"])
        m3.metric("📝 Total Objects", len(dets))

        # Download result (preserve your logic)
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, "rb") as f:
            st.download_button("📥 Download Result", f, f"result_{img_file.name}", "image/jpeg")

# ============================================================
# TAB 2: VIDEO DETECTION
# ============================================================
with tab2:
    st.markdown("### 🎥 Upload a Video")

    col_u, col_tip = st.columns([2, 1], gap="large")
    with col_tip:
        st.markdown(
            """
            <div class="info-box">
              <strong>Fast Mode</strong><br/>
              • Optimized frame skipping<br/>
              • Live inference preview<br/>
              • MP4, AVI, MOV
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_u:
        vid_file = st.file_uploader(
            "Choose video",
            ["mp4", "avi", "mov", "mkv"],
            key="vid",
            label_visibility="collapsed",
        )

    if vid_file:
        st.markdown("### 🎬 Processing")
        if st.button("▶️ Start Live Inference", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            st_frame = st.empty()
            st_metrics = st.empty()
            st_progress = st.progress(0)

            frame_count = 0
            cached_detections = []
            current_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # OPTIMIZATION LOGIC (preserved)
                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)

                annotated = draw_boxes(frame, cached_detections)
                out.write(annotated)

                if current_stats["alert"]:
                    play_alarm()

                caption = f"Processing Frame {frame_count}/{total_frames}" if total_frames else f"Processing Frame {frame_count}"
                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)

                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 Helmets", current_stats["helmet_count"])
                    c2.metric("🔴 Violations", current_stats["no_helmet_count"])
                    if total_frames:
                        c3.metric("⏱️ Progress", f"{int(frame_count / max(total_frames,1) * 100)}%")
                    else:
                        c3.metric("⏱️ Progress", "—")

                if total_frames:
                    st_progress.progress(min(frame_count / max(total_frames, 1), 1.0))

            cap.release()
            out.release()

            st.success("✅ Processing Complete!")
            st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

            with open(outfile.name, "rb") as f:
                st.download_button("📥 Download Result", f, "result.mp4", "video/mp4")

# ============================================================
# TAB 3: REAL-TIME DETECTION (WEBRTC)
# ============================================================
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")
    st.markdown(
        """
        <div class="info-box">
          <strong>Live Webcam</strong><br/>
          • Click "START" below<br/>
          • Uses optimized frame skipping for smoother performance<br/>
          • Works on mobile & desktop
        </div>
        """,
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_processor_factory=HelmetTransformer,
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(model, confidence_threshold)

        st.markdown("### 📊 Live Stats")
        m1, m2 = st.columns(2)
        m1.metric("🟢 Helmets", ctx.video_processor.helmet)
        m2.metric("🔴 Violations", ctx.video_processor.no_helmet)

        if ctx.video_processor.alert:
            st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">✅ Area Secure</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("HelmetNet | © 2025")
