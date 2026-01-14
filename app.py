"""
AI Helmet Detection System (CSC738)
CLEAN UI (DEMO READY): Minimal, corporate dashboard layout + model selector + improved hierarchy

IMPORTANT:
- All detection / processing logic is preserved (same YOLO inference, frame skipping, alarm, WebRTC TURN).
- Only UI/UX, layout, and styling have been changed.
- Added model selection dropdown with metadata + dynamic model path loading.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HelmetNet — AI Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL REGISTRY (NEW: dropdown + metadata)
# ============================================================
MODELS = {
    "Production — best.pt": {
        "path": "best.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Production-ready model (default)."
    },
    "Experiment 1 — Baseline": {
        "path": "models/experiment_1/best.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Baseline training run."
    },
    "Experiment 2 — Augmented": {
        "path": "models/experiment_2/best.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Improved with augmentation."
    },
    "Experiment 3 — Latest": {
        "path": "models/experiment_3/best.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Most recent experimental model."
    },
}

# ============================================================
# TWILIO TURN (UNCHANGED)
# ============================================================
@st.cache_resource
def get_twilio_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        ice_servers = token.ice_servers
        if not ice_servers:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
        return ice_servers
    except Exception as e:
        st.sidebar.error(f"TURN setup error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ICE_SERVERS = get_twilio_ice_servers()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": ICE_SERVERS})

# ============================================================
# CONFIGURATION (UNCHANGED)
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP = 3
DEFAULT_MODEL_PATH = "best.pt"

# ============================================================
# CLEAN MINIMAL CSS (KEY CHANGE)
# - remove Streamlit chrome
# - strict spacing + typography
# - minimal borders, almost no shadows
# ============================================================
def inject_css():
    st.markdown(
        """
<style>
/* ---------- Hide Streamlit chrome (makes it look less "default") ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ---------- Layout / typography ---------- */
:root{
  --bg: #0B0F17;
  --panel: #0F1623;
  --panel2:#0C1220;
  --text: #EAF0FA;
  --muted: rgba(234,240,250,0.70);
  --border: rgba(234,240,250,0.10);
  --border2: rgba(234,240,250,0.14);
  --primary:#3B82F6;
  --success:#22C55E;
  --danger:#EF4444;
  --radius: 14px;
  --radius2: 12px;
}

/* App background */
html, body, [data-testid="stAppViewContainer"]{
  background: var(--bg) !important;
  color: var(--text);
}

/* Global container width & padding */
.block-container{
  max-width: 1240px;
  padding-top: 1.2rem !important;
  padding-bottom: 2.5rem !important;
}

/* Reduce “default Streamlit” spacing */
[data-testid="stVerticalBlock"]{ gap: 0.85rem; }
div[data-testid="stMarkdownContainer"] p{ margin-bottom: 0.35rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: #0A0E16 !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container{
  padding-top: 1rem !important;
}

/* Cards */
.hn-card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 16px;
}
.hn-card.soft{
  background: var(--panel2);
  border: 1px solid var(--border);
}
.hn-row{
  display:flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.hn-title{
  font-size: 1.4rem;
  font-weight: 850;
  margin: 0;
  letter-spacing: 0.2px;
}
.hn-subtitle{
  margin: 0.25rem 0 0 0;
  color: var(--muted);
  font-weight: 520;
  font-size: 0.96rem;
}
.hn-kicker{
  color: var(--muted);
  font-size: 0.9rem;
  font-weight: 600;
  letter-spacing: 0.2px;
}
.hn-divider{
  height: 1px;
  background: var(--border);
  margin: 12px 0;
}

/* Make tabs look like a segmented control */
.stTabs [data-baseweb="tab-list"]{
  background: transparent;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 6px;
  gap: 6px;
}
.stTabs [data-baseweb="tab"]{
  height: 44px;
  border-radius: 12px;
  color: var(--muted);
  font-weight: 750;
  padding: 0 14px;
}
.stTabs [aria-selected="true"]{
  background: rgba(59,130,246,0.16);
  border: 1px solid rgba(59,130,246,0.35);
  color: var(--text);
}

/* Inputs / uploader */
[data-testid="stFileUploader"]{
  background: var(--panel);
  border: 1px dashed var(--border2);
  border-radius: var(--radius);
  padding: 14px;
}

/* Buttons (minimal + premium) */
.stButton > button{
  background: var(--primary);
  color: white;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 12px;
  padding: 0.62rem 1.0rem;
  font-weight: 800;
  transition: transform .12s ease, filter .12s ease;
  box-shadow: none;
}
.stButton > button:hover{
  transform: translateY(-1px);
  filter: brightness(1.04);
}

/* Download button secondary */
.stDownloadButton > button{
  background: transparent;
  color: var(--text);
  border: 1px solid var(--border2);
  border-radius: 12px;
  font-weight: 800;
  box-shadow: none;
}
.stDownloadButton > button:hover{
  background: rgba(234,240,250,0.06);
}

/* Metrics */
[data-testid="metric-container"]{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: var(--radius2);
  padding: 12px 12px;
  box-shadow: none;
}
[data-testid="stMetricValue"]{
  font-size: 1.55rem !important;
  font-weight: 900 !important;
  color: var(--text) !important;
}
[data-testid="stMetricLabel"]{
  color: var(--muted) !important;
  font-weight: 650 !important;
}

/* Alerts (clean) */
.hn-alert{
  display:flex;
  gap: 12px;
  align-items:flex-start;
  padding: 12px 12px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--panel);
}
.hn-dot{ width:10px;height:10px;border-radius:999px;margin-top:6px; }
.hn-dot.ok{ background: var(--success); }
.hn-dot.bad{ background: var(--danger); }
.hn-alert .t{ font-weight: 900; margin:0; }
.hn-alert .d{ margin:3px 0 0 0; color: var(--muted); font-weight: 520; }

audio{ display:none; }
</style>
""",
        unsafe_allow_html=True,
    )

inject_css()

# ============================================================
# UTILS & LOGIC (UNCHANGED)
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        if Path(path).exists():
            model = YOLO(path)
            st.sidebar.success("Model loaded")
            return model
        st.sidebar.warning("Model not found — using YOLOv8n")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None

def play_alarm():
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0
    if time.time() - st.session_state.last_alarm > 3:
        if Path("alert.mp3").exists():
            st.audio("alert.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()

def draw_boxes(frame, detections):
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        color = (0, 0, 139) if det["class"] in NO_HELMET_LABELS else (0, 100, 0)
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def detect_frame(frame, model, conf_threshold):
    # Keep original inference config (imgsz=640, device='cpu')
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
# WEBRTC CLASS (UNCHANGED)
# ============================================================
class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.helmet = 0
        self.no_helmet = 0
        self.frame_cnt = 0
        self.last_dets = []
        self.alert = False

    def set_model(self, model, conf):
        self.model = model
        self.conf = conf

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return img

        self.frame_cnt += 1

        if self.frame_cnt % FRAME_SKIP == 0:
            try:
                detections, stats = detect_frame(img, self.model, self.conf)
                self.last_dets = detections
                self.helmet = stats["helmet_count"]
                self.no_helmet = stats["no_helmet_count"]
                self.alert = stats["alert"]
            except Exception:
                pass

        return draw_boxes(img, self.last_dets)

# ============================================================
# SIDEBAR (clean + model selector + metadata)
# ============================================================
with st.sidebar:
    st.markdown("<div class='hn-kicker'>CONTROL PANEL</div>", unsafe_allow_html=True)
    st.markdown("<div class='hn-card'>", unsafe_allow_html=True)

    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    selected = MODELS[model_key]

    # Keep path override feature (unchanged functionality)
    with st.expander("Advanced", expanded=False):
        use_custom_path = st.checkbox("Use custom model path", value=False)
        custom_path = st.text_input("Custom path", DEFAULT_MODEL_PATH)

    model_path = custom_path if use_custom_path else selected["path"]
    confidence_threshold = st.slider("Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="hn-card soft">
  <div style="font-weight:900;">Model info</div>
  <div style="color:var(--muted); margin-top:6px;"><b>Name:</b> {model_key}</div>
  <div style="color:var(--muted);"><b>Path:</b> <code>{model_path}</code></div>
  <div style="color:var(--muted);"><b>Date:</b> {selected.get("date","—")}</div>
  <div style="color:var(--muted);"><b>Accuracy:</b> {selected.get("accuracy","—")}</div>
  <div style="color:var(--muted); margin-top:8px;">{selected.get("description","")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)

    with st.expander("WebRTC / TURN diagnostics", expanded=False):
        st.write("ICE servers loaded:", len(ICE_SERVERS))
        if len(ICE_SERVERS) > 0 and "urls" in ICE_SERVERS[0]:
            st.write("First ICE urls:", ICE_SERVERS[0]["urls"])

    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total detections", st.session_state.total_detections)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL (UNCHANGED)
# ============================================================
model = load_model(model_path)
if not model:
    st.sidebar.warning(f"Could not load {model_path}, using default YOLOv8n")
    model = YOLO("yolov8n.pt")

# ============================================================
# HEADER (very clean, no gradients)
# ============================================================
st.markdown(
    """
<div class="hn-card">
  <div class="hn-row">
    <div>
      <p class="hn-title">HelmetNet</p>
      <p class="hn-subtitle">AI Helmet Detection — Image • Video • Real-time</p>
    </div>
    <div style="text-align:right;">
      <div class="hn-kicker">SESSION</div>
      <div style="font-weight:900; font-size:1.05rem;">Ready</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Image", "Video", "Real-time"])

# ============================================================
# TAB 1: IMAGE DETECTION (same logic)
# ============================================================
with tab1:
    st.markdown("<div class='hn-card'><div style='font-weight:900;'>Image detection</div><div class='hn-subtitle'>Upload and compare original vs annotated output.</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1.0], gap="large")
    with right:
        st.markdown(
            """
<div class="hn-card soft">
  <div style="font-weight:900;">Tips</div>
  <ul style="margin-top:10px; color: var(--muted); font-weight: 520;">
    <li>Clear, well-lit images</li>
    <li>Rider visible in frame</li>
    <li>JPG / PNG / BMP supported</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )
    with left:
        img_file = st.file_uploader("Upload image", ["jpg", "jpeg", "png", "bmp"], key="img", label_visibility="collapsed")

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running inference…"):
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("<div class='hn-kicker'>ORIGINAL</div>", unsafe_allow_html=True)
            st.image(img_file, use_container_width=True)
        with b:
            st.markdown("<div class='hn-kicker'>ANNOTATED</div>", unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

        if stats["alert"]:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-dot bad"></div>
  <div>
    <p class="t">Violation detected</p>
    <p class="d">At least one rider appears without a helmet.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            play_alarm()
        else:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-dot ok"></div>
  <div>
    <p class="t">All clear</p>
    <p class="d">No violations detected in this image.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Helmets", stats["helmet_count"])
        m2.metric("Violations", stats["no_helmet_count"])
        m3.metric("Total objects", len(dets))

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, "rb") as f:
            st.download_button("Download result", f, f"result_{img_file.name}", "image/jpeg")

# ============================================================
# TAB 2: VIDEO DETECTION (same logic)
# ============================================================
with tab2:
    st.markdown("<div class='hn-card'><div style='font-weight:900;'>Video detection</div><div class='hn-subtitle'>Live preview + export with frame skipping.</div></div>", unsafe_allow_html=True)

    left, right = st.columns([1.7, 1.0], gap="large")
    with right:
        st.markdown(
            """
<div class="hn-card soft">
  <div style="font-weight:900;">Notes</div>
  <ul style="margin-top:10px; color: var(--muted); font-weight: 520;">
    <li>Frame skipping enabled</li>
    <li>MP4 / AVI / MOV / MKV</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )
    with left:
        vid_file = st.file_uploader("Upload video", ["mp4", "avi", "mov", "mkv"], key="vid", label_visibility="collapsed")

    if vid_file:
        if st.button("Start inference", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())

            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            st_frame = st.empty()
            st_metrics = st.empty()
            st_progress = st.progress(0)
            status = st.status("Processing…", expanded=False)

            frame_count = 0
            cached_detections = []
            current_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)

                annotated = draw_boxes(frame, cached_detections)
                out.write(annotated)

                if current_stats["alert"]:
                    play_alarm()

                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               caption=f"Frame {frame_count}/{total_frames}",
                               use_container_width=True)

                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Helmets", current_stats["helmet_count"])
                    c2.metric("Violations", current_stats["no_helmet_count"])
                    c3.metric("Progress", f"{int(frame_count/total_frames*100)}%")

                st_progress.progress(frame_count / total_frames)

            cap.release()
            out.release()

            status.update(label="Complete", state="complete", expanded=False)
            st.success("Processing complete.")
            st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

            with open(outfile.name, "rb") as f:
                st.download_button("Download result video", f, "result.mp4", "video/mp4")

# ============================================================
# TAB 3: REAL-TIME DETECTION (same logic)
# ============================================================
with tab3:
    st.markdown("<div class='hn-card'><div style='font-weight:900;'>Real-time detection</div><div class='hn-subtitle'>Start webcam stream. TURN enabled for hotspots.</div></div>", unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HelmetTransformer,
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(model, confidence_threshold)

        m1, m2 = st.columns(2)
        m1.metric("Helmets", ctx.video_processor.helmet)
        m2.metric("Violations", ctx.video_processor.no_helmet)

        if ctx.video_processor.alert:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-dot bad"></div>
  <div>
    <p class="t">Violation detected</p>
    <p class="d">At least one rider appears without a helmet.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            play_alarm()
        else:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-dot ok"></div>
  <div>
    <p class="t">All clear</p>
    <p class="d">No violations in the current live window.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

st.caption("HelmetNet • © 2025")
