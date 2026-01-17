"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Modern Safety Theme UI

UPDATED:
- Reliable WebRTC on hotspots using Twilio Network Traversal (TURN) tokens
- Use Streamlit Secrets (st.secrets) instead of os.environ
- Model dropdown from ./models/*.pt
- Fancy detections table shown under Image Detection (after Run)
- Table rendering forced via components.html to avoid "HTML printed as text" issues
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from twilio.rest import Client
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
    page_title="AI Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# TWILIO TURN (Network Traversal Token -> ICE servers)
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
# SAFETY THEME CSS + TABLE STYLES
# ============================================================
st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem !important; }
    .main-header {
        font-size: 2.5rem; font-weight: 800; color: var(--text-color);
        text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center; font-size: 1.1rem; color: var(--text-color);
        opacity: 0.8; font-weight: 500; margin-bottom: 1.5rem;
    }
    h2 {
        color: var(--text-color) !important; font-weight: 700 !important;
        border-bottom: 3px solid #FFD700; padding-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary-background-color); padding: 0.5rem;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 8px; color: var(--text-color);
        font-weight: 600; padding: 0 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #FFD700 !important; color: #1E3A8A !important;
    }
    .alert-danger {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: 700;
        animation: pulse 2s infinite; margin: 20px 0;
        box-shadow: 0 4px 6px rgba(239,68,68,0.3);
        border: 3px solid #FCA5A5;
    }
    .alert-success {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.3rem; font-weight: 700;
        margin: 20px 0; box-shadow: 0 4px 6px rgba(34,197,94,0.3);
        border: 3px solid #86EFAC;
    }
    @keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: 0.85; transform: scale(1.02);}}
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1E3A8A; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 700;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: #1E3A8A;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white; border: none;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700 !important; color: var(--text-color);
    }
    [data-testid="metric-container"] {
        background: var(--secondary-background-color); padding: 1rem;
        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #FFD700;
    }
    [data-testid="stFileUploader"] {
        background: var(--secondary-background-color); padding: 1.5rem;
        border-radius: 10px; border: 2px dashed #FFD700;
    }
    audio {display: none;}
    .info-box {
        background: rgba(59, 130, 246, 0.1); padding: 1rem;
        border-radius: 10px; border-left: 4px solid #1E3A8A;
        margin: 1rem 0; color: var(--text-color);
    }

    /* Fancy result table styles */
    .hn-card {
        background: white;
        border: 1px solid rgba(226,232,240,1);
        border-radius: 14px;
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
        overflow: hidden;
        margin-top: 1rem;
    }
    .hn-card-h {
        padding: 14px 18px;
        border-bottom: 1px solid rgba(226,232,240,1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        font-weight: 800;
        color: #0f172a;
    }
    .hn-pill {
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(226,232,240,1);
        background: rgba(248,250,252,1);
        font-weight: 800;
        font-size: 0.85rem;
        color: #334155;
        white-space: nowrap;
    }
    .hn-sub {
        font-weight: 500;
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 2px;
    }
    .hn-table-wrap { overflow-x: auto; padding: 0 0 6px 0; }
    .hn-table {
        width: 100%;
        border-collapse: collapse;
        min-width: 760px;
        background: white;
    }
    .hn-table thead th {
        text-align: left;
        padding: 12px;
        font-size: 0.8rem;
        color: #475569;
        font-weight: 900;
        border-top: 1px solid #eef2f7;
        border-bottom: 1px solid #eef2f7;
        background: white;
    }
    .hn-table tbody td {
        padding: 14px 12px;
        border-top: 1px solid #eef2f7;
        vertical-align: middle;
        font-variant-numeric: tabular-nums;
    }
    .hn-badge-ok {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 12px;
        border-radius: 999px;
        background: #dcfce7;
        color: #14532d;
        font-weight: 800;
        font-size: 0.85rem;
        border: 1px solid #86efac;
    }
    .hn-badge-bad {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 6px 12px;
        border-radius: 999px;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 800;
        font-size: 0.85rem;
        border: 1px solid #fca5a5;
    }
    .hn-foot {
        padding: 12px 18px;
        border-top: 1px solid #eef2f7;
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet", "nohelmet"]
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP = 3
MODELS_DIR = Path("models")
DEFAULT_MODEL_PATH = "model_1.pt"

# ============================================================
# UTILS & LOGIC
# ============================================================
@st.cache_resource
def load_model(model_file: str):
    try:
        candidate = MODELS_DIR / model_file
        if candidate.exists():
            model = YOLO(str(candidate))
            st.sidebar.success(f"✅ Model loaded: {model_file}")
            return model
        st.sidebar.warning("⚠️ Model not found in ./models, using YOLOv8n")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


def play_alarm():
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0.0
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
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def detect_frame(frame, model, conf_threshold):
    results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False, device="cpu")

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = str(model.names[cls_id]).lower()
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


def _badge_html(label: str) -> str:
    lab = (label or "").lower()
    non = lab in NO_HELMET_LABELS or lab.replace("_", "-") in NO_HELMET_LABELS
    return '<span class="hn-badge-bad">Non-compliant</span>' if non else '<span class="hn-badge-ok">Compliant</span>'


def render_detection_table(detections: list[dict], model_name: str) -> None:
    """
    Pretty results card + table. Uses components.html, so we inline CSS (components are isolated).
    """
    # ----- inline CSS for the component frame -----
    css = """
    <style>
      :root { --border:#e2e8f0; --muted:#64748b; --text:#0f172a; --bg:#ffffff; }
      body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; color:var(--text); background:transparent; }
      .card{
        background:var(--bg);
        border:1px solid var(--border);
        border-radius:16px;
        box-shadow:0 12px 30px rgba(15,23,42,.08);
        overflow:hidden;
      }
      .head{
        padding:16px 18px;
        border-bottom:1px solid var(--border);
        display:flex; align-items:flex-start; justify-content:space-between; gap:12px;
      }
      .title{ font-weight:900; font-size:18px; line-height:1.15; }
      .sub{ margin-top:4px; color:var(--muted); font-size:13px; font-weight:600; }
      .pill{
        margin-top:2px;
        padding:8px 12px;
        border-radius:999px;
        border:1px solid var(--border);
        background:#f8fafc;
        font-weight:900;
        font-size:13px;
        color:#334155;
        white-space:nowrap;
      }
      .section{
        padding:14px 18px;
        display:flex; align-items:center; justify-content:space-between; gap:12px;
      }
      .section .left{ font-weight:900; font-size:16px; }
      .section .right{ color:var(--muted); font-weight:800; font-size:13px; }

      .wrap{ overflow:auto; }
      table{ width:100%; border-collapse:collapse; min-width:820px; background:#fff; }
      thead th{
        text-align:left;
        padding:12px 14px;
        font-size:12px;
        letter-spacing:.02em;
        color:#475569;
        font-weight:900;
        background:#f8fafc;
        border-top:1px solid #eef2f7;
        border-bottom:1px solid #eef2f7;
      }
      tbody td{
        padding:12px 14px;
        border-bottom:1px solid #eef2f7;
        font-size:14px;
        vertical-align:middle;
      }
      tbody tr:nth-child(odd){ background:#ffffff; }
      tbody tr:nth-child(even){ background:#fbfdff; }

      .num{ color:#475569; width:46px; }
      .label{ font-weight:900; }
      .bbox{ color:#475569; font-variant-numeric: tabular-nums; }
      .ok{
        display:inline-flex; align-items:center; justify-content:center;
        padding:6px 12px; border-radius:999px;
        background:#dcfce7; color:#14532d; font-weight:900; font-size:13px;
        border:1px solid #86efac;
      }
      .bad{
        display:inline-flex; align-items:center; justify-content:center;
        padding:6px 12px; border-radius:999px;
        background:#fee2e2; color:#991b1b; font-weight:900; font-size:13px;
        border:1px solid #fca5a5;
      }
      .foot{
        padding:12px 18px;
        border-top:1px solid var(--border);
        color:var(--muted);
        font-size:13px;
        font-weight:600;
        background:#ffffff;
      }
    </style>
    """

    def badge_html(label: str) -> str:
        lab = (label or "").lower()
        non = lab in NO_HELMET_LABELS or lab.replace("_", "-") in NO_HELMET_LABELS
        return '<span class="bad">Non-compliant</span>' if non else '<span class="ok">Compliant</span>'

    if not detections:
        html = f"""
        {css}
        <div class="card">
          <div class="head">
            <div>
              <div class="title">Results</div>
              <div class="sub">No detections found.</div>
            </div>
            <div class="pill">Model: {model_name}</div>
          </div>
          <div class="foot">Tip: Try a clearer image for better detection.</div>
        </div>
        """
        components.html(html, height=200, scrolling=False)
        return

    dets = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)

    rows = []
    for i, det in enumerate(dets, start=1):
        label = str(det.get("class", ""))
        conf = float(det.get("confidence", 0.0))
        bbox = det.get("bbox", [0, 0, 0, 0])

        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
        except Exception:
            x1 = y1 = w = h = 0

        rows.append(
            f"""
            <tr>
              <td class="num">{i}</td>
              <td class="label">{label}</td>
              <td>{conf*100:.1f}%</td>
              <td>{badge_html(label)}</td>
              <td class="bbox">{x1}, {y1}, {w}, {h}</td>
            </tr>
            """
        )

    # height: more rows -> taller
    height = 520 if len(rows) <= 6 else 620

    html = f"""
    {css}
    <div class="card">
      <div class="head">
        <div>
          <div class="title">Results</div>
          <div class="sub">Populated from YOLO outputs (label, confidence, bbox).</div>
        </div>
        <div class="pill">Model: {model_name}</div>
      </div>

      <div class="section">
        <div class="left">Detections Table</div>
        <div class="right">Sorted by confidence</div>
      </div>

      <div class="wrap">
        <table>
          <thead>
            <tr>
              <th style="width:56px;">#</th>
              <th>LABEL</th>
              <th>CONFIDENCE</th>
              <th>COMPLIANCE</th>
              <th>BBOX (X,Y,W,H)</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>

      <div class="foot">
        Tip: For your final demo, add “export report” (model version, threshold, timestamp, detections).
      </div>
    </div>
    """
    components.html(html, height=height, scrolling=True)

# ============================================================
# WEBRTC CLASS
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
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**🤖 Model Settings**")
    model_files = sorted([p.name for p in MODELS_DIR.glob("*.pt")])

    if not model_files:
        st.warning("No .pt files found in ./models. Falling back to yolov8n.pt")
        model_files = ["yolov8n.pt"]

    default_index = model_files.index(DEFAULT_MODEL_PATH) if DEFAULT_MODEL_PATH in model_files else 0
    model_choice = st.selectbox("Select Model", options=model_files, index=default_index)

    confidence_threshold = st.slider("🎯 Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("**🌐 WebRTC / TURN Debug**")
    st.write("ICE servers loaded:", len(ICE_SERVERS))
    if len(ICE_SERVERS) > 0 and "urls" in ICE_SERVERS[0]:
        st.write("First ICE urls:", ICE_SERVERS[0]["urls"])

    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total Detections", st.session_state.total_detections)

    st.markdown("---")

# ============================================================
# LOAD MODEL
# ============================================================
if model_choice == "yolov8n.pt":
    model = YOLO("yolov8n.pt")
else:
    model = load_model(model_choice)

if not model:
    st.sidebar.warning(f"⚠️ Could not load {model_choice}, using YOLOv8n")
    model = YOLO("yolov8n.pt")

# ============================================================
# MAIN APP UI
# ============================================================
st.markdown('<h1 class="main-header">🛡️ HelmetNet </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI Helmet Detection System</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real-Time Detection"])

# --- TAB 1: IMAGE DETECTION ---
with tab1:
    st.markdown("### 📸 Upload an Image")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<div class="info-box"><strong>💡 Tips:</strong><br>• Clear, well-lit images<br>• JPG, PNG, BMP</div>',
            unsafe_allow_html=True,
        )

    with col1:
        img_file = st.file_uploader(
            "Choose image",
            ["jpg", "jpeg", "png", "bmp"],
            key="img",
            label_visibility="collapsed",
        )

    run_img = st.button("Run Detection", key="run_img")

    if img_file and run_img:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Analyzing..."):
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**📷 Original**")
            st.image(original_rgb, use_container_width=True)
        with c2:
            st.markdown("**🎯 Result**")
            st.image(annotated_rgb, use_container_width=True)

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

        # Fancy detections table (forced HTML render)
        render_detection_table(dets, model_choice)

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, "rb") as f:
            st.download_button("📥 Download Result", f, f"result_{img_file.name}", "image/jpeg")

# --- TAB 2: VIDEO DETECTION ---
with tab2:
    st.markdown("### 🎥 Upload a Video")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown(
            '<div class="info-box"><strong>💡 Fast Mode:</strong><br>• Optimized frame skipping<br>• Live inference preview<br>• MP4, AVI, MOV</div>',
            unsafe_allow_html=True,
        )

    with col1:
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
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

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
                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)

                annotated = draw_boxes(frame, cached_detections)
                out.write(annotated)

                if current_stats["alert"]:
                    play_alarm()

                st_frame.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Processing Frame {frame_count}/{total_frames}" if total_frames else f"Processing Frame {frame_count}",
                    use_container_width=True,
                )

                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 Helmets", current_stats["helmet_count"])
                    c2.metric("🔴 Violations", current_stats["no_helmet_count"])
                    c3.metric("⏱️ Progress", f"{int(frame_count/total_frames*100)}%" if total_frames else "-")

                if total_frames:
                    st_progress.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()

            st.success("✅ Processing Complete!")
            st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

            with open(outfile.name, "rb") as f:
                st.download_button("📥 Download Result", f, "result.mp4", "video/mp4")

# --- TAB 3: REAL-TIME DETECTION (WEBRTC) ---
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")

    if "start_live" not in st.session_state:
        st.session_state.start_live = False

    if not st.session_state.start_live:
        if st.button("▶️ START Live Detection"):
            st.session_state.start_live = True

    if st.session_state.start_live:
        ctx = webrtc_streamer(
            key="helmet-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=HelmetTransformer,
            async_processing=True,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )

        if ctx.video_processor:
            ctx.video_processor.set_model(model, confidence_threshold)

            st.markdown("### 📊 Live Stats")
            m1, m2 = st.columns(2)
            m1.metric("🟢 Helmets", ctx.video_processor.helmet)
            m2.metric("🔴 Violations", ctx.video_processor.no_helmet)

            if ctx.video_processor.alert:
                st.markdown(
                    '<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alert-success">✅ Area Secure</div>',
                    unsafe_allow_html=True,
                )

st.markdown("---")
st.caption("🚀 HelmetNet App | © 2025")
