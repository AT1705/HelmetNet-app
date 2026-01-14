"""
AI Helmet Detection System (CSC738)
UI REFRESH (PROFESSIONAL DASHBOARD): Modern layout + model selector + polished styling

IMPORTANT:
- All detection / processing logic is preserved (same YOLO inference, frame skipping, alarm, WebRTC TURN).
- Only UI/UX, layout, and styling have been changed.
- Added a model selection dropdown with metadata + dynamic model loading (no change to detection capability).
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration

# Twilio client for TURN credentials (ephemeral)
from twilio.rest import Client

# ============================================================
# PAGE CONFIG (UI-only changes)
# ============================================================
st.set_page_config(
    page_title="HelmetNet — AI Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MODEL REGISTRY (NEW FEATURE: dropdown + metadata)
# - Keep this list aligned with your trained model locations.
# - You can add more entries without touching core logic.
# ============================================================
MODELS = {
    "Experiment 1 — Baseline": {
        "path": "best.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Model 1."
    },
    "Experiment 2 — Augmented": {
        "path": "bestv2.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Trained with additional augmentation."
    },
    "Experiment 3 — Latest": {
        "path": "bestv3.pt",
        "accuracy": "—",
        "date": "—",
        "description": "Newest experiment model."
    },
}

# ============================================================
# TWILIO TURN (Network Traversal Token -> ICE servers)
# (UNCHANGED FUNCTIONALITY)
# ============================================================
@st.cache_resource
def get_twilio_ice_servers():
    """
    Gets ICE servers (STUN/TURN) from Twilio Network Traversal Service.
    Reliable for restrictive networks (hotspots).
    """
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)

        token = client.tokens.create()  # ephemeral TURN creds
        ice_servers = token.ice_servers

        if not ice_servers:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        return ice_servers
    except Exception as e:
        # UI note kept in sidebar (existing behavior)
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
# PROFESSIONAL ICONS (inline SVG; no external dependencies)
# - Consistent outline style (Heroicons-inspired)
# ============================================================
ICON_SHIELD = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M12 3l7 4v6c0 5-3 8-7 9-4-1-7-4-7-9V7l7-4z"
        stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_CPU = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M9 3v2M15 3v2M9 19v2M15 19v2M3 9h2M3 15h2M19 9h2M19 15h2"
        stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
  <rect x="7" y="7" width="10" height="10" rx="2"
        stroke="currentColor" stroke-width="1.8"/>
</svg>
"""

ICON_IMAGE = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" stroke-width="1.8"/>
  <path d="M8 13l2-2 4 4 3-3 2 2" stroke="currentColor" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M8.5 9.5h.01" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>
</svg>
"""

ICON_VIDEO = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <rect x="3" y="7" width="14" height="10" rx="2" stroke="currentColor" stroke-width="1.8"/>
  <path d="M17 10l4-2v8l-4-2v-4z" stroke="currentColor" stroke-width="1.8"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_LIVE = """
<svg width="20" height="20" viewBox="0 0 24 24" fill="none"
     xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M8 12a4 4 0 018 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
  <path d="M5 12a7 7 0 0114 0" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" opacity="0.7"/>
  <path d="M12 12v7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
</svg>
"""

# ============================================================
# THEME + CSS (UI-only changes)
# - Modern SaaS dashboard look
# - Cards, typography, button polish, refined alerts
# ============================================================
def inject_css(is_dark: bool):
    # Design decision:
    # We control the app’s visual identity with CSS variables.
    # This avoids fighting Streamlit defaults and makes it maintainable.
    if is_dark:
        bg = "#0B1220"
        panel = "#0F1B2D"
        panel2 = "#0B1628"
        text = "#E7EEF8"
        muted = "rgba(231, 238, 248, 0.72)"
        border = "rgba(231, 238, 248, 0.10)"
        shadow = "rgba(0, 0, 0, 0.45)"
        primary = "#3B82F6"
        accent = "#22C55E"
        warning = "#F59E0B"
        danger = "#EF4444"
    else:
        bg = "#F6F8FB"
        panel = "#FFFFFF"
        panel2 = "#F1F5F9"
        text = "#0F172A"
        muted = "rgba(15, 23, 42, 0.66)"
        border = "rgba(15, 23, 42, 0.10)"
        shadow = "rgba(2, 6, 23, 0.12)"
        primary = "#2563EB"
        accent = "#16A34A"
        warning = "#D97706"
        danger = "#DC2626"

    st.markdown(
        f"""
<style>
/* ---------- App-wide variables ---------- */
:root {{
  --hn-bg: {bg};
  --hn-panel: {panel};
  --hn-panel-2: {panel2};
  --hn-text: {text};
  --hn-muted: {muted};
  --hn-border: {border};
  --hn-shadow: {shadow};
  --hn-primary: {primary};
  --hn-accent: {accent};
  --hn-warning: {warning};
  --hn-danger: {danger};
  --hn-radius: 14px;
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--hn-bg) !important;
}}

.block-container {{
  padding-top: 1.25rem !important;
  max-width: 1250px;
}}

* {{
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}}

/* ---------- Sidebar polish ---------- */
section[data-testid="stSidebar"] {{
  background: var(--hn-panel) !important;
  border-right: 1px solid var(--hn-border);
}}

section[data-testid="stSidebar"] .block-container {{
  padding-top: 1rem !important;
}}

.hn-sidebar-title {{
  display: flex;
  gap: 10px;
  align-items: center;
  color: var(--hn-text);
  font-weight: 800;
  letter-spacing: 0.2px;
  margin: 0.25rem 0 0.75rem 0;
}}

.hn-card {{
  background: var(--hn-panel);
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius);
  padding: 16px 16px;
  box-shadow: 0 10px 24px var(--hn-shadow);
}}

.hn-card.soft {{
  background: var(--hn-panel-2);
  box-shadow: none;
}}

.hn-muted {{
  color: var(--hn-muted);
}}

.hn-divider {{
  height: 1px;
  background: var(--hn-border);
  margin: 14px 0;
}}

/* ---------- Header ---------- */
.hn-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(34,197,94,0.10));
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius);
  padding: 18px 18px;
  box-shadow: 0 12px 26px var(--hn-shadow);
  margin-bottom: 14px;
}}

.hn-brand {{
  display: flex;
  gap: 12px;
  align-items: center;
}}

.hn-title {{
  font-size: 1.45rem;
  font-weight: 900;
  margin: 0;
  color: var(--hn-text);
}}

.hn-subtitle {{
  margin: 2px 0 0 0;
  color: var(--hn-muted);
  font-weight: 500;
  font-size: 0.95rem;
}}

.hn-pill {{
  display: inline-flex;
  gap: 8px;
  align-items: center;
  background: var(--hn-panel);
  border: 1px solid var(--hn-border);
  padding: 8px 10px;
  border-radius: 999px;
  color: var(--hn-text);
  font-weight: 600;
}}

/* ---------- Tabs (clean, non-childish) ---------- */
.stTabs [data-baseweb="tab-list"] {{
  background: transparent;
  padding: 8px;
  border-radius: var(--hn-radius);
  border: 1px solid var(--hn-border);
}}

.stTabs [data-baseweb="tab"] {{
  height: 46px;
  border-radius: 12px;
  color: var(--hn-muted);
  font-weight: 700;
  padding: 0 16px;
}}

.stTabs [aria-selected="true"] {{
  background: var(--hn-panel);
  color: var(--hn-text);
  box-shadow: 0 10px 22px var(--hn-shadow);
  border: 1px solid var(--hn-border);
}}

/* ---------- Inputs / uploader as cards ---------- */
[data-testid="stFileUploader"] {{
  background: var(--hn-panel);
  border: 1px dashed var(--hn-border);
  padding: 14px;
  border-radius: var(--hn-radius);
}}

label, .stMarkdown, p, li {{
  color: var(--hn-text);
}}

/* ---------- Metrics ---------- */
[data-testid="metric-container"] {{
  background: var(--hn-panel);
  border: 1px solid var(--hn-border);
  border-radius: var(--hn-radius);
  padding: 12px 12px;
  box-shadow: 0 10px 22px var(--hn-shadow);
}}

[data-testid="stMetricValue"] {{
  font-size: 1.6rem !important;
  font-weight: 900 !important;
  color: var(--hn-text) !important;
}}

[data-testid="stMetricLabel"] {{
  color: var(--hn-muted) !important;
  font-weight: 650 !important;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
  background: var(--hn-primary);
  color: white;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 0.62rem 1.1rem;
  font-weight: 800;
  transition: transform .12s ease, box-shadow .12s ease, filter .12s ease;
  box-shadow: 0 12px 24px var(--hn-shadow);
}}

.stButton > button:hover {{
  transform: translateY(-1px);
  filter: brightness(1.03);
}}

.stDownloadButton > button {{
  background: transparent;
  color: var(--hn-text);
  border: 1px solid var(--hn-border);
  border-radius: 12px;
  font-weight: 800;
  box-shadow: none;
}}

.stDownloadButton > button:hover {{
  background: rgba(59,130,246,0.10);
}}

/* ---------- Alerts (professional “toast-style”) ---------- */
.hn-alert {{
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 14px 14px;
  border-radius: var(--hn-radius);
  border: 1px solid var(--hn-border);
  background: var(--hn-panel);
  box-shadow: 0 12px 26px var(--hn-shadow);
  margin: 14px 0;
}}

.hn-alert .title {{
  font-weight: 900;
  margin: 0;
}}

.hn-alert .desc {{
  margin: 2px 0 0 0;
  color: var(--hn-muted);
  font-weight: 550;
}}

.hn-badge {{
  width: 10px; height: 10px; border-radius: 999px; margin-top: 6px;
}}

.hn-badge.success {{ background: var(--hn-accent); }}
.hn-badge.danger  {{ background: var(--hn-danger); }}

/* ---------- Footer ---------- */
.hn-footer {{
  color: var(--hn-muted);
  font-weight: 550;
  margin-top: 14px;
  padding: 8px 2px;
}}
</style>
        """,
        unsafe_allow_html=True,
    )


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
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def detect_frame(frame, model, conf_threshold):
    # NOTE: Keeping your original inference config (device='cpu', imgsz=640)
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
# SIDEBAR (PRO UI + model dropdown + metadata)
# ============================================================
with st.sidebar:
    # Theme toggle (UI-only)
    is_dark = st.toggle("Dark mode", value=True)
    inject_css(is_dark)

    st.markdown(
        f"""
<div class="hn-sidebar-title">
  <span style="display:flex;align-items:center;opacity:0.95;">{ICON_SHIELD}</span>
  <span>HelmetNet Console</span>
</div>
<div class="hn-card">
  <div style="font-weight:900;font-size:1.05rem;">Configuration</div>
  <div class="hn-muted" style="margin-top:4px;">Select model and inference settings</div>
  <div class="hn-divider"></div>
""",
        unsafe_allow_html=True,
    )

    # NEW: Model selection dropdown
    model_key = st.selectbox("Model", list(MODELS.keys()), index=0)
    selected = MODELS[model_key]
    selected_path = selected["path"]

    # Optional: keep path override (so you don't lose the original "model_path" functionality)
    with st.expander("Advanced model path override", expanded=False):
        use_custom_path = st.checkbox("Use custom model path", value=False)
        custom_path = st.text_input("Custom path", DEFAULT_MODEL_PATH)

    # Resolve final model path (dynamic)
    model_path = custom_path if use_custom_path else selected_path

    confidence_threshold = st.slider("Confidence threshold", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    # Model metadata card
    st.markdown(
        f"""
<div class="hn-card soft" style="margin-top:12px;">
  <div style="font-weight:900;">Model details</div>
  <div class="hn-muted" style="margin-top:6px;"><b>Name:</b> {model_key}</div>
  <div class="hn-muted"><b>Path:</b> <code>{model_path}</code></div>
  <div class="hn-muted"><b>Training date:</b> {selected.get("date","—")}</div>
  <div class="hn-muted"><b>Accuracy:</b> {selected.get("accuracy","—")}</div>
  <div class="hn-muted" style="margin-top:8px;">{selected.get("description","")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)

    # TURN debug (kept as requested; UI polish only)
    with st.expander("WebRTC / TURN diagnostics", expanded=False):
        st.write("ICE servers loaded:", len(ICE_SERVERS))
        if len(ICE_SERVERS) > 0 and "urls" in ICE_SERVERS[0]:
            st.write("First ICE urls:", ICE_SERVERS[0]["urls"])

    # Session stats (unchanged)
    if "total_detections" not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total detections (session)", st.session_state.total_detections)

    st.markdown("</div>", unsafe_allow_html=True)  # close hn-card


# ============================================================
# LOAD MODEL (UNCHANGED behavior; now driven by dropdown)
# ============================================================
model = load_model(model_path)
if not model:
    st.sidebar.warning(f"Could not load {model_path}, using default YOLOv8n")
    model = YOLO("yolov8n.pt")


# ============================================================
# HEADER (PROFESSIONAL)
# ============================================================
st.markdown(
    f"""
<div class="hn-header">
  <div class="hn-brand">
    <div class="hn-pill" style="gap:10px;">
      <span style="display:flex;align-items:center;">{ICON_SHIELD}</span>
      <span style="font-weight:900;">HelmetNet</span>
    </div>
    <div>
      <p class="hn-title">AI Helmet Detection System</p>
      <p class="hn-subtitle">Image • Video • Real-time WebRTC — optimized with frame skipping & safety alerts</p>
    </div>
  </div>
  <div class="hn-pill" title="Inference device (fixed by your current code)">
    <span style="display:flex;align-items:center;">{ICON_CPU}</span>
    <span>Inference: CPU</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs (text-only labels for a cleaner/pro UI)
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Real-time"])


# ============================================================
# TAB 1: IMAGE DETECTION (same logic; better layout)
# ============================================================
with tab1:
    st.markdown(
        f"""
<div class="hn-card">
  <div style="display:flex;gap:10px;align-items:center;">
    <span style="display:flex;align-items:center;">{ICON_IMAGE}</span>
    <div style="font-weight:900;font-size:1.15rem;">Image detection</div>
  </div>
  <div class="hn-muted" style="margin-top:6px;">Upload a JPG/PNG/BMP. Results are shown side-by-side with summary metrics.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1.7, 1.0], gap="large")

    with c_right:
        st.markdown(
            """
<div class="hn-card soft">
  <div style="font-weight:900;">Guidelines</div>
  <ul style="margin-top:10px; color: var(--hn-muted); font-weight: 550;">
    <li>Use clear, well-lit images</li>
    <li>Ensure riders are visible</li>
    <li>Supported: JPG, PNG, BMP</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with c_left:
        img_file = st.file_uploader(
            "Upload image",
            ["jpg", "jpeg", "png", "bmp"],
            key="img",
            label_visibility="collapsed",
        )

    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Professional loading state (kept spinner, but less “emoji”)
        with st.spinner("Running inference on the image…"):
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        p1, p2 = st.columns(2, gap="large")
        with p1:
            st.markdown("<div class='hn-card'><div style='font-weight:900;margin-bottom:8px;'>Original</div></div>", unsafe_allow_html=True)
            st.image(img_file, use_container_width=True)
        with p2:
            st.markdown("<div class='hn-card'><div style='font-weight:900;margin-bottom:8px;'>Result</div></div>", unsafe_allow_html=True)
            st.image(annotated_rgb, use_container_width=True)

        # Alerts (same behavior; redesigned UI)
        if stats["alert"]:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-badge danger"></div>
  <div>
    <p class="title">Safety violation detected</p>
    <p class="desc">At least one rider appears without a helmet. Alarm will play (rate-limited).</p>
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
  <div class="hn-badge success"></div>
  <div>
    <p class="title">Area secure</p>
    <p class="desc">No “no-helmet” detections found in this image.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='hn-card'><div style='font-weight:900;margin-bottom:10px;'>Summary</div></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Helmets", stats["helmet_count"])
        m2.metric("Violations", stats["no_helmet_count"])
        m3.metric("Total objects", len(dets))

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, "rb") as f:
            st.download_button("Download annotated result", f, f"result_{img_file.name}", "image/jpeg")


# ============================================================
# TAB 2: VIDEO DETECTION (same logic; improved display + status)
# ============================================================
with tab2:
    st.markdown(
        f"""
<div class="hn-card">
  <div style="display:flex;gap:10px;align-items:center;">
    <span style="display:flex;align-items:center;">{ICON_VIDEO}</span>
    <div style="font-weight:900;font-size:1.15rem;">Video detection</div>
  </div>
  <div class="hn-muted" style="margin-top:6px;">Optimized frame skipping with live preview and export.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.7, 1.0], gap="large")
    with col2:
        st.markdown(
            """
<div class="hn-card soft">
  <div style="font-weight:900;">Performance mode</div>
  <ul style="margin-top:10px; color: var(--hn-muted); font-weight: 550;">
    <li>Frame skipping enabled</li>
    <li>Live preview during processing</li>
    <li>Formats: MP4, AVI, MOV, MKV</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col1:
        vid_file = st.file_uploader(
            "Upload video",
            ["mp4", "avi", "mov", "mkv"],
            key="vid",
            label_visibility="collapsed",
        )

    if vid_file:
        st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)

        # Design decision: keep a single clear primary action
        if st.button("Start live inference", type="primary"):
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

            # More professional processing state
            status = st.status("Processing video…", expanded=False)

            frame_count = 0
            cached_detections = []
            current_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

            status.write("Initializing decoder and writer…")
            status.write(f"Frames: {total_frames} | FPS: {fps} | Resolution: {width}×{height}")

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
                    caption=f"Frame {frame_count}/{total_frames}",
                    use_container_width=True,
                )

                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Helmets", current_stats["helmet_count"])
                    c2.metric("Violations", current_stats["no_helmet_count"])
                    c3.metric("Progress", f"{int(frame_count / total_frames * 100)}%")

                st_progress.progress(frame_count / total_frames)

            cap.release()
            out.release()

            status.update(label="Processing complete", state="complete", expanded=False)
            st.success("Video processed successfully.")
            st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

            with open(outfile.name, "rb") as f:
                st.download_button("Download annotated video", f, "result.mp4", "video/mp4")


# ============================================================
# TAB 3: REAL-TIME DETECTION (same logic; refined presentation)
# ============================================================
with tab3:
    st.markdown(
        f"""
<div class="hn-card">
  <div style="display:flex;gap:10px;align-items:center;">
    <span style="display:flex;align-items:center;">{ICON_LIVE}</span>
    <div style="font-weight:900;font-size:1.15rem;">Real-time detection</div>
  </div>
  <div class="hn-muted" style="margin-top:6px;">
    Start your webcam stream below. TURN is enabled for improved reliability on restrictive networks.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=HelmetTransformer,
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.set_model(model, confidence_threshold)

        st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='hn-card'><div style='font-weight:900;margin-bottom:10px;'>Live stats</div></div>", unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        m1.metric("Helmets", ctx.video_processor.helmet)
        m2.metric("Violations", ctx.video_processor.no_helmet)

        if ctx.video_processor.alert:
            st.markdown(
                """
<div class="hn-alert">
  <div class="hn-badge danger"></div>
  <div>
    <p class="title">Safety violation detected</p>
    <p class="desc">At least one rider appears without a helmet. Alarm will play (rate-limited).</p>
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
  <div class="hn-badge success"></div>
  <div>
    <p class="title">Area secure</p>
    <p class="desc">No “no-helmet” detections in the current live window.</p>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

# ============================================================
# FOOTER
# ============================================================
st.markdown("<div class='hn-footer'>HelmetNet App • © 2025</div>", unsafe_allow_html=True)
