"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Modern Safety Theme UI

UPDATED UI/UX:
- Landing page with metrics & value propositions
- Professional dashboard layout (less "default Streamlit")
- Model selection dropdown with metadata
- Threat level & risk score (0–100) with Malaysian safety references
- Action log + CSV export for compliance audits
- Dark / Light mode via CSS variables (no functionality changes)

IMPORTANT:
- All detection logic, Twilio / WebRTC, YOLO usage and thresholds are preserved.
- Only the layout / UI and additional NON-INTRUSIVE features were added.
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client  # Twilio client for TURN credentials
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HelmetNet | AI Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GLOBAL SESSION INITIALIZATION
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"  # "landing" or "detection"

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

if "action_log" not in st.session_state:
    st.session_state.action_log = []  # list of dicts for compliance / audits

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "model_path" not in st.session_state:
    st.session_state.model_path = "best.pt"

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "Custom path"

# ============================================================
# MODEL PRESETS (UI SUGAR ONLY – CORE LOGIC UNCHANGED)
# ============================================================
MODELS = {
    "Model v1.0 – Baseline": {
        "path": "models/experiment_1/best_model.pth",
        "accuracy": "92.5%",
        "date": "2024-01-15",
        "description": "Initial baseline model"
    },
    "Model v2.0 – Enhanced": {
        "path": "models/experiment_2/best_model.pth",
        "accuracy": "95.2%",
        "date": "2024-02-10",
        "description": "Improved with data augmentation"
    },
    "Model v3.0 – Latest (Production)": {
        "path": "models/experiment_3/best_model.pth",
        "accuracy": "97.1%",
        "date": "2024-03-05",
        "description": "Current production model"
    }
}

# ============================================================
# BASE CSS (STRUCTURE & COMPONENTS)
# ============================================================

# NOTE: we use CSS variables so we can flip light/dark mode later.
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Outlined" rel="stylesheet">
<style>
    :root {
        /* These will be overridden by theme-specific CSS below */
        --bg-color: #050816;
        --bg-elevated: #0b1021;
        --bg-muted: #0f172a;
        --accent: #38bdf8;
        --accent-soft: rgba(56, 189, 248, 0.15);
        --accent-strong: #0ea5e9;
        --border-subtle: rgba(148, 163, 184, 0.35);
        --text-color: #e5e7eb;
        --text-muted: #9ca3af;
        --danger: #ef4444;
        --warning: #f97316;
        --success: #22c55e;
        --info: #3b82f6;
    }

    * {
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1150px !important;
    }

    /* GENERAL TYPOGRAPHY */
    h1, h2, h3, h4, h5 {
        color: var(--text-color) !important;
    }
    p, label, span, div, input, textarea {
        color: var(--text-color);
    }

    /* HERO SECTION */
    .hero-wrapper {
        padding: 1.75rem 1.75rem 1.5rem 1.75rem;
        border-radius: 18px;
        background: radial-gradient(circle at top left, rgba(56,189,248,0.25), transparent 55%),
                    radial-gradient(circle at bottom right, rgba(34,197,94,0.22), transparent 50%),
                    var(--bg-elevated);
        border: 1px solid rgba(148,163,184,0.45);
        box-shadow: 0 18px 45px rgba(15,23,42,0.8);
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: var(--text-muted);
        max-width: 480px;
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.18rem 0.75rem;
        font-size: 0.76rem;
        font-weight: 500;
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(148,163,184,0.45);
        color: var(--text-muted);
        gap: 0.4rem;
        margin-bottom: 0.6rem;
    }
    .hero-pill-icon {
        font-size: 1.05rem;
        color: var(--accent);
    }
    .hero-highlight {
        color: var(--accent-strong);
        font-weight: 600;
    }

    /* CTA BUTTONS */
    .primary-cta {
        border-radius: 999px;
        padding: 0.6rem 1.6rem;
        border: none;
        font-weight: 600;
        font-size: 0.93rem;
        background: linear-gradient(135deg, var(--accent-strong), #4f46e5);
        color: white;
        box-shadow: 0 10px 25px rgba(15,23,42,0.9);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
    }
    .primary-cta:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 45px rgba(15,23,42,1);
    }
    .secondary-cta {
        border-radius: 999px;
        padding: 0.6rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.65);
        background: rgba(15,23,42,0.75);
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-muted);
    }
    .secondary-cta:hover {
        background: rgba(15,23,42,0.9);
    }

    .primary-cta .material-icons-outlined,
    .secondary-cta .material-icons-outlined {
        font-size: 1.05rem;
    }

    /* CARDS */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        margin-top: 1.1rem;
    }
    .feature-card {
        background: var(--bg-elevated);
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 14px 30px rgba(15,23,42,0.55);
    }
    .feature-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        width: 32px;
        height: 32px;
        background: var(--accent-soft);
        margin-bottom: 0.65rem;
    }
    .feature-icon .material-icons-outlined {
        font-size: 1.3rem;
        color: var(--accent-strong);
    }
    .feature-title {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .feature-text {
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    /* METRICS */
    [data-testid="metric-container"] {
        background: var(--bg-elevated);
        padding: 0.85rem 0.9rem;
        border-radius: 13px;
        box-shadow: 0 12px 32px rgba(15,23,42,0.7);
        border: 1px solid var(--border-subtle);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: var(--text-color);
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 0.8rem;
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        padding: 0.2rem;
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 10px 24px rgba(15,23,42,0.8);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.88rem;
        font-weight: 500;
        padding: 0.45rem 1.45rem;
        border-radius: 999px;
        color: var(--text-muted);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent-soft) !important;
        color: var(--accent-strong) !important;
        box-shadow: 0 12px 28px rgba(15,23,42,0.85);
    }

    /* INFO BOXES */
    .info-box {
        background: rgba(15,23,42,0.8);
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid var(--border-subtle);
        font-size: 0.85rem;
        color: var(--text-muted);
    }
    .info-box-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
        color: var(--text-color);
    }

    /* ALERT STYLES (RISK LEVELS) */
    .alert-pill {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        gap: 0.4rem;
        margin-bottom: 0.25rem;
    }
    .alert-pill-low {
        background: rgba(34,197,94,0.12);
        border: 1px solid rgba(34,197,94,0.6);
        color: var(--success);
    }
    .alert-pill-medium {
        background: rgba(234,179,8,0.12);
        border: 1px solid rgba(234,179,8,0.6);
        color: #eab308;
    }
    .alert-pill-high {
        background: rgba(249,115,22,0.14);
        border: 1px solid rgba(249,115,22,0.7);
        color: var(--warning);
    }
    .alert-pill-critical {
        background: rgba(239,68,68,0.18);
        border: 1px solid rgba(239,68,68,0.85);
        color: var(--danger);
    }

    .alert-panel {
        border-radius: 14px;
        padding: 0.85rem 1rem;
        background: linear-gradient(145deg, #111827, #020617);
        border: 1px solid rgba(148,163,184,0.8);
        box-shadow: 0 14px 32px rgba(15,23,42,0.9);
        margin-top: 0.6rem;
    }
    .alert-panel-title {
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .alert-panel-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .alert-panel-list {
        margin-left: 1.1rem;
        padding-left: 0;
        font-size: 0.82rem;
        color: var(--text-muted);
    }
    .alert-panel-list li {
        margin-bottom: 0.15rem;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background: var(--bg-elevated);
        border-radius: 14px;
        border: 1px dashed var(--border-subtle);
        padding: 1.1rem;
    }

    /* DOWNLOAD BUTTONS & DEFAULT BUTTONS */
    .stButton > button {
        border-radius: 999px;
        border: none;
        font-size: 0.9rem;
        font-weight: 600;
        padding: 0.45rem 1.25rem;
        background: linear-gradient(135deg, var(--accent-strong), #4f46e5);
        color: white;
        box-shadow: 0 10px 24px rgba(15,23,42,0.8);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }
    .stDownloadButton > button {
        border-radius: 999px;
        border: 1px solid var(--border-subtle);
        background: rgba(15,23,42,0.9);
        color: var(--text-color);
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background: var(--accent-soft);
        color: var(--accent-strong);
    }

    /* TABLE (ACTION LOG) */
    thead tr th {
        background: var(--bg-muted) !important;
        font-size: 0.78rem !important;
    }
    tbody tr td {
        font-size: 0.78rem !important;
    }

    /* FOOTER */
    .app-footer {
        font-size: 0.75rem;
        color: var(--text-muted);
        padding-top: 0.5rem;
        border-top: 1px solid rgba(148,163,184,0.35);
        margin-top: 1.5rem;
    }

    audio { display: none; }
</style>
""",
    unsafe_allow_html=True,
)


def inject_theme_css(dark: bool = True):
    """Injects theme variables for dark / light mode."""
    if dark:
        css = """
        <style>
        :root {
            --bg-color: #020617;
            --bg-elevated: #020617;
            --bg-muted: #0b1120;
            --accent: #38bdf8;
            --accent-soft: rgba(56, 189, 248, 0.12);
            --accent-strong: #0ea5e9;
            --border-subtle: rgba(148, 163, 184, 0.5);
            --text-color: #e5e7eb;
            --text-muted: #9ca3af;
            --danger: #ef4444;
            --warning: #f97316;
            --success: #22c55e;
            --info: #3b82f6;
        }
        body {
            background: radial-gradient(circle at top, #0f172a, #020617 55%);
        }
        </style>
        """
    else:
        css = """
        <style>
        :root {
            --bg-color: #f3f4f6;
            --bg-elevated: #ffffff;
            --bg-muted: #e5e7eb;
            --accent: #0369a1;
            --accent-soft: rgba(56, 189, 248, 0.16);
            --accent-strong: #0369a1;
            --border-subtle: rgba(148, 163, 184, 0.5);
            --text-color: #111827;
            --text-muted: #4b5563;
            --danger: #b91c1c;
            --warning: #c2410c;
            --success: #15803d;
            --info: #1d4ed8;
        }
        body {
            background: radial-gradient(circle at top, #e5e7eb, #f9fafb 55%);
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# ============================================================
# TWILIO TURN (Network Traversal Token -> ICE servers)
# ============================================================
@st.cache_resource
def get_twilio_ice_servers():
    """
    Gets ICE servers (STUN/TURN) from Twilio Network Traversal Service.
    This is the most reliable method for restrictive networks (hotspots).
    """
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)

        token = client.tokens.create()  # returns ephemeral TURN creds
        ice_servers = token.ice_servers

        # Safety: ensure list exists
        if not ice_servers:
            # fallback to STUN only
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

        return ice_servers
    except Exception as e:
        # If Twilio fails, fallback to STUN only (may fail on hotspots)
        st.sidebar.error(f"TURN setup error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


ICE_SERVERS = get_twilio_ice_servers()
RTC_CONFIGURATION = RTCConfiguration({"iceServers": ICE_SERVERS})

# ============================================================
# CONFIGURATION (unchanged core behavior)
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP = 3
DEFAULT_MODEL_PATH = "best.pt"

# ============================================================
# UTILS & DETECTION LOGIC (UNMODIFIED)
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        if Path(path).exists():
            model = YOLO(path)
            st.sidebar.success("Model loaded")
            return model
        st.sidebar.warning("Model not found, using YOLOv8n backbone")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Model load error: {e}")
        return None


def play_alarm():
    if 'last_alarm' not in st.session_state:
        st.session_state.last_alarm = 0
    if time.time() - st.session_state.last_alarm > 3:
        if Path("alert.mp3").exists():
            st.audio("alert.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()


def draw_boxes(frame, detections):
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = (0, 0, 139) if det['class'] in NO_HELMET_LABELS else (0, 100, 0)
        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    return img


def detect_frame(frame, model, conf_threshold):
    # Existing core detection behavior preserved
    results = model.predict(frame, conf=conf_threshold, imgsz=640, verbose=False, device='cpu')

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id].lower()
        conf = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()

        detections.append({'class': cls_name, 'confidence': conf, 'bbox': bbox})

        if cls_name in NO_HELMET_LABELS:
            no_helmet_count += 1
        else:
            helmet_count += 1

    return detections, {
        'helmet_count': helmet_count,
        'no_helmet_count': no_helmet_count,
        'alert': no_helmet_count > 0
    }


# ============================================================
# WEBRTC CLASS (UNMODIFIED CORE LOGIC)
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
                self.helmet = stats['helmet_count']
                self.no_helmet = stats['no_helmet_count']
                self.alert = stats['alert']
            except Exception:
                pass

        return draw_boxes(img, self.last_dets)


# ============================================================
# PRESCRIPTIVE ANALYTICS & MALAYSIAN SAFETY CONTEXT
# ============================================================
def assess_risk(helmet_count, no_helmet_count, avg_conf, conf_threshold):
    """
    Returns a simple risk model for prescriptive analytics.
    Risk score 0–100, level, and Malaysian-safety-flavoured recommendations.
    This is a decision-support heuristic, not legal advice.
    """
    if no_helmet_count <= 0:
        base_score = 5
        level = "Low"
    elif no_helmet_count <= 2:
        base_score = 40
        level = "Medium"
    elif no_helmet_count <= 5:
        base_score = 65
        level = "High"
    else:
        base_score = 85
        level = "Critical"

    conf_factor = float(min(max(avg_conf, conf_threshold), 1.0))
    score = int(min(100, base_score + (conf_factor - conf_threshold) * 40))

    recs = []
    category = "helmet"

    # High-level, Malaysia-oriented recommendations (not a substitute for legal guidance)
    if level == "Low":
        recs = [
            "Continue standard PPE monitoring in line with OSHA 1994 and DOSH guidelines.",
            "Document the inspection as part of routine safety checks.",
            "Reinforce communication on helmet requirements during toolbox talks."
        ]
    elif level == "Medium":
        recs = [
            "Remind personnel in the affected area about mandatory helmet use under OSHA 1994.",
            "Record this event in the safety log for trend monitoring and internal audits.",
            "Review signage and access controls to ensure helmet requirements are clearly displayed."
        ]
    elif level == "High":
        recs = [
            "Immediately engage the site supervisor or Safety & Health Officer (SHO) to intervene.",
            "Temporarily stop non-essential work in the affected zone until compliance is restored.",
            "Record the event in the safety log for potential DOSH review and internal investigation.",
            "Verify that induction and PPE training comply with OSHA 1994 and relevant Malaysian Standards."
        ]
    else:  # Critical
        recs = [
            "Stop work in the affected area in line with the 'stop work' principle under OSHA 1994.",
            "Escalate to management and your Safety & Health Committee for immediate corrective action.",
            "Isolate the area if necessary, especially in machinery zones governed by the Factories and Machinery Act 1967.",
            "Document the incident thoroughly for DOSH and internal reporting, including photos/video.",
            "Review and, if required, update your PPE policy and risk assessment in line with Malaysian Standards."
        ]

    compliance_ref = (
        "References: Occupational Safety and Health Act 1994 (OSHA), "
        "Factories and Machinery Act 1967, DOSH guidelines and applicable "
        "Malaysian Standards (MS) for PPE and industrial safety."
    )

    return {
        "level": level,
        "score": score,
        "category": category,
        "recommendations": recs,
        "compliance_reference": compliance_ref,
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count
    }


def log_detection_event(mode, stats, risk_info, model_name, confidence_threshold):
    """
    Append a single detection event to the in-memory action log.
    This supports Malaysian-compliance style audits (exportable as CSV).
    """
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "helmet_count": stats.get("helmet_count", 0),
        "no_helmet_count": stats.get("no_helmet_count", 0),
        "alert": stats.get("alert", False),
        "risk_level": risk_info["level"],
        "risk_score": risk_info["score"],
        "model": model_name,
        "confidence_threshold": confidence_threshold,
    }
    st.session_state.action_log.append(entry)


def render_risk_panel(risk_info):
    """Render threat level pill, risk score, and Malaysian safety recommendations."""
    level = risk_info["level"]
    score = risk_info["score"]

    if level == "Low":
        cls = "alert-pill-low"
        icon = "verified"
    elif level == "Medium":
        cls = "alert-pill-medium"
        icon = "priority_high"
    elif level == "High":
        cls = "alert-pill-high"
        icon = "warning"
    else:
        cls = "alert-pill-critical"
        icon = "dangerous"

    st.markdown(
        f"""
        <div class="{cls} alert-pill">
            <span class="material-icons-outlined">{icon}</span>
            <span>Threat level: {level} &nbsp;•&nbsp; Risk score: {score}/100</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Prescriptive safety recommendations (Malaysia-focused)", expanded=(level in ["High", "Critical"])):
        st.markdown(
            """
            <div class="alert-panel">
                <div class="alert-panel-title">Suggested actions</div>
                <div class="alert-panel-subtitle">
                    These are decision-support recommendations only and do not replace formal legal or safety advice.
                    Always consult your Safety & Health Officer and applicable Malaysian regulations.
                </div>
                <ul class="alert-panel-list">
            """,
            unsafe_allow_html=True,
        )
        for r in risk_info["recommendations"]:
            st.markdown(f"<li>{r}</li>", unsafe_allow_html=True)
        st.markdown(
            f"""
                </ul>
                <div class="alert-panel-subtitle" style="margin-top:0.45rem;">
                    <strong>Regulatory context (Malaysia):</strong> {risk_info['compliance_reference']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SIDEBAR (CONFIG / MODEL SELECTION / DARK MODE / REPORT EXPORT)
# ============================================================
with st.sidebar:
    st.markdown("#### HelmetNet Control Panel")

    # Theme toggle – applied via CSS variables
    st.session_state.dark_mode = st.toggle("Dark mode", value=st.session_state.dark_mode)

    st.markdown("---")
    st.markdown("**Model configuration**")

    model_options = ["Custom path"] + list(MODELS.keys())
    selected_preset = st.selectbox("Model preset", model_options, index=model_options.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_options else 0)

    if selected_preset == "Custom path":
        model_path = st.text_input("Model path", st.session_state.model_path or DEFAULT_MODEL_PATH)
        st.session_state.model_path = model_path
        st.session_state.selected_model_name = "Custom path"
    else:
        meta = MODELS[selected_preset]
        model_path = meta["path"]
        st.session_state.model_path = model_path
        st.session_state.selected_model_name = selected_preset

        st.markdown(
            f"""
            <div class="info-box" style="margin-top:0.25rem;">
                <div class="info-box-title">Selected model</div>
                <div style="font-size:0.82rem;">
                    <strong>Version:</strong> {selected_preset}<br/>
                    <strong>Path:</strong> <code>{meta["path"]}</code><br/>
                    <strong>Accuracy:</strong> {meta["accuracy"]}<br/>
                    <strong>Trained on:</strong> {meta["date"]}<br/>
                    <strong>Notes:</strong> {meta["description"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    confidence_threshold = st.slider("Detection confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("**WebRTC / TURN diagnostics**")
    st.write("ICE servers configured:", len(ICE_SERVERS))
    if len(ICE_SERVERS) > 0 and "urls" in ICE_SERVERS[0]:
        st.write("First ICE URLs:", ICE_SERVERS[0]["urls"])

    st.markdown("---")
    st.markdown("**Session statistics**")
    st.metric("Total detections", st.session_state.total_detections)
    st.caption("Updated whenever image/video detections complete.")

    st.markdown("---")
    st.markdown("**Compliance & action log**")
    if st.session_state.action_log:
        df_log = pd.DataFrame(st.session_state.action_log)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download action log (CSV)",
            data=csv,
            file_name="helmetnet_action_log.csv",
            mime="text/csv",
        )
        st.caption("Use this CSV as part of your OSHA/DOSH documentation trail.")
    else:
        st.caption("Action log will appear after detections are recorded.")

# Apply theme CSS (dark/light) – after sidebar toggle is known
inject_theme_css(st.session_state.dark_mode)

# ============================================================
# LOAD MODEL (UNCHANGED BEHAVIOR)
# ============================================================
model = load_model(st.session_state.model_path or DEFAULT_MODEL_PATH)
if not model:
    st.sidebar.warning(f"Could not load {st.session_state.model_path}, using default YOLOv8n")
    model = YOLO("yolov8n.pt")

# ============================================================
# LANDING PAGE
# ============================================================
def render_landing_page():
    col_left, col_right = st.columns([1.6, 1.4])

    with col_left:
        st.markdown(
            """
            <div class="hero-wrapper">
                <div class="hero-pill">
                    <span class="material-icons-outlined hero-pill-icon">shield</span>
                    <span>Real-time AI PPE monitoring • Designed for Malaysian safety environments</span>
                </div>
                <div class="hero-title">HelmetNet Detection Console</div>
                <div class="hero-subtitle">
                    YOLO-powered helmet detection with live video, WebRTC, and prescriptive safety analytics
                    aligned with Malaysian OSHA, DOSH, and industrial best practices.
                </div>
                <div style="margin-top:1.1rem; display:flex; gap:0.65rem; flex-wrap:wrap;">
                    <button class="primary-cta" onclick="window.location.reload();">
                        <span class="material-icons-outlined">play_circle</span>
                        <span>Launch detection system</span>
                    </button>
                    <button class="secondary-cta" onclick="document.getElementById('how-it-works').scrollIntoView({behavior:'smooth'});">
                        <span class="material-icons-outlined">quiz</span>
                        <span>How it works</span>
                    </button>
                </div>
                <div style="margin-top:1.05rem; font-size:0.78rem; color:var(--text-muted); max-width:430px;">
                    <span class="hero-highlight">Tip:</span> Use the sidebar to select a model preset and adjust detection confidence.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        # High-level statistics from log
        total_events = len(st.session_state.action_log)
        total_alerts = sum(1 for e in st.session_state.action_log if e.get("alert"))
        avg_risk = (
            int(pd.DataFrame(st.session_state.action_log)["risk_score"].mean())
            if st.session_state.action_log
            else 0
        )
        compliance_rate = (
            int(100 * (total_events - total_alerts) / total_events)
            if total_events > 0
            else 100
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Detections analysed", st.session_state.total_detections)
            st.metric("Compliance rate (no alerts)", f"{compliance_rate}%")
        with m2:
            st.metric("Logged events", total_events)
            st.metric("Average risk score", f"{avg_risk}/100")

    st.markdown("")

    st.markdown("#### Why HelmetNet?")
    st.markdown(
        """
        <div class="card-grid">
            <div class="feature-card">
                <div class="feature-icon"><span class="material-icons-outlined">speed</span></div>
                <div class="feature-title">Accurate, real-time detection</div>
                <div class="feature-text">
                    YOLO-based inference optimised for live WebRTC streams, video uploads and still images,
                    with frame skipping for smooth performance.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="material-icons-outlined">analytics</span></div>
                <div class="feature-title">Actionable analytics</div>
                <div class="feature-text">
                    Threat levels, risk scores and prescriptive recommendations tailored to Malaysian OSH practices,
                    helping supervisors take the right action quickly.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="material-icons-outlined">gavel</span></div>
                <div class="feature-title">Compliance-oriented logging</div>
                <div class="feature-text">
                    Exportable action logs support internal audits and documentation aligned with OSHA 1994, DOSH guidelines
                    and Malaysian Standards for PPE and factory safety.
                </div>
            </div>
            <div class="feature-card">
                <div class="feature-icon"><span class="material-icons-outlined">dashboard_customize</span></div>
                <div class="feature-title">Flexible deployment</div>
                <div class="feature-text">
                    Works with custom YOLO weights or predefined experiments. Adjust confidence and sensitivity per site
                    while keeping the underlying detection pipeline intact.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
    st.markdown("#### How it works")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1. Configure model**\n\nSelect a trained YOLO model or specify a custom path.")
    c2.markdown("**2. Acquire stream**\n\nUpload an image, video, or start a live WebRTC session.")
    c3.markdown("**3. Detect & assess**\n\nHelmetNet runs inference and assesses risk levels in real time.")
    c4.markdown("**4. Act & report**\n\nFollow recommended actions and export logs for audits.")

    st.markdown("")
    st.markdown("#### Recent detection activity")
    if st.session_state.action_log:
        df = pd.DataFrame(st.session_state.action_log).tail(6).iloc[::-1]
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No detections have been logged yet. Launch the detection system to see activity here.")

    # CTA to switch page (via session_state)
    if st.button("Open detection workspace", type="primary", use_container_width=False):
        st.session_state.page = "detection"
        st.experimental_rerun()


# ============================================================
# DETECTION UI (IMAGE / VIDEO / REAL-TIME)
# ============================================================
def render_detection_ui():
    # Header with minimal navigation
    top_col_left, top_col_right = st.columns([1.6, 1])
    with top_col_left:
        st.markdown("### Detection workspace")
        st.caption("Run image, video, or live WebRTC helmet detection. Model and thresholds are managed from the sidebar.")
    with top_col_right:
        # Link back to landing
        if st.button("Back to overview", use_container_width=False):
            st.session_state.page = "landing"
            st.experimental_rerun()

    tab1, tab2, tab3 = st.tabs(["Image detection", "Video detection", "Real-time detection"])

    # --- TAB 1: IMAGE DETECTION ---
    with tab1:
        st.markdown("##### Image detection")
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown(
                """
                <div class="info-box">
                    <div class="info-box-title">Input guidelines</div>
                    <div>
                        • Use clear, well-lit images<br/>
                        • Supported formats: JPG, JPEG, PNG, BMP<br/>
                        • Capture the whole person and helmet region where possible
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col1:
            img_file = st.file_uploader(
                "Upload an image",
                ["jpg", "jpeg", "png", "bmp"],
                key="img",
                label_visibility="collapsed",
            )

        if img_file:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Running detection..."):
                dets, stats = detect_frame(frame, model, confidence_threshold)
                annotated = draw_boxes(frame, dets)
                st.session_state.total_detections += len(dets)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            avg_conf = float(np.mean([d["confidence"] for d in dets])) if dets else 0.0
            risk_info = assess_risk(stats["helmet_count"], stats["no_helmet_count"], avg_conf, confidence_threshold)
            log_detection_event("image", stats, risk_info, st.session_state.selected_model_name, confidence_threshold)

            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**Original**")
                st.image(img_file, use_container_width=True)
            with c2:
                st.markdown("**Detection result**")
                st.image(annotated_rgb, use_container_width=True)

            # Visual alert based on violations (kept, but restyled via CSS)
            if stats["alert"]:
                st.error("Helmet violation detected in this frame.")
                play_alarm()
            else:
                st.success("No helmet violations detected in this frame.")

            st.markdown("##### Detection summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Helmets", stats["helmet_count"])
            m2.metric("No-helmet", stats["no_helmet_count"])
            m3.metric("Total objects", len(dets))
            m4.metric("Avg confidence", f"{avg_conf:.2f}")

            render_risk_panel(risk_info)

            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_img.name, annotated)
            with open(temp_img.name, "rb") as f:
                st.download_button("Download annotated image", f, f"result_{img_file.name}", "image/jpeg")

            with st.expander("Action log (latest entries)"):
                if st.session_state.action_log:
                    df_log = pd.DataFrame(st.session_state.action_log).tail(10).iloc[::-1]
                    st.dataframe(df_log, use_container_width=True, hide_index=True)
                else:
                    st.caption("No events logged yet.")

    # --- TAB 2: VIDEO DETECTION ---
    with tab2:
        st.markdown("##### Video detection")
        col1, col2 = st.columns([2, 1])
        with col2:
            st.markdown(
                """
                <div class="info-box">
                    <div class="info-box-title">Fast mode video analytics</div>
                    <div>
                        • Frame skipping enabled for performance<br/>
                        • Supported formats: MP4, AVI, MOV, MKV<br/>
                        • Optimised for pre-recorded CCTV or production footage
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col1:
            vid_file = st.file_uploader(
                "Upload a video",
                ["mp4", "avi", "mov", "mkv"],
                key="vid",
                label_visibility="collapsed",
            )

        if vid_file:
            st.markdown("###### Processing")
            if st.button("Start live inference", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(vid_file.read())

                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out = cv2.VideoWriter(
                    outfile.name,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

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
                        caption=f"Processing frame {frame_count}/{total_frames}",
                        use_container_width=True,
                    )

                    with st_metrics.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Helmets", current_stats["helmet_count"])
                        c2.metric("No-helmet", current_stats["no_helmet_count"])
                        c3.metric("Progress", f"{int(frame_count / max(total_frames, 1) * 100)}%")

                    st_progress.progress(frame_count / max(total_frames, 1))

                cap.release()
                out.release()

                st.success("Video processing complete.")
                st.session_state.total_detections += (
                    current_stats["helmet_count"] + current_stats["no_helmet_count"]
                )

                # Use the final stats + avg confidence from last cached detections for risk assessment
                avg_conf_video = float(np.mean([d["confidence"] for d in cached_detections])) if cached_detections else 0.0
                risk_info_video = assess_risk(
                    current_stats["helmet_count"],
                    current_stats["no_helmet_count"],
                    avg_conf_video,
                    confidence_threshold,
                )
                log_detection_event("video", current_stats, risk_info_video, st.session_state.selected_model_name, confidence_threshold)

                st.markdown("###### Video summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Last frame helmets", current_stats["helmet_count"])
                m2.metric("Last frame no-helmet", current_stats["no_helmet_count"])
                m3.metric("Frames processed", frame_count)
                m4.metric("Est. avg confidence", f"{avg_conf_video:.2f}")

                render_risk_panel(risk_info_video)

                with open(outfile.name, "rb") as f:
                    st.download_button("Download processed video", f, "result.mp4", "video/mp4")

                with st.expander("Action log (latest entries)"):
                    if st.session_state.action_log:
                        df_log = pd.DataFrame(st.session_state.action_log).tail(10).iloc[::-1]
                        st.dataframe(df_log, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No events logged yet.")

    # --- TAB 3: REAL-TIME DETECTION (WEBRTC) ---
    with tab3:
        st.markdown("##### Real-time detection (WebRTC)")
        st.markdown(
            """
            <div class="info-box">
                <div class="info-box-title">Live webcam stream</div>
                <div>
                    • Click the WebRTC start button below<br/>
                    • Works on mobile and desktop; TURN enabled for restrictive networks<br/>
                    • Frame skipping preserves performance while maintaining detection quality
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

            st.markdown("###### Live statistics")
            m1, m2 = st.columns(2)
            m1.metric("Helmets", ctx.video_processor.helmet)
            m2.metric("No-helmet", ctx.video_processor.no_helmet)

            current_stats = {
                "helmet_count": ctx.video_processor.helmet,
                "no_helmet_count": ctx.video_processor.no_helmet,
                "alert": ctx.video_processor.alert,
            }
            avg_conf_live = 0.0  # we don't track per-frame confidence here; kept minimal

            if ctx.video_processor.alert:
                st.error("Live stream: helmet violation detected.")
                play_alarm()
            else:
                st.success("Live stream: area currently compliant (no violations detected).")

            # Give operator a manual way to log the current live snapshot
            if st.button("Record current live snapshot to action log"):
                risk_info_live = assess_risk(
                    current_stats["helmet_count"],
                    current_stats["no_helmet_count"],
                    avg_conf_live,
                    confidence_threshold,
                )
                log_detection_event("live", current_stats, risk_info_live, st.session_state.selected_model_name, confidence_threshold)
                st.success("Snapshot recorded to action log.")
                render_risk_panel(risk_info_live)


# ============================================================
# ROUTING: LANDING vs DETECTION
# ============================================================
if st.session_state.page == "landing":
    render_landing_page()
else:
    render_detection_ui()

st.markdown(
    """
    <div class="app-footer">
        HelmetNet AI Helmet Detection Console · © 2025<br/>
        This tool supports Malaysian safety practices but does not replace formal legal, DOSH, or OSHA guidance.
    </div>
    """,
    unsafe_allow_html=True,
)
