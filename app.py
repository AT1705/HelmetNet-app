from __future__ import annotations

from pathlib import Path
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)
from twilio.rest import Client

import streamlit.components.v1 as components

# ============================================================
# PATHS
# ============================================================
APP_DIR = Path(__file__).resolve().parent
SITE_HTML_PATH = APP_DIR / "static" / "site" / "index.html"
MODELS_DIR = APP_DIR / "models"

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="HelmetNet",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Streamlit chrome removal + edge-to-edge canvas for marketing pages ---
st.markdown(
    """
    <style>
      /* Hide built-in Streamlit chrome */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      /* Default: full-bleed */
      [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ROUTING
# ============================================================

def _get_page() -> str:
    # Streamlit 1.30+ (preferred)
    try:
        return str(st.query_params.get("page", "home")).lower()
    except Exception:
        qp = st.experimental_get_query_params()
        return str(qp.get("page", ["home"])[0]).lower()


# ============================================================
# MARKETING SITE (HOME / ABOUT)
# ============================================================

def render_marketing_site(active: str) -> None:
    """Renders the exported Figma HTML (Home/About) as a marketing site.

    Notes:
    - Uses components.html (iframe) so external scripts (Tailwind CDN, Lucide) work reliably.
    - Navigation is handled by query params (href="?page=...") and server-side routing.
    """

    if not SITE_HTML_PATH.exists():
        st.error(f"Missing site HTML at: {SITE_HTML_PATH}")
        st.stop()

    render_top_nav(active)

    html = SITE_HTML_PATH.read_text(encoding="utf-8")

    # Hide the embedded HTML nav: navigation is handled by Streamlit (query params)
    # because content rendered via components.html runs inside an iframe.
    inject = (
        "<style>\n"
        "  #hn-app > nav{display:none !important;}\n"
        "  #page-home, #page-about, #page-demo{display:none !important;}\n"
        f"  #page-{active}{{display:block !important;}}\n"
        "</style>"
    )
    html = html.replace("</head>", inject + "\n</head>")

    # Set a generous height to avoid clipping. (No internal scrollbar.)
    components.html(html, height=5200, scrolling=False)


# ============================================================
# DEMO (REAL MODEL INFERENCE)
# ============================================================

NO_HELMET_LABELS = {"no helmet", "no_helmet", "no-helmet", "nohelmet"}
DEFAULT_CONFIDENCE = 0.50
FRAME_SKIP = 3


@st.cache_resource
def get_twilio_ice_servers():
    """Fetch STUN/TURN servers via Twilio Network Traversal (ephemeral token).

    If Twilio secrets are not configured, we fall back to a public STUN server.
    """
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        ice_servers = token.ice_servers
        if not ice_servers:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]
        return ice_servers
    except Exception:
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    """Load a YOLO model from a local path; fall back to yolov8n if missing."""
    p = Path(model_path)
    if p.exists():
        return YOLO(str(p))

    # Allow shorthand names like "model_1.pt"
    p2 = MODELS_DIR / model_path
    if p2.exists():
        return YOLO(str(p2))

    # Fallback
    return YOLO("yolov8n.pt")


def draw_boxes(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls = det["class"]
        conf = det["confidence"]
        color = (0, 0, 139) if cls in NO_HELMET_LABELS else (0, 100, 0)
        label = f"{cls} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def detect_frame(frame: np.ndarray, model: YOLO, conf_threshold: float):
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


class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model: YOLO | None = None
        self.conf: float = DEFAULT_CONFIDENCE
        self.helmet: int = 0
        self.no_helmet: int = 0
        self.frame_cnt: int = 0
        self.last_dets: list[dict] = []
        self.alert: bool = False

    def set_model(self, model: YOLO, conf: float):
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
                # Keep last detections on transient errors
                pass

        return draw_boxes(img, self.last_dets)


def inject_demo_css():
    """A minimal CSS layer to make the Streamlit demo resemble the Tailwind design."""
    st.markdown(
        """
        <style>
          [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }

          .hn-hero {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid #334155;
            padding: 3rem 2rem;
            margin-top: 4rem;
          }
          .hn-wrap { max-width: 80rem; margin: 0 auto; padding: 2rem; }
          .hn-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 0.75rem; box-shadow: 0 8px 16px rgba(15,23,42,0.08); }
          .hn-card-h { padding: 1rem 1.25rem; border-bottom: 1px solid #e2e8f0; font-weight: 700; color: #0f172a; }
          .hn-card-b { padding: 1.25rem; }

          /* Buttons */
          .stButton > button {
            background: #f59e0b !important;
            color: #0f172a !important;
            border: none !important;
            border-radius: 0.75rem !important;
            font-weight: 700 !important;
            padding: 0.75rem 1rem !important;
          }
          .stButton > button:hover { background: #fbbf24 !important; }

          /* Tabs */
          .stTabs [data-baseweb="tab-list"] { background: #ffffff; padding: 0.4rem; border-radius: 0.9rem; border: 1px solid #e2e8f0; box-shadow: 0 6px 14px rgba(15,23,42,0.06); }
          .stTabs [data-baseweb="tab"] { height: 48px; border-radius: 0.75rem; font-weight: 650; }
          .stTabs [aria-selected="true"] { background: #f59e0b !important; color: #0f172a !important; }

          /* Uploader */
          [data-testid="stFileUploader"] { background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 0.75rem; padding: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(active: str):
    """A lightweight Streamlit-native nav that matches the marketing nav targets."""
    st.markdown(
        f"""
        <div style="position:fixed; top:0; left:0; right:0; z-index:999; background:rgba(255,255,255,0.95); backdrop-filter: blur(6px); border-bottom:1px solid #e2e8f0; box-shadow:0 2px 10px rgba(15,23,42,0.06);">
          <div style="max-width:80rem; margin:0 auto; padding:0 1rem; height:64px; display:flex; align-items:center; justify-content:space-between;">
            <a href="?page=home" style="display:flex; align-items:center; gap:0.5rem; text-decoration:none;">
              <span style="font-weight:800; font-size:1.25rem; color:#0f172a;">HelmetNet</span>
            </a>
            <div style="display:flex; align-items:center; gap:1.75rem;">
              <a href="?page=home" style="text-decoration:none; font-weight:{700 if active=='home' else 500}; color:{'#0f172a' if active=='home' else '#475569'};">Home</a>
              <a href="?page=about" style="text-decoration:none; font-weight:{700 if active=='about' else 500}; color:{'#0f172a' if active=='about' else '#475569'};">About</a>
              <a href="?page=demo" style="text-decoration:none; font-weight:700; background:#f59e0b; color:#0f172a; padding:0.6rem 1.1rem; border-radius:0.75rem; box-shadow:0 6px 12px rgba(15,23,42,0.10);">Start Demo</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_page():
    inject_demo_css()
    render_top_nav("demo")

    # Header
    st.markdown(
        """
        <div class="hn-hero">
          <div style="max-width:80rem; margin:0 auto;">
            <h1 style="font-size:2.25rem; font-weight:800; color:white; margin:0 0 0.5rem 0;">HelmetNet Detection System</h1>
            <p style="color:#cbd5e1; font-size:1.05rem; margin:0;">AI-powered helmet compliance detection</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout
    wrap = st.container()
    with wrap:
        st.markdown('<div class="hn-wrap">', unsafe_allow_html=True)
        col_cfg, col_main = st.columns([1, 2.4], gap="large")

        # --- Left config ---
        with col_cfg:
            st.markdown('<div class="hn-card">', unsafe_allow_html=True)
            st.markdown('<div class="hn-card-h">Configuration</div>', unsafe_allow_html=True)
            st.markdown('<div class="hn-card-b">', unsafe_allow_html=True)

            # Model selection
            model_files = sorted([p.name for p in MODELS_DIR.glob("*.pt")])
            default_model = model_files[0] if model_files else "best.pt"

            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#334155; margin-bottom:0.5rem;'>Model Settings</div>", unsafe_allow_html=True)
            model_choice = st.selectbox(
                "Model",
                options=model_files if model_files else [default_model],
                index=0,
                label_visibility="collapsed",
            )

            conf = st.slider(
                "Confidence Threshold",
                min_value=0.10,
                max_value=1.00,
                value=DEFAULT_CONFIDENCE,
                step=0.05,
            )

            # Session stats
            if "total_detections" not in st.session_state:
                st.session_state.total_detections = 0

            st.markdown("<div style='margin-top:1.25rem; padding-top:1rem; border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#334155; margin-bottom:0.5rem;'>Session Stats</div>", unsafe_allow_html=True)
            st.metric("Total Detections", st.session_state.total_detections)
            st.markdown("</div>", unsafe_allow_html=True)

            # TURN debug
            ice_servers = get_twilio_ice_servers()
            st.markdown("<div style='margin-top:1rem; font-size:0.8rem; color:#64748b;'>", unsafe_allow_html=True)
            st.write("ICE servers loaded:", len(ice_servers))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)

        # --- Right content ---
        with col_main:
            model = load_model(model_choice)

            tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real Time Detection"])

            # TAB 1: Image
            with tab1:
                st.markdown('<div class="hn-card">', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-h">Upload an Image</div>', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-b">', unsafe_allow_html=True)

                img_file = st.file_uploader(
                    "Choose image",
                    type=["jpg", "jpeg", "png", "bmp"],
                    key="img",
                    label_visibility="collapsed",
                )

                run = st.button("Run Detection", use_container_width=True)

                if img_file is not None and run:
                    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    with st.spinner("Analyzing..."):
                        dets, stats = detect_frame(frame, model, conf)
                        annotated = draw_boxes(frame, dets)

                    st.session_state.total_detections += len(dets)

                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.markdown("**Original**")
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with c2:
                        st.markdown("**Result**")
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Helmets", stats["helmet_count"])
                    m2.metric("Violations", stats["no_helmet_count"])
                    m3.metric("Total Objects", len(dets))

                    # Download
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    cv2.imwrite(tmp.name, annotated)
                    with open(tmp.name, "rb") as f:
                        st.download_button("Download Result", f, f"result_{img_file.name}", "image/jpeg")

                st.markdown('</div></div>', unsafe_allow_html=True)

            # TAB 2: Video
            with tab2:
                st.markdown('<div class="hn-card">', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-h">Upload a Video</div>', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-b">', unsafe_allow_html=True)

                vid_file = st.file_uploader(
                    "Choose video",
                    type=["mp4", "avi", "mov", "mkv"],
                    key="vid",
                    label_visibility="collapsed",
                )

                if vid_file is not None and st.button("Start Live Inference", type="primary"):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(vid_file.read())

                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                    outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    out = cv2.VideoWriter(
                        outfile.name,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                    st_frame = st.empty()
                    st_progress = st.progress(0)

                    frame_count = 0
                    cached_dets: list[dict] = []
                    current_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                            cached_dets, current_stats = detect_frame(frame, model, conf)

                        annotated = draw_boxes(frame, cached_dets)
                        out.write(annotated)

                        st_frame.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=(
                                f"Processing frame {frame_count}/{total_frames}" if total_frames else f"Processing frame {frame_count}"
                            ),
                            use_container_width=True,
                        )

                        if total_frames:
                            st_progress.progress(min(frame_count / total_frames, 1.0))

                    cap.release()
                    out.release()

                    st.session_state.total_detections += int(current_stats["helmet_count"] + current_stats["no_helmet_count"])
                    st.success("Processing complete")

                    with open(outfile.name, "rb") as f:
                        st.download_button("Download Result Video", f, "result.mp4", "video/mp4")

                st.markdown('</div></div>', unsafe_allow_html=True)

            # TAB 3: Real-time
            with tab3:
                st.markdown('<div class="hn-card">', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-h">Real-Time Live Detection</div>', unsafe_allow_html=True)
                st.markdown('<div class="hn-card-b">', unsafe_allow_html=True)

                ice_servers = get_twilio_ice_servers()
                rtc_conf = RTCConfiguration({"iceServers": ice_servers})

                ctx = webrtc_streamer(
                    key="helmet-live",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_conf,
                    video_processor_factory=HelmetTransformer,
                    async_processing=True,
                )

                if ctx.video_processor:
                    ctx.video_processor.set_model(model, conf)

                    c1, c2 = st.columns(2)
                    c1.metric("Helmets", ctx.video_processor.helmet)
                    c2.metric("Violations", ctx.video_processor.no_helmet)

                    if ctx.video_processor.alert:
                        st.error("NO HELMET DETECTED")
                    else:
                        st.success("Area Secure")

                st.markdown('</div></div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================
page = _get_page()

if page == "demo":
    render_demo_page()
elif page in {"home", "about"}:
    render_marketing_site(page)
else:
    # Unknown -> Home
    render_marketing_site("home")
