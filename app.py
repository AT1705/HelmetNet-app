from __future__ import annotations

from pathlib import Path
import base64
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client


APP_DIR = Path(__file__).resolve().parent
SITE_HTML_PATH = APP_DIR / "static" / "site" / "index.html"
MODELS_DIR = APP_DIR / "models"

# -----------------------------------------------------------------------------
# Routing helpers (query params)
# -----------------------------------------------------------------------------

def get_qp(key: str, default: str) -> str:
    try:
        return st.query_params.get(key, default)
    except Exception:
        qp = st.experimental_get_query_params()
        return qp.get(key, [default])[0]


def set_qp(**kwargs):
    # Works across Streamlit versions
    try:
        st.query_params.update(kwargs)
    except Exception:
        st.experimental_set_query_params(**kwargs)


# -----------------------------------------------------------------------------
# Global chrome removal + layout
# -----------------------------------------------------------------------------

def inject_global_css():
    st.markdown(
        """
        <style>
          /* Hide Streamlit chrome */
          #MainMenu {visibility: hidden;}
          header[data-testid="stHeader"] {display: none;}
          footer {display: none;}

          /* True edge-to-edge */
          .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
          [data-testid="stMainBlockContainer"] { padding: 0 !important; }
          [data-testid="stMainBlockContainer"] { max-width: 100% !important; }

          /* Hide Streamlit sidebar entirely (we render our own left panel in Demo) */
          [data-testid="stSidebar"] {display: none !important;}

          /* Avoid extra whitespace between blocks */
          .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }

          /* Base background matches the site */
          body { background: #f8fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_nav(active: str):
    # Pixel-match the Tailwind nav from index.html using plain CSS.
    # Uses query params so the URL changes and Streamlit reruns.
    def cls(is_active: bool) -> str:
        return "hn-navlink hn-active" if is_active else "hn-navlink"

    nav_html = f"""
    <style>
      .hn-nav {{
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 9999;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      }}
      .hn-nav-inner {{
        height: 64px;
        max-width: 80rem; /* ~max-w-7xl */
        margin: 0 auto;
        padding: 0 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }}
      @media (min-width: 640px) {{ .hn-nav-inner {{ padding: 0 24px; }} }}
      @media (min-width: 1024px) {{ .hn-nav-inner {{ padding: 0 32px; }} }}

      .hn-brand {{
        display: flex;
        gap: 8px;
        align-items: center;
        font-weight: 800;
        font-size: 20px;
        color: #0f172a;
        text-decoration: none;
      }}
      .hn-links {{ display: flex; align-items: center; gap: 28px; }}

      .hn-navlink {{
        font-weight: 600;
        color: #334155;
        text-decoration: none;
        padding: 8px 6px;
        border-radius: 10px;
        transition: background 150ms ease, color 150ms ease;
      }}
      .hn-navlink:hover {{ background: #f1f5f9; color: #0f172a; }}
      .hn-active {{ color: #0f172a; }}

      .hn-cta {{
        background: #f59e0b; /* amber-500 */
        color: #0f172a;
        padding: 10px 18px;
        border-radius: 12px;
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
      }}
      .hn-cta:hover {{ background: #fbbf24; }}

      /* Reserve space so content isn't hidden behind the fixed nav */
      .hn-nav-spacer {{ height: 64px; }}
    </style>

    <div class="hn-nav">
      <div class="hn-nav-inner">
        <a class="hn-brand" href="?page=home">
          <span style="display:inline-flex; width:32px; height:32px; align-items:center; justify-content:center;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z" stroke="#334155" stroke-width="2" stroke-linejoin="round"/>
            </svg>
          </span>
          HelmetNet
        </a>

        <div class="hn-links">
          <a class="{cls(active=='home')}" href="?page=home">Home</a>
          <a class="{cls(active=='about')}" href="?page=about">About</a>
          <a class="hn-navlink hn-cta" href="?page=demo">Start Demo</a>
        </div>
      </div>
    </div>
    <div class="hn-nav-spacer"></div>
    """

    st.markdown(nav_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Render the HTML marketing site (Home/About) while keeping Streamlit routing
# -----------------------------------------------------------------------------

def render_site_page(show_main_id: str):
    if not SITE_HTML_PATH.exists():
        st.error(f"Missing site HTML at: {SITE_HTML_PATH}")
        st.stop()

    html = SITE_HTML_PATH.read_text(encoding="utf-8")

    # Hide the HTML's built-in nav (we render our own Streamlit nav),
    # and show only the requested <main> section.
    css_gate = f"""
    <style>
      nav{{display:none !important;}}
      #page-home, #page-about, #page-demo{{display:none !important;}}
      #{show_main_id}{{display:block !important;}}
      /* Remove the internal pt-16 on demo wrapper if needed; harmless on home/about */
    </style>
    """

    payload = css_gate + html

    if hasattr(st, "html"):
        # Non-iframe render so the page scrolls normally.
        st.html(payload, unsafe_allow_javascript=True, width="stretch")
    else:
        import streamlit.components.v1 as components
        components.html(payload, height=2200, scrolling=True)


# -----------------------------------------------------------------------------
# Demo: model + inference (integrated with the marketing site's look)
# -----------------------------------------------------------------------------

NO_HELMET_LABELS = {"no helmet", "no_helmet", "no-helmet"}
DEFAULT_CONFIDENCE = 0.50
FRAME_SKIP_DEFAULT = 3


@st.cache_resource
def get_twilio_ice_servers():
    """Ephemeral ICE server list from Twilio TURN; falls back to STUN."""
    try:
        account_sid = st.secrets.get("TWILIO_ACCOUNT_SID")
        auth_token = st.secrets.get("TWILIO_AUTH_TOKEN")
        if not account_sid or not auth_token:
            return [{"urls": ["stun:stun.l.google.com:19302"]}]

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
    p = Path(model_path)
    if not p.is_absolute():
        p = APP_DIR / model_path

    if p.exists():
        return YOLO(str(p))

    # Fallback (keeps app alive even if user selects wrong path)
    return YOLO("yolov8n.pt")


def detect_frame(frame_bgr: np.ndarray, model: YOLO, conf: float):
    results = model.predict(frame_bgr, conf=conf, imgsz=640, verbose=False, device="cpu")

    helmet_count = 0
    no_helmet_count = 0
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        cls_name = str(model.names[cls_id]).lower()
        confidence = float(box.conf)
        bbox = box.xyxy[0].cpu().numpy().tolist()

        detections.append({"class": cls_name, "confidence": confidence, "bbox": bbox})

        if cls_name in NO_HELMET_LABELS:
            no_helmet_count += 1
        else:
            helmet_count += 1

    stats = {
        "helmet_count": helmet_count,
        "no_helmet_count": no_helmet_count,
        "alert": no_helmet_count > 0,
    }

    return detections, stats


def draw_boxes(frame_bgr: np.ndarray, detections):
    img = frame_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        is_violation = det["class"] in NO_HELMET_LABELS
        color = (0, 0, 139) if is_violation else (0, 100, 0)

        label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(0, y1 - 20)), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def play_alarm():
    # Optional: keep it simple and non-annoying
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0.0

    if time.time() - st.session_state.last_alarm > 3:
        alert_path = APP_DIR / "alert.mp3"
        if alert_path.exists():
            st.audio(str(alert_path), format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()


class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.frame_skip = FRAME_SKIP_DEFAULT
        self.frame_cnt = 0
        self.last_dets = []
        self.helmet = 0
        self.no_helmet = 0
        self.alert = False

    def set_model(self, model, conf: float, frame_skip: int):
        self.model = model
        self.conf = conf
        self.frame_skip = max(1, int(frame_skip))

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None:
            return img

        self.frame_cnt += 1

        if self.frame_cnt % self.frame_skip == 0:
            try:
                dets, stats = detect_frame(img, self.model, self.conf)
                self.last_dets = dets
                self.helmet = stats["helmet_count"]
                self.no_helmet = stats["no_helmet_count"]
                self.alert = stats["alert"]
            except Exception:
                pass

        return draw_boxes(img, self.last_dets)


def inject_demo_css():
    st.markdown(
        """
        <style>
          /* Demo-specific typography and spacing */
          .hn-wrap { background:#f8fafc; }
          .hn-max { max-width:80rem; margin:0 auto; padding:32px 16px; }
          @media (min-width: 640px){ .hn-max { padding:32px 24px; } }
          @media (min-width: 1024px){ .hn-max { padding:32px 32px; } }

          .hn-card { background:#fff; border:1px solid #e2e8f0; border-radius:12px; box-shadow:0 8px 20px rgba(15,23,42,0.06); }
          .hn-card-soft { background:#fff; border:1px solid #e2e8f0; border-radius:12px; box-shadow:0 4px 12px rgba(15,23,42,0.06); }

          /* Make Streamlit widgets visually match index.html */
          div[data-testid="stFileUploader"] section {
            border: 2px dashed #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 36px 24px !important;
            background: #ffffff !important;
          }
          div[data-testid="stFileUploader"] section:hover {
            border-color: #f59e0b !important;
            background: rgba(251,191,36,0.08) !important;
          }

          div.stButton > button {
            background: #f59e0b !important;
            color: #0f172a !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 12px 18px !important;
            box-shadow: 0 4px 12px rgba(15,23,42,0.12) !important;
          }
          div.stButton > button:hover {
            background: #fbbf24 !important;
            box-shadow: 0 6px 16px rgba(15,23,42,0.14) !important;
            transform: translateY(-1px);
          }

          /* Select + slider to look like Tailwind inputs */
          div[data-testid="stSelectbox"] div[role="combobox"],
          div[data-testid="stTextInput"] input {
            background: #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
          }

          /* Tabs (we avoid st.tabs and render our own) */
          .hn-tabbar { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:6px; box-shadow:0 4px 12px rgba(15,23,42,0.08); display:flex; gap:8px; }
          .hn-tab { flex:1; display:block; text-align:center; padding:12px 14px; border-radius:10px; font-weight:600; text-decoration:none; color:#475569; transition: all 150ms ease; }
          .hn-tab:hover { background:#f8fafc; color:#0f172a; }
          .hn-tab-active { background:#f59e0b; color:#0f172a; box-shadow:0 8px 18px rgba(245,158,11,0.35); }

          .hn-pill { background:#f1f5f9; color:#475569; border-radius:10px; padding:6px 10px; font-size:12px; }

          .hn-alert-danger { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color:#fff; padding:16px; border-radius:12px; font-weight:800; text-align:center; border: 2px solid #fca5a5; }
          .hn-alert-success { background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); color:#fff; padding:16px; border-radius:12px; font-weight:800; text-align:center; border: 2px solid #86efac; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_demo():
    inject_demo_css()

    # Header (same as index.html demo header)
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom: 1px solid #334155;">
          <div class="hn-max" style="padding-top:48px; padding-bottom:48px;">
            <h1 style="font-size:36px; font-weight:800; color:white; margin:0 0 12px 0;">HelmetNet Detection System</h1>
            <p style="color:#cbd5e1; font-size:18px; margin:0;">AI-powered helmet compliance detection</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Demo layout
    st.markdown('<div class="hn-wrap"><div class="hn-max">', unsafe_allow_html=True)

    # Two-column layout (config + content) to match the HTML demo mock
    col_left, col_right = st.columns([1, 2.5], gap="large")

    # ---------- Left: Configuration panel ----------
    with col_left:
        st.markdown(
            """
            <div class="hn-card" style="position:sticky; top:96px;">
              <div style="padding:24px; border-bottom:1px solid #e2e8f0;">
                <div style="font-weight:700; font-size:18px; color:#0f172a;">Configuration</div>
              </div>
              <div style="padding:24px; border-bottom:1px solid #e2e8f0;">
                <div style="font-size:13px; font-weight:700; color:#334155; margin-bottom:14px;">Model Settings</div>
            """,
            unsafe_allow_html=True,
        )

        # model dropdown from /models
        pt_files = sorted([p.name for p in MODELS_DIR.glob("*.pt")])
        if not pt_files:
            pt_files = ["best.pt"]

        selected_model = st.selectbox(
            "Model Path",
            pt_files,
            index=min(len(pt_files) - 1, 0),
            label_visibility="collapsed",
        )

        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:13px; color:#475569; margin-bottom:8px;">Confidence Threshold <span style="float:right; color:#d97706; font-weight:800;">{:.0f}%</span></div>'.format(
                get_qp("conf", str(int(DEFAULT_CONFIDENCE * 100))) and float(get_qp("conf", str(int(DEFAULT_CONFIDENCE * 100))))
            ),
            unsafe_allow_html=True,
        )

        conf_val = st.slider(
            "",
            10,
            100,
            int(float(get_qp("conf", str(int(DEFAULT_CONFIDENCE * 100))))),
            step=5,
            label_visibility="collapsed",
        )
        conf = conf_val / 100.0

        frame_skip = st.slider(
            "Frame skip (Real-time)",
            1,
            10,
            int(get_qp("skip", str(FRAME_SKIP_DEFAULT))),
            step=1,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Session stats
        if "total_detections" not in st.session_state:
            st.session_state.total_detections = 0

        st.markdown(
            """
              <div style="padding:24px;">
                <div style="font-size:13px; font-weight:700; color:#334155; margin-bottom:14px;">Session Stats</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:10px;">
              <span style="color:#475569;">Total Detections</span>
              <span style="color:#0f172a; font-weight:700;">{st.session_state.total_detections}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="display:flex; justify-content:space-between; font-size:14px;">
              <span style="color:#475569;">Model Status</span>
              <span style="color:#16a34a; font-weight:700; display:flex; align-items:center; gap:6px;">
                <span style="width:8px; height:8px; background:#22c55e; border-radius:99px; display:inline-block"></span>
                Loaded
              </span>
            </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Persist config to URL for shareability
        set_qp(page="demo", mode=get_qp("mode", "image"), conf=str(conf_val), skip=str(frame_skip))

    # ---------- Right: Demo content ----------
    with col_right:
        mode = get_qp("mode", "image")

        st.markdown(
            f"""
            <div class="hn-tabbar" style="margin-bottom:24px;">
              <a class="hn-tab {'hn-tab-active' if mode=='image' else ''}" href="?page=demo&mode=image&conf={conf_val}&skip={frame_skip}">Image Detection</a>
              <a class="hn-tab {'hn-tab-active' if mode=='video' else ''}" href="?page=demo&mode=video&conf={conf_val}&skip={frame_skip}">Video Detection</a>
              <a class="hn-tab {'hn-tab-active' if mode=='realtime' else ''}" href="?page=demo&mode=realtime&conf={conf_val}&skip={frame_skip}">Real Time Detection</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        model = load_model(str(MODELS_DIR / selected_model))

        if mode == "image":
            st.markdown(
                """
                <div class="hn-card-soft" style="padding:24px; margin-bottom:20px;">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:18px;">
                    <div>
                      <div style="font-size:20px; font-weight:800; color:#0f172a;">Upload an Image</div>
                      <div style="font-size:13px; color:#475569; margin-top:6px;">Supported formats: JPG, PNG, BMP</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:10px 12px; border-radius:10px; text-align:right;">
                      <div style="font-size:12px; color:#64748b; font-weight:700;">Quick Tips</div>
                      <div style="font-size:12px; color:#334155; margin-top:4px;">Clear, well-lit images</div>
                      <div style="font-size:12px; color:#334155;">Max size: 10MB</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            img_file = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png", "bmp"],
                label_visibility="collapsed",
                key="img",
            )

            st.markdown("</div>", unsafe_allow_html=True)

            if img_file:
                run = st.button("Run Detection", use_container_width=True)

                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if run:
                    with st.spinner("Analyzing..."):
                        dets, stats = detect_frame(frame, model, conf)
                        annotated = draw_boxes(frame, dets)
                        st.session_state.total_detections += len(dets)

                    # Two-column preview (matches typical demo layout)
                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.markdown("<div class='hn-card-soft' style='padding:18px;'><div style='font-weight:800; color:#0f172a; margin-bottom:10px;'>Original</div>", unsafe_allow_html=True)
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    with c2:
                        st.markdown("<div class='hn-card-soft' style='padding:18px;'><div style='font-weight:800; color:#0f172a; margin-bottom:10px;'>Result</div>", unsafe_allow_html=True)
                        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    if stats["alert"]:
                        st.markdown('<div class="hn-alert-danger">NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                        play_alarm()
                    else:
                        st.markdown('<div class="hn-alert-success">All Safe!</div>', unsafe_allow_html=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Helmets", stats["helmet_count"])
                    m2.metric("Violations", stats["no_helmet_count"])
                    m3.metric("Total Objects", len(dets))

                    # Download annotated image
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    cv2.imwrite(tmp.name, annotated)
                    with open(tmp.name, "rb") as f:
                        st.download_button(
                            "Download Result",
                            f,
                            file_name=f"result_{img_file.name}",
                            mime="image/jpeg",
                        )

        elif mode == "video":
            st.markdown(
                """
                <div class="hn-card-soft" style="padding:24px; margin-bottom:20px;">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:16px; margin-bottom:18px;">
                    <div>
                      <div style="font-size:20px; font-weight:800; color:#0f172a;">Upload a Video</div>
                      <div style="font-size:13px; color:#475569; margin-top:6px;">Supported formats: MP4, AVI, MOV, MKV</div>
                    </div>
                    <div style="background:#f8fafc; border:1px solid #e2e8f0; padding:10px 12px; border-radius:10px; text-align:right;">
                      <div style="font-size:12px; color:#64748b; font-weight:700;">Fast Mode</div>
                      <div style="font-size:12px; color:#334155; margin-top:4px;">Frame skipping</div>
                      <div style="font-size:12px; color:#334155;">Live preview</div>
                    </div>
                  </div>
                """,
                unsafe_allow_html=True,
            )

            vid_file = st.file_uploader(
                "Choose video",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed",
                key="vid",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            if vid_file:
                if st.button("Start Live Inference", use_container_width=True):
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(vid_file.read())

                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

                    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    out = cv2.VideoWriter(out_file.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

                    frame_ph = st.empty()
                    prog = st.progress(0)

                    frame_count = 0
                    cached_dets = []
                    cached_stats = {"helmet_count": 0, "no_helmet_count": 0, "alert": False}

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        if frame_count % max(1, frame_skip) == 0 or frame_count == 1:
                            cached_dets, cached_stats = detect_frame(frame, model, conf)

                        annotated = draw_boxes(frame, cached_dets)
                        out.write(annotated)

                        if cached_stats["alert"]:
                            play_alarm()

                        frame_ph.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_count}/{total_frames}",
                            use_container_width=True,
                        )
                        prog.progress(min(1.0, frame_count / max(1, total_frames)))

                    cap.release()
                    out.release()

                    st.session_state.total_detections += cached_stats["helmet_count"] + cached_stats["no_helmet_count"]
                    st.success("Processing complete")

                    with open(out_file.name, "rb") as f:
                        st.download_button("Download Result Video", f, file_name="result.mp4", mime="video/mp4")

        else:  # realtime
            st.markdown(
                """
                <div class="hn-card-soft" style="padding:24px; margin-bottom:20px;">
                  <div style="font-size:20px; font-weight:800; color:#0f172a;">Real-Time Live Detection</div>
                  <div style="font-size:13px; color:#475569; margin-top:6px;">
                    Click START to enable webcam. TURN is supported via Twilio for restrictive networks.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ice_servers = get_twilio_ice_servers()
            rtc_config = RTCConfiguration({"iceServers": ice_servers})

            ctx = webrtc_streamer(
                key="helmet-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=HelmetTransformer,
                async_processing=True,
            )

            if ctx.video_processor:
                ctx.video_processor.set_model(model, conf, frame_skip)

                m1, m2 = st.columns(2)
                m1.metric("Helmets", ctx.video_processor.helmet)
                m2.metric("Violations", ctx.video_processor.no_helmet)

                if ctx.video_processor.alert:
                    st.markdown('<div class="hn-alert-danger">NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                    play_alarm()
                else:
                    st.markdown('<div class="hn-alert-success">Area Secure</div>', unsafe_allow_html=True)

                # Debug info (compact)
                st.markdown(
                    f"<div class='hn-pill'>ICE servers loaded: {len(ice_servers)}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("</div></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------

st.set_page_config(page_title="HelmetNet", layout="wide", initial_sidebar_state="collapsed")
inject_global_css()

page = get_qp("page", "home")

render_top_nav(page)

if page == "about":
    render_site_page("page-about")
elif page == "demo":
    render_demo()
else:
    render_site_page("page-home")
