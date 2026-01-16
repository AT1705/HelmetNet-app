from __future__ import annotations

from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from streamlit_webrtc import (
    VideoTransformerBase,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
)
from twilio.rest import Client

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

# Hide Streamlit chrome + full-bleed canvas
st.markdown(
    """
    <style>
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}

      .stApp { padding: 0 !important; }
      [data-testid="stAppViewContainer"] { padding: 0 !important; }
      [data-testid="stMain"] { padding: 0 !important; }
      [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
      .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# ROUTING
# ============================================================
def _get_page() -> str:
    # Streamlit 1.30+ preferred
    try:
        return str(st.query_params.get("page", "home")).lower()
    except Exception:
        qp = st.experimental_get_query_params()
        return str(qp.get("page", ["home"])[0]).lower()


def _set_top_query_param_js(page: str) -> str:
    # Sets the TOP window (Streamlit) query param from inside the iframe.
    # This is the key to make HTML navbar buttons work.
    return f"""
    <script>
      (function () {{
        function go(p) {{
          try {{
            // Preserve current path, just change query
            window.top.location.search = "?page=" + encodeURIComponent(p);
          }} catch (e) {{
            // fallback
            window.location.search = "?page=" + encodeURIComponent(p);
          }}
        }}

        // Navbar buttons in index.html
        var navHome = document.getElementById("nav-home");
        var navAbout = document.getElementById("nav-about");
        var navDemo = document.getElementById("nav-demo");
        var navBrand = document.getElementById("nav-brand");

        if (navHome)  navHome.addEventListener("click", function(e){{ e.preventDefault(); go("home"); }});
        if (navAbout) navAbout.addEventListener("click", function(e){{ e.preventDefault(); go("about"); }});
        if (navDemo)  navDemo.addEventListener("click", function(e){{ e.preventDefault(); go("demo"); }});
        if (navBrand) navBrand.addEventListener("click", function(e){{ e.preventDefault(); go("home"); }});

        // Hero CTA buttons in index.html
        var heroTry = document.getElementById("hero-try-demo");
        var heroLearn = document.getElementById("hero-learn-more");

        if (heroTry) heroTry.addEventListener("click", function(e){{ e.preventDefault(); go("demo"); }});
        if (heroLearn) heroLearn.addEventListener("click", function(e){{ e.preventDefault(); go("about"); }});

        // Any other CTAs (safe no-op if not present)
        var cta = document.getElementById("cta-launch-demo");
        if (cta) cta.addEventListener("click", function(e){{ e.preventDefault(); go("demo"); }});
      }})();
    </script>
    """


# ============================================================
# MARKETING SITE (HOME / ABOUT) — keep index.html design intact
# ============================================================
def render_marketing_site(active: str) -> None:
    if not SITE_HTML_PATH.exists():
        st.error(f"Missing site HTML at: {SITE_HTML_PATH}")
        st.stop()

    html = SITE_HTML_PATH.read_text(encoding="utf-8")

    # 1) Force only one "page" visible inside the HTML (Home OR About)
    #    Demo section in index.html remains a mock; real demo is Streamlit page.
    inject_css = f"""
    <style>
      /* Show only selected section */
      #page-home, #page-about, #page-demo {{ display: none !important; }}
      #page-{active} {{ display: block !important; }}

      /* Keep fixed nav usable; ensure content isn't hidden by it */
      #page-home, #page-about, #page-demo {{ padding-top: 64px; }}
    </style>
    """

    # 2) Inject JS to make HTML navbar buttons change Streamlit URL (?page=...)
    inject_js = _set_top_query_param_js(active)

    # Insert right before </head>
    html = html.replace("</head>", inject_css + "\n" + "</head>")
    # Insert right before </body> so DOM exists
    html = html.replace("</body>", inject_js + "\n</body>")

    # Render in iframe (reliable for Tailwind CDN + Lucide)
    components.html(html, height=5200, scrolling=False)


# ============================================================
# DEMO (REAL MODEL INFERENCE) — Streamlit must own this page for WebRTC
# ============================================================
NO_HELMET_LABELS = {"no helmet", "no_helmet", "no-helmet", "nohelmet"}
DEFAULT_CONFIDENCE = 0.50
FRAME_SKIP = 3


@st.cache_resource
def get_twilio_ice_servers():
    """
    Fetch STUN/TURN servers via Twilio Network Traversal (ephemeral token).
    Falls back to public STUN if secrets are missing.
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
    p = Path(model_path)
    if p.exists():
        return YOLO(str(p))

    p2 = MODELS_DIR / model_path
    if p2.exists():
        return YOLO(str(p2))

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


def play_alarm():
    # throttle alarm
    if "last_alarm" not in st.session_state:
        st.session_state.last_alarm = 0.0
    if time.time() - st.session_state.last_alarm > 3:
        # If you have alert.mp3 in repo root, it will play.
        alert = APP_DIR / "alert.mp3"
        if alert.exists():
            st.audio(str(alert), format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()


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
                pass

        return draw_boxes(img, self.last_dets)


def inject_demo_css():
    """
    Tighten spacing + align with your Tailwind marketing design.
    This avoids the 'messy' look (no sidebar, consistent max width, clean cards).
    """
    st.markdown(
        """
        <style>
          .hn-section-title{
            font-size: 0.95rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0.25rem 0 0.35rem 0;
          }
          .hn-divider{
            height: 1px;
            background: #e2e8f0;
            margin: 1rem 0;
          }
    
          /* Ensure widgets don't "escape" the card visually */
          .hn-card-b > div { margin-bottom: 0.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_demo_nav():
    """
    Streamlit-native nav for Demo.
    (We keep marketing nav inside index.html for Home/About, but Demo needs Streamlit UI anyway.)
    """
    st.markdown(
        """
        <div style="position:fixed; top:0; left:0; right:0; z-index:999;
                    background:rgba(255,255,255,0.95); backdrop-filter: blur(6px);
                    border-bottom:1px solid #e2e8f0; box-shadow:0 2px 10px rgba(15,23,42,0.06);">
          <div style="max-width:80rem; margin:0 auto; padding:0 1rem; height:64px;
                      display:flex; align-items:center; justify-content:space-between;">
            <a href="?page=home" style="text-decoration:none;">
              <span style="font-weight:800; font-size:1.25rem; color:#0f172a;">HelmetNet</span>
            </a>
            <div style="display:flex; align-items:center; gap:1.75rem;">
              <a href="?page=home" style="text-decoration:none; font-weight:500; color:#475569;">Home</a>
              <a href="?page=about" style="text-decoration:none; font-weight:500; color:#475569;">About</a>
              <a href="?page=demo" style="text-decoration:none; font-weight:800; background:#f59e0b; color:#0f172a;
                                        padding:0.6rem 1.1rem; border-radius:0.75rem;
                                        box-shadow:0 6px 12px rgba(15,23,42,0.10);">Start Demo</a>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_page():
    inject_demo_css()
    render_demo_nav()

    # spacer below fixed nav
    st.markdown("<div class='hn-topspacer'></div>", unsafe_allow_html=True)

    # Hero (matches the marketing tone)
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                    border-bottom: 1px solid #334155; padding: 2.25rem 1rem;">
          <div style="max-width:80rem; margin:0 auto;">
            <h1 style="font-size:2.25rem; font-weight:900; color:white; margin:0 0 0.35rem 0;">
              HelmetNet Detection System
            </h1>
            <p style="color:#cbd5e1; font-size:1.05rem; margin:0;">
              AI-powered helmet compliance detection (Image, Video, Real-Time WebRTC)
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main layout
    st.markdown("<div class='hn-wrap'>", unsafe_allow_html=True)
    col_cfg, col_main = st.columns([1, 2.4], gap="large")

    # LEFT: Configuration
    with col_cfg:
        # Single, properly wrapped card for ALL configuration controls
        st.markdown(
            """
            <div class="hn-card">
              <div class="hn-card-h">Configuration</div>
              <div class="hn-card-b">
            """,
            unsafe_allow_html=True,
        )
    
        # Model picker
        model_files = sorted([p.name for p in MODELS_DIR.glob("*.pt")])
        if not model_files:
            model_files = ["best.pt"]
    
        st.markdown(
            "<div class='hn-section-title'>Model Settings</div>",
            unsafe_allow_html=True,
        )
        model_choice = st.selectbox(
            "Model",
            options=model_files,
            index=0,
            label_visibility="visible",
        )
    
        # Confidence slider
        conf = st.slider(
            "Confidence Threshold",
            0.10,
            1.00,
            DEFAULT_CONFIDENCE,
            0.05,
        )
    
        # Session stats
        if "total_detections" not in st.session_state:
            st.session_state.total_detections = 0
    
        st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hn-section-title'>Session Stats</div>",
            unsafe_allow_html=True,
        )
        st.metric("Total Detections", st.session_state.total_detections)
    
        # ICE debug
        ice_servers = get_twilio_ice_servers()
        st.markdown("<div class='hn-divider'></div>", unsafe_allow_html=True)
        st.caption(f"ICE servers loaded: {len(ice_servers)}")
    
        # Close card
        st.markdown("</div></div>", unsafe_allow_html=True)
    # RIGHT: Content
    with col_main:
        model = load_model(model_choice)

        tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real Time Detection"])

        # TAB 1: Image
        with tab1:
            st.markdown("<div class='hn-card'><div class='hn-card-h'>Upload an Image</div><div class='hn-card-b'>", unsafe_allow_html=True)

            img_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png", "bmp"], key="img")

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

                if stats["alert"]:
                    st.error("NO HELMET DETECTED")
                    play_alarm()
                else:
                    st.success("All Safe")

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(tmp.name, annotated)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download Result", f, f"result_{img_file.name}", "image/jpeg")

            st.markdown("</div></div>", unsafe_allow_html=True)

        # TAB 2: Video
        with tab2:
            st.markdown("<div class='hn-card'><div class='hn-card-h'>Upload a Video</div><div class='hn-card-b'>", unsafe_allow_html=True)

            vid_file = st.file_uploader("Choose video", type=["mp4", "avi", "mov", "mkv"], key="vid")

            if vid_file is not None and st.button("Start Live Inference", type="primary"):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(vid_file.read())

                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                outfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

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
                        caption=(f"Processing {frame_count}/{total_frames}" if total_frames else f"Processing {frame_count}"),
                        use_container_width=True,
                    )
                    if total_frames:
                        st_progress.progress(min(frame_count / total_frames, 1.0))

                    if current_stats["alert"]:
                        play_alarm()

                cap.release()
                out.release()

                st.session_state.total_detections += int(current_stats["helmet_count"] + current_stats["no_helmet_count"])
                st.success("Processing complete")

                with open(outfile.name, "rb") as f:
                    st.download_button("Download Result Video", f, "result.mp4", "video/mp4")

            st.markdown("</div></div>", unsafe_allow_html=True)

        # TAB 3: Real-Time WebRTC
        with tab3:
            st.markdown("<div class='hn-card'><div class='hn-card-h'>Real-Time Live Detection (WebRTC)</div><div class='hn-card-b'>", unsafe_allow_html=True)

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
                    play_alarm()
                else:
                    st.success("Area Secure")

            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close hn-wrap


# ============================================================
# MAIN
# ============================================================
page = _get_page()

if page == "demo":
    render_demo_page()
elif page in {"home", "about"}:
    render_marketing_site(page)
else:
    render_marketing_site("home")
