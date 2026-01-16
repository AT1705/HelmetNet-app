"""
HelmetNet Demo Page (Fixed Portal UI)
- Configuration widgets are truly inside a card (st.container(border=True))
- Single mode control (tabs) to avoid duplicated UI
- Detection logic preserved from original app.py
"""

import base64
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Demo | HelmetNet",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# ============================================================
# ICONS (Optional, you provide from Flaticon)
# Place files in: assets/icons/
#  - shield.svg
#  - image.svg
#  - video.svg
#  - realtime.svg
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]
ICONS_DIR = ROOT_DIR / "assets" / "icons"

def _b64_file(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("utf-8")

def icon_html(stem: str, size: int = 18) -> str:
    for ext in (".svg", ".png", ".jpg", ".jpeg", ".webp"):
        p = ICONS_DIR / f"{stem}{ext}"
        if p.exists():
            mime = "image/svg+xml" if ext == ".svg" else f"image/{ext.lstrip('.')}"
            b64 = _b64_file(p)
            return (
                f'<img src="data:{mime};base64,{b64}" '
                f'style="width:{size}px;height:{size}px;vertical-align:middle;opacity:0.95;" />'
            )
    return ""


# ============================================================
# THEME
# ============================================================
BRAND = {
    "bg": "#F8FAFC",
    "text": "#0F172A",
    "muted": "#475569",
    "border": "rgba(148,163,184,0.28)",
    "card": "rgba(255,255,255,0.92)",
    "heroA": "#0B1220",
    "heroB": "#1F2937",
    "amber": "#F59E0B",
    "amberHover": "#FBBF24",
}

# ============================================================
# GLOBAL CSS (IMPORTANT: style Streamlit's real containers)
# ============================================================
st.markdown(
    f"""
<style>
  .stApp {{
    background: {BRAND["bg"]};
    color: {BRAND["text"]};
  }}

  /* Tighten layout */
  .block-container {{
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1280px;
  }}

  /* Hide Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stStatusWidget"] {{ display: none; }}

  /* Top Nav */
  .hn-nav {{
    position: sticky;
    top: 0;
    z-index: 9999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(226,232,240,1);
    box-shadow: 0 1px 8px rgba(15,23,42,0.06);
  }}
  .hn-nav-inner {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 0.85rem 1.1rem;
    display:flex;
    align-items:center;
    justify-content:space-between;
  }}
  .hn-brand {{
    display:flex;
    align-items:center;
    gap:0.65rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    font-size: 1.15rem;
    color: {BRAND["text"]};
  }}
  .hn-brand-badge {{
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg, #334155, #0F172A);
    color: white;
    box-shadow: 0 10px 25px rgba(15,23,42,0.18);
    overflow: hidden;
  }}
  .hn-links {{
    display:flex;
    align-items:center;
    gap:1.1rem;
  }}
  .hn-link {{
    font-weight: 700;
    color: {BRAND["muted"]};
    text-decoration:none;
    padding: 0.35rem 0.2rem;
  }}
  .hn-link:hover {{ color: {BRAND["text"]}; }}
  .hn-cta {{
    display:inline-flex;
    align-items:center;
    justify-content:center;
    padding: 0.55rem 1.05rem;
    border-radius: 14px;
    font-weight: 900;
    text-decoration:none;
    background: {BRAND["amber"]};
    color: #0B1220;
    box-shadow: 0 10px 25px rgba(245,158,11,0.25);
    border: 1px solid rgba(245,158,11,0.35);
  }}
  .hn-cta:hover {{
    background: {BRAND["amberHover"]};
    color: #0B1220;
  }}

  /* Hero */
  .hn-hero {{
    background: radial-gradient(1100px 380px at 18% 15%, rgba(245,158,11,0.18), transparent 60%),
                linear-gradient(135deg, {BRAND["heroA"]}, {BRAND["heroB"]});
    color: white;
    padding: 3.4rem 1.1rem 3.0rem 1.1rem;
    border-bottom: 1px solid rgba(148,163,184,0.18);
  }}
  .hn-hero-inner {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 1.1rem;
  }}
  .hn-hero-title {{
    font-size: 3.0rem;
    font-weight: 950;
    letter-spacing: -0.03em;
    line-height: 1.1;
  }}
  .hn-hero-sub {{
    margin-top: 0.8rem;
    font-size: 1.15rem;
    color: rgba(226,232,240,0.92);
    font-weight: 650;
  }}

  /* Main wrap spacing */
  .hn-wrap {{
    max-width: 1280px;
    margin: 0 auto;
    padding: 1.7rem 1.1rem 0 1.1rem;
  }}

  /* IMPORTANT: style Streamlit bordered containers as cards */
  /* st.container(border=True) renders a border wrapper; style it. */
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {BRAND["card"]};
    border: 1px solid {BRAND["border"]};
    border-radius: 18px;
    box-shadow: 0 20px 45px rgba(15,23,42,0.08);
  }}

  /* Make headings look like portal */
  .hn-card-title {{
    font-size: 1.15rem;
    font-weight: 900;
    color: {BRAND["text"]};
    margin-bottom: 0.8rem;
  }}
  .hn-section-h {{
    font-size: 0.9rem;
    font-weight: 900;
    color: #334155;
    margin: 0.6rem 0 0.45rem 0;
  }}

  /* Tabs look like the segmented control */
  .stTabs [data-baseweb="tab-list"] {{
    background: rgba(255,255,255,0.92);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 18px;
    padding: 0.5rem;
    box-shadow: 0 16px 35px rgba(15,23,42,0.06);
    gap: 0.5rem;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 14px;
    height: 48px;
    font-weight: 900;
    color: #334155;
    padding: 0 1rem;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BRAND["amber"]} !important;
    color: #0B1220 !important;
    border-radius: 14px !important;
  }}

  /* Uploader */
  [data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.92);
    padding: 1.4rem;
    border-radius: 18px;
    border: 2px dashed rgba(148,163,184,0.55);
    box-shadow: 0 16px 35px rgba(15,23,42,0.05);
  }}

  /* Buttons */
  .stButton > button {{
    background: {BRAND["amber"]};
    color: #0B1220;
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 14px;
    padding: 0.62rem 1.1rem;
    font-weight: 950;
    box-shadow: 0 12px 28px rgba(245,158,11,0.18);
  }}
  .stButton > button:hover {{
    background: {BRAND["amberHover"]};
    color: #0B1220;
  }}
  .stDownloadButton > button {{
    background: linear-gradient(135deg, #334155, #0F172A);
    color: white;
    border: none;
    border-radius: 14px;
    font-weight: 950;
  }}

  /* Alerts */
  .alert-danger {{
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 950;
    margin: 16px 0;
    box-shadow: 0 18px 40px rgba(239,68,68,0.18);
    border: 2px solid rgba(252,165,165,0.9);
    animation: pulse 2s infinite;
  }}
  .alert-success {{
    background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%);
    color: white;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    font-size: 1.1rem;
    font-weight: 950;
    margin: 16px 0;
    box-shadow: 0 18px 40px rgba(34,197,94,0.14);
    border: 2px solid rgba(134,239,172,0.9);
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.9; transform: scale(1.01); }}
  }}

  /* Tips pill */
  .hn-tips {{
    background: rgba(241,245,249,0.8);
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 14px;
    padding: 0.75rem 0.9rem;
    color: #334155;
    font-weight: 700;
    font-size: 0.92rem;
  }}
  .hn-tips strong {{ font-weight: 950; }}

  audio {{ display: none; }}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# TOP NAV + HERO
# ============================================================
shield = icon_html("shield", 18) or "🛡️"
st.markdown(
    f"""
<div class="hn-nav">
  <div class="hn-nav-inner">
    <div class="hn-brand">
      <div class="hn-brand-badge">{shield}</div>
      HelmetNet
    </div>
    <div class="hn-links">
      <a class="hn-link" href="/">Home</a>
      <a class="hn-link" href="/About">About</a>
      <a class="hn-cta" href="/Demo">Start Demo</a>
    </div>
  </div>
</div>

<div class="hn-hero">
  <div class="hn-hero-inner">
    <div class="hn-hero-title">HelmetNet Detection System</div>
    <div class="hn-hero-sub">AI-powered helmet compliance detection</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# ORIGINAL DETECTION LOGIC (UNCHANGED)
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 3
DEFAULT_MODEL_PATH = "best.pt"

@st.cache_resource
def load_model(path):
    try:
        if Path(path).exists():
            model = YOLO(path)
            return model, True
        return YOLO("yolov8n.pt"), False
    except Exception:
        return None, False

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
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def detect_frame(frame, model, conf_threshold):
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
# STATE
# ============================================================
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

# ============================================================
# PORTAL BODY (Left config card + Right workspace)
# ============================================================
st.markdown('<div class="hn-wrap">', unsafe_allow_html=True)

left, right = st.columns([0.34, 0.66], gap="large")

# ---------------------------
# LEFT: Configuration Card
# ---------------------------
with left:
    with st.container(border=True):
        st.markdown('<div class="hn-card-title">Configuration</div>', unsafe_allow_html=True)

        st.markdown('<div class="hn-section-h">Model Settings</div>', unsafe_allow_html=True)

        MODEL_OPTIONS = {
            "YOLOv8 v3.2 (Recommended)": "best.pt",
            "YOLOv8 Nano (yolov8n.pt)": "yolov8n.pt",
            "YOLOv8 Small (yolov8s.pt)": "yolov8s.pt",
        }
        model_choice = st.selectbox("Model Path", list(MODEL_OPTIONS.keys()))
        model_path = MODEL_OPTIONS[model_choice]

        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.1, 1.0,
            float(CONFIDENCE_THRESHOLD),
            0.05,
        )

        st.markdown("---")
        st.markdown('<div class="hn-section-h">Session Stats</div>', unsafe_allow_html=True)

        model, loaded_ok = load_model(model_path)
        if not model:
            st.warning("Model load failed. Falling back to YOLOv8n.")
            model = YOLO("yolov8n.pt")
            loaded_ok = True

        a, b = st.columns(2)
        with a:
            st.metric("Total Detections", st.session_state.total_detections)
        with b:
            st.markdown("**Model Status**")
            st.write("🟢 Loaded" if loaded_ok else "🟠 Fallback")

# ---------------------------
# RIGHT: Workspace Card + Tabs
# ---------------------------
with right:
    with st.container(border=True):
        tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real Time Detection"])

        # ========== IMAGE ==========
        with tab1:
            head, tips = st.columns([0.7, 0.3], gap="large")
            with head:
                st.markdown("### Upload an Image")
                st.caption("Supported formats: JPG, PNG, BMP")
            with tips:
                st.markdown(
                    """
                    <div class="hn-tips">
                      <strong>Quick Tips</strong><br/>
                      Clear, well-lit images<br/>
                      Max size: 10MB
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            img_file = st.file_uploader(
                "Drag and drop file here",
                ["jpg", "jpeg", "png", "bmp"],
                key="img_upl",
                label_visibility="collapsed",
            )

            if img_file:
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                with st.spinner("Analyzing..."):
                    dets, stats = detect_frame(frame, model, confidence_threshold)
                    annotated = draw_boxes(frame, dets)
                    st.session_state.total_detections += len(dets)

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                out_col, table_col = st.columns([0.62, 0.38], gap="large")
                with out_col:
                    st.markdown("#### Result")
                    st.image(annotated_rgb, use_container_width=True)

                with table_col:
                    st.markdown("#### Detections")
                    if dets:
                        rows = []
                        for d in dets:
                            x1, y1, x2, y2 = d["bbox"]
                            rows.append(
                                {
                                    "class": d["class"],
                                    "confidence": round(float(d["confidence"]), 3),
                                    "x1": int(x1), "y1": int(y1),
                                    "x2": int(x2), "y2": int(y2),
                                }
                            )
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    else:
                        st.info("No detections above the confidence threshold.")

                if stats["alert"]:
                    st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                    play_alarm()
                else:
                    st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Helmets", stats["helmet_count"])
                m2.metric("Violations", stats["no_helmet_count"])
                m3.metric("Total Objects", len(dets))

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                cv2.imwrite(tmp.name, annotated)
                with open(tmp.name, "rb") as f:
                    st.download_button("Download Result", f, f"result_{img_file.name}", "image/jpeg")

        # ========== VIDEO ==========
        with tab2:
            head, tips = st.columns([0.7, 0.3], gap="large")
            with head:
                st.markdown("### Upload a Video")
                st.caption("Supported formats: MP4, AVI, MOV, MKV")
            with tips:
                st.markdown(
                    """
                    <div class="hn-tips">
                      <strong>Quick Tips</strong><br/>
                      Short clips run faster<br/>
                      Frame-skip enabled
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            vid_file = st.file_uploader(
                "Drag and drop file here",
                ["mp4", "avi", "mov", "mkv"],
                key="vid_upl",
                label_visibility="collapsed",
            )

            if vid_file:
                if st.button("Start Live Inference"):
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

                        if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                            cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)

                        annotated = draw_boxes(frame, cached_detections)
                        out.write(annotated)

                        if current_stats["alert"]:
                            play_alarm()

                        caption = (
                            f"Processing Frame {frame_count}/{total_frames}"
                            if total_frames else f"Processing Frame {frame_count}"
                        )
                        st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                       caption=caption,
                                       use_container_width=True)

                        with st_metrics.container():
                            a, b, c = st.columns(3)
                            a.metric("Helmets", current_stats["helmet_count"])
                            b.metric("Violations", current_stats["no_helmet_count"])
                            c.metric("Progress", f"{int(frame_count / max(total_frames,1) * 100)}%" if total_frames else "—")

                        if total_frames:
                            st_progress.progress(min(frame_count / max(total_frames, 1), 1.0))

                    cap.release()
                    out.release()

                    st.success("Processing Complete!")
                    st.session_state.total_detections += (current_stats["helmet_count"] + current_stats["no_helmet_count"])

                    with open(outfile.name, "rb") as f:
                        st.download_button("Download Result Video", f, "result.mp4", "video/mp4")

        # ========== REAL TIME ==========
        with tab3:
            st.markdown("### Real-Time Live Detection")
            st.caption("Click START below to begin. Optimized frame skipping is enabled.")

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

                m1, m2 = st.columns(2)
                m1.metric("Helmets", ctx.video_processor.helmet)
                m2.metric("Violations", ctx.video_processor.no_helmet)

                if ctx.video_processor.alert:
                    st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                    play_alarm()
                else:
                    st.markdown('<div class="alert-success">✅ Area Secure</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # end wrap
st.markdown("---")
st.caption("HelmetNet | © 2025")
