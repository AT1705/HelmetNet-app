"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Fast UI
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
import os
from datetime import timedelta
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Helmet Detection",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; color: var(--text-color); text-align: center;}
    .alert-danger {background: #EF4444; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; animation: pulse 2s infinite;}
    .alert-success {background: #22C55E; color: white; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold;}
    @keyframes pulse {0% {opacity: 1;} 50% {opacity: 0.8;}}
    [data-testid="stMetricValue"] {font-size: 1.5rem !important;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 3  # OPTIMIZATION: Process 1 frame, skip 2 (3x faster)

# ============================================================
# UTILS
# ============================================================
@st.cache_resource
def load_model(model_source):
    try:
        model = YOLO(model_source)
        return model
    except Exception as e:
        return None

def play_alarm():
    # Simple rate-limited alarm
    if 'last_alarm' not in st.session_state: st.session_state.last_alarm = 0
    if time.time() - st.session_state.last_alarm > 3:
        if Path("alert.mp3").exists():
            st.audio("alert.mp3", format="audio/mp3", autoplay=True)
        st.session_state.last_alarm = time.time()

def draw_boxes(frame, detections):
    """
    Manually draws bounding boxes on a frame.
    Used for 'skipped' frames to maintain visual continuity without running AI.
    """
    img = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = (0, 0, 255) if det['class'] in NO_HELMET_LABELS else (0, 255, 0)
        label = f"{det['class']} {det['confidence']:.2f}"
        
        # Draw Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw Label Background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def detect_frame(frame, model, conf_threshold):
    # OPTIMIZATION: imgsz=640 speeds up inference on large images
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
# WEBRTC CLASS
# ============================================================
class HelmetTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf = 0.25
        self.helmet = 0
        self.no_helmet = 0
        self.frame_cnt = 0
        self.last_dets = [] # Cache detections
        
    def set_model(self, model, conf):
        self.model = model
        self.conf = conf
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None: return img
        
        self.frame_cnt += 1
        
        # OPTIMIZATION: Run AI only every 3rd frame
        if self.frame_cnt % FRAME_SKIP == 0:
            try:
                detections, stats = detect_frame(img, self.model, self.conf)
                self.last_dets = detections
                self.helmet = stats['helmet_count']
                self.no_helmet = stats['no_helmet_count']
            except: pass
            
        # Draw the (new or cached) boxes
        return draw_boxes(img, self.last_dets)

# ============================================================
# APP UI
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Uploader
    model_file = st.file_uploader("Upload Model (best.pt)", type=['pt'])
    if model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            f.write(model_file.read())
            model_path = f.name
        st.success("Custom model loaded!")
    else:
        model_path = st.text_input("Or Model Path", "best.pt")

    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    
    st.divider()
    st.metric("Total Detections", st.session_state.get('total_detections', 0))

# LOAD MODEL
model = load_model(model_path)
if not model:
    st.error("❌ Model not found. Please upload 'best.pt' in the sidebar.")
    st.stop()

st.markdown('<h1 class="main-header">🛵 AI Helmet Detection</h1>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video (Fast)", "📱 Webcam"])

# --- IMAGE TAB ---
with tab1:
    img_file = st.file_uploader("Upload Image", ["jpg", "png"], key="img")
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner("Processing..."):
            dets, stats = detect_frame(frame, model, conf)
            annotated = draw_boxes(frame, dets)
            
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("🟢 Helmets", stats['helmet_count'])
        c2.metric("🔴 Violations", stats['no_helmet_count'])
        
        if stats['alert']:
            st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)

# --- VIDEO TAB (OPTIMIZED LIVE INFERENCE) ---
with tab2:
    st.info("💡 **Fast Mode:** Processing is optimized for speed using frame skipping.")
    vid_file = st.file_uploader("Upload Video", ["mp4", "avi", "mov"], key="vid")
    
    if vid_file:
        if st.button("▶️ Start Live Inference", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(vid_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prepare Output
            outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            # UI Elements
            st_frame = st.empty()
            st_metrics = st.empty()
            st_progress = st.progress(0)
            
            frame_count = 0
            cached_detections = [] # To store boxes for skipped frames
            current_stats = {'helmet_count': 0, 'no_helmet_count': 0, 'alert': False}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                
                # --- OPTIMIZATION LOGIC ---
                # 1. Run AI only every N frames
                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, conf)
                
                # 2. Draw boxes (using fresh or cached data)
                annotated = draw_boxes(frame, cached_detections)
                
                # 3. Write to video file
                out.write(annotated)
                
                # 4. Trigger Alarm
                if current_stats['alert']: play_alarm()
                
                # 5. Display Live (Every frame)
                # Convert to RGB for Streamlit
                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                             caption=f"Frame {frame_count}/{total_frames}",
                             use_container_width=True)
                
                with st_metrics.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🟢 Helmets", current_stats['helmet_count'])
                    m2.metric("🔴 Violations", current_stats['no_helmet_count'])
                    m3.metric("⏱️ Progress", f"{int(frame_count/total_frames*100)}%")
                
                st_progress.progress(frame_count / total_frames)
            
            cap.release()
            out.release()
            st.success("✅ Processing Complete")
            
            with open(outfile.name, 'rb') as f:
                st.download_button("📥 Download Result", f, "result.mp4", "video/mp4")

# --- WEBCAM TAB ---
with tab3:
    st.write("Click START to use webcam.")
    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=HelmetTransformer,
        async_processing=True
    )
    
    if ctx.video_processor:
        ctx.video_processor.set_model(model, conf)
        c1, c2 = st.columns(2)
        c1.metric("🟢 Helmets", ctx.video_processor.helmet)
        c2.metric("🔴 Violations", ctx.video_processor.no_helmet)
