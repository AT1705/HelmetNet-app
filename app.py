"""
AI Helmet Detection System (CSC738)
OPTIMIZED: Live Inference + Frame Skipping + Modern Safety Theme UI
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
# SAFETY THEME CSS (FROM APP.PY)
# ============================================================
st.markdown("""
<style>
    /* Global Styles using Streamlit Variables */
    .block-container {
        padding-top: 1.5rem !important; 
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem; 
        font-weight: 800; 
        color: var(--text-color); 
        text-align: center; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        text-align: center; 
        font-size: 1.1rem; 
        color: var(--text-color); 
        opacity: 0.8;
        font-weight: 500; 
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: var(--text-color) !important; 
        font-weight: 700 !important; 
        border-bottom: 3px solid #FFD700; 
        padding-bottom: 0.5rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary-background-color); 
        padding: 0.5rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; 
        border-radius: 8px; 
        color: var(--text-color); 
        font-weight: 600; 
        padding: 0 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #FFD700 !important; 
        color: #1E3A8A !important; 
    }
    
    /* Alerts */
    .alert-danger {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 12px;
        text-align: center; 
        font-size: 1.3rem; 
        font-weight: 700; 
        animation: pulse 2s infinite; 
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(239,68,68,0.3); 
        border: 3px solid #FCA5A5;
    }
    .alert-success {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); 
        color: white; 
        padding: 20px; 
        border-radius: 12px;
        text-align: center; 
        font-size: 1.3rem; 
        font-weight: 700; 
        margin: 20px 0; 
        box-shadow: 0 4px 6px rgba(34,197,94,0.3); 
        border: 3px solid #86EFAC;
    }
    @keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: 0.85; transform: scale(1.02);}}
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); 
        color: #1E3A8A; 
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem; 
        font-weight: 700; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        transform: translateY(-2px); 
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: #1E3A8A;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); 
        color: white;
        border: none;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; 
        font-weight: 700 !important; 
        color: var(--text-color);
    }
    [data-testid="metric-container"] {
        background: var(--secondary-background-color); 
        padding: 1rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); 
        border-left: 4px solid #FFD700;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--secondary-background-color); 
        padding: 1.5rem; 
        border-radius: 10px; 
        border: 2px dashed #FFD700;
    }
    
    audio {display: none;}
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1); 
        padding: 1rem; 
        border-radius: 10px; 
        border-left: 4px solid #1E3A8A; 
        margin: 1rem 0;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURATION
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.25
FRAME_SKIP = 3  # Optimization from app_2.py
DEFAULT_MODEL_PATH = "best.pt"

# ============================================================
# UTILS & LOGIC (FROM APP_2.PY)
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
    if 'last_alarm' not in st.session_state: st.session_state.last_alarm = 0
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
        x1, y1, x2, y2 = map(int, det['bbox'])
        # Red for no helmet, Green for helmet
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
    # Logic from app_2.py
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
        self.alert = False
        
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
                self.alert = stats['alert']
            except: pass
            
        # Draw the (new or cached) boxes
        return draw_boxes(img, self.last_dets)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("**🤖 Model Settings**")
    
    st.text_input("Model Path", DEFAULT_MODEL_PATH)

    confidence_threshold = st.slider("🎯 Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    
    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total Detections", st.session_state.total_detections)
    
    st.markdown("---")

# LOAD MODEL
model = load_model(DEFAULT_MODEL_PATH)
if not model:
    st.sidebar.warning(f"⚠️ Could not load {model_path}, using default YOLOv8n")
    model = YOLO("yolov8n.pt")

# ============================================================
# MAIN APP UI
# ============================================================
st.markdown('<h1 class="main-header">🛵 HelmetNet </h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI Helmet Detection System</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Real-Time Detection"])

# --- TAB 1: IMAGE DETECTION ---
with tab1:
    st.markdown("### 📸 Upload an Image")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-box"><strong>💡 Tips:</strong><br>• Clear, well-lit images<br>• JPG, PNG, BMP</div>', unsafe_allow_html=True)
    
    with col1:
        img_file = st.file_uploader("Choose image", ["jpg", "jpeg", "png", "bmp"], key="img", label_visibility="collapsed")
    
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        with st.spinner("🔍 Analyzing..."):
            # Use logic from app_2 (draw_boxes)
            dets, stats = detect_frame(frame, model, confidence_threshold)
            annotated = draw_boxes(frame, dets)
            st.session_state.total_detections += len(dets)
            
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**📷 Original**")
            st.image(img_file, use_container_width=True)
        with c2:
            st.markdown("**🎯 Result**")
            st.image(annotated_rgb, use_container_width=True)
        
        # Alerts (UI from app.py)
        if stats['alert']:
            st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)
            
        st.markdown("### 📊 Summary")
        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 Helmets", stats['helmet_count'])
        m2.metric("🔴 No Helmets", stats['no_helmet_count'])
        m3.metric("📝 Total Objects", len(dets))
        
        # Download
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(temp_img.name, annotated)
        with open(temp_img.name, 'rb') as f:
            st.download_button("📥 Download Result", f, f"result_{img_file.name}", "image/jpeg")

# --- TAB 2: VIDEO DETECTION (OPTIMIZED LOOP) ---
with tab2:
    st.markdown("### 🎥 Upload a Video")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-box"><strong>💡 Fast Mode:</strong><br>• Optimized frame skipping<br>• Live inference preview<br>• MP4, AVI, MOV</div>', unsafe_allow_html=True)
    
    with col1:
        vid_file = st.file_uploader("Choose video", ["mp4", "avi", "mov", "mkv"], key="vid", label_visibility="collapsed")
    
    if vid_file:
        st.markdown("### 🎬 Processing")
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
            
            # Layout for live updates
            st_frame = st.empty()
            st_metrics = st.empty()
            st_progress = st.progress(0)
            
            frame_count = 0
            cached_detections = [] 
            current_stats = {'helmet_count': 0, 'no_helmet_count': 0, 'alert': False}
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                
                # --- OPTIMIZATION LOGIC (From app_2) ---
                if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                    cached_detections, current_stats = detect_frame(frame, model, confidence_threshold)
                
                annotated = draw_boxes(frame, cached_detections)
                out.write(annotated)
                
                if current_stats['alert']: play_alarm()
                
                # Update UI
                st_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), 
                             caption=f"Processing Frame {frame_count}/{total_frames}",
                             use_container_width=True)
                
                # Metric Container styling
                with st_metrics.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 Helmets", current_stats['helmet_count'])
                    c2.metric("🔴 Violations", current_stats['no_helmet_count'])
                    c3.metric("⏱️ Progress", f"{int(frame_count/total_frames*100)}%")
                
                st_progress.progress(frame_count / total_frames)
            
            cap.release()
            out.release()
            
            st.success("✅ Processing Complete!")
            st.session_state.total_detections += (current_stats['helmet_count'] + current_stats['no_helmet_count'])
            
            with open(outfile.name, 'rb') as f:
                st.download_button("📥 Download Result", f, "result.mp4", "video/mp4")

# --- TAB 3: REAL-TIME DETECTION (WEBRTC) ---
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")
    st.markdown("""
    <div class="info-box">
    <strong>🎥 Live Webcam:</strong><br>
    • Click "START" below<br>
    • Uses optimized frame skipping for smoother performance<br>
    • Works on mobile & desktop
    </div>
    """, unsafe_allow_html=True)
    
    # Webrtc Logic from app_2 wrapped in UI
    ctx = webrtc_streamer(
        key="helmet-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=HelmetTransformer,
        async_processing=True,
    )
    
    if ctx.video_processor:
        ctx.video_processor.set_model(model, confidence_threshold)
        
        st.markdown("### 📊 Live Stats")
        m1, m2 = st.columns(2)
        
        # These will update automatically via Streamlit re-runs triggered by WebRTC
        m1.metric("🟢 Helmets", ctx.video_processor.helmet)
        m2.metric("🔴 Violations", ctx.video_processor.no_helmet)
        
        if ctx.video_processor.alert:
            st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
            play_alarm()
        else:
            st.markdown('<div class="alert-success">✅ Area Secure</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("🚀 HelmetNet App | © 2025")





