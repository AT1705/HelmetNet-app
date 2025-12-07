"""
AI Helmet Detection System (CSC738)
Fixed: Model Loading Issues & Live Video Inference
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
# SAFETY THEME CSS (ADAPTIVE)
# ============================================================
st.markdown("""
<style>
    /* Global Styles */
    .block-container {padding-top: 1.5rem !important;}
    
    /* Headers */
    .main-header {
        font-size: 2.5rem; font-weight: 800; color: var(--text-color); 
        text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center; font-size: 1.1rem; color: var(--text-color); 
        opacity: 0.8; font-weight: 500; margin-bottom: 1.5rem;
    }
    
    /* Tabs & Cards */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--secondary-background-color); 
        padding: 0.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 8px; color: var(--text-color); font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #FFD700 !important; color: #1E3A8A !important;
    }
    
    /* Alerts */
    .alert-danger {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
        color: white; padding: 20px; border-radius: 12px; text-align: center; 
        font-weight: 700; animation: pulse 2s infinite; margin: 20px 0;
    }
    .alert-success {
        background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); 
        color: white; padding: 20px; border-radius: 12px; text-align: center; 
        font-weight: 700; margin: 20px 0;
    }
    @keyframes pulse {0%, 100% {opacity: 1;} 50% {opacity: 0.9;}}
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: var(--secondary-background-color); 
        padding: 1rem; border-radius: 10px; border-left: 4px solid #FFD700;
    }
    
    /* Info Box */
    .info-box {
        background: rgba(59, 130, 246, 0.1); padding: 1rem; 
        border-radius: 10px; border-left: 4px solid #1E3A8A; 
        margin: 1rem 0; color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIG
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# LOAD MODEL & ALARM
# ============================================================
@st.cache_resource
def load_model(path_or_file):
    """
    Loads model from a path string or a file object
    """
    try:
        # If it's a temporary file path from uploader
        model = YOLO(path_or_file)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_alarm():
    if Path("alert.mp3").exists():
        with open("alert.mp3", "rb") as f:
            return f.read()
    return None

alarm_audio = load_alarm()

def play_alarm():
    if alarm_audio:
        if 'last_alarm_time' not in st.session_state:
            st.session_state.last_alarm_time = 0
        current_time = time.time()
        if current_time - st.session_state.last_alarm_time > 3: 
            st.audio(alarm_audio, format="audio/mp3", autoplay=True)
            st.session_state.last_alarm_time = current_time

# ============================================================
# DETECTION FUNCTION
# ============================================================
def detect_frame(frame, model, conf_threshold):
    results = model.predict(frame, conf=conf_threshold, verbose=False, device='cpu')
    annotated = results[0].plot()
    
    helmet_count = no_helmet_count = 0
    detections = []
    
    for box in results[0].boxes:
        cls_index = int(box.cls)
        class_name = model.names[cls_index].lower()
        confidence = float(box.conf)
        
        detections.append({'class': class_name, 'confidence': confidence, 'bbox': box.xyxy[0].cpu().numpy().tolist()})
        
        if class_name in NO_HELMET_LABELS:
            no_helmet_count += 1
        else:
            helmet_count += 1
    
    return annotated, {
        'helmet_count': helmet_count,
        'no_helmet_count': no_helmet_count,
        'detections': detections,
        'alert': no_helmet_count > 0
    }

def process_image(uploaded_file, model, conf_threshold):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None: return None, None
        return detect_frame(frame, model, conf_threshold)
    except Exception as e:
        return None, None

# ============================================================
# REAL-TIME WEBRTC TRANSFORMER
# ============================================================
class HelmetDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.frame_count = 0
        self.alert = False
    
    def set_model(self, model, conf_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is None: return img
        self.frame_count += 1
        
        if self.frame_count % 2 == 0:
            try:
                annotated, res = detect_frame(img, self.model, self.conf_threshold)
                self.helmet_count = res['helmet_count']
                self.no_helmet_count = res['no_helmet_count']
                self.alert = res['alert']
                return annotated
            except: return img
        return img

# ============================================================
# SIDEBAR - MODEL UPLOADER (NEW)
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("**🤖 Model Selection**")
    
    # NEW: File Uploader for Model
    uploaded_model_file = st.file_uploader("Upload 'best.pt' File", type=['pt'], help="Upload your trained YOLO model here")
    
    # Fallback to text input if no file uploaded
    if uploaded_model_file:
        # Save uploaded model to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            f.write(uploaded_model_file.read())
            model_path_to_load = f.name
        st.success("Custom model uploaded!")
    else:
        model_path_to_load = st.text_input("Or enter path manually", "best.pt")
        st.caption(f"Looking in: `{os.getcwd()}`")
        
    confidence_threshold = st.slider("🎯 Confidence", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)
    
    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    if 'total_detections' not in st.session_state:
        st.session_state.total_detections = 0
    st.metric("Total Detections", st.session_state.total_detections)
    
    st.markdown("---")
    st.caption("🚀 CSC738 Project")

# ============================================================
# MAIN APP
# ============================================================
st.markdown('<h1 class="main-header">🛵 AI Helmet Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🎯 Advanced AI-Powered Safety Monitoring</p>', unsafe_allow_html=True)

# LOAD MODEL
model = load_model(model_path_to_load)

# CHECK IF MODEL LOADED SUCCESSFULLY
if model is None:
    st.error("❌ Failed to load model!")
    with st.expander("🔎 Troubleshooting Tips (Open Me)"):
        st.markdown("""
        **Why is this happening?**
        1. The file `best.pt` might not be in the same folder as this script.
        2. The file path might be incorrect.
        
        **How to fix it:**
        - **Option 1 (Recommended):** Use the **"Upload 'best.pt' File"** button in the Sidebar (left) to upload your model directly.
        - **Option 2:** Ensure `best.pt` is inside this folder: 
        """)
        st.code(os.getcwd())
    st.stop() # Stop execution here if no model

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📱 Real-Time Detection"])

# IMAGE TAB
with tab1:
    st.markdown("### 📸 Upload an Image")
    col1, col2 = st.columns([2, 1])
    with col2: st.markdown('<div class="info-box"><strong>💡 Tips:</strong><br>• JPG, PNG<br>• Clear lighting</div>', unsafe_allow_html=True)
    with col1: uploaded_image = st.file_uploader("Choose image", ["jpg", "png"], key="img", label_visibility="collapsed")
    
    if uploaded_image:
        with st.spinner("Analyzing..."):
            annotated, results = process_image(uploaded_image, model, confidence_threshold)
            
        if annotated is not None:
            st.session_state.total_detections += len(results['detections'])
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2)
            c1.image(uploaded_image, caption="Original", use_container_width=True)
            c2.image(annotated_rgb, caption="Result", use_container_width=True)
            
            if results['alert']: 
                st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                play_alarm()
            else:
                st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)
                
            m1, m2 = st.columns(2)
            m1.metric("🟢 Helmets", results['helmet_count'])
            m2.metric("🔴 Violations", results['no_helmet_count'])

# VIDEO TAB (LIVE INFERENCE)
with tab2:
    st.markdown("### 🎥 Video Live Inference")
    col1, col2 = st.columns([2, 1])
    with col2: st.markdown('<div class="info-box"><strong>💡 Info:</strong><br>• Real-time View<br>• Auto-save</div>', unsafe_allow_html=True)
    with col1: uploaded_video = st.file_uploader("Choose video", ["mp4", "mov", "avi"], key="vid", label_visibility="collapsed")
    
    if uploaded_video:
        if st.button("▶️ Start Processing", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            
            # Setup output
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            out = cv2.VideoWriter(outfile.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            # UI Placeholders
            vid_place = st.empty()
            metric_place = st.empty()
            prog_bar = st.progress(0)
            
            frame_cnt = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_cnt += 1
                annotated, res = detect_frame(frame, model, confidence_threshold)
                out.write(annotated)
                
                if res['alert']: play_alarm()
                
                # Update UI
                vid_place.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                with metric_place.container():
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🟢 Helmets", res['helmet_count'])
                    c2.metric("🔴 Violations", res['no_helmet_count'])
                    c3.metric("⏱️ Progress", f"{int(frame_cnt/total_frames*100)}%")
                
                prog_bar.progress(frame_cnt / total_frames)
            
            cap.release()
            out.release()
            st.success("✅ Done!")
            with open(outfile.name, 'rb') as f:
                st.download_button("📥 Download Result", f, "processed.mp4", "video/mp4")

# REAL-TIME TAB
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    col1, col2 = st.columns([1, 4])
    with col1:
        start = st.button("▶️ Start Camera", type="primary")
        stop = st.button("⏹️ Stop Camera")
    
    if 'cam_run' not in st.session_state: st.session_state.cam_run = False
    if start: st.session_state.cam_run = True
    if stop: st.session_state.cam_run = False; st.rerun()
    
    if st.session_state.cam_run:
        ctx = webrtc_streamer(
            key="helmet-live", 
            mode=WebRtcMode.SENDRECV, 
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=HelmetDetectionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
        if ctx.video_processor:
            ctx.video_processor.set_model(model, confidence_threshold)
            c1, c2 = st.columns(2)
            c1.metric("🟢 Helmets", ctx.video_processor.helmet_count)
            c2.metric("🔴 Violations", ctx.video_processor.no_helmet_count)
            if ctx.video_processor.alert and ctx.video_processor.frame_count % 30 == 0:
                play_alarm()
    else:
        st.info("Click 'Start Camera' to begin.")
