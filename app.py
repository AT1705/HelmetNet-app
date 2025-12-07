"""
AI Helmet Detection System (CSC738)
Enhanced with Real-Time Detection & Modern Safety Theme UI (Dark/Light Mode Compatible)
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
import time
from datetime import timedelta
from PIL import Image
import av
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
# CONFIG
# ============================================================
NO_HELMET_LABELS = ["no helmet", "no_helmet", "no-helmet"]
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# LOAD MODEL & ALARM
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        if Path(path).exists():
            model = YOLO(path)
            st.sidebar.success("✅ Model loaded")
            return model
        st.sidebar.warning("⚠️ Model not found, using yolov8n")
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
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
        # Use a timestamp to prevent spamming audio every millisecond
        if 'last_alarm_time' not in st.session_state:
            st.session_state.last_alarm_time = 0
        
        current_time = time.time()
        if current_time - st.session_state.last_alarm_time > 3: # Play max once every 3 seconds
            st.audio(alarm_audio, format="audio/mp3", autoplay=True)
            st.session_state.last_alarm_time = current_time

# ============================================================
# DETECTION FUNCTION (SINGLE FRAME)
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
        if frame is None:
            st.error("❌ Invalid image")
            return None, None
        return detect_frame(frame, model, conf_threshold)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None

# ============================================================
# REAL-TIME VIDEO TRANSFORMER (WEBRTC)
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
        
        if self.model is None:
            return img
        
        self.frame_count += 1
        
        # Process every 2nd frame for performance
        if self.frame_count % 2 == 0:
            try:
                results = self.model.predict(img, conf=self.conf_threshold, verbose=False, device='cpu')
                annotated = results[0].plot()
                
                # Count detections
                helmet_count = 0
                no_helmet_count = 0
                
                for box in results[0].boxes:
                    cls_index = int(box.cls)
                    class_name = self.model.names[cls_index].lower()
                    
                    if class_name in NO_HELMET_LABELS:
                        no_helmet_count += 1
                    else:
                        helmet_count += 1
                
                self.helmet_count = helmet_count
                self.no_helmet_count = no_helmet_count
                self.alert = no_helmet_count > 0
                
                return annotated
            except Exception as e:
                return img
        
        return img

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    st.markdown("**🤖 Model Settings**")
    model_path_input = st.text_input("Model Path", MODEL_PATH, label_visibility="collapsed")
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

model = load_model(model_path_input)
if model is None:
    st.error("❌ Failed to load model")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📸 Image Detection", "🎥 Video Detection", "📱 Real-Time Detection"])

# IMAGE TAB
with tab1:
    st.markdown("### 📸 Upload an Image")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-box"><strong>💡 Tips:</strong><br>• Clear, well-lit images<br>• JPG, PNG, BMP<br>• Max 200MB</div>', unsafe_allow_html=True)
    
    with col1:
        uploaded_image = st.file_uploader("Choose image", ["jpg", "jpeg", "png", "bmp"], key="img_up", label_visibility="collapsed")
    
    if uploaded_image:
        with st.spinner("🔍 Analyzing..."):
            annotated, results = process_image(uploaded_image, model, confidence_threshold)
        
        if annotated is not None and results is not None:
            st.session_state.total_detections += len(results['detections'])
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown("**📷 Original**")
                st.image(uploaded_image, use_container_width=True)
            with c2:
                st.markdown("**🎯 Result**")
                st.image(annotated_rgb, use_container_width=True)
            
            if results['alert']:
                st.markdown('<div class="alert-danger">⚠️ NO HELMET DETECTED!</div>', unsafe_allow_html=True)
                play_alarm()
            else:
                st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)
            
            # Metrics
            st.markdown("### 📊 Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🟢 Helmets", results['helmet_count'])
            m2.metric("🔴 No Helmets", results['no_helmet_count'])
            m3.metric("📝 Total", len(results['detections']))
            avg_conf = np.mean([d['confidence'] for d in results['detections']]) if results['detections'] else 0
            m4.metric("🎯 Confidence", f"{avg_conf:.1%}")
            
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_img.name, annotated)
            with open(temp_img.name, 'rb') as f:
                st.download_button("📥 Download", f, f"result_{uploaded_image.name}", "image/jpeg")

# VIDEO TAB (UPDATED FOR LIVE INFERENCE)
with tab2:
    st.markdown("### 🎥 Upload a Video")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-box"><strong>💡 Live Inference:</strong><br>• See detections in real-time<br>• Auto-saves result<br>• Download when done</div>', unsafe_allow_html=True)
    
    with col1:
        uploaded_video = st.file_uploader("Choose video", ["mp4", "mov", "avi", "mkv"], key="vid_up", label_visibility="collapsed")
    
    if uploaded_video:
        # Create a button to start processing so it doesn't auto-start
        if st.button("▶️ Start Processing", type="primary"):
            try:
                # Save uploaded file to temp
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()

                cap = cv2.VideoCapture(tfile.name)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Prepare output writer for download later
                output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(output_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

                # Layout for Live View
                st.markdown("### 🎬 Live Inference View")
                
                # Placeholders for dynamic content
                video_placeholder = st.empty()
                metrics_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                frame_count = 0
                total_helmet = 0
                total_no_helmet = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    
                    # Detect
                    annotated, results = detect_frame(frame, model, confidence_threshold)
                    
                    # Write to file
                    out.write(annotated)
                    
                    # Update counts
                    total_helmet += results['helmet_count']
                    total_no_helmet += results['no_helmet_count']
                    
                    # Alert Logic (Live)
                    if results['alert']:
                        play_alarm()
                    
                    # Convert for Display (BGR -> RGB)
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    
                    # Update UI - The 'Live' part
                    video_placeholder.image(frame_rgb, caption=f"Processing Frame {frame_count}/{total_frames}", use_container_width=True)
                    
                    # Update Live Metrics in columns
                    with metrics_placeholder.container():
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🟢 Current Helmets", results['helmet_count'])
                        c2.metric("🔴 Current Violations", results['no_helmet_count'])
                        c3.metric("⏱️ Progress", f"{int((frame_count/total_frames)*100)}%")
                    
                    progress_bar.progress(frame_count / total_frames)
                
                cap.release()
                out.release()
                
                st.success("✅ Processing Complete!")
                
                # Download Button
                with open(output_file.name, 'rb') as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )
                    
            except Exception as e:
                st.error(f"Error processing video: {e}")

# REAL-TIME TAB
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")
    
    st.markdown("""
    <div class="info-box">
    <strong>🎥 Live Webcam Detection:</strong><br>
    • Click "START" to enable your webcam<br>
    • Real-time bounding boxes appear automatically<br>
    • Click "STOP" to turn off the camera<br>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    if 'webrtc_ctx' not in st.session_state:
        st.session_state.webrtc_ctx = None
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    
    with col_btn1:
        start_webcam = st.button("▶️ START WEBCAM", use_container_width=True, type="primary")
    with col_btn2:
        stop_webcam = st.button("⏹️ STOP WEBCAM", use_container_width=True)
    
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_webcam:
        st.session_state.webcam_running = True
    if stop_webcam:
        st.session_state.webcam_running = False
        st.rerun()
    
    if st.session_state.webcam_running:
        st.markdown("### 📹 Live Detection Stream")
        
        webrtc_ctx = webrtc_streamer(
            key="helmet-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=HelmetDetectionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_model(model, confidence_threshold)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🟢 Helmets", webrtc_ctx.video_processor.helmet_count)
            with m2:
                st.metric("🔴 No Helmets", webrtc_ctx.video_processor.no_helmet_count)
            with m3:
                st.metric("🎞️ Frames", webrtc_ctx.video_processor.frame_count)
            
            if webrtc_ctx.video_processor.alert:
                if webrtc_ctx.video_processor.frame_count % 30 == 0:
                    play_alarm()
    
    else:
        st.markdown("### 🎥 Webcam Inactive")
        st.markdown("""
        <div style="background: var(--secondary-background-color); padding: 100px; border-radius: 15px; text-align: center; border: 3px dashed #FFD700;">
            <h2 style="color: var(--text-color); opacity: 0.5;">📷 Webcam Feed Will Appear Here</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🚀 CSC738 | Helmet Safety Detection | © 2025")
