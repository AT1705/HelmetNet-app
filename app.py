"""
AI Helmet Detection System (CSC738)
Enhanced with Real-Time Detection & Modern Safety Theme UI
"""

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import tempfile
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
# SAFETY THEME CSS
# ============================================================
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem !important; background: #F5F5F5;}
    .main-header {font-size: 2.5rem; font-weight: 800; color: #1E3A8A; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);}
    .sub-header {text-align: center; font-size: 1.1rem; color: #4B5563; font-weight: 500; margin-bottom: 1.5rem;}
    
    /* Tabs with Safety Yellow */
    .stTabs [data-baseweb="tab-list"] {background: white; padding: 0.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .stTabs [data-baseweb="tab"] {height: 50px; border-radius: 8px; color: #1E3A8A; font-weight: 600; padding: 0 1.5rem;}
    .stTabs [aria-selected="true"] {background: #FFD700 !important; color: #1E3A8A !important;}
    
    /* Alerts */
    .alert-danger {background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); color: white; padding: 20px; border-radius: 12px;
                    text-align: center; font-size: 1.3rem; font-weight: 700; animation: pulse 2s infinite; margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(239,68,68,0.3); border: 3px solid #FCA5A5;}
    .alert-success {background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); color: white; padding: 20px; border-radius: 12px;
                     text-align: center; font-size: 1.3rem; font-weight: 700; margin: 20px 0; box-shadow: 0 4px 6px rgba(34,197,94,0.3); border: 3px solid #86EFAC;}
    @keyframes pulse {0%, 100% {opacity: 1; transform: scale(1);} 50% {opacity: 0.85; transform: scale(1.02);}}
    
    /* Headers */
    h2 {color: #1E3A8A !important; font-weight: 700 !important; border-bottom: 3px solid #FFD700; padding-bottom: 0.5rem;}
    
    /* Buttons */
    .stButton > button {background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: #1E3A8A; border-radius: 10px;
                        padding: 0.6rem 2rem; font-weight: 700; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .stButton > button:hover {transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.15);}
    .stDownloadButton > button {background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); color: white;}
    
    /* Metrics */
    [data-testid="stMetricValue"] {font-size: 1.8rem !important; font-weight: 700 !important; color: #1E3A8A;}
    [data-testid="metric-container"] {background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #FFD700;}
    
    /* File uploader */
    [data-testid="stFileUploader"] {background: white; padding: 1.5rem; border-radius: 10px; border: 2px dashed #FFD700;}
    
    audio {display: none;}
    .info-box {background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #1E3A8A; margin: 1rem 0;}
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
        st.sidebar.warning("⚠️ Model not found")
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
        if 'alarm_counter' not in st.session_state:
            st.session_state.alarm_counter = 0
        st.session_state.alarm_counter += 1
        st.audio(alarm_audio, format="audio/mp3", autoplay=True)

# ============================================================
# DETECTION FUNCTIONS
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

def process_video(uploaded_file, model, conf_threshold, progress_placeholder):
    try:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_input.write(uploaded_file.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            st.error("❌ Failed to open video")
            return None, None

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_num = helmet_count = no_helmet_count = 0
        timestamps = []
        prog_bar = progress_placeholder.progress(0)
        status = progress_placeholder.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            annotated, results = detect_frame(frame, model, conf_threshold)
            out.write(annotated)

            helmet_count += results['helmet_count']
            no_helmet_count += results['no_helmet_count']
            if results['no_helmet_count'] > 0:
                timestamps.append(frame_num / fps)

            prog_bar.progress(frame_num / total_frames)
            status.text(f"Frame {frame_num}/{total_frames}")

        cap.release()
        out.release()

        return temp_output.name, {
            'total_frames': total_frames,
            'helmet_count': helmet_count,
            'no_helmet_count': no_helmet_count,
            'no_helmet_timestamps': timestamps,
            'fps': fps,
            'duration': total_frames / fps
        }
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None, None

# ============================================================
# REAL-TIME VIDEO TRANSFORMER
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
                
                # Update counts
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
            
            st.markdown("### 📊 Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🟢 Helmets", results['helmet_count'])
            m2.metric("🔴 No Helmets", results['no_helmet_count'])
            m3.metric("📝 Total", len(results['detections']))
            avg_conf = np.mean([d['confidence'] for d in results['detections']]) if results['detections'] else 0
            m4.metric("🎯 Confidence", f"{avg_conf:.1%}")
            
            if results['detections']:
                with st.expander("🔍 Details"):
                    for i, d in enumerate(results['detections'], 1):
                        c_a, c_b, c_c = st.columns([1, 2, 2])
                        c_a.markdown(f"**#{i}**")
                        emoji = "🔴" if d['class'] in NO_HELMET_LABELS else "🟢"
                        c_b.markdown(f"{emoji} `{d['class']}`")
                        c_c.markdown(f"**{d['confidence']:.1%}**")
            
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            cv2.imwrite(temp_img.name, annotated)
            with open(temp_img.name, 'rb') as f:
                st.download_button("📥 Download", f, f"result_{uploaded_image.name}", "image/jpeg")

# VIDEO TAB
with tab2:
    st.markdown("### 🎥 Upload a Video")
    
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-box"><strong>💡 Tips:</strong><br>• MP4, MOV, AVI<br>• Live progress<br>• Downloadable</div>', unsafe_allow_html=True)
    
    with col1:
        uploaded_video = st.file_uploader("Choose video", ["mp4", "mov", "avi", "mkv"], key="vid_up", label_visibility="collapsed")
    
    if uploaded_video:
        prog = st.empty()
        with st.spinner("🎬 Processing..."):
            output, summary = process_video(uploaded_video, model, confidence_threshold, prog)
        prog.empty()
        
        if output is not None and summary is not None:
            st.success("✅ Complete!")
            st.session_state.total_detections += summary['helmet_count'] + summary['no_helmet_count']
            
            st.markdown("### 🎬 Result")
            st.video(output)
            
            if summary['no_helmet_count'] > 0:
                st.markdown('<div class="alert-danger">⚠️ VIOLATIONS DETECTED!</div>', unsafe_allow_html=True)
                play_alarm()
            else:
                st.markdown('<div class="alert-success">✅ All Safe!</div>', unsafe_allow_html=True)
            
            st.markdown("### 📊 Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎞️ Frames", summary['total_frames'])
            c2.metric("🟢 Helmets", summary['helmet_count'])
            c3.metric("🔴 No Helmets", summary['no_helmet_count'])
            c4.metric("⏱️ Duration", str(timedelta(seconds=int(summary['duration']))))
            
            if summary['no_helmet_timestamps']:
                with st.expander(f"⚠️ Violations ({len(summary['no_helmet_timestamps'])})"):
                    unique = []
                    last = -999
                    for ts in summary['no_helmet_timestamps']:
                        if ts - last > 1:
                            unique.append(ts)
                            last = ts
                    for i, ts in enumerate(unique[:15], 1):
                        st.write(f"{i}. `{str(timedelta(seconds=int(ts)))}`")
                    if len(unique) > 15:
                        st.caption(f"...+{len(unique)-15} more")
            
            with open(output, 'rb') as f:
                st.download_button("📥 Download", f, "result.mp4", "video/mp4")

# REAL-TIME TAB
with tab3:
    st.markdown("### 📱 Real-Time Live Detection")
    
    st.markdown("""
    <div class="info-box">
    <strong>🎥 Live Webcam Detection:</strong><br>
    • Click "START" to enable your webcam<br>
    • Real-time bounding boxes appear automatically<br>
    • Click "STOP" to turn off the camera<br>
    • Works on desktop and mobile browsers<br><br>
    <em>⚠️ Note: Allow camera permission when prompted</em>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # WebRTC Configuration for better connectivity
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Initialize session state for WebRTC
    if 'webrtc_ctx' not in st.session_state:
        st.session_state.webrtc_ctx = None
    
    # Start/Stop buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    
    with col_btn1:
        start_webcam = st.button("▶️ START WEBCAM", use_container_width=True, type="primary")
    
    with col_btn2:
        stop_webcam = st.button("⏹️ STOP WEBCAM", use_container_width=True)
    
    st.markdown("---")
    
    # Webcam state management
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    if start_webcam:
        st.session_state.webcam_running = True
    
    if stop_webcam:
        st.session_state.webcam_running = False
        st.rerun()
    
    # Display webcam stream
    if st.session_state.webcam_running:
        st.markdown("### 📹 Live Detection Stream")
        
        # Create video transformer
        webrtc_ctx = webrtc_streamer(
            key="helmet-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=HelmetDetectionTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Set model after stream starts
        if webrtc_ctx.video_processor:
            webrtc_ctx.video_processor.set_model(model, confidence_threshold)
        
        # Live statistics
        st.markdown("### 📊 Live Detection Statistics")
        
        metric_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        # Update stats if stream is active
        if webrtc_ctx.video_processor:
            m1, m2, m3 = metric_placeholder.columns(3)
            
            with m1:
                st.metric("🟢 Helmets Detected", webrtc_ctx.video_processor.helmet_count)
            with m2:
                st.metric("🔴 No Helmets", webrtc_ctx.video_processor.no_helmet_count)
            with m3:
                st.metric("🎞️ Frames Processed", webrtc_ctx.video_processor.frame_count)
            
            # Alert display
            if webrtc_ctx.video_processor.alert:
                alert_placeholder.markdown(
                    '<div class="alert-danger">⚠️ WARNING: NO HELMET DETECTED!</div>',
                    unsafe_allow_html=True
                )
                if webrtc_ctx.video_processor.frame_count % 30 == 0:  # Play alarm every 30 frames
                    play_alarm()
            else:
                alert_placeholder.markdown(
                    '<div class="alert-success">✅ All Safe - Helmets Detected</div>',
                    unsafe_allow_html=True
                )
        
        st.info("💡 Camera is active. Click 'STOP WEBCAM' to turn it off.")
    
    else:
        st.markdown("### 🎥 Webcam Inactive")
        st.info("👆 Click 'START WEBCAM' button above to begin real-time detection")
        
        # Show placeholder
        st.markdown("""
        <div style="background: #E5E7EB; padding: 100px; border-radius: 15px; text-align: center; border: 3px dashed #FFD700;">
            <h2 style="color: #6B7280;">📷 Webcam Feed Will Appear Here</h2>
            <p style="color: #9CA3AF;">Press START to enable live detection with bounding boxes</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

st.caption("🚀 CSC738 | Helmet Safety Detection | © 2025")

