"""
Demo Page for HelmetNet Streamlit App
"""

import streamlit as st
import time
from PIL import Image
import io

def render():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                border-bottom: 1px solid #475569; padding: 3rem 2rem; margin-top: 4rem;">
        <div class="container">
            <h1 style="font-size: 2.5rem; font-weight: 700; color: white; margin-bottom: 0.75rem;">
                HelmetNet Detection System
            </h1>
            <p style="color: #cbd5e1; font-size: 1.125rem;">
                AI-powered helmet compliance detection
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="section-spacing">', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Create layout
    col_sidebar, col_main = st.columns([1, 2.5])
    
    # Left Sidebar - Configuration
    with col_sidebar:
        st.markdown('<div class="config-panel">', unsafe_allow_html=True)
        
        st.markdown("""
        <h2 style="font-weight: 600; color: #0f172a; font-size: 1.125rem; margin-bottom: 1.5rem;">
            Configuration
        </h2>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style="font-size: 0.875rem; font-weight: 600; color: #475569; margin-bottom: 1rem;">
            Model Settings
        </h3>
        """, unsafe_allow_html=True)
        
        # Model selection
        model = st.selectbox(
            "Model Path",
            ["YOLOv8 v3.2 (Recommended)", "Faster R-CNN", "EfficientDet-D4"],
            key="model_select"
        )
        
        # Confidence threshold
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <label style="font-size: 0.875rem; color: #64748b; display: block; margin-bottom: 0.5rem;">
                Confidence Threshold
            </label>
        </div>
        """, unsafe_allow_html=True)
        
        confidence = st.slider("", 0, 100, 50, label_visibility="collapsed", key="confidence_slider")
        
        st.markdown(f"""
        <div style="text-align: right; margin-top: -0.5rem; margin-bottom: 1.5rem;">
            <span style="color: #d97706; font-weight: 700; font-size: 1rem;">{confidence}%</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Session Stats
        st.markdown('<div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0;">', unsafe_allow_html=True)
        st.markdown("""
        <h3 style="font-size: 0.875rem; font-weight: 600; color: #475569; margin-bottom: 1rem;">
            Session Stats
        </h3>
        """, unsafe_allow_html=True)
        
        # Initialize detection count in session state
        if 'detection_count' not in st.session_state:
            st.session_state.detection_count = 0
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem; font-size: 0.875rem;">
            <span style="color: #64748b;">Total Detections</span>
            <span style="color: #0f172a; font-weight: 600;">{st.session_state.detection_count}</span>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
            <span style="color: #64748b;">Model Status</span>
            <span style="color: #16a34a; font-weight: 600; display: flex; align-items: center; gap: 0.25rem;">
                <span style="width: 0.5rem; height: 0.5rem; background: #22c55e; border-radius: 50%; display: inline-block;"></span>
                Loaded
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Right Content Area
    with col_main:
        # Mode Tabs
        st.markdown('<div class="detection-tabs">', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📤 Image Detection", "🎥 Video Detection", "📡 Real Time Detection"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image Detection Tab
        with tab1:
            # Upload Section
            st.markdown("""
            <div class="card" style="margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1.5rem;">
                    <div>
                        <h3 style="font-size: 1.25rem; font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;">
                            Upload an Image
                        </h3>
                        <p style="font-size: 0.875rem; color: #64748b;">
                            Supported formats: JPG, PNG, BMP
                        </p>
                    </div>
                    <div style="background: #f8fafc; padding: 0.75rem 1rem; border-radius: 0.5rem;">
                        <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; margin-bottom: 0.25rem;">
                            Quick Tips
                        </div>
                        <div style="font-size: 0.75rem; color: #475569;">Clear, well-lit images</div>
                        <div style="font-size: 0.75rem; color: #475569;">Max size: 10MB</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                key="image_uploader",
                label_visibility="collapsed"
            )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Run Detection Button
            if uploaded_file is not None:
                run_detection = st.button("🔍 Run Detection", use_container_width=True, type="primary")
                
                # Display uploaded image
                st.markdown("""
                <div class="card" style="margin-top: 1.5rem; margin-bottom: 1.5rem;">
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #0f172a; margin-bottom: 1rem;">
                        Uploaded Image
                    </h3>
                """, unsafe_allow_html=True)
                
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Process detection
                if run_detection:
                    with st.spinner("🔄 Processing..."):
                        time.sleep(2)  # Simulate processing
                        st.session_state.detection_count += 1
                        st.session_state.detections_made = True
                        st.rerun()
            
            # Results Section
            st.markdown("""
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding-bottom: 1rem; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.5rem;">
                    <h3 style="font-size: 1.25rem; font-weight: 600; color: #0f172a;">Results</h3>
                    <span style="font-size: 0.875rem; color: #64748b; background: #f1f5f9; 
                                padding: 0.25rem 0.75rem; border-radius: 0.5rem;">
                        Model: HelmetNet
                    </span>
                </div>
                
                <h4 style="font-weight: 600; color: #0f172a; margin-bottom: 0.75rem; display: flex; 
                          align-items: center; gap: 0.5rem;">
                    Detections Table
                    <span style="font-size: 0.75rem; color: #64748b; font-weight: 400;">
                        Sorted by confidence
                    </span>
                </h4>
            """, unsafe_allow_html=True)
            
            # Display detection results if available
            if hasattr(st.session_state, 'detections_made') and st.session_state.detections_made:
                # Mock detection data
                detections = [
                    {"id": 1, "label": "Helmet", "confidence": 96.8, "compliance": "COMPLIANT", "bbox": "245, 120, 180, 160"},
                    {"id": 2, "label": "Motorcycle", "confidence": 98.5, "compliance": "N/A", "bbox": "150, 200, 400, 350"},
                    {"id": 3, "label": "Person", "confidence": 97.2, "compliance": "N/A", "bbox": "220, 100, 200, 380"}
                ]
                
                # Create table HTML
                table_html = """
                <div style="border: 1px solid #e2e8f0; border-radius: 0.75rem; overflow: hidden;">
                    <table class="detection-table">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>LABEL</th>
                                <th>CONFIDENCE</th>
                                <th>COMPLIANCE</th>
                                <th>BBOX (X,Y,W,H)</th>
                            </tr>
                        </thead>
                        <tbody>
                """
                
                for det in detections:
                    compliance_html = ""
                    if det['compliance'] == 'COMPLIANT':
                        compliance_html = f'<span style="color: #16a34a; font-weight: 500;">✓ {det["compliance"]}</span>'
                    else:
                        compliance_html = f'<span style="color: #64748b;">{det["compliance"]}</span>'
                    
                    table_html += f"""
                        <tr>
                            <td style="color: #64748b;">{det['id']}</td>
                            <td style="color: #0f172a; font-weight: 500;">{det['label']}</td>
                            <td style="color: #16a34a; font-weight: 600;">{det['confidence']}%</td>
                            <td>{compliance_html}</td>
                            <td style="color: #64748b; font-family: monospace; font-size: 0.75rem;">{det['bbox']}</td>
                        </tr>
                    """
                
                table_html += """
                        </tbody>
                    </table>
                </div>
                """
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                # Integration note
                st.markdown("""
                <div style="background: #fffbeb; border: 1px solid #fcd34d; padding: 1.25rem; 
                            border-radius: 0.75rem; margin-top: 1.5rem; font-size: 0.875rem; color: #475569;">
                    <p style="margin-bottom: 0.5rem;">
                        <strong style="color: #0f172a;">Integration approach:</strong> Replace mock generation 
                        with a backend endpoint returning 
                        <code style="background: white; border: 1px solid #fcd34d; padding: 0.125rem 0.5rem; 
                                    border-radius: 0.25rem; color: #1e293b;">
                            {'{ detections: [{ label, conf, x, y, w, h }] }'}
                        </code>.
                    </p>
                    <p style="color: #64748b; margin: 0;">
                        Tip: For your final demo, add "export report" (model version, threshold, timestamp, detections).
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-center; padding: 3rem; border: 1px solid #e2e8f0; 
                            border-radius: 0.75rem; background: #f8fafc;">
                    <p style="color: #64748b;">No results yet. Upload an image and click "Run detection".</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Video Detection Tab
        with tab2:
            st.markdown("""
            <div class="card" style="padding: 4rem; text-align: center;">
                <div style="font-size: 5rem; color: #cbd5e1; margin-bottom: 1rem;">🎥</div>
                <h3 style="font-size: 1.5rem; font-weight: 600; color: #0f172a; margin-bottom: 0.75rem;">
                    Video Detection
                </h3>
                <p style="color: #64748b; max-width: 28rem; margin: 0 auto 1.5rem;">
                    Upload a video file to process frame-by-frame helmet detection
                </p>
                <button class="btn-primary">Coming Soon</button>
            </div>
            """, unsafe_allow_html=True)
        
        # Real-time Detection Tab
        with tab3:
            st.markdown("""
            <div class="card" style="padding: 4rem; text-align: center;">
                <div style="font-size: 5rem; color: #cbd5e1; margin-bottom: 1rem;">📡</div>
                <h3 style="font-size: 1.5rem; font-weight: 600; color: #0f172a; margin-bottom: 0.75rem;">
                    Real Time Detection
                </h3>
                <p style="color: #64748b; max-width: 28rem; margin: 0 auto 1.5rem;">
                    Connect to a webcam or RTSP stream for live helmet detection
                </p>
                <button class="btn-primary">Coming Soon</button>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
