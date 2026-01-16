"""
About Page for HelmetNet Streamlit App
"""

import streamlit as st

def render():
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">About HelmetNet</h1>
            <p class="hero-subtitle">
                A Computer Vision pipeline designed to detect helmet compliance for motorcycle riders 
                through iterative research and development.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About HelmetNet Section
    st.markdown('<div class="section-spacing" style="background: white;">', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);">
        <h2 style="font-size: 2.5rem; font-weight: 700; color: #0f172a; margin-bottom: 2rem;">
            About HelmetNet
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card" style="height: 100%; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);">
            <p style="font-size: 1.125rem; color: #374151; line-height: 1.8; margin-bottom: 1.5rem;">
                HelmetNet is a Computer Vision pipeline designed to detect helmet compliance for motorcycle riders. 
                This portal demonstrates how model quality evolves through iterative dataset labeling, class definitions, 
                and annotation discipline.
            </p>
            <p style="font-size: 1.125rem; color: #374151; line-height: 1.8; margin-bottom: 1.5rem;">
                The system supports inference across <strong style="color: #1e293b;">Images</strong>, 
                <strong style="color: #1e293b;">Videos</strong>, and 
                <strong style="color: #1e293b;">Real-time streams</strong>.
            </p>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #f59e0b;">
                <p style="color: #374151; line-height: 1.8;">
                    The redesign you are viewing emphasizes a professional "government portal" experience: 
                    guided navigation, clean configuration, and compliance-oriented insights.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1569932353341-b518d82f8a54?w=600&h=400&fit=crop", 
                 use_container_width=True)
    
    # Compliance Orientation
    st.markdown('<div style="margin-top: 3rem;">', unsafe_allow_html=True)
    st.markdown("""
    <h3 style="font-size: 1.875rem; font-weight: 600; color: #0f172a; margin-bottom: 1.5rem;">
        Compliance Orientation
    </h3>
    <p style="color: #374151; line-height: 1.8; margin-bottom: 1.5rem;">
        HelmetNet is positioned as a compliance-support tool rather than a purely technical demo. 
        The portal integrates prescriptive guidance referencing:
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #475569;">
            <h4 style="font-weight: 600; font-size: 1.125rem; margin-bottom: 0.5rem; color: #1e293b;">
                Section 119(2) Road Transport Act 1987
            </h4>
            <p style="color: #64748b;">Non-compliance signaling and legal framework for helmet enforcement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #475569;">
            <h4 style="font-weight: 600; font-size: 1.125rem; margin-bottom: 0.5rem; color: #1e293b;">
                SIRIM MS 1:2011
            </h4>
            <p style="color: #64748b;">Helmet compliance standard reference for safety certification</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown('<div style="margin-top: 3rem;">', unsafe_allow_html=True)
    st.markdown("""
    <h3 style="font-size: 1.875rem; font-weight: 600; color: #0f172a; margin-bottom: 1.5rem;">
        Technology Stack
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0;">
            <div style="font-size: 2.5rem; color: #475569; margin-bottom: 0.75rem;">🧠</div>
            <h4 style="font-weight: 600; margin-bottom: 0.5rem; color: #0f172a;">Deep Learning</h4>
            <p style="font-size: 0.875rem; color: #64748b;">
                CNN-based models trained for helmet detection with 99.2% accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0;">
            <div style="font-size: 2.5rem; color: #475569; margin-bottom: 0.75rem;">🛡️</div>
            <h4 style="font-weight: 600; margin-bottom: 0.5rem; color: #0f172a;">Computer Vision</h4>
            <p style="font-size: 0.875rem; color: #64748b;">
                Multi-angle detection with real-time image processing capabilities
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0;">
            <div style="font-size: 2.5rem; color: #475569; margin-bottom: 0.75rem;">👥</div>
            <h4 style="font-weight: 600; margin-bottom: 0.5rem; color: #0f172a;">Edge Computing</h4>
            <p style="font-size: 0.875rem; color: #64748b;">
                Sub-second detection time with local processing for privacy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div></div></div>', unsafe_allow_html=True)
    
    # The 4 Experiments
    st.markdown('<div class="section-spacing" style="background: #f8fafc;">', unsafe_allow_html=True)
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    st.markdown("""
    <h2 class="section-title">The 4 Experiments</h2>
    <p class="section-subtitle">
        Research progression addressing concrete failure modes and refining the dataset
    </p>
    """, unsafe_allow_html=True)
    
    experiments = [
        {
            "number": 1,
            "title": "Poor cap detection (baseline limitations)",
            "issue": "The model frequently confused caps / head coverings with helmets, producing poor discrimination.",
            "learning": "Model performance is bottlenecked by labeling policy quality more than raw architecture."
        },
        {
            "number": 2,
            "title": "Helmet type classification refinement",
            "issue": "System struggled to distinguish between different helmet types and safety standards.",
            "learning": "Dataset diversity across helmet types and viewing angles is critical for robust detection."
        },
        {
            "number": 3,
            "title": "Real-time stream optimization",
            "issue": "Processing latency exceeded acceptable thresholds for live traffic monitoring.",
            "learning": "Architecture optimization and edge computing integration essential for real-time applications."
        },
        {
            "number": 4,
            "title": "Multi-angle detection enhancement",
            "issue": "Detection accuracy dropped significantly for side and rear viewing angles.",
            "learning": "Comprehensive multi-angle dataset coverage ensures consistent performance across deployment scenarios."
        }
    ]
    
    col1, col2 = st.columns(2)
    
    for idx, exp in enumerate(experiments):
        with col1 if idx % 2 == 0 else col2:
            st.markdown(f"""
            <div class="card" style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1rem;">
                    <div class="experiment-badge">E{exp['number']}</div>
                    <h3 style="font-size: 1.125rem; font-weight: 600; color: #0f172a; padding-top: 0.5rem;">
                        {exp['title']}
                    </h3>
                </div>
                <div style="font-size: 0.875rem;">
                    <div style="margin-bottom: 0.75rem;">
                        <span style="font-weight: 600; color: #1e293b;">Issue:</span>
                        <p style="color: #64748b; margin-top: 0.25rem; margin-bottom: 0;">{exp['issue']}</p>
                    </div>
                    <div>
                        <span style="font-weight: 600; color: #1e293b;">Learning:</span>
                        <p style="color: #374151; margin-top: 0.25rem; margin-bottom: 0;">{exp['learning']}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div></div>', unsafe_allow_html=True)
