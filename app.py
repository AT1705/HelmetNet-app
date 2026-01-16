"""
HelmetNet - AI-Powered Helmet Detection System
Main Streamlit Application
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HelmetNet - AI-Powered Helmet Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Navigation Bar */
    .nav-container {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid #e2e8f0;
        padding: 1rem 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .nav-content {
        max-width: 1280px;
        margin: 0 auto;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        cursor: pointer;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .nav-link {
        color: #64748b;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s;
        cursor: pointer;
    }
    
    .nav-link:hover {
        color: #0f172a;
    }
    
    .nav-link.active {
        color: #0f172a;
        font-weight: 600;
    }
    
    .nav-button {
        background: #f59e0b;
        color: #0f172a;
        padding: 0.625rem 1.5rem;
        border-radius: 0.75rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    
    .nav-button:hover {
        background: #fbbf24;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 8rem 2rem 6rem;
        margin-top: 4rem;
        border-radius: 0;
    }
    
    .hero-content {
        max-width: 1280px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3.75rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 1.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #cbd5e1;
        margin-bottom: 2rem;
        max-width: 48rem;
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1.5rem;
    }
    
    /* Buttons */
    .btn-primary {
        background: #f59e0b;
        color: #0f172a;
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        font-weight: 600;
        border: none;
        cursor: pointer;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
        display: inline-block;
        text-decoration: none;
        margin-right: 1rem;
    }
    
    .btn-primary:hover {
        background: #fbbf24;
        transform: scale(1.05);
    }
    
    .btn-secondary {
        background: transparent;
        color: white;
        padding: 1rem 2rem;
        border-radius: 0.75rem;
        font-weight: 600;
        border: 2px solid white;
        cursor: pointer;
        transition: all 0.3s;
        display: inline-block;
        text-decoration: none;
    }
    
    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 0.75rem;
        padding: 2rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s;
    }
    
    .card:hover {
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }
    
    /* Stats */
    .stat-container {
        background: white;
        padding: 3rem 2rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #64748b;
        font-size: 1rem;
    }
    
    /* Section Titles */
    .section-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .section-subtitle {
        font-size: 1.25rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Feature Cards */
    .feature-icon {
        width: 3rem;
        height: 3rem;
        background: #f1f5f9;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        color: #475569;
        font-size: 1.5rem;
    }
    
    /* Experiment Cards */
    .experiment-badge {
        width: 3rem;
        height: 3rem;
        background: linear-gradient(135deg, #475569 0%, #1e293b 100%);
        color: white;
        border-radius: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.25rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Detection Dashboard */
    .config-panel {
        background: white;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    .detection-tabs {
        background: white;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        padding: 0.375rem;
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .tab-button {
        flex: 1;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: none;
        background: transparent;
        color: #64748b;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .tab-button.active {
        background: #f59e0b;
        color: #0f172a;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Upload Area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 0.75rem;
        padding: 3rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #f59e0b;
        background: #fffbf0;
    }
    
    /* Detection Table */
    .detection-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }
    
    .detection-table th {
        background: #f8fafc;
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        color: #475569;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .detection-table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .detection-table tr:hover {
        background: #f8fafc;
    }
    
    /* Spacing */
    .section-spacing {
        padding: 4rem 2rem;
    }
    
    /* Container */
    .container {
        max-width: 1280px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    /* Streamlit specific overrides */
    .stButton > button {
        background: #f59e0b;
        color: #0f172a;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: #fbbf24;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide default padding */
    .block-container {
        padding-top: 0;
        padding-bottom: 0;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation
def render_navigation():
    st.markdown("""
    <div class="nav-container">
        <div class="nav-content">
            <div class="nav-logo">
                🛡️ <span>HelmetNet</span>
            </div>
            <div class="nav-links">
                <span class="nav-link" id="nav-home">Home</span>
                <span class="nav-link" id="nav-about">About</span>
                <button class="nav-button" id="nav-demo">Start Demo</button>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Load custom CSS
load_css()

# Render navigation
render_navigation()


if __name__ == "__main__":
    main()
