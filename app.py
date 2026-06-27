import streamlit as st
import os
import pandas as pd
from utils import load_model_assets

# Ensure Session State variables are initialized
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# Helper: Load CSS file
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Centralized Page Setup
def setup_page_layout():
    """Initializes styling, sidebar navigation widgets, and applies theme overrides."""
    # Ensure Session State variables are initialized for every page session
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    if 'dark_mode' not in st.session_state:
        st.session_state['dark_mode'] = False

    # Load default styles
    local_css(os.path.join("assets", "styles.css"))
    
    # Sidebar Logo and Title
    st.sidebar.image(os.path.join("assets", "heart.png"), width=80)
    st.sidebar.title("MediPredict AI")
    st.sidebar.markdown("### AI-Powered Screening")
    
    # Theme switch toggle (using toggle switch for modern sidebar styling)
    dark_mode = st.sidebar.toggle("Dark Mode", value=st.session_state.get('dark_mode', False))
    st.session_state['dark_mode'] = dark_mode
    
    if dark_mode:
        st.markdown("""
            <style>
                /* Force Black Dark Theme */
                .stApp {
                    --bg-light: #000000;
                    --bg-card-light: #121212;
                    --text-primary: #ffffff;
                    --text-secondary: #a0a0a0;
                    --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.5);
                    --shadow-medium: 0 8px 30px rgba(0, 0, 0, 0.7);
                    --border-color: #2c2c2c;
                    --bg-simulator: #1a2a44;
                    --bg-sidebar: #0f0f0f;
                    --border-sidebar: #2c2c2c;
                    --text-nav: #a0aec0;
                    --bg-nav-hover: rgba(11, 87, 208, 0.15);
                    --text-nav-hover: #ffffff;
                    --primary-color: #ffffff;
                    background-color: #000000 !important;
                    color: #ffffff !important;
                }
                /* Hide Streamlit top white header bar in dark mode */
                header[data-testid="stHeader"], 
                div[data-testid="stHeader"] {
                    background-color: transparent !important;
                    background: transparent !important;
                }
                /* Tabs styling in dark mode */
                button[data-baseweb="tab"] {
                    color: #a0a0a0 !important;
                    background-color: transparent !important;
                }
                button[data-baseweb="tab"][aria-selected="true"] {
                    color: #ffffff !important;
                    border-bottom-color: #0b57d0 !important;
                }
                /* Input widgets adaptivity in dark mode */
                div[data-baseweb="select"] > div,
                div[data-baseweb="popover"] ul,
                div[data-baseweb="popover"] li,
                div[data-baseweb="popover"] span,
                div[data-baseweb="popover"] div {
                    background-color: #121212 !important;
                    color: #ffffff !important;
                }
                div[data-baseweb="popover"] li:hover {
                    background-color: #0b57d0 !important;
                    color: #ffffff !important;
                }
                input {
                    background-color: #121212 !important;
                    color: #ffffff !important;
                    border: 1px solid #2c2c2c !important;
                }
                .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                    color: #ffffff !important;
                }
                .workflow-step {
                    background-color: #121212 !important;
                    color: #ffffff !important;
                    border: 1px solid #2c2c2c !important;
                }
                /* Tables and Dataframes in dark mode */
                div[data-testid="stTable"] table {
                    color: #ffffff !important;
                    background-color: #121212 !important;
                    border: 1px solid #2c2c2c !important;
                }
                div[data-testid="stTable"] th {
                    color: #ffffff !important;
                    background-color: #1a1a1a !important;
                    border-bottom: 2px solid #2c2c2c !important;
                }
                div[data-testid="stTable"] td {
                    color: #ffffff !important;
                    background-color: #121212 !important;
                    border-bottom: 1px solid #2c2c2c !important;
                }
                .stDataFrame div {
                    color: #ffffff !important;
                }
                /* Streamlit default alert boxes text visibility */
                div[data-testid="stNotification"] * {
                    color: #ffffff !important;
                }
                /* About Page Hero developer card adaptivity */
                div[data-element-id="about_hero_card"],
                div[class*="st-key-about_hero_card"] {
                    background: linear-gradient(135deg, #121212 0%, #1e1e1e 100%) !important;
                    color: #ffffff !important;
                }
                div[data-element-id="about_hero_card"] *,
                div[class*="st-key-about_hero_card"] * {
                    color: #ffffff !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
                /* Force Clean Light Theme */
                .stApp {
                    --bg-light: #f8fafd;
                    --bg-card-light: #ffffff;
                    --text-primary: #1f1f1f;
                    --text-secondary: #5f6368;
                    --shadow-soft: 0 4px 20px rgba(0, 0, 0, 0.05);
                    --shadow-medium: 0 8px 30px rgba(0, 0, 0, 0.08);
                    --border-color: rgba(0, 0, 0, 0.05);
                    --bg-simulator: #f0f7ff;
                    --bg-sidebar: #ffffff;
                    --border-sidebar: #e2e8f0;
                    --text-nav: #4a5568;
                    --bg-nav-hover: rgba(11, 87, 208, 0.05);
                    --text-nav-hover: #0b57d0;
                    --primary-color: #0b57d0;
                    background-color: #f8fafd !important;
                    color: #1f1f1f !important;
                }
                /* Hide Streamlit top header bar border in light mode */
                header[data-testid="stHeader"], 
                div[data-testid="stHeader"] {
                    background-color: transparent !important;
                    background: transparent !important;
                }
                /* Input widgets adaptivity in light mode */
                div[data-baseweb="select"] > div,
                div[data-baseweb="popover"] ul,
                div[data-baseweb="popover"] li {
                    background-color: #ffffff !important;
                    color: #1f1f1f !important;
                }
                input {
                    background-color: #ffffff !important;
                    color: #1f1f1f !important;
                    border: 1px solid rgba(0, 0, 0, 0.05) !important;
                }
                .stMarkdown p, .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                    color: #1f1f1f !important;
                }
                .workflow-step {
                    background-color: #ffffff !important;
                    color: #1f1f1f !important;
                    border: 1px solid rgba(0, 0, 0, 0.05) !important;
                }
                /* About Page Hero developer card adaptivity */
                div[data-element-id="about_hero_card"],
                div[class*="st-key-about_hero_card"] {
                    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
                    color: #1f1f1f !important;
                }
            </style>
        """, unsafe_allow_html=True)
        
    return dark_mode

# Footer rendering helper
def render_footer():
    current_time = pd.Timestamp.now().strftime("%B %d, %Y - %H:%M")
    st.markdown(f"""
        <div class="footer-text">
            MediPredict AI • Cardiovascular Diagnosis Support System<br>
            Developed by Dharanidharan T (Sri Eshwar College of Engineering)<br>
            CodeAlpha Machine Learning Internship Project • Last Refreshed: {current_time}
        </div>
    """, unsafe_allow_html=True)

# Home page rendering function
def render_home_page():
    # Hero Section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 style='color: var(--primary-color); font-size: 3rem; margin-top:0;'>MediPredict AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #5f6368; margin-top:0;'>AI-Powered Heart Disease Prediction System</h3>", unsafe_allow_html=True)
        st.markdown("""
        Welcome to **MediPredict AI**, a diagnostic decision support tool designed to aid clinicians 
        and individuals in early screening and detection of cardiovascular risk factors. 
        Using advanced machine learning, we process multi-modal health attributes to analyze patient heart health.
        """)
    with col2:
        doc_path = os.path.join("assets", "doctor.png")
        if os.path.exists(doc_path):
            st.image(doc_path, width=220)

    st.write("---")

    # Statistics Cards
    st.markdown("### Key System Statistics")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown("""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Total Features</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">13 Attributes</p>
            </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">ML Algorithm</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">Random Forest</p>
            </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown("""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Prediction Accuracy</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--accent-green); margin: 10px 0 0 0;">80.20%</p>
            </div>
        """, unsafe_allow_html=True)
    with sc4:
        st.markdown("""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Disease Classes</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--accent-red); margin: 10px 0 0 0;">2 Categories</p>
            </div>
        """, unsafe_allow_html=True)

    # Overview Section
    st.markdown("### Project Overview")
    st.write("""
    Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.9 million lives each year. 
    Early identification of high-risk patients can prevent adverse events and improve outcomes. 
    
    **MediPredict AI** integrates a Random Forest Classifier trained on clinical patient data (the Cleveland Heart Disease Dataset) 
    to automatically evaluate and predict heart health. By entering patient demographic, clinical, and stress test parameters, 
    the system calculates a quantitative risk score and generates actionable recommendations based on standard clinical guidelines.
    """)

    # Workflow Section
    st.markdown("### Machine Learning Workflow Pipeline")
    st.markdown("""
        <div class="timeline-wrapper">
            <div class="timeline-steps">
                <div class="timeline-item">
                    <div class="timeline-node">1</div>
                    <div class="timeline-info">
                        <div class="timeline-title">Patient Data</div>
                        <div class="timeline-desc">Demographics, clinical vitals & blood sugar</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-node">2</div>
                    <div class="timeline-info">
                        <div class="timeline-title">Preprocessing</div>
                        <div class="timeline-desc">Auto-scaling & normalization</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-node">3</div>
                    <div class="timeline-info">
                        <div class="timeline-title">Random Forest</div>
                        <div class="timeline-desc">Ensemble of 200 decision trees</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-node">4</div>
                    <div class="timeline-info">
                        <div class="timeline-title">AI Inference</div>
                        <div class="timeline-desc">Binary classification & risk probability</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-node">5</div>
                    <div class="timeline-info">
                        <div class="timeline-title">Analysis</div>
                        <div class="timeline-desc">Risk profile index stratification</div>
                    </div>
                </div>
                <div class="timeline-item">
                    <div class="timeline-node timeline-active">6</div>
                    <div class="timeline-info">
                        <div class="timeline-title">Action Plan</div>
                        <div class="timeline-desc">Clinical lifestyle & medical guidelines</div>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Date/Time rendering in footer
    render_footer()

if __name__ == "__main__":
    st.set_page_config(
        page_title="MediPredict AI - Heart Disease Prediction",
        page_icon="https://img.icons8.com/color/48/heart-monitor.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    setup_page_layout()
    render_home_page()
