import streamlit as st

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

st.set_page_config(
    page_title="MediPredict AI - Home",
    page_icon="https://img.icons8.com/color/48/heart-monitor.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os

# Ensure the root directory is in python path for page imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import render_home_page, setup_page_layout

# Run the home page (must be at module level — Streamlit pages are NOT run as __main__)
setup_page_layout()
render_home_page()
