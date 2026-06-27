import streamlit as st

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

st.set_page_config(
    page_title="CardioMind AI - About",
    page_icon="https://img.icons8.com/color/48/heart-monitor.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import pandas as pd

# Ensure root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import setup_page_layout, render_footer

# Apply centralized layout and theme
setup_page_layout()

# Title
st.markdown("<h1 style='color: var(--primary-color); font-size: 2.2rem;'>About CardioMind AI Project</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #5f6368; margin-top: -10px;'>Learn more about the technology stack, dataset parameters, and developer.</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1.8, 1.2])

with col1:
    # Project Description Card
    with st.container(key="about_card_objectives", border=True):
        st.markdown("<h3 style='color: var(--primary-color); margin-top:0;'>Project Objectives & Context</h3>", unsafe_allow_html=True)
        st.write("""
        **CardioMind AI** is a professional cardiovascular screening application developed to support early detection 
        of heart disease. Identifying risk parameters is crucial for proactive patient management. 
        By leveraging a Random Forest model, the application serves as a diagnostic dashboard for clinic screenings.
        """)
        st.markdown("**Core Aims:**")
        st.markdown("- **Clinical Screening:** Offer a fast, automated assessment of a patient's risk based on standard clinical indicators.")
        st.markdown("- **Risk Stratification:** Classify patient status into precise risk categories (Low, Moderate, High) with corresponding clinical recommendations.")
        st.markdown("- **Interpretability:** Make the model's classifications transparent by displaying feature importances and validation metrics.")

    # Dataset Parameters Card
    with st.container(key="about_card_dataset", border=True):
        st.markdown("<h3 style='color: var(--primary-color); margin-top:0;'>Dataset Specifications</h3>", unsafe_allow_html=True)
        st.write("""
        The system is calibrated on the standard **UCI Cleveland Heart Disease Dataset**, which includes 303 patient cases (297 after removing missing records). 
        The 13 clinical parameters used in prediction are:
        """)
        
        # Grid of features
        df_features = pd.DataFrame([
            {"Feature": "Age", "Type": "Continuous", "Clinical Significance": "Primary factor in heart vascular degeneration."},
            {"Feature": "Sex (Gender)", "Type": "Binary", "Clinical Significance": "Males exhibit statistically higher early-onset risk rates."},
            {"Feature": "Chest Pain (CP)", "Type": "Categorical", "Clinical Significance": "Pain severity ranging from typical to asymptomatic."},
            {"Feature": "Resting BP", "Type": "Continuous", "Clinical Significance": "High resting blood pressure indicates cardiovascular stress."},
            {"Feature": "Cholesterol", "Type": "Continuous", "Clinical Significance": "High serum cholesterol leads to arterial blockages."},
            {"Feature": "Fasting Blood Sugar", "Type": "Binary", "Clinical Significance": "Indicator of insulin resistance or diabetic complications."},
            {"Feature": "Rest ECG", "Type": "Categorical", "Clinical Significance": "Electrical cardiac signals (normal, abnormal, hypertrophy)."},
            {"Feature": "Max Heart Rate", "Type": "Continuous", "Clinical Significance": "Maximum heart output during exercise tests."},
            {"Feature": "Exercise Induced Angina", "Type": "Binary", "Clinical Significance": "Angina triggered by exercise implies ischemic vessels."},
            {"Feature": "Oldpeak", "Type": "Continuous", "Clinical Significance": "ST depression showing decreased myocardial myocardial blood flow."},
            {"Feature": "ST Slope", "Type": "Categorical", "Clinical Significance": "Slope orientation of the ST segment peak exercise."},
            {"Feature": "Vessels (Fluoroscopy)", "Type": "Continuous", "Clinical Significance": "Number of major blood vessels exhibiting blockages."},
            {"Feature": "Thalassemia", "Type": "Categorical", "Clinical Significance": "Underlying genetic blood flow characteristics."}
        ])
        st.table(df_features)

with col2:
    # Developer Card
    with st.container(key="about_hero_card", border=True):
        st.markdown("<h3 style='margin-top:0; color: var(--primary-color);'>Developer Information</h3>", unsafe_allow_html=True)
        st.markdown("**Dharanidharan T**")
        st.write("Sri Eshwar College of Engineering")
        st.write("Department of Computer Science & Business Systems")
        st.markdown("---")
        st.markdown("**CodeAlpha Machine Learning Internship**")
        st.write("Project: Heart Disease Prediction System")
        st.write("Evaluation Level: Production-Ready Dashboard")

    # Technologies Used Card
    with st.container(key="about_card_tech", border=True):
        st.markdown("<h3 style='color: var(--primary-color); margin-top:0;'>Technology Stack</h3>", unsafe_allow_html=True)
        st.markdown("- **Web UI:** Streamlit (Python Dashboard Framework)")
        st.markdown("- **Core Logic:** Python 3.11")
        st.markdown("- **Machine Learning:** Scikit-learn (Random Forest Classifier)")
        st.markdown("- **Data Processing:** Pandas & NumPy")
        st.markdown("- **Interactions & Plots:** Plotly Express & Plotly Graph Objects")
        st.markdown("- **Serialization:** Joblib")
        st.markdown("- **Report Generation:** ReportLab PDF Toolkit")

    # Future Enhancements Card
    with st.container(key="about_card_future", border=True):
        st.markdown("<h3 style='color: var(--primary-color); margin-top:0;'>Future Enhancements</h3>", unsafe_allow_html=True)
        st.write("""
        1. **Clinical EHR Integration:** Add FHIR API pipelines to retrieve patient vitals directly from hospital systems.
        2. **Deep Learning Integration:** Deploy Neural Network models alongside Random Forests for ensemble voting.
        3. **Multi-Disease Panel:** Expand the prediction panel to include diabetes and hypertension risk assessments.
        """)

# Footer
render_footer()
