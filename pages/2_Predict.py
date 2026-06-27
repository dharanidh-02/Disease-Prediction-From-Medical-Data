import streamlit as st

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

st.set_page_config(
    page_title="MediPredict AI - Predict Risk",
    page_icon="https://img.icons8.com/color/48/heart-monitor.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import os
import io
import time
import sys
import plotly.express as px

# Ensure root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import setup_page_layout, render_footer
from utils import (
    load_model_assets, predict_risk, generate_pdf_report, check_scaling_required, FEATURE_NAMES,
    MAP_GENDER, MAP_CP, MAP_FBS, MAP_ECG, MAP_EXANG, MAP_SLOPE, MAP_THAL,
    save_prediction_to_db, fetch_prediction_history, clear_prediction_history
)

# Apply centralized layout and theme
setup_page_layout()

# Title
st.markdown("<h1 style='color: var(--primary-color); font-size: 2.2rem;'>Cardiovascular Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #5f6368; margin-top: -10px;'>Complete the clinical parameters below to evaluate heart disease risk, or perform bulk screening.</p>", unsafe_allow_html=True)
st.write("---")

# Load model assets
model, scaler = load_model_assets()

# Form Defaults
defaults = {
    'age': 50,
    'sex': 'Male',
    'chest_pain_type': 'Atypical Angina',
    'resting_blood_pressure': 130,
    'cholestoral': 240,
    'fasting_blood_sugar': '<= 120 mg/dl (Normal)',
    'rest_ecg': 'Normal',
    'Max_heart_rate': 150,
    'exercise_induced_angina': 'No',
    'oldpeak': 1.0,
    'slope': 'Flat',
    'vessels_colored_by_flourosopy': 0,
    'thalassemia': 'Fixed Defect'
}

# Form Reset Logic
def reset_form():
    for key in defaults.keys():
        st.session_state[f'form_{key}'] = defaults[key]
    if 'prediction_result' in st.session_state:
        del st.session_state['prediction_result']
    st.rerun()

# Helper for Bulk Mapping
def map_bulk_column(series, mapping_dict, col_name):
    def map_val(val):
        if pd.isna(val):
            return 0.0
        val_str = str(val).strip()
        for key, numeric_code in mapping_dict.items():
            if key.lower() == val_str.lower():
                return numeric_code
        try:
            return float(val)
        except ValueError:
            return list(mapping_dict.values())[0]
    return series.apply(map_val)

# Define Tabs
tab1, tab2 = st.tabs(["Individual Screening", "Bulk Clinic Screening"])

with tab1:
    # Layout: Split into input columns
    col1, col2 = st.columns([2, 1.2])

    with col1:
        st.markdown("### Patient Clinical Attributes")
        
        # Rounded Card 1: Demographics & Vitals
        with st.container(key="predict_card_demographics", border=True):
            st.markdown("<h4 style='color: var(--primary-color); margin-top:0;'>Demographics & Physical Vitals</h4>", unsafe_allow_html=True)
            
            c_dem1, c_dem2 = st.columns(2)
            with c_dem1:
                age = st.slider(
                    "Patient Age (Years)", 
                    min_value=20, max_value=100, 
                    value=st.session_state.get('form_age', defaults['age']), 
                    key='form_age'
                )
            with c_dem2:
                sex = st.radio(
                    "Gender", 
                    options=["Female", "Male"], 
                    index=0 if st.session_state.get('form_sex', defaults['sex']) == "Female" else 1, 
                    horizontal=True,
                    key='form_sex'
                )
                
            c_vit1, c_vit2 = st.columns(2)
            with c_vit1:
                trestbps = st.slider(
                    "Resting Blood Pressure (mmHg)", 
                    min_value=80, max_value=220, 
                    value=st.session_state.get('form_resting_blood_pressure', defaults['resting_blood_pressure']), 
                    key='form_resting_blood_pressure',
                    help="Resting blood pressure in mmHg upon admission."
                )
            with c_vit2:
                chol = st.slider(
                    "Serum Cholesterol (mg/dl)", 
                    min_value=100, max_value=600, 
                    value=st.session_state.get('form_cholestoral', defaults['cholestoral']), 
                    key='form_cholestoral',
                    help="Serum cholesterol level in mg/dl."
                )
                
            fbs = st.radio(
                "Fasting Blood Sugar",
                options=["<= 120 mg/dl (Normal)", "> 120 mg/dl (High)"],
                index=0 if st.session_state.get('form_fasting_blood_sugar', defaults['fasting_blood_sugar']) == "<= 120 mg/dl (Normal)" else 1,
                horizontal=True,
                key='form_fasting_blood_sugar',
                help="Fasting blood sugar level relative to 120 mg/dl."
            )

        # Rounded Card 2: Symptoms & Cardiac Stress Vitals
        with st.container(key="predict_card_symptoms", border=True):
            st.markdown("<h4 style='color: var(--primary-color); margin-top:0;'>Cardiac & Stress Attributes</h4>", unsafe_allow_html=True)
            
            cp = st.selectbox(
                "Chest Pain Type (CP)",
                options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                index=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(st.session_state.get('form_chest_pain_type', defaults['chest_pain_type'])),
                key='form_chest_pain_type',
                help="Nature of pain reported in the chest area."
            )
            
            c_str1, c_str2 = st.columns(2)
            with c_str1:
                thalach = st.slider(
                    "Maximum Heart Rate Achieved (bpm)",
                    min_value=60, max_value=220,
                    value=st.session_state.get('form_Max_heart_rate', defaults['Max_heart_rate']),
                    key='form_Max_heart_rate',
                    help="Maximum heart rate achieved during exercise stress test."
                )
            with c_str2:
                exang = st.radio(
                    "Exercise Induced Angina",
                    options=["No", "Yes"],
                    index=0 if st.session_state.get('form_exercise_induced_angina', defaults['exercise_induced_angina']) == "No" else 1,
                    horizontal=True,
                    key='form_exercise_induced_angina',
                    help="Angina (chest pain) induced by exercise stress test."
                )
                
            c_ecg1, c_ecg2 = st.columns(2)
            with c_ecg1:
                restecg = st.selectbox(
                    "Resting Electrocardiographic (ECG) Results",
                    options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                    index=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(st.session_state.get('form_rest_ecg', defaults['rest_ecg'])),
                    key='form_rest_ecg',
                    help="ECG results at rest."
                )
            with c_ecg2:
                oldpeak = st.slider(
                    "ST Depression (Oldpeak)",
                    min_value=0.0, max_value=8.0, step=0.1,
                    value=float(st.session_state.get('form_oldpeak', defaults['oldpeak'])),
                    key='form_oldpeak',
                    help="ST depression induced by exercise relative to rest."
                )
                
            c_ecg3, c_ecg4 = st.columns(2)
            with c_ecg3:
                slope = st.selectbox(
                    "Slope of Peak Exercise ST Segment",
                    options=["Upsloping", "Flat", "Downsloping"],
                    index=["Upsloping", "Flat", "Downsloping"].index(st.session_state.get('form_slope', defaults['slope'])),
                    key='form_slope',
                    help="Slope of the peak exercise ST segment."
                )
            with c_ecg4:
                ca = st.slider(
                    "Major Vessels Colored by Fluoroscopy",
                    min_value=0, max_value=4,
                    value=int(st.session_state.get('form_vessels_colored_by_flourosopy', defaults['vessels_colored_by_flourosopy'])),
                    key='form_vessels_colored_by_flourosopy',
                    help="Number of major vessels (0-4) colored by fluoroscopy."
                )
                
            thal = st.selectbox(
                "Thalassemia Type",
                options=["Normal / None", "Fixed Defect", "Reversable Defect", "Irreversable Defect"],
                index=["Normal / None", "Fixed Defect", "Reversable Defect", "Irreversable Defect"].index(st.session_state.get('form_thalassemia', defaults['thalassemia'])),
                key='form_thalassemia',
                help="Blood disorder screening classification."
            )

        # Action Buttons
        c_btn1, c_btn2 = st.columns([2, 1])
        with c_btn1:
            predict_clicked = st.button("Predict Heart Disease")
        with c_btn2:
            st.button("Reset Form", on_click=reset_form, type="secondary")

    with col2:
        st.markdown("### Risk Analysis Output")
        
        # Process prediction if clicked
        if predict_clicked:
            with st.spinner("Processing clinical metrics through AI..."):
                time.sleep(1.2) # Realistic clinic diagnostics wait
                
                # Map input text back to training numeric codes
                inputs_numeric = {
                    'age': float(age),
                    'sex': MAP_GENDER[sex],
                    'chest_pain_type': MAP_CP[cp],
                    'resting_blood_pressure': float(trestbps),
                    'cholestoral': float(chol),
                    'fasting_blood_sugar': MAP_FBS[fbs],
                    'rest_ecg': MAP_ECG[restecg],
                    'Max_heart_rate': float(thalach),
                    'exercise_induced_angina': MAP_EXANG[exang],
                    'oldpeak': float(oldpeak),
                    'slope': MAP_SLOPE[slope],
                    'vessels_colored_by_flourosopy': float(ca),
                    'thalassemia': MAP_THAL[thal]
                }
                
                # Run prediction
                pred, risk_score, healthy_score = predict_risk(inputs_numeric, model, scaler)
                
                # Format clinical recommendations based on risk
                if risk_score < 30:
                    recs = [
                        "Maintain healthy diet (low saturated fats, high fiber, lots of vegetables).",
                        "Regular physical exercise (at least 150 minutes of moderate activity per week).",
                        "Perform routine annual health checkups and blood pressure screenings."
                    ]
                    status_class = "status-card-green"
                    verdict = "No Heart Disease Detected"
                    verdict_desc = "Patient profile indicates a low risk of cardiovascular diseases."
                elif risk_score <= 70:
                    recs = [
                        "Improve diet (restrict sodium, saturated fats, and refined sugars).",
                        "Monitor blood pressure and cholesterol levels regularly (every 3-6 months).",
                        "Exercise regularly under professional guidance (e.g. brisk walking, cycling).",
                        "Consult a physician if any warning symptoms (chest tightness, shortness of breath) appear."
                    ]
                    status_class = "status-card-yellow"
                    verdict = "Heart Disease Risk Detected (Moderate)"
                    verdict_desc = "Patient profile indicates a moderate risk of cardiovascular issues. Lifestyle changes advised."
                else:
                    recs = [
                        "Consult a cardiologist immediately for a comprehensive cardiac evaluation.",
                        "Reduce cholesterol intake and adhere strictly to cardiac dietary guidelines.",
                        "Avoid smoking and limit alcohol consumption entirely.",
                        "Monitor blood pressure daily and record readings.",
                        "Take all prescribed cardiovascular medications strictly as directed."
                    ]
                    status_class = "status-card-red"
                    verdict = "Heart Disease Risk Detected (High)"
                    verdict_desc = "Patient profile indicates a high risk. Clinical diagnosis and treatment protocol are highly recommended."
                
                # Store in session state
                st.session_state['prediction_result'] = {
                    'pred': pred,
                    'risk_score': risk_score,
                    'healthy_score': healthy_score,
                    'recs': recs,
                    'status_class': status_class,
                    'verdict': verdict,
                    'verdict_desc': verdict_desc,
                    'inputs': inputs_numeric
                }
                
                # Add to SQLite database
                save_prediction_to_db(inputs_numeric, risk_score, verdict)

        # Render Prediction Output
        if 'prediction_result' in st.session_state:
            res = st.session_state['prediction_result']
            
            # Verdict Card
            st.markdown(f"""
                <div class="{res['status_class']}">
                    <h3 style="margin-top: 0; margin-bottom: 5px; font-size: 1.3rem;">{res['verdict']}</h3>
                    <span style="font-size: 0.9rem;">{res['verdict_desc']}</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Risk Meter
            with st.container(key="predict_card_meter", border=True):
                st.markdown("<h4 style='color: var(--primary-color); margin-top:0; text-align: center;'>Heart Disease Risk Meter</h4>", unsafe_allow_html=True)
                
                # Progress Bar visual
                st.progress(res['risk_score'] / 100.0)
                
                st.markdown(f"""
                    <div class="gauge-container">
                        <span class="gauge-percentage" style="color: {'#34a853' if res['risk_score'] < 30 else '#fbbc05' if res['risk_score'] <= 70 else '#ea4335'};">
                            {res['risk_score']:.1f}%
                        </span>
                        <span style="color: var(--text-secondary); font-size: 0.9rem;">Heart Disease Risk Probability</span>
                    </div>
                """, unsafe_allow_html=True)
                
            # Clinical Recommendations Card
            with st.container(key="predict_card_recs", border=True):
                st.markdown("<h4 style='color: var(--primary-color); margin-top:0;'>Medical Recommendations</h4>", unsafe_allow_html=True)
                for r in res['recs']:
                    st.markdown(f"- {r}")

            # Scenario Simulator (What-If Analysis)
            st.write("---")
            st.markdown("<h4 style='color: var(--primary-color); margin-top:0;'>Risk Factor Scenario Simulator</h4>", unsafe_allow_html=True)
            st.write("Simulate changes in key cardiac metrics to see the risk score response in real time:")
            
            sim_bp = st.slider("Simulated BP (mmHg)", 80, 220, int(res['inputs']['resting_blood_pressure']), key="sim_bp")
            sim_chol = st.slider("Simulated Cholesterol (mg/dl)", 100, 600, int(res['inputs']['cholestoral']), key="sim_chol")
            sim_hr = st.slider("Simulated Max HR (bpm)", 60, 220, int(res['inputs']['Max_heart_rate']), key="sim_hr")
            
            # Re-predict
            sim_inputs = res['inputs'].copy()
            sim_inputs['resting_blood_pressure'] = float(sim_bp)
            sim_inputs['cholestoral'] = float(sim_chol)
            sim_inputs['Max_heart_rate'] = float(sim_hr)
            
            sim_pred, sim_risk, sim_healthy = predict_risk(sim_inputs, model, scaler)
            
            delta = sim_risk - res['risk_score']
            delta_color = "#ea4335" if delta > 0.05 else "#34a853" if delta < -0.05 else "#5f6368"
            delta_symbol = "+" if delta > 0 else ""
            
            st.markdown(f"""
                <div class="simulator-card">
                    <p style="margin: 0; color: #5f6368; font-size: 0.85rem;">Simulated Risk Probability</p>
                    <h3 style="margin: 5px 0; font-size: 1.8rem; color: {'#ea4335' if sim_risk > 70 else '#fbbc05' if sim_risk >= 30 else '#34a853'};">
                        {sim_risk:.1f}% 
                        <span style="font-size: 0.95rem; font-weight: normal; color: {delta_color};">
                            ({delta_symbol}{delta:.1f}% variance)
                        </span>
                    </h3>
                    <p style="margin: 0; font-size: 0.8rem; color: #7f8c8d;">
                        Simulated attributes: BP {sim_bp} mmHg, Cholesterol {sim_chol} mg/dl, Max Heart Rate {sim_hr} bpm.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Report Downloads Row
            st.write("---")
            st.markdown("### Export Diagnostic Data")
            
            # Generate PDF bytes
            pdf_filename = "MediPredict_AI_Report.pdf"
            generate_pdf_report(res['inputs'], res['risk_score'], res['recs'], pdf_filename)
            
            with open(pdf_filename, "rb") as f:
                pdf_bytes = f.read()
                
            # Generate CSV string
            csv_df = pd.DataFrame([res['inputs']])
            csv_df['risk_score'] = f"{res['risk_score']:.2f}%"
            csv_df['verdict'] = res['verdict']
            csv_buffer = io.StringIO()
            csv_df.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            dc1, dc2 = st.columns(2)
            with dc1:
                st.download_button(
                    label="Download Report (PDF)",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )
            with dc2:
                st.download_button(
                    label="Export Data (CSV)",
                    data=csv_bytes,
                    file_name="patient_cardiac_record.csv",
                    mime="text/csv"
                )
                
            # Remove local file after reading it to bytes
            if os.path.exists(pdf_filename):
                try:
                    os.remove(pdf_filename)
                except Exception:
                    pass
        else:
            # Pre-prediction state
            with st.container(key="predict_card_pre", border=True):
                st.markdown("<h4 style='margin: 0 0 5px 0; text-align: center;'>No Diagnostics Run Yet</h4>", unsafe_allow_html=True)
                st.markdown("<p style='color: var(--text-secondary); font-size: 0.9rem; text-align: center;'>Fill in the parameters on the left and click <b>Predict Heart Disease</b> to trigger AI clinical screening.</p>", unsafe_allow_html=True)

with tab2:
    st.markdown("### Bulk Cohort Evaluation")
    st.markdown("<p style='color: #5f6368; margin-top:-10px;'>Upload a patient registry CSV spreadsheet to perform batch inference and diagnostic risk profiling.</p>", unsafe_allow_html=True)
    
    # Download template CSV
    sample_df = pd.DataFrame([
        {
            'age': 55, 'sex': 'Male', 'chest_pain_type': 'Atypical Angina',
            'resting_blood_pressure': 140, 'cholestoral': 250, 'fasting_blood_sugar': '<= 120 mg/dl (Normal)',
            'rest_ecg': 'ST-T Wave Abnormality', 'Max_heart_rate': 145, 'exercise_induced_angina': 'No',
            'oldpeak': 1.5, 'slope': 'Flat', 'vessels_colored_by_flourosopy': 1, 'thalassemia': 'Fixed Defect'
        },
        {
            'age': 42, 'sex': 'Female', 'chest_pain_type': 'Typical Angina',
            'resting_blood_pressure': 120, 'cholestoral': 210, 'fasting_blood_sugar': '<= 120 mg/dl (Normal)',
            'rest_ecg': 'Normal', 'Max_heart_rate': 168, 'exercise_induced_angina': 'No',
            'oldpeak': 0.0, 'slope': 'Upsloping', 'vessels_colored_by_flourosopy': 0, 'thalassemia': 'Normal / None'
        },
        {
            'age': 65, 'sex': 'Male', 'chest_pain_type': 'Asymptomatic',
            'resting_blood_pressure': 155, 'cholestoral': 290, 'fasting_blood_sugar': '> 120 mg/dl (High)',
            'rest_ecg': 'Left Ventricular Hypertrophy', 'Max_heart_rate': 115, 'exercise_induced_angina': 'Yes',
            'oldpeak': 2.8, 'slope': 'Flat', 'vessels_colored_by_flourosopy': 3, 'thalassemia': 'Reversable Defect'
        }
    ])
    
    template_buffer = io.StringIO()
    sample_df.to_csv(template_buffer, index=False)
    
    st.download_button(
        label="Download CSV Screening Template",
        data=template_buffer.getvalue(),
        file_name="medipredict_bulk_template.csv",
        mime="text/csv",
        type="secondary"
    )
    
    st.write("")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Patient Registry (CSV)", type=["csv"], key="bulk_uploader")
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = FEATURE_NAMES
            missing_cols = [col for col in required_cols if col not in bulk_df.columns]
            
            if missing_cols:
                st.error(f"Invalid file format. Missing columns: {', '.join(missing_cols)}")
            else:
                # Map values to numeric codes
                mapped_df = pd.DataFrame()
                
                # Copy numeric features
                mapped_df['age'] = bulk_df['age'].astype(float)
                mapped_df['sex'] = map_bulk_column(bulk_df['sex'], MAP_GENDER, 'sex')
                mapped_df['chest_pain_type'] = map_bulk_column(bulk_df['chest_pain_type'], MAP_CP, 'chest_pain_type')
                mapped_df['resting_blood_pressure'] = bulk_df['resting_blood_pressure'].astype(float)
                mapped_df['cholestoral'] = bulk_df['cholestoral'].astype(float)
                mapped_df['fasting_blood_sugar'] = map_bulk_column(bulk_df['fasting_blood_sugar'], MAP_FBS, 'fasting_blood_sugar')
                mapped_df['rest_ecg'] = map_bulk_column(bulk_df['rest_ecg'], MAP_ECG, 'rest_ecg')
                mapped_df['Max_heart_rate'] = bulk_df['Max_heart_rate'].astype(float)
                mapped_df['exercise_induced_angina'] = map_bulk_column(bulk_df['exercise_induced_angina'], MAP_EXANG, 'exercise_induced_angina')
                mapped_df['oldpeak'] = bulk_df['oldpeak'].astype(float)
                mapped_df['slope'] = map_bulk_column(bulk_df['slope'], MAP_SLOPE, 'slope')
                mapped_df['vessels_colored_by_flourosopy'] = bulk_df['vessels_colored_by_flourosopy'].astype(float)
                mapped_df['thalassemia'] = map_bulk_column(bulk_df['thalassemia'], MAP_THAL, 'thalassemia')
                
                # Preprocess (scale if needed)
                scaling_req = check_scaling_required(model)
                if scaling_req and scaler is not None:
                    scaled_feats = scaler.transform(mapped_df)
                    input_data = pd.DataFrame(scaled_feats, columns=FEATURE_NAMES)
                else:
                    input_data = mapped_df
                
                # Predict
                preds = model.predict(input_data)
                probs = model.predict_proba(input_data)
                
                # Class 0 is Disease Risk, Class 1 is Healthy
                risk_scores = probs[:, 0] * 100
                
                # Risk Category Assigning
                verdicts = []
                for score in risk_scores:
                    if score < 30:
                        verdicts.append("Low Risk")
                    elif score <= 70:
                        verdicts.append("Moderate Risk")
                    else:
                        verdicts.append("High Risk")
                
                # Output Dataframe construction
                output_df = bulk_df.copy()
                output_df['Calculated Risk Probability (%)'] = np.round(risk_scores, 2)
                output_df['Risk Classification'] = verdicts
                
                # Save bulk predictions to SQLite
                for idx in range(len(mapped_df)):
                    row_inputs = mapped_df.iloc[idx].to_dict()
                    save_prediction_to_db(row_inputs, risk_scores[idx], verdicts[idx])
                
                # Stats Display
                st.markdown("### Cohort Screening Analytics")
                
                total_patients = len(output_df)
                high_risk_cnt = sum(score > 70 for score in risk_scores)
                mod_risk_cnt = sum(30 <= score <= 70 for score in risk_scores)
                low_risk_cnt = sum(score < 30 for score in risk_scores)
                
                c_an1, c_an2, c_an3, c_an4 = st.columns(4)
                with c_an1:
                    st.markdown(f"""
                        <div class="predict-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 15px; border-radius: 12px;">
                            <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">Total Screened</h4>
                            <p style="font-size: 1.6rem; font-weight: 700; color: var(--primary-color); margin: 8px 0 0 0;">{total_patients} Patients</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c_an2:
                    st.markdown(f"""
                        <div class="predict-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 15px; border-radius: 12px;">
                            <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">High Risk</h4>
                            <p style="font-size: 1.6rem; font-weight: 700; color: var(--accent-red); margin: 8px 0 0 0;">{high_risk_cnt} Cases</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c_an3:
                    st.markdown(f"""
                        <div class="predict-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 15px; border-radius: 12px;">
                            <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">Moderate Risk</h4>
                            <p style="font-size: 1.6rem; font-weight: 700; color: var(--accent-yellow); margin: 8px 0 0 0;">{mod_risk_cnt} Cases</p>
                        </div>
                    """, unsafe_allow_html=True)
                with c_an4:
                    st.markdown(f"""
                        <div class="predict-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 15px; border-radius: 12px;">
                            <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.85rem;">Low Risk</h4>
                            <p style="font-size: 1.6rem; font-weight: 700; color: var(--accent-green); margin: 8px 0 0 0;">{low_risk_cnt} Cases</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Charts
                st.write("")
                col_chart1, col_chart2 = st.columns(2)
                
                # Define dynamic theme variables for Plotly charts
                is_dark = st.session_state.get('dark_mode', False)
                plotly_template = "plotly_dark" if is_dark else "plotly"
                paper_bgcolor = "rgba(0,0,0,0)"
                plot_bgcolor = "rgba(0,0,0,0)"
                grid_color = "#333333" if is_dark else "#e5e5e5"
                text_color = "#ffffff" if is_dark else "#1f1f1f"

                with col_chart1:
                    st.markdown("#### Risk Distribution Breakdown")
                    risk_counts = pd.Series(verdicts).value_counts()
                    fig_pie = px.pie(
                        names=risk_counts.index,
                        values=risk_counts.values,
                        color=risk_counts.index,
                        color_discrete_map={
                            'Low Risk': '#34a853',
                            'Moderate Risk': '#fbbc05',
                            'High Risk': '#ea4335'
                        },
                        hole=0.4
                    )
                    fig_pie.update_layout(
                        template=plotly_template,
                        paper_bgcolor=paper_bgcolor,
                        plot_bgcolor=plot_bgcolor,
                        font=dict(color=text_color),
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=300
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                with col_chart2:
                    st.markdown("#### Risk Score vs. Patient Age")
                    fig_scatter = px.scatter(
                        output_df,
                        x='age',
                        y='Calculated Risk Probability (%)',
                        color='Risk Classification',
                        color_discrete_map={
                            'Low Risk': '#34a853',
                            'Moderate Risk': '#fbbc05',
                            'High Risk': '#ea4335'
                        }
                    )
                    fig_scatter.update_layout(
                        template=plotly_template,
                        paper_bgcolor=paper_bgcolor,
                        plot_bgcolor=plot_bgcolor,
                        font=dict(color=text_color),
                        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title='Patient Age'),
                        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title='Risk Probability (%)'),
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=300
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Results Grid
                st.markdown("### Processed Results Overview")
                st.dataframe(output_df, use_container_width=True)
                
                # Export csv
                csv_buffer = io.StringIO()
                output_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Processed Patient Cohort CSV",
                    data=csv_buffer.getvalue(),
                    file_name="processed_patient_screenings.csv",
                    mime="text/csv",
                    type="primary"
                )
        except Exception as e:
            st.error(f"Error parsing uploaded file: {str(e)}")

# Prediction History Table
st.write("---")
st.markdown("### Patient Prediction History (Persistent SQLite)")

history_df = fetch_prediction_history()

if not history_df.empty:
    display_history_df = history_df.copy()
    display_history_df = display_history_df.rename(columns={
        'age': 'Age',
        'sex': 'Gender (Code)',
        'chest_pain_type': 'Chest Pain Type (Code)',
        'resting_blood_pressure': 'BP (mmHg)',
        'cholestoral': 'Chol (mg/dl)',
        'fasting_blood_sugar': 'FBS (Code)',
        'rest_ecg': 'Rest ECG (Code)',
        'Max_heart_rate': 'Max HR (bpm)',
        'exercise_induced_angina': 'Ex Angina (Code)',
        'oldpeak': 'Oldpeak',
        'slope': 'Slope (Code)',
        'vessels_colored_by_flourosopy': 'Vessels (Code)',
        'thalassemia': 'Thal (Code)',
        'risk_score': 'Risk Score',
        'verdict': 'Verdict',
        'timestamp': 'Timestamp'
    })
    
    display_history_df['Risk Score'] = display_history_df['Risk Score'].apply(lambda x: f"{x:.1f}%")
    
    cols_order = ['Timestamp', 'Age', 'BP (mmHg)', 'Chol (mg/dl)', 'Max HR (bpm)', 'Oldpeak', 'Risk Score', 'Verdict']
    st.dataframe(display_history_df[cols_order], use_container_width=True)
    
    # Download and Clear Row
    c_hist1, c_hist2 = st.columns([3, 1])
    with c_hist1:
        full_csv_buffer = io.StringIO()
        history_df.to_csv(full_csv_buffer, index=False)
        st.download_button(
            label="Download Full History as CSV",
            data=full_csv_buffer.getvalue().encode('utf-8'),
            file_name="medipredict_history.csv",
            mime="text/csv"
        )
    with c_hist2:
        if st.button("Clear History Logs", type="secondary", use_container_width=True):
            clear_prediction_history()
            st.success("Clinical logs cleared.")
            st.rerun()
else:
    st.info("No persistent patient diagnostic logs recorded yet.")

# Footer
render_footer()
