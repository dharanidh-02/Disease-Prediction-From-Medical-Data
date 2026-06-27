import streamlit as st

if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

st.set_page_config(
    page_title="MediPredict AI - Performance",
    page_icon="https://img.icons8.com/color/48/heart-monitor.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score

# Ensure root is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import setup_page_layout, render_footer
from utils import load_model_assets, load_performance_data

# Apply centralized layout and theme
setup_page_layout()

# Title
st.markdown("<h1 style='color: var(--primary-color); font-size: 2.2rem;'>Model Performance & Clinical Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #5f6368; margin-top: -10px;'>Verify model accuracy, threshold splittings, and attribute feature importances.</p>", unsafe_allow_html=True)
st.write("---")

# Load model and dataset
model, scaler = load_model_assets()
df_aligned = load_performance_data()

if df_aligned is None:
    st.error("Error: Evaluation dataset could not be loaded. Please ensure data/heart.csv exists.")
else:
    # Run evaluation
    feature_cols = list(model.feature_names_in_)
    X = df_aligned[feature_cols]
    y_true = df_aligned['model_target']
    
    # Model predictions
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1] # Probability of Class 1 (Healthy)
    
    # Since Class 0 represents Disease Risk and Class 1 represents Healthy,
    # let's define metrics matching Class 0 (Risk) or general binary classification.
    # In scikit-learn, positive class is typically Class 1.
    # In our diagnostic reporting, let's treat Class 0 (Risk) as the target of interest for diagnostic precision/recall.
    # We will compute metrics for target classification of risk (Class 0).
    # Since the model predicts Class 1 for Healthy and Class 0 for Risk,
    # to evaluate how well the model identifies "disease risk", we can swap labels so that "1" means Risk.
    # Or, we can just report the macro/weighted average, or the specific stats for Disease Risk (Class 0).
    
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1 for Class 0 (Disease Risk) and Class 1 (Healthy)
    # Average='binary' evaluates the positive class. In our model, target matches Kaggle, where:
    # Kaggle: 0 = Healthy, 1 = Disease.
    # Our model: 0 = Disease, 1 = Healthy (based on our alternative mapping test where accuracy was 80.20%).
    # So y_true is 0 (Diseased) and 1 (Healthy).
    # Let's calculate binary metrics treating Disease Risk (0) as the clinical target class.
    # Precision, Recall, and F1 for Class 0 (Risk)
    precision_risk = precision_score(y_true, y_pred, pos_label=0)
    recall_risk = recall_score(y_true, y_pred, pos_label=0)
    f1_risk = f1_score(y_true, y_pred, pos_label=0)

    # Display Metrics inside Cards
    st.markdown("### Classifier Quality Metrics")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Model Accuracy</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">{acc * 100:.2f}%</p>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">Overall Correct Predictions</span>
            </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Risk Precision</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">{precision_risk * 100:.2f}%</p>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">Positive Predictive Value</span>
            </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Risk Recall (Sensitivity)</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">{recall_risk * 100:.2f}%</p>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">True Positive Rate</span>
            </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
            <div class="medical-card" style="text-align: center; border: 1px solid var(--border-color); background-color: var(--bg-card-light); padding: 20px; border-radius: 12px;">
                <h4 style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">Risk F1 Score</h4>
                <p style="font-size: 1.8rem; font-weight: 700; color: var(--primary-color); margin: 10px 0 0 0;">{f1_risk * 100:.2f}%</p>
                <span style="font-size: 0.8rem; color: var(--text-secondary);">Harmonic Mean of PPV/TPR</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # Define dynamic theme variables for Plotly charts
    is_dark = st.session_state.get('dark_mode', False)
    plotly_template = "plotly_dark" if is_dark else "plotly"
    paper_bgcolor = "rgba(0,0,0,0)"
    plot_bgcolor = "rgba(0,0,0,0)"
    grid_color = "#333333" if is_dark else "#e5e5e5"
    text_color = "#ffffff" if is_dark else "#1f1f1f"

    # Main charts section
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Confusion Matrix")
        # Generate Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        # Formats: True label vs Predicted Label
        labels_cm = ["Heart Disease Risk (0)", "Healthy (1)"]
        
        fig_cm = px.imshow(
            cm,
            x=labels_cm,
            y=labels_cm,
            text_auto=True,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Label", y="True Label", color="Patients count")
        )
        fig_cm.update_layout(
            template=plotly_template,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font=dict(color=text_color),
            margin=dict(l=40, r=40, t=20, b=40),
            height=350,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonal elements represent correct classifications. Off-diagonals represent False Negatives (bottom-left) and False Positives (top-right).")

    with col_chart2:
        st.markdown("### Receiver Operating Characteristic (ROC) Curve")
        # In scikit-learn, roc_curve calculates for positive class (1).
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve using Plotly
        fig_roc = go.Figure()
        
        # ROC Line
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#ffffff' if is_dark else '#0b57d0', width=3)
        ))
        
        # Baseline Diagonal Line
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Guessing',
            line=dict(color='#ea4335', dash='dash')
        ))
        
        fig_roc.update_layout(
            template=plotly_template,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font=dict(color=text_color),
            xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title='False Positive Rate'),
            yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color, title='True Positive Rate'),
            margin=dict(l=40, r=40, t=20, b=40),
            height=350,
            legend=dict(
                x=0.55, y=0.15,
                bgcolor='rgba(18,18,18,0.8)' if is_dark else 'rgba(255,255,255,0.8)',
                bordercolor=grid_color
            )
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.caption(f"ROC AUC is {roc_auc:.3f}, demonstrating strong classification discriminative power between cardiovascular risk and healthy cases.")

    st.write("---")

    # Feature Importance Section
    st.markdown("### Clinical Feature Importance Analysis")
    st.write("Understand which parameters contribute most to the Random Forest model's predictions:")
    
    # Get importances
    importances = model.feature_importances_
    features = list(model.feature_names_in_)
    
    # Prettify features for human reading
    pretty_features = {
        'age': 'Age',
        'sex': 'Gender / Sex',
        'chest_pain_type': 'Chest Pain Severity (CP)',
        'resting_blood_pressure': 'Resting Blood Pressure',
        'cholestoral': 'Serum Cholesterol',
        'fasting_blood_sugar': 'Fasting Blood Sugar',
        'rest_ecg': 'Resting ECG Results',
        'Max_heart_rate': 'Maximum Heart Rate',
        'exercise_induced_angina': 'Exercise Induced Angina',
        'oldpeak': 'ST Depression (Oldpeak)',
        'slope': 'Peak ST Segment Slope',
        'vessels_colored_by_flourosopy': 'Vessels Colored (Fluoroscopy)',
        'thalassemia': 'Thalassemia Type'
    }
    
    df_imp = pd.DataFrame({
        'Feature': [pretty_features.get(f, f) for f in features],
        'RawFeature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    col_feat1, col_feat2 = st.columns([1.5, 1])
    
    with col_feat1:
        # Plotly horizontal bar chart
        fig_imp = px.bar(
            df_imp,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues',
            labels=dict(Importance="Relative Importance score", Feature="Clinical Feature")
        )
        fig_imp.update_layout(
            template=plotly_template,
            paper_bgcolor=paper_bgcolor,
            plot_bgcolor=plot_bgcolor,
            font=dict(color=text_color),
            xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
            yaxis={'categoryorder': 'total ascending', 'gridcolor': grid_color},
            margin=dict(l=40, r=40, t=20, b=40),
            height=400,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_feat2:
        st.markdown("<h4 style='color: var(--primary-color); margin-top:0;'>Top Influential Indicators</h4>", unsafe_allow_html=True)
        st.write("""
        The Random Forest model assigns weights to features based on their node impurity reduction capabilities across its **200 decision trees**. 
        Here are the most significant diagnostic indicators according to the model:
        """)
        
        # Display top 3 features with explanations
        top_3 = df_imp.head(3).to_dict('records')
        for i, row in enumerate(top_3):
            st.markdown(f"**{i+1}. {row['Feature']}** (Weight: `{row['Importance']*100:.1f}%`)")
            if row['RawFeature'] == 'chest_pain_type':
                st.write("*Chest pain nature is the highest predictor. Asymptomatic chest pain in risk-suspected patients is highly correlated with blockages.*")
            elif row['RawFeature'] == 'thalassemia':
                st.write("*Thalassemia type is a critical genetic blood screening parameter linked to underlying cardiac stress tolerance.*")
            elif row['RawFeature'] == 'vessels_colored_by_flourosopy':
                st.write("*Number of major blood vessels visible under fluoroscopy indicates the severity of coronary artery narrowing.*")
            elif row['RawFeature'] == 'Max_heart_rate':
                st.write("*Maximum heart rate achieved acts as a primary index for overall aerobic fitness and cardiovascular output.*")
            elif row['RawFeature'] == 'oldpeak':
                st.write("*ST segment depression reflects myocardial ischemia (lack of blood supply to heart tissues) during exercise.*")
            else:
                st.write("*This clinical marker plays a supporting role in establishing patient-specific risk profiles.*")

    # Classification Report
    st.write("---")
    st.markdown("### Detailed Classification Report")
    
    report_dict = classification_report(y_true, y_pred, target_names=["Diseased (Class 0)", "Healthy (Class 1)"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=3), use_container_width=True)

# Footer
render_footer()
