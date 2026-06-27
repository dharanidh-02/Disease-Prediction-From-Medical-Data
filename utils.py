import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
# Suppress scikit-learn unpickle version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass
# Suppress UserWarning related to unpickling models across different sklearn versions
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Define the exact feature names in the order expected by the model
FEATURE_NAMES = [
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholestoral',
    'fasting_blood_sugar', 'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina',
    'oldpeak', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia'
]

# Mapping human-readable terms to dataset codes
MAP_GENDER = {"Female": 0.0, "Male": 1.0}
MAP_CP = {
    "Typical Angina": 0.0,
    "Atypical Angina": 1.0,
    "Non-anginal Pain": 2.0,
    "Asymptomatic": 3.0
}
MAP_FBS = {"<= 120 mg/dl (Normal)": 0.0, "> 120 mg/dl (High)": 1.0}
MAP_ECG = {
    "Normal": 0.0,
    "ST-T Wave Abnormality": 1.0,
    "Left Ventricular Hypertrophy": 2.0
}
MAP_EXANG = {"No": 0.0, "Yes": 1.0}
MAP_SLOPE = {
    "Upsloping": 0.0,
    "Flat": 1.0,
    "Downsloping": 2.0
}
MAP_THAL = {
    "Normal / None": 0.0,
    "Fixed Defect": 1.0,
    "Reversable Defect": 2.0,
    "Irreversable Defect": 3.0
}

@st.cache_resource
def load_model_assets():
    """Load model and scaler from files."""
    model_path = os.path.join("models", "heart_disease_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(model_path)
        
    scaler = None
    if os.path.exists(scaler_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scaler = joblib.load(scaler_path)
        except Exception:
            pass
            
    return model, scaler

def check_scaling_required(model):
    """
    Programmatically detect if scaling is required.
    Inspects thresholds for the 'age' feature (index 0) in the random forest estimators.
    If splits are made at values > 10.0 (like 45.0, 52.0), the model was trained on raw data.
    If splits are made at values < 5.0 (like -0.8, 1.2), the model expects scaled data.
    """
    if not hasattr(model, "estimators_"):
        return False # Fallback for non-tree models
        
    thresholds = []
    for tree in model.estimators_:
        tree_ = tree.tree_
        for i in range(tree_.node_count):
            if tree_.feature[i] == 0: # Age is index 0
                thresholds.append(tree_.threshold[i])
                
    if thresholds:
        mean_threshold = np.mean(thresholds)
        return mean_threshold < 10.0
    return False

def predict_risk(raw_input_dict, model, scaler):
    """
    Format raw inputs, preprocess (scale if needed), and run prediction.
    Returns:
        prediction (int): 0 (Diseased/Risk) or 1 (Healthy)
        risk_prob (float): Percentage risk of heart disease (Class 0 probability)
        healthy_prob (float): Percentage healthiness (Class 1 probability)
    """
    # Create DataFrame with correct column names and ordering
    df = pd.DataFrame([raw_input_dict], columns=FEATURE_NAMES)
    
    # Check if scaling is required
    scaling_req = check_scaling_required(model)
    if scaling_req and scaler is not None:
        scaled_features = scaler.transform(df)
        input_data = pd.DataFrame(scaled_features, columns=FEATURE_NAMES)
    else:
        input_data = df
        
    # Get model outputs
    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    
    # Class 0 represents Disease Risk, Class 1 represents Healthy/No Disease
    risk_prob = probs[0] * 100
    healthy_prob = probs[1] * 100
    
    return pred, risk_prob, healthy_prob

@st.cache_data
def load_performance_data():
    """Load Kaggle dataset and align columns for diagnostics page."""
    data_path = os.path.join("data", "heart.csv")
    if not os.path.exists(data_path):
        return None
        
    df = pd.read_csv(data_path)
    
    # Map Kaggle column names to model feature names
    df_aligned = df.rename(columns={
        'cp': 'chest_pain_type',
        'trestbps': 'resting_blood_pressure',
        'chol': 'cholestoral',
        'fbs': 'fasting_blood_sugar',
        'restecg': 'rest_ecg',
        'thalach': 'Max_heart_rate',
        'exang': 'exercise_induced_angina',
        'ca': 'vessels_colored_by_flourosopy',
        'thal': 'thalassemia'
    })
    
    # Ground truth mapping:
    # In Kaggle's heart.csv target=1 is diseased and target=0 is healthy.
    # In our model class 0 is diseased, class 1 is healthy.
    # Therefore, true model targets should be: 0 if Kaggle target==1 else 1.
    df_aligned['model_target'] = df_aligned['target'].apply(lambda x: 0 if x == 1 else 1)
    
    return df_aligned

def generate_pdf_report(patient_data, risk_score, recommendation, filepath):
    """Generate a clean, beautiful clinical report PDF for the patient."""
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#0b57d0'),
        alignment=0,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#5f6368'),
        spaceAfter=25
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1f1f1f'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1f1f1f')
    )
    
    label_style = ParagraphStyle(
        'TableLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#5f6368')
    )
    
    val_style = ParagraphStyle(
        'TableVal',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1f1f1f')
    )
    
    story = []
    
    # Header Banner
    story.append(Paragraph("MediPredict AI", title_style))
    story.append(Paragraph("Cardiovascular Health Analysis Report — Clinical Summary", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Patient Vitals Table
    story.append(Paragraph("Patient Clinical Attributes", section_title))
    
    # Formatted input representation
    def map_back_gender(v): return "Male" if v == 1.0 else "Female"
    def map_back_cp(v): return [k for k, val in MAP_CP.items() if val == v][0]
    def map_back_fbs(v): return "FBS > 120 mg/dl" if v == 1.0 else "FBS <= 120 mg/dl"
    def map_back_ecg(v): return [k for k, val in MAP_ECG.items() if val == v][0]
    def map_back_exang(v): return "Yes" if v == 1.0 else "No"
    def map_back_slope(v): return [k for k, val in MAP_SLOPE.items() if val == v][0]
    def map_back_thal(v): return [k for k, val in MAP_THAL.items() if val == v][0]

    data = [
        [Paragraph("Age", label_style), Paragraph(str(int(patient_data['age'])), val_style),
         Paragraph("Gender", label_style), Paragraph(map_back_gender(patient_data['sex']), val_style)],
        [Paragraph("Chest Pain Type", label_style), Paragraph(map_back_cp(patient_data['chest_pain_type']), val_style),
         Paragraph("Resting BP", label_style), Paragraph(f"{int(patient_data['resting_blood_pressure'])} mmHg", val_style)],
        [Paragraph("Serum Cholesterol", label_style), Paragraph(f"{int(patient_data['cholestoral'])} mg/dl", val_style),
         Paragraph("Fasting Blood Sugar", label_style), Paragraph(map_back_fbs(patient_data['fasting_blood_sugar']), val_style)],
        [Paragraph("Rest ECG Results", label_style), Paragraph(map_back_ecg(patient_data['rest_ecg']), val_style),
         Paragraph("Max Heart Rate", label_style), Paragraph(f"{int(patient_data['Max_heart_rate'])} bpm", val_style)],
        [Paragraph("Exercise Induced Angina", label_style), Paragraph(map_back_exang(patient_data['exercise_induced_angina']), val_style),
         Paragraph("ST Depression (Oldpeak)", label_style), Paragraph(f"{patient_data['oldpeak']:.1f}", val_style)],
        [Paragraph("Peak Exercise Slope", label_style), Paragraph(map_back_slope(patient_data['slope']), val_style),
         Paragraph("Major Colored Vessels", label_style), Paragraph(str(int(patient_data['vessels_colored_by_flourosopy'])), val_style)],
        [Paragraph("Thalassemia Type", label_style), Paragraph(map_back_thal(patient_data['thalassemia']), val_style),
         Paragraph("Report Date", label_style), Paragraph(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), val_style)]
    ]
    
    t = Table(data, colWidths=[140, 120, 140, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafd')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    
    # Results block
    story.append(Paragraph("AI Diagnostic Summary", section_title))
    
    # Define color based on risk severity
    if risk_score < 30:
        card_color = colors.HexColor('#d4edda')
        text_color = colors.HexColor('#155724')
        border_color = colors.HexColor('#c3e6cb')
        verdict = "LOW RISK - No Heart Disease Detected"
    elif risk_score <= 70:
        card_color = colors.HexColor('#fff3cd')
        text_color = colors.HexColor('#856404')
        border_color = colors.HexColor('#ffeeba')
        verdict = "MODERATE RISK - Clinical Monitoring Advised"
    else:
        card_color = colors.HexColor('#f8d7da')
        text_color = colors.HexColor('#721c24')
        border_color = colors.HexColor('#f5c6cb')
        verdict = "HIGH RISK - Heart Disease Risk Detected"
        
    result_style = ParagraphStyle(
        'ResultVerdict',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=text_color
    )
    
    result_desc = ParagraphStyle(
        'ResultDesc',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=text_color
    )
    
    result_data = [
        [Paragraph(verdict, result_style)],
        [Paragraph(f"AI Model Heart Disease Risk Score: <b>{risk_score:.1f}%</b>", result_desc)]
    ]
    
    rt = Table(result_data, colWidths=[520])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), card_color),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 2, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(rt)
    story.append(Spacer(1, 20))
    
    # Recommendation Block
    story.append(Paragraph("Clinical Recommendations", section_title))
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontSize=10,
        leading=15,
        leftIndent=15,
        firstLineIndent=-15,
        spaceAfter=8
    )
    
    for rec in recommendation:
        story.append(Paragraph(f"• {rec}", bullet_style))
        
    story.append(Spacer(1, 30))
    
    # Footer disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#94a3b8'),
        alignment=1
    )
    story.append(Paragraph("Disclaimer: This report is generated by a Machine Learning model (Random Forest Classifier) for screening purposes. It is NOT a diagnostic medical certificate. Please consult a qualified cardiologist or healthcare professional for clinical decisions.", disclaimer_style))
    
    doc.build(story)


def db_init():
    """Initializes the SQLite database and creates the prediction history table if not exists."""
    import sqlite3
    db_path = os.path.join("data", "predictions.db")
    # Ensure data folder exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patient_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age REAL,
            sex REAL,
            chest_pain_type REAL,
            resting_blood_pressure REAL,
            cholestoral REAL,
            fasting_blood_sugar REAL,
            rest_ecg REAL,
            Max_heart_rate REAL,
            exercise_induced_angina REAL,
            oldpeak REAL,
            slope REAL,
            vessels_colored_by_flourosopy REAL,
            thalassemia REAL,
            risk_score REAL,
            verdict TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction_to_db(inputs, risk_score, verdict):
    """Saves a patient prediction record to the SQLite database."""
    import sqlite3
    db_init() # Ensure table is created
    db_path = os.path.join("data", "predictions.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patient_predictions (
            age, sex, chest_pain_type, resting_blood_pressure, cholestoral,
            fasting_blood_sugar, rest_ecg, Max_heart_rate, exercise_induced_angina,
            oldpeak, slope, vessels_colored_by_flourosopy, thalassemia, risk_score, verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        inputs['age'], inputs['sex'], inputs['chest_pain_type'], inputs['resting_blood_pressure'], inputs['cholestoral'],
        inputs['fasting_blood_sugar'], inputs['rest_ecg'], inputs['Max_heart_rate'], inputs['exercise_induced_angina'],
        inputs['oldpeak'], inputs['slope'], inputs['vessels_colored_by_flourosopy'], inputs['thalassemia'],
        risk_score, verdict
    ))
    conn.commit()
    conn.close()


def fetch_prediction_history():
    """Fetches the complete prediction history from the SQLite database and returns it as a DataFrame."""
    import sqlite3
    db_init()
    db_path = os.path.join("data", "predictions.db")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM patient_predictions ORDER BY timestamp DESC", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def clear_prediction_history():
    """Clears all records in the patient_predictions SQLite table."""
    import sqlite3
    db_init()
    db_path = os.path.join("data", "predictions.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_predictions")
    conn.commit()
    conn.close()

