import os
os.environ["PYTHONWARNINGS"] = "ignore"
import unittest
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from utils import (
    load_model_assets, check_scaling_required, predict_risk, generate_pdf_report,
    save_prediction_to_db, fetch_prediction_history, clear_prediction_history
)

class TestMediPredictLogic(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load model and scaler once
        cls.model, cls.scaler = load_model_assets()
        
    def test_model_loaded(self):
        """Test that the pre-trained Random Forest model is loaded correctly."""
        self.assertIsNotNone(self.model, "Failed to load model file.")
        self.assertEqual(self.model.__class__.__name__, "RandomForestClassifier")
        
    def test_scaling_check(self):
        """Test that the auto-scaling checker runs and accurately detects that raw inputs are expected."""
        scaling_req = check_scaling_required(self.model)
        self.assertFalse(scaling_req, "System incorrectly flagged raw model as needing scaled inputs.")

    def test_healthy_patient_prediction(self):
        """Test model outputs for a mock healthy profile."""
        healthy_patient = {
            'age': 30.0,
            'sex': 0.0, # Female (lower risk)
            'chest_pain_type': 2.0, # Non-anginal pain
            'resting_blood_pressure': 110.0,
            'cholestoral': 180.0,
            'fasting_blood_sugar': 0.0,
            'rest_ecg': 0.0,
            'Max_heart_rate': 190.0,
            'exercise_induced_angina': 0.0,
            'oldpeak': 0.0,
            'slope': 2.0,
            'vessels_colored_by_flourosopy': 0.0,
            'thalassemia': 2.0
        }
        pred, risk_score, healthy_score = predict_risk(healthy_patient, self.model, self.scaler)
        
        # In our model Class 1 is Healthy (highest prob for healthy patient)
        self.assertEqual(pred, 1, "Healthy patient predicted as high risk.")
        self.assertLess(risk_score, 50.0, "Healthy patient risk score exceeds 50%.")
        self.assertGreater(healthy_score, 50.0, "Healthy patient health score is below 50%.")
        self.assertAlmostEqual(risk_score + healthy_score, 100.0, places=2, msg="Probabilities do not sum to 100%")

    def test_diseased_patient_prediction(self):
        """Test model outputs for a high-risk profile."""
        diseased_patient = {
            'age': 68.0,
            'sex': 1.0, # Male (higher risk)
            'chest_pain_type': 0.0, # Asymptomatic (highly dangerous CP category)
            'resting_blood_pressure': 160.0,
            'cholestoral': 320.0,
            'fasting_blood_sugar': 1.0,
            'rest_ecg': 1.0,
            'Max_heart_rate': 105.0, # Low heart rate
            'exercise_induced_angina': 1.0, # Yes
            'oldpeak': 3.5, # High ST depression
            'slope': 0.0,
            'vessels_colored_by_flourosopy': 3.0, # Multiple vessels colored
            'thalassemia': 3.0
        }
        pred, risk_score, healthy_score = predict_risk(diseased_patient, self.model, self.scaler)
        
        # Class 0 is Disease Risk (highest prob for high-risk patient)
        self.assertEqual(pred, 0, "High-risk patient predicted as healthy.")
        self.assertGreater(risk_score, 50.0, "High-risk patient risk score is below 50%.")
        self.assertLess(healthy_score, 50.0, "High-risk patient health score exceeds 50%.")
        self.assertAlmostEqual(risk_score + healthy_score, 100.0, places=2, msg="Probabilities do not sum to 100%")

    def test_pdf_generation(self):
        """Test that the PDF report generates successfully and contains data."""
        mock_patient = {
            'age': 45.0, 'sex': 1.0, 'chest_pain_type': 1.0, 'resting_blood_pressure': 120.0,
            'cholestoral': 210.0, 'fasting_blood_sugar': 0.0, 'rest_ecg': 0.0, 'Max_heart_rate': 160.0,
            'exercise_induced_angina': 0.0, 'oldpeak': 0.8, 'slope': 1.0, 'vessels_colored_by_flourosopy': 0.0,
            'thalassemia': 2.0
        }
        recs = ["Maintain diet", "Exercise"]
        test_pdf = "test_clinical_report.pdf"
        
        # If exists, delete first
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
            
        try:
            generate_pdf_report(mock_patient, 25.5, recs, test_pdf)
            self.assertTrue(os.path.exists(test_pdf), "PDF report file was not created.")
            self.assertGreater(os.path.getsize(test_pdf), 0, "PDF report file is empty.")
        finally:
            if os.path.exists(test_pdf):
                os.remove(test_pdf)

    def test_sqlite_operations(self):
        """Test database operations (saving, fetching, clearing history)."""
        # Clear database history first
        clear_prediction_history()
        
        # Check initial empty state
        initial_history = fetch_prediction_history()
        self.assertTrue(initial_history.empty, "Database history should be empty initially.")
        
        # Save a mock prediction
        mock_inputs = {
            'age': 40.0, 'sex': 1.0, 'chest_pain_type': 2.0, 'resting_blood_pressure': 120.0,
            'cholestoral': 200.0, 'fasting_blood_sugar': 0.0, 'rest_ecg': 0.0, 'Max_heart_rate': 160.0,
            'exercise_induced_angina': 0.0, 'oldpeak': 0.5, 'slope': 1.0, 'vessels_colored_by_flourosopy': 0.0,
            'thalassemia': 2.0
        }
        mock_risk = 15.4
        mock_verdict = "Low Risk"
        
        save_prediction_to_db(mock_inputs, mock_risk, mock_verdict)
        
        # Fetch predictions and assert
        history_df = fetch_prediction_history()
        self.assertEqual(len(history_df), 1, "Failed to record prediction row in database.")
        self.assertEqual(history_df.iloc[0]['age'], 40.0)
        self.assertEqual(history_df.iloc[0]['risk_score'], 15.4)
        self.assertEqual(history_df.iloc[0]['verdict'], "Low Risk")
        
        # Clear records
        clear_prediction_history()
        history_after_clear = fetch_prediction_history()
        self.assertTrue(history_after_clear.empty, "Failed to clear records in database.")

if __name__ == '__main__':
    unittest.main()
