import pandas as pd
import numpy as np
import pickle
import shap
import os
import logging
import warnings

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

class InsiderThreatSystem:
    """
    Unified class for Insider Threat Detection.
    Handles:
    1. Model Loading
    2. Hybrid Inference (Unsupervised + Supervised)
    3. Two-Stage XAI
    """
    
    # --- CONFIGURATION ---
    # The calculated optimal threshold from training (F1-score maximization)
    OPTIMAL_THRESHOLD = 0.3207 

    def __init__(self, iforest_path, xgboost_path):
        self.iforest_path = iforest_path
        self.xgboost_path = xgboost_path
        
        # Load Models
        self.if_model, self.xgb_model = self._load_models()
        
        # Initialize Explainers
        logging.info("Initializing SHAP explainers (this maximizes inference speed later)...")
        try:
            self.xgb_explainer = shap.TreeExplainer(self.xgb_model)
            self.if_explainer = shap.TreeExplainer(self.if_model)
            logging.info(f"System Ready. Optimal Threshold set to: {self.OPTIMAL_THRESHOLD}")
        except Exception as e:
            logging.error(f"Failed to initialize explainers: {e}")
            raise e

    def _load_models(self):
        try:
            if not os.path.exists(self.iforest_path) or not os.path.exists(self.xgboost_path):
                raise FileNotFoundError("Model files not found.")
                
            with open(self.iforest_path, 'rb') as f:
                if_model = pickle.load(f)
            with open(self.xgboost_path, 'rb') as f:
                xgb_model = pickle.load(f)
                
            return if_model, xgb_model
        except Exception as e:
            logging.error(f"Error loading models: {e}")
            raise e

    def _validate_columns(self, df, model):
        """Reorders columns to match model expectations."""
        expected = model.feature_names_in_
        missing = set(expected) - set(df.columns)
        if missing:
            raise ValueError(f"Input missing columns: {missing}")
        return df[expected]

    def _format_shap(self, shap_vals, features, row_values):
        """Helper to format SHAP output for JSON response."""
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        elif len(shap_vals.shape) > 1 and shap_vals.shape[1] > 1:
            vals = shap_vals[:, 1]
        else:
            vals = shap_vals
            
        if len(vals.shape) > 1: vals = vals[0]
        
        contributions = []
        for name, impact, val in zip(features, vals, row_values):
            contributions.append({
                "feature": name,
                "impact": float(impact),
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val)
            })
        
        contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
        return contributions

    def _normalize_anomaly_score(self, raw_decision):
        """Converts Isolation Forest decision_function to 0-100 scale."""
        score = 0.5 - raw_decision 
        score = np.clip(score, 0, 1)
        return score * 100

    def analyze_user(self, data_dict):
        """
        Main entry point. Accepts a dictionary of features (one user), 
        runs inference, and returns a result dictionary.
        """
        # Convert dict to DataFrame (1 row)
        df = pd.DataFrame([data_dict])
        
        result = {
            "user_id": data_dict.get('user_id', 'Unknown'),
            "risk_score": 0.0,
            "anomaly_score": 0.0,
            "verdict": "Safe",
            "threshold_used": self.OPTIMAL_THRESHOLD, # Returned for UI reference
            "explanation": {}
        }

        # --- 1. INFERENCE ---
        try:
            # A. Unsupervised (Isolation Forest)
            if_feats = self._validate_columns(df.drop(columns=['user_id'], errors='ignore'), self.if_model)
            raw_if_score = self.if_model.decision_function(if_feats)
            anomaly_score_scaled = self._normalize_anomaly_score(raw_if_score)
            
            # Add raw score to dataframe for XGBoost
            df['anomaly_score'] = raw_if_score * -1 
            
            # B. Supervised (XGBoost)
            xgb_feats = self._validate_columns(df.drop(columns=['user_id'], errors='ignore'), self.xgb_model)
            
            # Get Probability (0.0 to 1.0)
            risk_prob = self.xgb_model.predict_proba(xgb_feats)[:, 1]
            raw_probability = float(risk_prob[0])
            
            # Assign Results
            result['anomaly_score'] = float(anomaly_score_scaled[0])
            result['risk_score'] = raw_probability * 100 # Display as percentage (0-100)
            
            # --- verdict logic (Using Optimal Threshold) ---
            # Any probability >= optimal threshold is considered an Insider
            if raw_probability >= self.OPTIMAL_THRESHOLD:
                result['verdict'] = "High Risk"
            else:
                # We also add a 'Medium' buffer zone (within 10% of threshold)
                if raw_probability >= (self.OPTIMAL_THRESHOLD - 0.10):
                    result['verdict'] = "Medium Risk" # Close to threshold
                else:
                    result['verdict'] = "Safe"
                
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # --- 2. XAI (Two-Stage) ---
        try:
            # Stage 1: XGBoost Explanation
            xgb_shap = self.xgb_explainer.shap_values(xgb_feats)
            stage_1_drivers = self._format_shap(xgb_shap, xgb_feats.columns, xgb_feats.iloc[0])
            result['explanation']['risk_drivers'] = stage_1_drivers[:5]
            
            # Stage 2: Check if Anomaly Score is a top driver
            top_3_features = [x['feature'] for x in stage_1_drivers[:3]]
            
            if 'anomaly_score' in top_3_features:
                result['explanation']['anomaly_drill_down'] = "Triggered"
                if_shap = self.if_explainer.shap_values(if_feats)
                stage_2_drivers = self._format_shap(if_shap, if_feats.columns, if_feats.iloc[0])
                result['explanation']['anomaly_reasons'] = stage_2_drivers[:5]
            else:
                result['explanation']['anomaly_drill_down'] = "Skipped"
                
        except Exception as e:
            logging.error(f"XAI failed: {e}")
            result['explanation']['error'] = "Explanation generation failed"

        return result