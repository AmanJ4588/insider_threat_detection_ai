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
    
    # --- CONFIGURATION ---
    OPTIMAL_THRESHOLD = 0.3207 
    
    # --- RULE 1: COMPREHENSIVE DICTIONARY MAPPING ---
    # Covers all ~35 features from dataset
    FEATURE_MAP = {
        # --- LOGON ACTIVITY ---
        "total_logon_events": {"name": "Total Login Volume", "unit": "events"},
        "logon_unique_pcs": {"name": "Unique PCs Accessed", "unit": "devices"},
        "logon_after_hours_ratio": {"name": "After-Hours Work Rate", "unit": "%", "is_ratio": True},
        "logon_weekend_ratio": {"name": "Weekend Login Rate", "unit": "%", "is_ratio": True},
        "is_weekend": {"name": "Weekend Activity", "unit": "bool"},
        "after_hours_logons": {"name": "Late Night Logins", "unit": "events"},
        "weekend_logons": {"name": "Weekend Logins", "unit": "events"},
        "logon_ratio": {"name": "Daily Login Intensity", "unit": "ratio"},

        # --- FILE & HTTP ACTIVITY ---
        "file_access_count": {"name": "File Access Volume", "unit": "files"},
        "unique_urls_visited": {"name": "Web Browsing Diversity", "unit": "URLs"},
        "total_http_events": {"name": "Total Web Traffic", "unit": "requests"},
        "http_unique_pcs": {"name": "PCs used for Web", "unit": "devices"},
        "after_hours_http": {"name": "Late Night Browsing", "unit": "events"},
        "weekend_http": {"name": "Weekend Browsing", "unit": "events"},
        "http_after_hours_ratio": {"name": "After-Hours Web Rate", "unit": "%", "is_ratio": True},
        "http_weekend_ratio": {"name": "Weekend Web Rate", "unit": "%", "is_ratio": True},

        # --- EMAIL ACTIVITY ---
        "total_emails": {"name": "Email Volume", "unit": "emails"},
        "total_email_size": {"name": "Total Email Data Size", "unit": "bytes"},
        "avg_email_size": {"name": "Avg Email Size", "unit": "bytes"},
        "total_attachments": {"name": "Attachment Volume", "unit": "files"},
        "avg_attachments": {"name": "Avg Attachments per Email", "unit": "files"},
        "unique_recipients": {"name": "Recipient Diversity", "unit": "people"},
        "unique_senders": {"name": "Sender Diversity", "unit": "people"},
        "unique_cc": {"name": "CC Diversity", "unit": "people"},
        "unique_bcc": {"name": "BCC Diversity (Hidden)", "unit": "people"},
        "email_unique_pcs": {"name": "PCs used for Email", "unit": "devices"},

        # --- ORGANIZATION / ROLE ---
        "unique_teams": {"name": "Team Hopping Count", "unit": "teams"},
        "unique_departments": {"name": "Department Access Count", "unit": "depts"},
        "unique_business_units": {"name": "Business Unit Access", "unit": "units"},
        "unique_functional_units": {"name": "Functional Unit Access", "unit": "units"},
        "unique_roles": {"name": "Role Switching Count", "unit": "roles"},
        "unique_supervisors": {"name": "Supervisor Switching", "unit": "people"},

        # --- PSYCHOMETRICS (Big 5) ---
        "neuroticism": {"name": "Neuroticism Score", "unit": "/5"},
        "conscientiousness": {"name": "Conscientiousness Score", "unit": "/5"},
        "agreeableness": {"name": "Agreeableness Score", "unit": "/5"},
        "extraversion": {"name": "Extraversion Score", "unit": "/5"},
        "openness": {"name": "Openness Score", "unit": "/5"},

        # --- MODEL SCORES ---
        "anomaly_score": {"name": "Behavioral Anomaly Score", "unit": "/100"}
    }

    def __init__(self, iforest_path, xgboost_path):
        self.iforest_path = iforest_path
        self.xgboost_path = xgboost_path
        self.if_model, self.xgb_model = self._load_models()
        
        logging.info("Initializing SHAP explainers...")
        try:
            self.xgb_explainer = shap.TreeExplainer(self.xgb_model)
            self.if_explainer = shap.TreeExplainer(self.if_model)
            logging.info(f"System Ready. Threshold: {self.OPTIMAL_THRESHOLD}")
        except Exception as e:
            logging.error(f"Failed to initialize explainers: {e}")
            raise e

    def _load_models(self):
        try:
            if not os.path.exists(self.iforest_path) or not os.path.exists(self.xgboost_path):
                raise FileNotFoundError("Model files not found.")
            with open(self.iforest_path, 'rb') as f: if_model = pickle.load(f)
            with open(self.xgboost_path, 'rb') as f: xgb_model = pickle.load(f)
            return if_model, xgb_model
        except Exception as e:
            raise e

    def _validate_columns(self, df, model):
        expected = model.feature_names_in_
        missing = set(expected) - set(df.columns)
        if missing: raise ValueError(f"Input missing columns: {missing}")
        return df[expected]

    def _humanize_value(self, feature_key, raw_value):
        """Rule 2: Convert raw numbers to human units (%, MB, Yes/No)"""
        # FALLBACK: If key not in map, return empty dict {}
        config = self.FEATURE_MAP.get(feature_key, {})
        
        if config.get("unit") == "bool":
            return "Yes" if raw_value == 1 else "No"
        
        if config.get("is_ratio", False):
            return f"{raw_value:.1%}"
        
        if "size" in feature_key and raw_value > 1024:
            return f"{raw_value/1024:.1f} KB"
        
        if isinstance(raw_value, float):
            return f"{raw_value:.2f}"
            
        return str(raw_value)

    def _generate_narrative(self, feature_name, impact, formatted_value):
        """Rule 3: Create a sentence explaining the risk direction."""
        direction = "Increased Risk" if impact > 0 else "Decreased Risk"
        strength = ""
        if abs(impact) > 1.0: strength = "Strongly "
        elif abs(impact) < 0.1: strength = "Slightly "
        
        return f"{feature_name} was {formatted_value}, which {strength}{direction.lower()}."

    def _format_shap(self, shap_vals, features, row_values, raw_probability):
        """
        Applies Humanization rules and Robust Directional Sorting.
        Fixed logic to handle SHAP array shapes correctly.
        """
        # --- ROBUST SHAP EXTRACTION ---
        # XGBoost TreeExplainer returns different shapes depending on version/config.
        # Case 1: List of arrays (Binary/Multi-class in some versions) -> [Array(Class0), Array(Class1)]
        if isinstance(shap_vals, list):
            vals = shap_vals[1]
        else:
            vals = shap_vals

        # Case 2: 3D Array (samples, features, classes) -> rare for standard XGBoost binary
        if len(vals.shape) == 3:
             vals = vals[:, :, 1]

        # Case 3: 2D Array (samples, features) -> Standard XGBoost Binary case
        # We just want the first row (the single user we are analyzing)
        if len(vals.shape) == 2:
            vals = vals[0] 
        
        contributions = []
        
        # Safe Zip: Iterate through features
        for key, impact, raw_val in zip(features, vals, row_values):
            config = self.FEATURE_MAP.get(key, {})
            display_name = config.get("name", key.replace("_", " ").title())
            human_val = self._humanize_value(key, raw_val)
            narrative = self._generate_narrative(display_name, float(impact), human_val)
            
            contributions.append({
                "feature_id": key,
                "display_name": display_name,
                "impact_score": float(impact),
                "raw_value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "human_value": human_val,
                "narrative": narrative
            })
        
        # --- ROBUST SORTING STRATEGY ---
        
        concern_threshold = self.OPTIMAL_THRESHOLD - 0.12 
        
        if raw_probability >= concern_threshold:
            # CASE: High/Medium Risk -> Show what INCREASED risk (Positive Impact)
            positives = sorted(
                [x for x in contributions if x['impact_score'] > 0],
                key=lambda x: x['impact_score'], 
                reverse=True
            )
            
            negatives = sorted(
                [x for x in contributions if x['impact_score'] <= 0],
                key=lambda x: x['impact_score'], 
                reverse=True 
            )
            
            # Combine: Aggravating factors first
            contributions = positives + negatives
            
        else:
            # CASE: Safe -> Show what DECREASED risk (Negative Impact)
            negatives = sorted(
                [x for x in contributions if x['impact_score'] < 0],
                key=lambda x: x['impact_score'], 
                reverse=False 
            )
            
            positives = sorted(
                [x for x in contributions if x['impact_score'] >= 0],
                key=lambda x: x['impact_score'], 
                reverse=False 
            )
            
            # Combine: Mitigating factors first
            contributions = negatives + positives
            
        return contributions

    def _normalize_anomaly_score(self, raw_decision):
        score = 0.5 - raw_decision 
        score = np.clip(score, 0, 1)
        return score * 100

    def analyze_user(self, data_dict):
        df = pd.DataFrame([data_dict])
        
        result = {
            "user_id": data_dict.get('user_id', 'Unknown'),
            "risk_score": 0.0,
            "anomaly_score": 0.0,
            "verdict": "Safe",
            "threshold_used": self.OPTIMAL_THRESHOLD * 100,
            "explanation": {}
        }

        try:
            # Inference
            if_feats = self._validate_columns(df.drop(columns=['user_id'], errors='ignore'), self.if_model)
            raw_if_score = self.if_model.decision_function(if_feats)
            anomaly_score_scaled = self._normalize_anomaly_score(raw_if_score)
            
            df['anomaly_score'] = raw_if_score * -1 
            
            xgb_feats = self._validate_columns(df.drop(columns=['user_id'], errors='ignore'), self.xgb_model)
            risk_prob = self.xgb_model.predict_proba(xgb_feats)[:, 1]
            raw_probability = float(risk_prob[0])
            
            result['anomaly_score'] = float(anomaly_score_scaled[0])
            result['risk_score'] = raw_probability * 100 

            # Verdict
            if raw_probability >= self.OPTIMAL_THRESHOLD:
                result['verdict'] = "High Risk"
                result['action'] = "Immediate Investigation Required"
            elif raw_probability >= (self.OPTIMAL_THRESHOLD - 0.12):
                result['verdict'] = "Medium Risk"
                result['action'] = "Add to Watchlist"
            else:
                result['verdict'] = "Safe"
                result['action'] = "None"
                
        except Exception as e:
            return {"error": f"Inference failed: {str(e)}"}

        # XAI
        try:
            xgb_shap = self.xgb_explainer.shap_values(xgb_feats)
            
            # UPDATED CALL: Passing raw_probability
            stage_1_drivers = self._format_shap(
                xgb_shap, 
                xgb_feats.columns, 
                xgb_feats.iloc[0], 
                raw_probability 
            )
            
            result['explanation']['risk_drivers'] = stage_1_drivers[:5]
            
            top_3_features = [x['feature_id'] for x in stage_1_drivers[:3]]
            
            if 'anomaly_score' in top_3_features:
                result['explanation']['anomaly_drill_down'] = "Triggered"
                if_shap = self.if_explainer.shap_values(if_feats)
                
                # UPDATED CALL: Passing raw_probability (or 1.0 to force showing anomalies)
                # For anomaly drill-down, we ALWAYS want to see what caused the anomaly
                # So we pass 1.0 to force "Descending" sort (Bad things first)
                stage_2_drivers = self._format_shap(
                    if_shap, 
                    if_feats.columns, 
                    if_feats.iloc[0], 
                    1.0 
                )
                result['explanation']['anomaly_reasons'] = stage_2_drivers[:5]
            else:
                result['explanation']['anomaly_drill_down'] = "Skipped"
                
        except Exception as e:
            logging.error(f"XAI failed: {e}")
            result['explanation']['error'] = "Explanation generation failed"

        return result