import pandas as pd
import numpy as np
import pickle
import shap
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
IFOREST_MODEL_PATH = 'models/unsupervised/iforest_model.pkl'
XGBOOST_MODEL_PATH = 'models/supervised/XGBClassifier_model.pkl'
DATA_PATH = 'dataset/data_for_inference/fake_data_mixed_users.csv'

def load_resources():
    print("Loading models and data...")
    with open(IFOREST_MODEL_PATH, 'rb') as f:
        iforest = pickle.load(f)
    with open(XGBOOST_MODEL_PATH, 'rb') as f:
        xgb = pickle.load(f)
    
    df = pd.read_csv(DATA_PATH)
    print("Resources loaded.")
    return iforest, xgb, df

def print_feature_importance(shap_vals, feature_names, title):
    """Helper to print sorted feature contributions textually."""
    # Create a list of (feature, impact) tuples
    feature_importance = list(zip(feature_names, shap_vals))
    
    # Sort by absolute impact (magnitude)
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n--- {title} ---")
    print(f"{'Feature':<30} | {'Impact (SHAP Value)':<20}")
    print("-" * 55)
    
    # Print top 5 drivers
    for feature, impact in feature_importance[:5]:
        direction = "(Increases Risk)" if impact > 0 else "(Decreases Risk)"
        print(f"{feature:<30} | {impact:>10.4f} {direction}")

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. SETUP
    iforest_model, xgb_model, df = load_resources()

    # 2. PREPARE A TARGET USER
    # We need to run inference first to find a "High Risk" user to explain
    print("\nRunning preliminary inference to find a candidate...")
    
    # Prepare data for IForest
    feats_if = df[iforest_model.feature_names_in_]
    
    # Get Anomaly Score
    raw_scores = iforest_model.decision_function(feats_if)
    df['anomaly_score'] = raw_scores * -1
    
    # Prepare data for XGBoost (must include anomaly_score)
    feats_xgb = df[xgb_model.feature_names_in_]
    
    # Get Risk Score
    df['risk_score'] = xgb_model.predict_proba(feats_xgb)[:, 1]
    
    # Pick the user with the HIGHEST risk score for this demo
    target_row_index = df['risk_score'].idxmax()
    target_row = df.loc[[target_row_index]] # Keep as DataFrame (1 row)
    
    print(f"\nAnalyzing Target User: {target_row['user_id'].values[0]}")
    print(f"Calculated Risk Score:   {target_row['risk_score'].values[0]:.4f}")
    print(f"Calculated Anomaly Score: {target_row['anomaly_score'].values[0]:.4f}")

    # =========================================================================
    # STAGE 1: EXPLAIN THE RISK (XGBoost)
    # Goal: Did the model flag this person because of their Role/Dept 
    #       or because of their Anomaly Score?
    # =========================================================================
    print("\n" + "="*60)
    print("STAGE 1: SUPERVISED EXPLANATION (Why is Risk High?)")
    print("="*60)

    # 1. Get input vector for XGBoost
    input_xgb = target_row[xgb_model.feature_names_in_]
    
    # 2. Create Explainer
    # rapid optimization: we don't pass the whole dataset to TreeExplainer, just the model
    xgb_explainer = shap.TreeExplainer(xgb_model)
    
    # 3. Get SHAP values
    xgb_shap_values = xgb_explainer.shap_values(input_xgb)
    
    # Handle list vs array (binary classification output varies by version)
    if isinstance(xgb_shap_values, list):
        xgb_shap_vals_target = xgb_shap_values[1][0] # Class 1 (Insider), Row 0
    else:
        xgb_shap_vals_target = xgb_shap_values[0] # Row 0

    # 4. Print Report
    print_feature_importance(xgb_shap_vals_target, input_xgb.columns, "Top Risk Drivers")

    # =========================================================================
    # STAGE 2: EXPLAIN THE ANOMALY (Isolation Forest)
    # Goal: If 'anomaly_score' was a top driver in Stage 1, we must explain
    #       WHAT raw behaviors caused that anomaly score.
    # =========================================================================
    
    # Check if anomaly_score is in the top 3 contributors
    # (Re-calculating sort locally to check rank)
    impacts = list(zip(input_xgb.columns, xgb_shap_vals_target))
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3_features = [x[0] for x in impacts[:3]]
    
    print("\n" + "="*60)
    print("STAGE 2: UNSUPERVISED EXPLANATION (Why is Anomaly Score High?)")
    print("="*60)

    if 'anomaly_score' in top_3_features:
        print(">> Trigger: 'anomaly_score' is a key driver of risk. Drilling down...")
        
        # 1. Get input vector for IForest (Raw logs only)
        input_if = target_row[iforest_model.feature_names_in_]
        
        # 2. Create Explainer
        if_explainer = shap.TreeExplainer(iforest_model)
        
        # 3. Get SHAP values
        if_shap_values = if_explainer.shap_values(input_if)
        
        # Isolation forest usually returns shape (n_samples, n_features)
        if_shap_vals_target = if_shap_values[0]
        
        # 4. Print Report
        print_feature_importance(if_shap_vals_target, input_if.columns, "Top Anomaly Drivers")
        
    else:
        print(">> Skip: 'anomaly_score' was NOT a major factor in the risk calculation.")
        print("   The risk is likely driven by static attributes (Dept, Role, etc.) rather than behavior.")