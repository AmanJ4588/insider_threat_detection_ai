import pandas as pd
import numpy as np
import pickle
import warnings
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
IFOREST_MODEL_PATH = 'models/unsupervised/iforest_model.pkl'
XGBOOST_MODEL_PATH = 'models/supervised/XGBClassifier_model.pkl'
DATA_PATH = 'dataset/data_for_inference/fake_data_mixed_users.csv'

def load_models():
    """Loads both models from disk."""
    print("Loading models...")
    try:
        if not os.path.exists(IFOREST_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {IFOREST_MODEL_PATH}")
        if not os.path.exists(XGBOOST_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {XGBOOST_MODEL_PATH}")

        with open(IFOREST_MODEL_PATH, 'rb') as f:
            iforest_model = pickle.load(f)
        
        with open(XGBOOST_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
            
        print("Models loaded successfully.")
        return iforest_model, xgb_model
        
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def run_inference_pipeline(df, iforest_model, xgb_model):
    """
    Runs the full inference pipeline (Unsupervised -> Supervised) on a DataFrame.
    """
    # Create a copy to work on
    results_df = df.copy()
    
    # --- 1. Unsupervised Inference ---
    print("--- Starting Unsupervised Inference ---")
    
    # Prepare features: Drop IDs and Labels
    features_unsupervised = results_df.drop(columns=['user_id', 'is_insider'], errors='ignore')
    
    # Check columns for Isolation Forest
    expected_cols_if = iforest_model.feature_names_in_
    missing_cols = set(expected_cols_if) - set(features_unsupervised.columns)
    if missing_cols:
        raise ValueError(f"Input data missing columns for Unsupervised model: {missing_cols}")
    
    # Reorder columns
    features_unsupervised = features_unsupervised[expected_cols_if]
    
    # Predict Anomaly Score
    raw_score = iforest_model.decision_function(features_unsupervised)
    anomaly_score = raw_score * -1 # Invert so higher is more anomalous
    
    # Add to DataFrame
    results_df['anomaly_score'] = anomaly_score
    print(f"Anomaly Scores generated. Mean: {np.mean(anomaly_score):.4f}")

    # --- 2. Supervised Inference ---
    print("--- Starting Supervised Inference ---")
    
    # Prepare features: Drop IDs and Labels (keep anomaly_score)
    features_supervised = results_df.drop(columns=['user_id', 'is_insider'], errors='ignore')
    
    # Check columns for XGBoost
    expected_cols_xgb = xgb_model.feature_names_in_
    missing_cols = set(expected_cols_xgb) - set(features_supervised.columns)
    if missing_cols:
        raise ValueError(f"Input data missing columns for Supervised model: {missing_cols}")
        
    # Reorder columns
    features_supervised = features_supervised[expected_cols_xgb]
    
    # Predict Risk Probability
    # Class 1 is 'Insider', so we take column [:, 1]
    risk_prob = xgb_model.predict_proba(features_supervised)[:, 1]
    
    # Add to DataFrame
    results_df['risk_score'] = risk_prob
    print(f"Risk Scores generated. Mean: {np.mean(risk_prob):.4f}")
    
    return results_df

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Models
    iforest, xgb = load_models()
    
    if iforest is not None and xgb is not None:
        # 2. Load Test Data
        if os.path.exists(DATA_PATH):
            print(f"\nLoading test data from {DATA_PATH}...")
            test_df = pd.read_csv(DATA_PATH)
            
            # 3. Select a Sample (Mix of Benign and Insider if labels exist)
            if 'is_insider' in test_df.columns:
                benign_sample = test_df[test_df['is_insider'] == 0].head(10)
                insider_sample = test_df[test_df['is_insider'] == 1].head(10)
                sample_data = pd.concat([benign_sample, insider_sample])
            else:
                sample_data = test_df.head(20)

            print(f"Running inference on {len(sample_data)} sample rows...")
            
            # 4. Run the Pipeline
            try:
                final_results = run_inference_pipeline(sample_data, iforest, xgb)
                
                # 5. Print Report
                print("\n" + "="*60)
                print("FINAL INFERENCE REPORT (Sample)")
                print("="*60)
                
                # Select columns to display
                display_cols = ['user_id', 'anomaly_score', 'risk_score']
                if 'is_insider' in final_results.columns:
                    display_cols.append('is_insider')
                
                print(final_results[display_cols].to_string(index=False))
                print("="*60)
                
                # Stats
                print(f"\nAnomaly Score Range: Min={final_results['anomaly_score'].min():.4f}, Max={final_results['anomaly_score'].max():.4f}")
                print(f"Risk Score Range:    Min={final_results['risk_score'].min():.4f}, Max={final_results['risk_score'].max():.4f}")

            except Exception as e:
                print(f"Inference failed: {e}")
                
        else:
            print(f"Error: Data file '{DATA_PATH}' not found. Run auto_generate_data_code.py first.")