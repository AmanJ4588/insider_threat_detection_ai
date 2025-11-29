import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
IFOREST_MODEL_PATH = 'models/unsupervised/iforest_model.pkl'
XGBOOST_MODEL_PATH = 'models/supervised/XGBClassifier_model.pkl'

def generate_fake_user_row(is_insider=False):
    """
    Generates a single row of fake aggregated data for testing.
    The values are random but within plausible ranges based on the dataset.
    
    Args:
        is_insider (bool): If True, generates an anomalous "insider" profile.
    """
    fake_data = {
        # --- ID (Not used for prediction but needed for tracking) ---
        'user_id': 'TEST_USER_INSIDER' if is_insider else 'TEST_USER_BENIGN',
        
        # --- Logon Features ---
        'total_logon_events': np.random.randint(500, 2000),
        'logon_unique_pcs': np.random.randint(1, 5),
        'after_hours_logons': np.random.randint(0, 100),
        'weekend_logons': np.random.randint(0, 50),
        'logon_ratio': np.random.uniform(0.4, 0.6), # ~50% logons vs logoffs
        'logon_after_hours_ratio': np.random.uniform(0.0, 0.2),
        'logon_weekend_ratio': np.random.uniform(0.0, 0.1),
        
        # --- HTTP Features ---
        'total_http_events': np.random.randint(10000, 50000),
        'http_unique_pcs': np.random.randint(1, 3),
        'unique_urls_visited': np.random.randint(100, 500),
        'after_hours_http': np.random.randint(0, 500),
        'weekend_http': np.random.randint(0, 200),
        'http_after_hours_ratio': np.random.uniform(0.0, 0.1),
        'http_weekend_ratio': np.random.uniform(0.0, 0.05),
        
        # --- Email Features ---
        'total_emails': np.random.randint(1000, 5000),
        'email_unique_pcs': np.random.randint(1, 2),
        'unique_recipients': np.random.randint(50, 500),
        'unique_cc': np.random.randint(10, 100),
        'unique_bcc': np.random.randint(0, 5),
        'unique_senders': np.random.randint(1, 5), # Usually just themselves
        'total_email_size': np.random.uniform(1e7, 1e9), # 10MB - 1GB
        'avg_email_size': np.random.uniform(10000, 50000), # 10KB - 50KB
        'total_attachments': np.random.randint(0, 500),
        'avg_attachments': np.random.uniform(0.0, 1.5),
        
        # --- Psychometric Features (1-5 scale) ---
        'openness': np.random.randint(20, 50),
        'conscientiousness': np.random.randint(20, 50),
        'extraversion': np.random.randint(20, 50),
        'agreeableness': np.random.randint(20, 50),
        'neuroticism': np.random.randint(20, 50),
        
        # --- LDAP Features ---
        'unique_roles': 1,
        'unique_business_units': 1,
        'unique_functional_units': 1,
        'unique_departments': 1,
        'unique_teams': 1,
        'unique_supervisors': 1
    }
    
    # --- Inject Insider Anomalies ---
    if is_insider:
        # Increase suspicious activity significantly
        fake_data['after_hours_logons'] = np.random.randint(200, 500)  # High after-hours activity
        fake_data['logon_after_hours_ratio'] = np.random.uniform(0.4, 0.8)
        
        fake_data['total_http_events'] = np.random.randint(80000, 150000) # Excessive browsing
        fake_data['unique_urls_visited'] = np.random.randint(800, 2000)   # Visiting many new sites
        
        fake_data['total_email_size'] = np.random.uniform(2e9, 5e9) # Large data exfiltration via email
        fake_data['total_attachments'] = np.random.randint(800, 2000) # Many attachments
        
        # Lower conscientiousness (often associated with rule-breaking)
        fake_data['conscientiousness'] = np.random.randint(10, 25)

    # Convert to DataFrame (1 row)
    return pd.DataFrame([fake_data])


def run_unsupervised_inference(df_row):
    """
    Loads the Isolation Forest model and assigns an anomaly score.
    """
    print("--- Unsupervised Inference Started ---")
    
    try:
        # 1. Load Model
        with open(IFOREST_MODEL_PATH, 'rb') as f:
            iforest_model = pickle.load(f)
        print("Model loaded successfully.")
        
        # 2. Select Features 
        # drop user_id as the model doesn't use it
        features_for_model = df_row.drop(columns=['user_id'])
        
        # Verify columns match training data
        expected_cols = iforest_model.feature_names_in_
        missing_cols = set(expected_cols) - set(features_for_model.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing columns expected by the model: {missing_cols}")
            
        extra_cols = set(features_for_model.columns) - set(expected_cols)
        if extra_cols:
            print(f"Warning: Input data has extra columns that will be ignored: {extra_cols}")
            
        
        features_for_model = features_for_model[expected_cols]
        
        # 3. Predict Score [.decision_function() gives raw score (negative = anomaly)]
        raw_score = iforest_model.decision_function(features_for_model)
        
        # 4. Invert Score (Higher = More Anomalous)
        anomaly_score = raw_score[0] * -1
        
        print(f"Raw Score: {raw_score[0]:.4f}")
        print(f"Calculated Anomaly Score: {anomaly_score:.4f}")
        
        # 5. Add to DataFrame
        df_row['anomaly_score'] = anomaly_score
        
        return df_row

    except FileNotFoundError:
        print(f"ERROR: Model file '{IFOREST_MODEL_PATH}' not found.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def run_supervised_inference(df_row):
    """
    Loads the XGBoost model and predicts the Insider Risk Score.
    Requires 'anomaly_score' to be present in df_row.
    """
    print("\n--- Supervised Inference Started ---")
    
    try:
        # 1. Check for anomaly_score
        if 'anomaly_score' not in df_row.columns:
            raise ValueError("Missing 'anomaly_score'. Run unsupervised inference first.")

        # 2. Load Model
        with open(XGBOOST_MODEL_PATH, 'rb') as f:
            xgb_model = pickle.load(f)
        print("Supervised Model loaded successfully.")
        
        # 3. Select Features
        # Drop user_id, keep everything else (including anomaly_score)
        features_for_model = df_row.drop(columns=['user_id'])
        
        # Reorder columns to match training data
        expected_cols = xgb_model.feature_names_in_
        features_for_model = features_for_model[expected_cols]
        
        # 4. Predict Probability (Risk Score)
        # Class 1 is 'Insider', so we take the second probability [:, 1]
        risk_probability = xgb_model.predict_proba(features_for_model)[:, 1][0]
        
        print(f"Calculated Insider Risk Score: {risk_probability:.4f}")
        
        # 5. Add to DataFrame
        df_row['risk_score'] = risk_probability
        
        return df_row

    except FileNotFoundError:
        print(f"ERROR: Model file '{XGBOOST_MODEL_PATH}' not found.")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None



if __name__ == "__main__":
    # Generate Fake Data
    print("Generating fake user data...") 
    test_row = generate_fake_user_row(False) #true to generate fake data for a insider, false to generate fake data for a normal user
    print("Data generated:")
    print(test_row.T) # Transpose for easier reading
    
    # Run Unsupervised Inference
    result_row = run_unsupervised_inference(test_row)
    
    # 3. Run Supervised Inference (Only if step 2 succeeded)
    if test_row is not None:
        test_row = run_supervised_inference(test_row)
    
    # 4. Show Final Result
    if test_row is not None:
        print("\n" + "="*40)
        print("FINAL INFERENCE REPORT")
        print("="*40)
        print(f"User ID:       {test_row['user_id'].values[0]}")
        print(f"Anomaly Score: {test_row['anomaly_score'].values[0]:.4f}")
        print(f"Probability of Insider {test_row['risk_score'].values[0]:.4f} ")
        print(f"Risk Score:    { (test_row['risk_score'].values[0] * 100):.4f} ")
        print("="*40) 