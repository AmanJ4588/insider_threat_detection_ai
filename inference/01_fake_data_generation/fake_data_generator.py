import pandas as pd
import numpy as np

def create_fake_data_function(file_path):
    # 1. Load the data
    try:
        df = pd.read_csv(file_path)
        print(f"# Successfully loaded data with {len(df)} rows.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Identify Features
    # Exclude non-feature columns
    exclude_cols = ['user_id', 'anomaly_score', 'is_insider', 
                    'psychometric_employee_name', 'ldap_employee_name', 
                    'role_changed', 'department_changed', 'team_changed'] 
    
    features = [c for c in df.columns if c not in exclude_cols]
    
    # 3. Separate Classes
    benign_df = df[df['is_insider'] == 0]
    insider_df = df[df['is_insider'] == 1]
    
    # --- PART A: Print the Python Code ---
    print("\n    def generate_fake_user_row(self, is_insider=False):")
    print('        """')
    print('        Auto-generated function based on real training data statistics.')
    print('        """')
    print("        if not is_insider:")
    print("            # --- Generate BENIGN User (Class 0) ---")
    print("            fake_data = {")
    print(f"                'user_id': 'TEST_USER_BENIGN',")
    
    # Generate Benign Logic
    benign_stats = {}
    for col in features:
        min_val = benign_df[col].min()
        max_val = benign_df[col].max()
        benign_stats[col] = (min_val, max_val, pd.api.types.is_integer_dtype(df[col]))
        
        if min_val == max_val:
             print(f"                '{col}': {min_val},")
        elif pd.api.types.is_integer_dtype(df[col]):
            print(f"                '{col}': np.random.randint({int(min_val)}, {int(max_val) + 1}),")
        else:
            print(f"                '{col}': np.random.uniform({min_val:.6f}, {max_val:.6f}),")
            
    print("            }")
    
    print("        else:")
    print("            # --- Generate INSIDER User (Class 1) ---")
    print("            fake_data = {")
    print(f"                'user_id': 'TEST_USER_INSIDER',")
    
    # Generate Insider Logic
    insider_stats = {}
    for col in features:
        min_val = insider_df[col].min()
        max_val = insider_df[col].max()
        insider_stats[col] = (min_val, max_val, pd.api.types.is_integer_dtype(df[col]))
        
        if min_val == max_val:
             print(f"                '{col}': {min_val},")
        elif pd.api.types.is_integer_dtype(df[col]):
            print(f"                '{col}': np.random.randint({int(min_val)}, {int(max_val) + 1}),")
        else:
            print(f"                '{col}': np.random.uniform({min_val:.6f}, {max_val:.6f}),")
            
    print("            }")
    print("        ")
    print("        # Add back any static required columns with defaults")
    print("        fake_data['role_changed'] = 0")
    print("        fake_data['department_changed'] = 0")
    print("        fake_data['team_changed'] = 0")
    print("        ")
    print("        return pd.DataFrame([fake_data])")

    # --- PART B: Generate and Save Fake Data to CSV ---
    print("\n# Generating fake_data_mixed_users.csv...")
    
    mixed_data = []
    
    # Generate 500 Benign Users
    for i in range(500):
        row = {'user_id': f'FAKE_BENIGN_{i}', 'is_insider': 0}
        for col, (min_v, max_v, is_int) in benign_stats.items():
            if min_v == max_v:
                row[col] = min_v
            elif is_int:
                row[col] = np.random.randint(int(min_v), int(max_v) + 1)
            else:
                row[col] = np.random.uniform(min_v, max_v)
        # Add defaults
        row['role_changed'] = 0
        row['department_changed'] = 0
        row['team_changed'] = 0
        mixed_data.append(row)
        
    # Generate 500 Insider Users
    for i in range(500):
        row = {'user_id': f'FAKE_INSIDER_{i}', 'is_insider': 1}
        for col, (min_v, max_v, is_int) in insider_stats.items():
            if min_v == max_v:
                row[col] = min_v
            elif is_int:
                row[col] = np.random.randint(int(min_v), int(max_v) + 1)
            else:
                row[col] = np.random.uniform(min_v, max_v)
        # Add defaults
        row['role_changed'] = 0
        row['department_changed'] = 0
        row['team_changed'] = 0
        mixed_data.append(row)
        
    # Create DataFrame and Save
    fake_df = pd.DataFrame(mixed_data)
    
    # Reorder columns to match original if possible (optional but nice)
    cols = ['user_id'] + features + ['role_changed', 'department_changed', 'team_changed', 'is_insider']
    fake_df = fake_df[cols]
    
    fake_df.to_csv('dataset/data_for_inference/fake_data_mixed_users.csv', index=False)
    print(f"# Successfully saved 1000 rows to 'dataset/data_for_inference/fake_data_mixed_users.csv'")

if __name__ == "__main__":
    file_path = "dataset/training_validation_set_v2/train_val_set.csv" 
    create_fake_data_function(file_path)