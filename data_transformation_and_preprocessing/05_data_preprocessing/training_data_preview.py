import duckdb
import pandas as pd
import os
import sys


DUCKDB_FILE_PATH = 'dataset/supervised_dataset.duckdb' 
TRAINING_TABLE_NAME = 'training_data_table'

# ------------------------------------\

def review_data(db_path, table_name):
    """
    Connects to the DuckDB file and runs multiple queries to
    review the final training_data_table.
    """
    
    # Check if DuckDB file exists
    if not os.path.exists(db_path):
        print(f"Error: DuckDB file not found at {db_path}")
        print("Please update DUCKDB_FILE_PATH at the top of the script.")
        return

    print(f"--- Starting Review of '{table_name}' in {db_path} ---")

    try:
        # Connect to your persistent DuckDB file in read-only mode
        con = duckdb.connect(database=db_path, read_only=True)

        # 1. Show first 10 rows
        print("\n--- 1. First 10 Rows (Sample) ---")
        sample_df = con.sql(f"SELECT * FROM {table_name} LIMIT 10").df()
        print(sample_df.to_string()) # .to_string() gives better formatting

        # 2. Show schema (column names and types)
        print("\n--- 2. Table Schema (Columns and Data Types) ---")
        schema_df = con.sql(f"PRAGMA table_info('{table_name}')").df()
        print(schema_df.to_string())

        # 3. Show table statistics
        print("\n--- 3. Table Statistics ---")
        total_rows = con.sql(f"SELECT COUNT(*) FROM {table_name}").df().iloc[0, 0]
        print(f"Total rows in table: {total_rows:,}")

        # 4. Show label distribution
        print("\n--- 4. Label Distribution ---")
        label_dist = con.sql(f"SELECT label, COUNT(*) AS count FROM {table_name} GROUP BY label ORDER BY label").df()
        label_dist['percentage'] = (label_dist['count'] / total_rows) * 100
        label_dist['percentage'] = label_dist['percentage'].map('{:,.2f}%'.format)
        print(label_dist.to_string())

        # --- 4a. Show scenario distribution ---
        print("\n--- 4a. Scenario Distribution (0 = Normal) ---")
        try:
            scenario_dist = con.sql(f"SELECT scenario, COUNT(*) AS count FROM {table_name} GROUP BY scenario ORDER BY scenario").df()
            scenario_dist['percentage'] = (scenario_dist['count'] / total_rows) * 100
            scenario_dist['percentage'] = scenario_dist['percentage'].map('{:,.2f}%'.format)
            print(scenario_dist.to_string())
        except duckdb.CatalogException:
            print("  'scenario' column not found (this is ok if you haven't run the scenario join yet).")
        # --------------------------------------------

        # 5. Show unique users
        print("\n--- 5. Unique Users ---")
        unique_users = con.sql(f"SELECT COUNT(DISTINCT user_id) FROM {table_name}").df().iloc[0, 0]
        print(f"Total unique users: {unique_users:,}")

        # 6. Show date range
        print("\n--- 6. Data Time Range ---")
        date_range = con.sql(f"SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}").df()
        print(f"  Oldest event: {date_range.iloc[0, 0]}")
        print(f"  Newest event: {date_range.iloc[0, 1]}")

        # 7. Check for any NULL values (should be 0)
        print("\n--- 7. Null Value Check (per column) ---")
        null_checks = []
        # Get columns from the PRAGMA call, as sample_df might be empty
        all_columns = schema_df['name'].tolist() 
        
        for col in all_columns:
            null_count = con.sql(f"SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL").df().iloc[0, 0]
            null_checks.append({'column': col, 'null_count': null_count})
        
        null_df = pd.DataFrame(null_checks)
        print(null_df.to_string())

        con.close()
        print("\n--- Review Complete ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if 'con' in locals():
            con.close()
        sys.exit(1)

def main():
    review_data(DUCKDB_FILE_PATH, TRAINING_TABLE_NAME)

if __name__ == "__main__":
    main()