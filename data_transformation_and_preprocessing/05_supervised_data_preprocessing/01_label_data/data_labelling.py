# Setup and Configuration
import duckdb
import pandas as pd

# --- Configuration ---
DB_PATH = 'dataset/supervised_dataset.duckdb'
FEATURE_TABLE = 'training_table_new_approach' 
LABEL_TABLE = 'insiders'
OUTPUT_TABLE = 'labeled_training_data_new_approach'

print("--- Data Labeling Script Started ---")
print(f"Database: {DB_PATH}")


# Connect and Inspect
try:
    con = duckdb.connect(database=DB_PATH, read_only=False)
    print("Database connection successful.")
    
    # Check if our source tables exist
    tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
    print(f"\nTables in database: {tables}")

    if FEATURE_TABLE not in tables:
        raise ValueError(f"Missing feature table: '{FEATURE_TABLE}'. Did script 03 run?")
    if LABEL_TABLE not in tables:
        raise ValueError(f"Missing label table: '{LABEL_TABLE}'. Please load it.")

    # Inspect the schemas
    print(f"\nSchema for '{FEATURE_TABLE}':")
    print(con.table(FEATURE_TABLE).limit(0).describe())
    
    print(f"\nSchema for '{LABEL_TABLE}' (our labels):")
    print(con.table(LABEL_TABLE).limit(0).describe())
    
    print(f"\nWe will join '{FEATURE_TABLE}.user_id' with '{LABEL_TABLE}.user'")

except Exception as e:
    print(f"--- CRITICAL ERROR: Could not connect or find tables ---")
    print(f"Error: {e}")
    raise


# Labeling Data via SQL JOIN
print(f"\nCreating labeled table: '{OUTPUT_TABLE}'...")


sql_query = f"""
DROP TABLE IF EXISTS {OUTPUT_TABLE};

CREATE TABLE {OUTPUT_TABLE} AS
SELECT
    t.*,  -- Select all columns from the feature table
    CASE
        WHEN i.user IS NOT NULL THEN 1
        ELSE 0
    END AS is_insider  -- New label column
FROM
    {FEATURE_TABLE} t
LEFT JOIN
    (SELECT DISTINCT user FROM {LABEL_TABLE}) i ON t.user_id = i.user;
"""

try:
    con.execute(sql_query)
    print("Successfully created labeled table.")
    
    # Show the schema of the new table
    print(f"\nNew schema for '{OUTPUT_TABLE}':")
    con.table(OUTPUT_TABLE).limit(0).describe()

except Exception as e:
    print(f"--- ERROR during SQL labeling: {e} ---")
    raise


# Validation Step
print("\n--- Running Validation Step ---")

try:
    validation_query = f"""
    SELECT
        is_insider,
        COUNT(*) as user_count
    FROM
        {OUTPUT_TABLE}
    GROUP BY
        is_insider
    ORDER BY
        is_insider;
    """
    
    df_validation = con.query(validation_query).to_df()
    
    print("Label counts in new table:")
    print(df_validation.to_markdown(index=False))
    
    # --- Programmatic Check ---
    
    # Convert results to a dictionary for easy checking: {0: 930, 1: 70}
    counts = df_validation.set_index('is_insider')['user_count'].to_dict()
    
    insider_count = counts.get(1, 0) # Get count for key '1', default to 0
    benign_count = counts.get(0, 0)  # Get count for key '0', default to 0
    
    if insider_count == 70 and benign_count == 930:
        print("\nVALIDATION SUCCESS!")
        print("Dataset contains exactly 70 insiders and 930 benign users.")
    else:
        print("\nVALIDATION FAILED!")
        print(f"  Expected 70 insiders, but found: {insider_count}")
        print(f"  Expected 930 benign users, but found: {benign_count}")
        raise ValueError("Labeling validation failed. Check 'user_id' matching.")

except Exception as e:
    print(f"--- ERROR during validation: {e} ---")
    raise
finally:
    con.close()
    print("\nDatabase connection closed. Script finished.")