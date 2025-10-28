import duckdb
import json
import pandas as pd
import os


DUCKDB_FILE_PATH = 'dataset/supervised_dataset.duckdb' 
SOURCE_TABLE_NAME = 'transformed_data'

VOCAB_FILE_PATH = 'data_transformation_and_preprocessing/05_data_preprocessing/insider_vocab.json'

FINAL_TABLE_NAME = 'training_data_table'




def build_vocabularies(db_path, source_table, vocab_path):
    """
    Phase 1: Connects to DuckDB, queries the source table for unique
    categorical values, and saves them to a JSON vocabulary file.
    
    We still need this JSON file for our Keras model later.
    """
    print(f"--- Starting Phase 1: Building Vocabularies ---")
    print(f"Reading from table '{source_table}' in {db_path}...")

    try:
        con = duckdb.connect(database=db_path, read_only=True)

        # Query the table 
        event_type_vocab = con.sql(f"SELECT DISTINCT event_type FROM {source_table}").df()['event_type'].tolist()
        pc_vocab = con.sql(f"SELECT DISTINCT pc FROM {source_table}").df()['pc'].tolist()

        con.close()
        
        print(f"Found {len(event_type_vocab)} unique event_types and {len(pc_vocab)} unique PCs.")

        # --- Convert lists to dictionaries ---
        # 0 = "__UNKNOWN__", 1 = "__PADDING__"
        event_type_map = {val: i+2 for i, val in enumerate(event_type_vocab)}
        event_type_map["__UNKNOWN__"] = 0
        event_type_map["__PADDING__"] = 1

        pc_map = {val: i+2 for i, val in enumerate(pc_vocab)}
        pc_map["__UNKNOWN__"] = 0
        pc_map["__PADDING__"] = 1

        final_vocab = {
            'event_type': event_type_map,
            'pc': pc_map
        }

        # --- Save the vocabulary locally ---
        with open(vocab_path, 'w') as f:
            json.dump(final_vocab, f, indent=4)

        print(f"Successfully saved vocabulary to {vocab_path}")
        print("--- Phase 1 Complete ---")
        return final_vocab

    except Exception as e:
        print(f"Error in Phase 1: {e}")
        return None

def transform_data(db_path, source_table, vocab_dict, final_table):
    """
    Phase 2: Uses the vocabularies to transform the entire source table
    and save the clean data to a new table IN-PLACE inside the DuckDB file.
    """
    print(f"\n--- Starting Phase 2: Transforming Data ---")
    
    try:
        # --- 1. Load vocabs into pandas DataFrames for DuckDB ---
        event_type_df = pd.DataFrame(list(vocab_dict['event_type'].items()), columns=['event_type', 'event_type_encoded'])
        pc_df = pd.DataFrame(list(vocab_dict['pc'].items()), columns=['pc', 'pc_encoded'])

        # --- 2. Connect to DuckDB and register vocab tables ---
        con = duckdb.connect(database=db_path, read_only=False) 
        con.register('event_type_map', event_type_df)
        con.register('pc_map', pc_df)
        print(f"Registered vocab maps as temporary tables.")

        # --- 3. Define the main transformation SQL ---
        # This query creates the new, clean table in one go.
        transform_sql = f"""
        CREATE OR REPLACE TABLE {final_table} AS
        SELECT 
            t.user_id,
            t.timestamp AS timestamp, -- Just select the timestamp directly
            t.label,
            HOUR(t.timestamp) AS hour, -- Use HOUR() directly
            DAYOFWEEK(t.timestamp) AS day_of_week, -- Use DAYOFWEEK() directly
            
            COALESCE(e.event_type_encoded, 0) AS event_type_encoded, 
            COALESCE(p.pc_encoded, 0) AS pc_encoded
            
        FROM {source_table} AS t
        
        LEFT JOIN event_type_map AS e ON t.event_type = e.event_type
        LEFT JOIN pc_map AS p ON t.pc = p.pc
        
        WHERE t.timestamp IS NOT NULL;
        """

        print(f"Executing transformation... (This may take a few minutes for 34M rows)")
        con.execute(transform_sql)
        print(f"Successfully created table '{final_table}' in {db_path}")
        
        con.close()
        print("--- Phase 2 Complete ---")

    except Exception as e:
        print(f"Error in Phase 2: {e}")

def main():
    # Check if DuckDB file exists
    if not os.path.exists(DUCKDB_FILE_PATH):
        print(f"Error: DuckDB file not found at {DUCKDB_FILE_PATH}")
        print("Please update DUCKDB_FILE_PATH at the top of the script.")
        return

    # Run Phase 1
    vocab = build_vocabularies(DUCKDB_FILE_PATH, SOURCE_TABLE_NAME, VOCAB_FILE_PATH)
    
    # Run Phase 2 only if Phase 1 was successful
    if vocab:
        transform_data(DUCKDB_FILE_PATH, SOURCE_TABLE_NAME, vocab, FINAL_TABLE_NAME)
        print(f"\nPreprocessing is complete!")
        print(f"Your training data is now in the table '{FINAL_TABLE_NAME}' inside {DUCKDB_FILE_PATH}")

if __name__ == "__main__":
    main()