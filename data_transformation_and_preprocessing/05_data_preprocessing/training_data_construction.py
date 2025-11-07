import duckdb
import json
import pandas as pd
import os

# --- Constants ---
DUCKDB_FILE_PATH = 'dataset/supervised_dataset.duckdb' 
SOURCE_TABLE_NAME = 'transformed_data'
VOCAB_FILE_PATH = 'data_transformation_and_preprocessing/05_data_preprocessing/insider_vocab.json'
FINAL_TABLE_NAME = 'training_data_table'
INSIDERS_CSV_PATH = 'dataset/answers/insiders.csv' 


def build_vocabularies(db_path, source_table, vocab_path):
    """
    Phase 1: Connects to DuckDB, queries the source table for unique
    categorical values, and saves them to a JSON vocabulary file.
    
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

        # Create 0-based vocabularies with 'UNK' at index 0
        event_type_map = {val: i+1 for i, val in enumerate(event_type_vocab)}
        event_type_map['UNK'] = 0

        pc_map = {val: i+1 for i, val in enumerate(pc_vocab)}
        pc_map['UNK'] = 0

        vocab_data = {
            "event_type": event_type_map,
            "pc": pc_map
        }
        
        # Save to JSON
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=4)
        
        print(f"Successfully saved vocabularies to {vocab_path}")
        print("--- Phase 1 Complete ---")
        return vocab_data

    except Exception as e:
        print(f"Error in Phase 1: {e}")
        return None

def create_training_table(db_path, source_table, final_table, vocab_data):
    """
    Phase 2: Connects to DuckDB, creates mapping tables from the vocab,
    loads insider scenario data, and builds the final training table
    by joining all data sources.
    """
    print(f"\n--- Starting Phase 2: Creating Final Training Table ---")
    if vocab_data is None:
        print("Error: Vocab data is missing. Cannot proceed.")
        return

    try:
        con = duckdb.connect(database=db_path, read_only=False)
        
        
        print("Cleaning up old tables if they exist...")
        con.execute(f"DROP TABLE IF EXISTS {final_table}")
        con.execute("DROP TABLE IF EXISTS event_type_map")
        con.execute("DROP TABLE IF EXISTS pc_map")
        con.execute("DROP TABLE IF EXISTS insider_scenario_map")
        

        # 1. Create mapping tables for event_type and pc
        con.execute("CREATE TABLE event_type_map (event_type VARCHAR, event_type_encoded INTEGER)")
        con.executemany("INSERT INTO event_type_map VALUES (?, ?)", list(vocab_data['event_type'].items()))
        
        con.execute("CREATE TABLE pc_map (pc VARCHAR, pc_encoded INTEGER)")
        con.executemany("INSERT INTO pc_map VALUES (?, ?)", list(vocab_data['pc'].items()))

        # --- Step 2. Load insiders.csv and create scenario map ---
        print(f"Loading insider scenario data from {INSIDERS_CSV_PATH}...")
        try:
            insiders_df = pd.read_csv(INSIDERS_CSV_PATH)
            # Select only the user and scenario, rename user to match main table
            insiders_map_df = insiders_df[['user', 'scenario']].rename(columns={'user': 'user_id'})
            
            # Create the mapping table in DuckDB
            con.execute("CREATE TABLE insider_scenario_map AS SELECT * FROM insiders_map_df")
            print("Successfully created 'insider_scenario_map' table.")
            
        except Exception as e:
            print(f"Error loading {INSIDERS_CSV_PATH}: {e}")
            raise # Stop if this file is missing


        # 3. Build the final training table SQL
        transform_sql = f"""
        CREATE OR REPLACE TABLE {final_table} AS
        SELECT 
            t.user_id,  
            t.sequence_id,                        
            t.timestamp,
            t.label,                            
            COALESCE(s.scenario, 0) AS scenario, 
            
            EXTRACT(HOUR FROM t.timestamp) AS hour,
            DAYOFWEEK(t.timestamp) AS day_of_week,
            
            COALESCE(e.event_type_encoded, 0) AS event_type_encoded, 
            COALESCE(p.pc_encoded, 0) AS pc_encoded
            
        FROM {source_table} AS t
        
        LEFT JOIN event_type_map AS e ON t.event_type = e.event_type
        LEFT JOIN pc_map AS p ON t.pc = p.pc
        LEFT JOIN insider_scenario_map AS s ON t.user_id = s.user_id -- (NEW JOIN)
        
        WHERE t.timestamp IS NOT NULL;
        """

        print(f"Executing transformation... (This may take a few minutes)")
        con.execute(transform_sql)
        print(f"Successfully created table '{final_table}' in {db_path}")
        
        # Clean up mapping tables
        con.execute("DROP TABLE event_type_map")
        con.execute("DROP TABLE pc_map")
        con.execute("DROP TABLE insider_scenario_map")
        
        con.close()
        print("--- Phase 2 Complete ---")

    except Exception as e:
        print(f"Error in Phase 2: {e}")
        con.close()

def main():
    # Check if DuckDB file exists
    if not os.path.exists(DUCKDB_FILE_PATH):
        print(f"Error: DuckDB file not found at {DUCKDB_FILE_PATH}")
        print("Please update DUCKDB_FILE_PATH at the top of the script.")
        return

    # Run Phase 1
    vocab = build_vocabularies(DUCKDB_FILE_PATH, SOURCE_TABLE_NAME, VOCAB_FILE_PATH)
    
    # Run Phase 2
    if vocab:
        create_training_table(DUCKDB_FILE_PATH, SOURCE_TABLE_NAME, FINAL_TABLE_NAME, vocab)

if __name__ == "__main__":
    main()