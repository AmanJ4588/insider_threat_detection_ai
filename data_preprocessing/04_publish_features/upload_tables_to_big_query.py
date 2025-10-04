import os 
from google.cloud import bigquery 
import dotenv 
import duckdb

dotenv.load_dotenv() 

client = bigquery.Client() 
dataset_id = f"{client.project}.cert_v42"
db_path = "dataset/supervised_dataset.duckdb"

#connect to duckdb
print(f"Connecting to DuckDB at: {db_path}")
db_con = duckdb.connect(db_path)

#create a bigquery dataset
print(f"Ensuring BigQuery dataset '{dataset_id}' exists...")
client.create_dataset(bigquery.Dataset(dataset_id), exists_ok=True)

#tables to be uploaded in bigquery
tables = ("feature_table" , "insiders") 

for t in tables: 
    print(f"\n--- Processing table: {t} ---")
    
    # Define the full BigQuery table ID
    table_id = f"{dataset_id}.{t}"
    
    # Use the table name to create a temporary Parquet file.
    temp_parquet_file = f"{t}_temp.parquet"

    try:
        # Export the current table to the temporary Parquet file.
        print(f"Exporting '{t}' to '{temp_parquet_file}'...")
        db_con.execute(f"COPY {t} TO '{temp_parquet_file}' (FORMAT 'PARQUET');")

        # Configure the BigQuery load job
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition="WRITE_TRUNCATE", # will overwrite the table if it already exists
        )

        # Upload the Parquet file to BigQuery
        print(f"Uploading to BigQuery table: '{table_id}'...")
        with open(temp_parquet_file, "rb") as source_file:
            load_job = client.load_table_from_file(source_file, table_id, job_config=job_config)

        load_job.result() 

        destination_table = client.get_table(table_id)
        print(f"Successfully loaded {destination_table.num_rows} rows into '{table_id}'.")

    except Exception as e:
        print(f"An error occurred while processing table '{t}': {e}")
    finally:
        # Clean up the temporary file, whether the upload succeeded or failed.
        if os.path.exists(temp_parquet_file):
            os.remove(temp_parquet_file)
            print(f"Removed temporary file: '{temp_parquet_file}'")


#close the connection
db_con.close()
print("\n--- All tables processed. ---")
