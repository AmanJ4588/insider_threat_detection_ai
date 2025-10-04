import os
import duckdb

base_dir = r"CERT_dataset_parquet\r4.2" #path to dataset (parquet format)

con = duckdb.connect("dataset/cert42.duckdb") #creating a local duck db file
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".parquet"):
            file_path = os.path.join(root, file)

            # Make table name: remove base_dir and replace slashes with _
            relative_path = os.path.relpath(file_path, base_dir)
            table_name = (
                relative_path.replace("\\", "_")
                             .replace("/", "_")
                             .replace(".parquet", "")
            )

            print(f"Loading {file_path} → table `{table_name}`")

            # Create table from parquet file
            con.execute(f"""
                CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT * FROM parquet_scan('{file_path}')
            """)

            print(f"✅ Loaded {table_name}")

#Listing all tables after creation
tables = con.execute("SHOW TABLES").df()
print("Tables in cert42.duckdb:")
print(tables)
con.close()