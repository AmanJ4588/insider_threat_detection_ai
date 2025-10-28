import pandas as pd
from pathlib import Path

input_root = Path("dataset/CERT_dataset/r4.2") #path to original csv files
output_root = Path("dataset/CERT_dataset_parquet/r4.2") #path to parquet files

for csv_file in input_root.rglob("*.csv"):
    
    relative_path = csv_file.relative_to(input_root) #compute relative path
    
    parquet_file = output_root / relative_path.with_suffix(".parquet") #create the new path for parquet file (new folder, new extension, same name and structure)
    
    parquet_file.parent.mkdir(parents=True, exist_ok=True) # Ensure the parent directories exist
    
    df = pd.read_csv(csv_file) 
    df.to_parquet(parquet_file, index=False)
    
    print(f"Converted {csv_file} -> {parquet_file}")
