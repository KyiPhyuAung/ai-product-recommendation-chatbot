import pandas as pd
import os

# Load the dataset
data_path = "models/products_index.csv"
df = pd.read_csv(data_path)

# Calculate memory usage
memory_bytes = df.memory_usage(deep=True).sum()
memory_mb = memory_bytes / (1024 * 1024)

print("="*40)
print("     DATASET USAGE REPORT     ")
print("="*40)
print(f"File Path:       {data_path}")
print(f"Total Rows:      {len(df):,}")
print(f"Total Columns:   {len(df.columns)}")
print(f"Memory Footprint:{memory_mb:.2f} MB")
print("="*40)