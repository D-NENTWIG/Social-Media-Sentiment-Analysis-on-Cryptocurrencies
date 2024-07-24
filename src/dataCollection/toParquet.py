import pandas as pd
import json

# Load JSON data from the file
file_path = 'example.json'
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Normalizing the main level of the JSON
df_main = pd.json_normalize(json_data)

# Flatten nested structures as needed
# E.g.: Flatten 'author' and merge it back to the main DataFrame
# Adjust 'record_path' and 'meta' as per your JSON structure
if 'author' in df_main.columns:
    df_author = pd.json_normalize(json_data, record_path=['author'], meta=['id'])
    df_main = df_main.merge(df_author, on='id', suffixes=('', '_author'), how='left')

# Replace dots in column names with underscores to avoid issues in Parquet format
df_main.columns = df_main.columns.str.replace('.', '_', regex=False)

# Save the normalized DataFrame as a Parquet file
normalized_parquet_path = 'normalized_output.parquet'
df_main.to_parquet(normalized_parquet_path)


#To Read The Parquet
"""

import pyarrow.parquet as pq

# Path to your Parquet file
parquet_file_path = 'normalized_output.parquet'

# Read the Parquet file
parquet_file = pq.read_table(parquet_file_path)

# Print the first few rows of the file
print(parquet_file.to_pandas().head())

#####################

import pandas as pd
import pyarrow.parquet as pq

# Replace with the path to your Parquet file
parquet_file_path = 'normalized_output.parquet'

# Read the Parquet file
parquet_file = pq.read_table(parquet_file_path)

# Convert to a pandas DataFrame
df = parquet_file.to_pandas()

# Replace with the path where you want to save the CSV file
csv_file_path = 'path_to_save_csv_file.csv'

# Save DataFrame as CSV
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at: {csv_file_path}")


"""