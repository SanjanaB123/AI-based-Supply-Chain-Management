import pandas as pd
from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-key.json"

# Download and read the parquet from GCS
client = storage.Client()
bucket = client.bucket("supply-chain-pipeline")

# List all feature files
blobs = list(bucket.list_blobs(prefix="features/"))
for blob in blobs:
    print(blob.name)

# Read the latest one
latest = sorted(blobs, key=lambda b: b.updated, reverse=True)[0]
print(f"\nReading: {latest.name}")

latest.download_to_filename("temp_features.parquet")
df = pd.read_parquet("temp_features.parquet")

# Check it
print(f"\nShape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nNull counts:\n{df.isnull().sum()}")