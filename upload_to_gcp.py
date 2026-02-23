from google.cloud import storage


def upload_to_gcs(file_path: str, bucket_name: str, destination_blob_name: str = None):
    
    if destination_blob_name is None:
        destination_blob_name = file_path.split("/")[-1]
    
    # Point to your service account key file
    client = storage.Client.from_service_account_json(
        r"gcp-key.json"
    )
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    
    print(f"✅ File uploaded successfully!")
    print(f"   Local:  {file_path}")
    print(f"   GCS:    gs://{bucket_name}/{destination_blob_name}")

# --- How to use it ---
upload_to_gcs(
    file_path="data/test_data.txt",
    bucket_name="ai-chain-supply-gcp-bucket",
    #destination_blob_name="processed/cleaned_data.csv"   # optional
    destination_blob_name="test_data.txt"   # optional, same as original filename
)