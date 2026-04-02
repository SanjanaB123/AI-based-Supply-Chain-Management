# import os
# from dotenv import load_dotenv
# from google.cloud import storage

# def upload_to_gcs(file_path: str, bucket_name: str, destination_blob_name: str = None):
    
#     # Load environment variables from .env
#     load_dotenv()
    
#     # Get credentials path from environment variable
#     credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
#     if not credentials_path:
#         raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
    
#     if destination_blob_name is None:
#         destination_blob_name = os.path.basename(file_path)
    
#     # Create client using env-based credentials
#     client = storage.Client.from_service_account_json(credentials_path)
    
#     bucket = client.bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#     blob.upload_from_filename(file_path)
    
#     print("✅ File uploaded successfully!")
#     print(f"   Local:  {file_path}")
#     print(f"   GCS:    gs://{bucket_name}/{destination_blob_name}")

''' The local to gcp code is commented out because it is not needed for the current implementation. 
The code is left here for future reference if we need to upload files to gcp in the future. '''

import os
from google.cloud import storage

def upload_to_gcs(file_path: str, bucket_name: str, destination_blob_name: str = None):
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(file_path)

    # Use Application Default Credentials (ADC)
    # - On Cloud Run: picks up the attached service account automatically
    # - Locally: uses GOOGLE_APPLICATION_CREDENTIALS env var or gcloud auth
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path and os.path.exists(credentials_path):
        client = storage.Client.from_service_account_json(credentials_path)
    else:
        client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)

    print("File uploaded successfully!")
    print(f"Local:  {file_path}")
    print(f"GCS:    gs://{bucket_name}/{destination_blob_name}")
