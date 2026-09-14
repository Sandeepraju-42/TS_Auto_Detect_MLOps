import yaml
from google.cloud import storage

BUCKET_NAME = "fashionmvforecast"
BLOB_PATH = "config/params.yaml"

def load_params() -> dict:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(BLOB_PATH)
    
    yaml_text = blob.download_as_text()
    return yaml.safe_load(yaml_text)