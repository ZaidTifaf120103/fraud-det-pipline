import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Load environment variables
load_dotenv()

# Retrieve credentials securely
connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Conteneur et fichier
container_name = "datasets"
blob_name = "fraudTrain.csv"
local_file_path = r"C:\Users\zaid2\Desktop\fraud-detection-pipeline\fraud-detection-pipeline\data\raw\fraudTrain.csv"

# Téléchargement du fichier
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

with open(local_file_path, "rb") as data:
    blob_client.upload_blob(data)

print(f"{blob_name} téléchargé avec succès dans {container_name}.")