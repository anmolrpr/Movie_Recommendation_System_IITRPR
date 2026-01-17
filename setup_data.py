import os
import requests
import zipfile
import io

DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = "data"

def download_and_extract():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print(f"Downloading data from {DATA_URL}...")
    response = requests.get(DATA_URL)
    response.raise_for_status()

    print("Extracting data...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)
    
    print("Done!")
    print(f"Data available in: {os.path.join(DATA_DIR, 'ml-100k')}")

if __name__ == "__main__":
    download_and_extract()
