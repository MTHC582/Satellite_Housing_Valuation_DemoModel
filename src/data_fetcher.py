import pandas as pd
import requests
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor  # Forparallel downloads, speedup.

# ================= CONFIGURATION =================
# MAPBOX API-KEY.
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_KEY")

if not MAPBOX_TOKEN:
    raise ValueError("MapBox key error, check you .env file!.")

# File Setup
INPUT_FILE = "data/train(1).xlsx"
IMAGE_FOLDER = "data/images"
LOG_FILE = "download_log.txt"

# Image Settings
ZOOM = 18
SIZE = "600x600"
STYLE = "mapbox/satellite-v9"
# =================================================

"""
Since each image takes about 100 KB.
22k files store about nearly ~ 2.2 GB.
(Number of rows btw)
"""


def download_image(row):
    """Downloads a single image given a row."""
    try:
        prop_id = str(row["id"])
        lat = row["lat"]
        long = row["long"]

        filename = os.path.join(IMAGE_FOLDER, f"{prop_id}.jpg")

        # Skip if already exists
        if os.path.exists(filename):
            return None

        # Mapbox API URL
        url = f"https://api.mapbox.com/styles/v1/{STYLE}/static/{long},{lat},{ZOOM},0,0/{SIZE}?access_token={MAPBOX_TOKEN}"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            return None  # Success
        else:
            return f"Error {response.status_code} on ID {prop_id}"

    except Exception as e:
        return f"Failed ID {row.get('id', 'Unknown')}: {e}"


def main():
    # 1_Setup Folders
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # 2_Load Excel
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_excel(INPUT_FILE)
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: File {INPUT_FILE} not found. Check the name!")
        return

    # 3_Parallel Download
    print("Starting download... (Check data/images folder!)")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_image, [row for _, row in df.iterrows()]))

    # 4_Summary
    errors = [r for r in results if r is not None]
    print(f"\nDONE! Processed {len(df)} rows.")
    if errors:
        print(f"{len(errors)} errors occurred. Saving to {LOG_FILE}.")
        with open(LOG_FILE, "w") as f:
            f.write("\n".join(errors))


if __name__ == "__main__":
    main()
