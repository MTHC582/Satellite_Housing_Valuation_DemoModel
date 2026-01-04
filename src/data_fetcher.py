"""
I'm using MapBox for this, since it comes in handy for fetchin about
50k images per month and since my data-set has an overall count of below 25k.
THis is found to be more suitable.
This file fetches images using mapbox api, to both train and test sets.
"""

import pandas as pd
import requests
import os
from dotenv import load_dotenv  # to prevent the misusage of api key provided by mapbox
from concurrent.futures import ThreadPoolExecutor  # Forparallel downloads, speedup.

# ================= CONFIGURATION =================
# MAPBOX API-KEY.
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_KEY")

if not MAPBOX_TOKEN:
    raise ValueError("MapBox key error, check you .env file!.")

# File Setup
IMAGE_FOLDER = (
    "data/images"  # Choosing the folder as .\data\images to store the downloads.
)
# INPUT_FILE variable removed because we now pass it into the function directly. # <<< MODIFIED

# Image Settings
ZOOM = 18
SIZE = "600x600"
STYLE = "mapbox/satellite-v9"
# =================================================

"""
Since each image takes about 100 KB.
22k files store about nearly ~ 2.2 GB.
(Number of rows btw)

Total size of images is 2.3 GB (approx)
"""


def download_image(row):
    """Downloads a single image given a row."""
    try:
        prop_id = str(row["id"])
        lat = row["lat"]
        long = row["long"]

        filename = os.path.join(IMAGE_FOLDER, f"{prop_id}.jpg")

        # (Skip if already exists)
        # This is an important 'if', as this is the one that prevents duplicates.
        # If the program ever crsashes mid-way, this skips over the recieved images n continues further till the end.
        if os.path.exists(filename):
            return None

        # Mapbox API URL
        url = f"https://api.mapbox.com/styles/v1/{STYLE}/static/{long},{lat},{ZOOM},0,0/{SIZE}?access_token={MAPBOX_TOKEN}"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filename, "wb") as f:  # Since it's for images, its wb mode.
                f.write(response.content)
            return None  # Success
        else:
            return (
                f"Error {response.status_code} on ID {prop_id}"  # Proper Error throw.
            )

    except Exception as e:
        return f"Failed ID {row.get('id', 'Unknown')}: {e}"


# I changed 'main' to 'process_excel' so one can reuse it for both files. # <<< MODIFIED
def process_excel(file_path):

    # 1_Setup Folders
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)

    # 2_Load Excel
    print(f"\nLoading {file_path}...")  # <<< MODIFIED to print the specific file name
    try:
        df = pd.read_excel(file_path)  # <<< MODIFIED to use the argument variable
        print(f"Loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"ERROR: File {file_path} not found. Check the name!")
        return

    # 3_Parallel Download
    print("Starting download... (Check data/images folder!)")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(download_image, [row for _, row in df.iterrows()]))

    # 4_Summary
    errors = [r for r in results if r is not None]
    print(f"DONE! Processed {len(df)} rows form {file_path}.")  # <<< MODIFIED

    if errors:
        # Saving errors to a unique log_file to prevent overwritiing each other # <<< MODIFIED
        log_file = f"errors_{os.path.basename(file_path)}.txt"
        print(f"{len(errors)} errors occurred. Saving to {log_file}.")
        with open(log_file, "w") as f:
            f.write("\n".join(errors))


if __name__ == "__main__":

    # List of files to process (Train AND Test) # <<< MODIFIED
    files_list = ["data/train(1).xlsx", "data/test2.xlsx"]

    # Changed from main to this so that both test n train data-sets can be done in one single stretch
    # Loop through the list and run the process for each # <<< MODIFIED
    for file_name in files_list:
        process_excel(file_name)

    print("\nAll files processed successfully.")
