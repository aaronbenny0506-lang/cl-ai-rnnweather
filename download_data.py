"""
Downloads the historical London weather dataset (london_weather.csv) from the
Google Drive link given in the task, and saves it to data/london_weather.csv.

Usage:
    python download_data.py
"""
import os
import gdown

FILE_ID = "1mRTi_ZiuFinPnqHpm_XFFuGUdjnY87Hq"
OUT_PATH = os.path.join("data", "london_weather.csv")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} already exists, skipping download.")
    else:
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, OUT_PATH, quiet=False)
        print(f"Saved dataset to {OUT_PATH}")
