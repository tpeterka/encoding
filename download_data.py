"""
Usage:
    python download_data.py
Author: Jianxin Sun
Email: sunjianxin66@gmail.com
Description:
    Download the Argon Bubble toy dataset from GitHub release and extract it
"""

import urllib.request
import zipfile
import os
from pathlib import Path

URL = "https://github.com/sunjianxin/F-Hash/releases/download/v1.0/data.zip"
ZIP_NAME = "data.zip"
OUT_DIR = "data"

def download(url, filename):
    if Path(filename).exists():
        print(f"{filename} already exists, skipping download.")
        return

    print("Downloading dataset...")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded / total_size * 100, 100)
        print(f"\rProgress: {percent:5.1f}%", end="")

    urllib.request.urlretrieve(url, filename, progress)
    print("\nDownload complete.")


def unzip(zip_path, out_dir):
    if Path(out_dir).exists():
        print(f"{out_dir}/ already exists, skipping extraction.")
        return

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    print("Extraction complete.")


def main():
    download(URL, ZIP_NAME)
    unzip(ZIP_NAME, OUT_DIR)

if __name__ == "__main__":
    main()
