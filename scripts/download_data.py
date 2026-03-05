#!/usr/bin/env python3
"""
Download datasets from Kaggle.
Requires: pip install kaggle, and kaggle.json in ~/.kaggle/
"""

import subprocess
import sys
from pathlib import Path

DATASETS = [
    ("datasets", "xdxd003/ff-c23", "data/raw/ff_c23"),
    ("competitions", "deepfake-detection-challenge", "data/raw/deepfake_detection"),
    ("datasets", "awsaf49/asvpoof-2019-dataset", "data/raw/asvspoof2019"),
]


def main():
    for dtype, name, dest in DATASETS:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        cmd = ["kaggle", dtype, "download", "-d" if dtype == "datasets" else "-c", name, "-p", str(dest)]
        if dtype == "datasets":
            cmd.append("--unzip")
        print(f"Downloading {name}...")
        subprocess.run(cmd, check=True)
    print("Done. Run scripts/preprocess.py to create n=200 subsets.")


if __name__ == "__main__":
    main()
