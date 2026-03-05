#!/bin/bash
# Download datasets from Kaggle (requires kaggle CLI + API credentials)
# Install: pip install kaggle
# Setup: kaggle.json in ~/.kaggle/ (from Kaggle Account → Create New API Token)

set -e
mkdir -p data/raw/ff_c23 data/raw/deepfake_detection data/raw/asvspoof2019

echo "Downloading from Kaggle..."

# FF-c23 (FaceForensics++ C23)
kaggle datasets download -d xdxd003/ff-c23 -p data/raw/ff_c23 --unzip

# Deepfake Detection Challenge (requires accepting competition rules)
kaggle competitions download -c deepfake-detection-challenge -p data/raw/deepfake_detection

# ASVspoof 2019
kaggle datasets download -d awsaf49/asvpoof-2019-dataset -p data/raw/asvspoof2019 --unzip

echo "Done. Run scripts/preprocess.py to create n=200 subsets."
