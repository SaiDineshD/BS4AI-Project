#!/usr/bin/env python3
"""
Preprocess datasets and create n=200 subsets.
Run after downloading raw data.
"""

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.sampling import create_n200_subset, load_data_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--output-dir", default="data/sampling")
    args = parser.parse_args()

    config = load_data_config(args.config)
    n = config["sampling"]["n_per_dataset"]
    seed = config["sampling"]["seed"]
    output_dir = Path(args.output_dir)

    # Placeholder: In practice, build indices_by_category from dataset metadata
    # For now, create empty structure as template
    print(f"Preprocessing with n={n} per dataset, seed={seed}")
    print("Implement dataset-specific index building after data download.")
    print("Output will be saved to:", output_dir)


if __name__ == "__main__":
    main()
