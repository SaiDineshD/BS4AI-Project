#!/usr/bin/env python3
"""
Preprocess datasets: create n=200 sample indices and verify data integrity.
Run after downloading raw data.

Usage:
    python scripts/preprocess.py --config config/data_config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv

import pandas as pd

from src.data.sampling import create_n200_subset, load_data_config


def preprocess_ff_c23(config: dict, output_dir: Path):
    """Build FF++ C23 sample indices from CSV metadata."""
    csv_dir = Path(config["ff_c23"]["csv_dir"])
    n = config["sampling"]["n_per_dataset"]
    seed = config["sampling"]["seed"]

    categories = {
        "real": config["ff_c23"]["categories"]["real"],
        "fake": config["ff_c23"]["categories"]["fake"],
    }

    all_indices = {"real": [], "fake": []}
    video_manifest = []

    for label_type, cats in categories.items():
        for cat in cats:
            csv_path = csv_dir / f"{cat}.csv"
            if not csv_path.exists():
                print(f"  [SKIP] CSV not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            root = Path(config["paths"]["ff_c23"])
            valid_count = 0

            for idx, row in df.iterrows():
                video_path = root / row["File Path"]
                if video_path.exists():
                    entry = {
                        "index": len(video_manifest),
                        "path": str(video_path),
                        "label": 0 if label_type == "real" else 1,
                        "category": cat,
                    }
                    video_manifest.append(entry)
                    all_indices[label_type].append(entry["index"])
                    valid_count += 1

            print(f"  {cat}: {valid_count} valid videos found")

    n_real = n // 2
    n_fake = n - n_real

    real_subset = create_n200_subset(
        all_indices["real"], n=n_real, seed=seed,
        output_path=output_dir / "ff_c23_real_indices.json",
    )
    fake_subset = create_n200_subset(
        all_indices["fake"], n=n_fake, seed=seed,
        output_path=output_dir / "ff_c23_fake_indices.json",
    )

    with open(output_dir / "ff_c23_manifest.json", "w") as f:
        json.dump(video_manifest, f, indent=2)

    print(f"  Selected {len(real_subset)} real + {len(fake_subset)} fake = {len(real_subset) + len(fake_subset)} total")
    return real_subset, fake_subset


def preprocess_asvspoof(config: dict, output_dir: Path):
    """Build ASVspoof 2019 sample indices from protocol files."""
    root = Path(config["paths"]["asvspoof2019"])
    n = config["sampling"]["n_per_dataset"]
    seed = config["sampling"]["seed"]

    for split_name in ["train", "dev", "eval"]:
        split_cfg = config["asvspoof2019"]["splits"][split_name]
        protocol_path = root / split_cfg["protocol"]
        audio_dir = root / split_cfg["audio_dir"]

        bonafide_indices = []
        spoof_indices = []
        manifest = []

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                audio_id = parts[1]
                key = parts[4]
                audio_path = audio_dir / f"{audio_id}.flac"

                if audio_path.exists():
                    entry_idx = len(manifest)
                    manifest.append({
                        "index": entry_idx,
                        "audio_id": audio_id,
                        "path": str(audio_path),
                        "label": 0 if key == "bonafide" else 1,
                        "speaker": parts[0],
                        "system": parts[3],
                    })
                    if key == "bonafide":
                        bonafide_indices.append(entry_idx)
                    else:
                        spoof_indices.append(entry_idx)

        n_each = n // 2
        bonafide_subset = create_n200_subset(
            bonafide_indices, n=n_each, seed=seed,
            output_path=output_dir / f"asvspoof_{split_name}_bonafide_indices.json",
        )
        spoof_subset = create_n200_subset(
            spoof_indices, n=n - n_each, seed=seed,
            output_path=output_dir / f"asvspoof_{split_name}_spoof_indices.json",
        )

        with open(output_dir / f"asvspoof_{split_name}_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(
            f"  {split_name}: {len(bonafide_subset)} bonafide + {len(spoof_subset)} spoof "
            f"(from {len(bonafide_indices)} + {len(spoof_indices)} available)"
        )


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets and create sampling indices")
    parser.add_argument("--config", default="config/data_config.yaml")
    parser.add_argument("--output-dir", default="data/sampling")
    args = parser.parse_args()

    config = load_data_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = config["sampling"]["n_per_dataset"]
    seed = config["sampling"]["seed"]
    print(f"Preprocessing with n={n} per dataset, seed={seed}")
    print(f"Output directory: {output_dir}\n")

    print("=== FaceForensics++ C23 ===")
    preprocess_ff_c23(config, output_dir)

    print("\n=== ASVspoof 2019 LA ===")
    preprocess_asvspoof(config, output_dir)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
