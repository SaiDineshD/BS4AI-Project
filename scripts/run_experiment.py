#!/usr/bin/env python3
"""
Run liveness detection experiment.
Trains visual, audio, and fusion modules; runs fairness evaluation.
"""

import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model_config.yaml")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    print("Experiment runner — implement training loop after data loaders.")
    print("Config:", args.config)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
