"""
Sampling: 200 samples from each dataset.
Ensures reproducible subsets across FaceForensics++, ASVspoof 2019, FairFace.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import yaml


def create_n200_subset(
    indices: List[int],
    n: int = 200,
    seed: int = 42,
    output_path: Optional[Path] = None,
) -> List[int]:
    """
    Sample n indices from a dataset.
    If dataset has fewer than n samples, use all available.
    """
    random.seed(seed)
    if len(indices) <= n:
        subset = indices
    else:
        subset = sorted(random.sample(indices, n))

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(subset, f, indent=2)

    return subset


def load_data_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load data configuration including n=200 setting."""
    with open(config_path) as f:
        return yaml.safe_load(f)
