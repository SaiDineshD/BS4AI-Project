"""
Deepfake Detection Challenge data loader.
Source: https://www.kaggle.com/competitions/deepfake-detection-challenge/data
Loads visual stream with n=200 subset.
"""

from pathlib import Path
from typing import Optional

# Placeholder — implement with DFDC structure from Kaggle
# Real vs. fake videos; metadata in train_sample_videos/ or similar


def load_deepfake_subset(
    root: Path,
    subset_indices: Optional[list] = None,
    n: int = 200,
) -> list:
    """Load Deepfake Detection Challenge samples according to n=200 subset."""
    raise NotImplementedError("Implement after downloading from Kaggle")
