"""
ASVspoof 2019 Logical Access data loader.
Source: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
Loads audio stream with n=200 subset.
"""

from pathlib import Path
from typing import Optional

# Placeholder — implement with ASVspoof 2019 LA structure from Kaggle


def load_asvspoof_subset(
    root: Path,
    subset_indices: Optional[list] = None,
    n: int = 200,
) -> list:
    """Load ASVspoof 2019 LA samples according to n=200 subset."""
    raise NotImplementedError("Implement after downloading from Kaggle")
