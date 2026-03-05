"""
FF-c23 (FaceForensics++ C23) data loader.
Source: https://www.kaggle.com/datasets/xdxd003/ff-c23
Loads visual stream with n=200 subset.
"""

from pathlib import Path
from typing import Optional

# Placeholder — implement with actual FF-c23 structure from Kaggle
# Categories: original, deepfakes, face2face, faceswap, neural_textures


def load_ff_c23_subset(
    root: Path,
    subset_indices: Optional[list] = None,
    n: int = 200,
) -> list:
    """Load FF-c23 samples according to n=200 subset."""
    raise NotImplementedError("Implement after downloading from Kaggle")
