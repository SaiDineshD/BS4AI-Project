"""
FairFace image dataset for fairness evaluation (genuine-only protocol).

Expects the FairFace CSV layout (e.g. fairface_label_train.csv) with at least
file path, race, gender, and age columns. Configure column names in data_config.yaml.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.ff_c23_loader import get_eval_transform


def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for name in [preferred] + fallbacks:
        if name.lower() in cols:
            return cols[name.lower()]
    raise KeyError(f"No column matching {preferred} (tried {fallbacks}) in {list(df.columns)}")


class FairFaceImageDataset(Dataset):
    """Single-frame faces (T=1) with aligned demographic rows in ``self.meta``."""

    def __init__(
        self,
        root: str,
        csv_path: str,
        face_size: int = 224,
        max_samples: Optional[int] = None,
        seed: int = 42,
        img_path_column: str = "file",
        race_column: str = "race",
        gender_column: str = "gender",
        age_column: str = "age",
    ):
        self.root = Path(root)
        csv_file = Path(csv_path)
        if not csv_file.is_file():
            csv_file = self.root / csv_path
        if not csv_file.is_file():
            raise FileNotFoundError(
                f"FairFace CSV not found: tried {csv_path} and {csv_file}. "
                "Set paths.fairface in config/data_config.yaml to your FairFace root and CSV."
            )

        df = pd.read_csv(csv_file)
        if len(df) == 0:
            raise ValueError(f"FairFace CSV is empty: {csv_file}")

        img_col = _resolve_column(df, img_path_column, ["file", "img_path", "path"])
        race_col = _resolve_column(df, race_column, ["race"])
        gender_col = _resolve_column(df, gender_column, ["gender"])
        age_col = _resolve_column(df, age_column, ["age"])

        df = df.copy()
        df["_img"] = df[img_col].astype(str)
        df["_race"] = df[race_col].astype(str)
        df["_gender"] = df[gender_col].astype(str)
        df["_age"] = df[age_col].astype(str)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(int(max_samples), random_state=seed).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        self.meta = df[["_img", "_race", "_gender", "_age"]].rename(
            columns={"_img": "file", "_race": "race", "_gender": "gender", "_age": "age"}
        )
        self.transform = get_eval_transform(face_size)
        self.face_size = face_size

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.meta.iloc[idx]
        rel = row["file"]
        path = self.root / rel if not Path(rel).is_absolute() else Path(rel)
        if not path.is_file():
            raise FileNotFoundError(f"FairFace image missing: {path}")

        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        tensor_chw = self.transform(arr)
        # One temporal frame: (1, C, H, W) for visual backbone (B, T, C, H, W)
        video = tensor_chw.unsqueeze(0)
        return video, 0
