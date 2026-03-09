"""
Deepfake Detection data loader.
The DeepFakeDetection subset is included within the FaceForensics++ C23 dataset
and is handled by ff_c23_loader.py as part of the 'fake' categories.

This module provides a convenience wrapper if you need to load
DeepFakeDetection videos separately.
"""

from pathlib import Path
from typing import Optional

from .ff_c23_loader import FF_C23_Dataset


class DeepFakeDetectionDataset(FF_C23_Dataset):
    """Loads only the DeepFakeDetection subset from FaceForensics++ C23."""

    def _load_metadata(self):
        """Override to load only DeepFakeDetection category."""
        import csv

        csv_path = self.csv_dir / "DeepFakeDetection.csv"
        if not csv_path.exists():
            return

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = self.root / row["File Path"]
                if video_path.exists():
                    self.samples.append(
                        {
                            "path": str(video_path),
                            "label": 1,  # all DeepFakeDetection are FAKE
                            "category": "DeepFakeDetection",
                            "frame_count": int(row["Frame Count"]),
                        }
                    )
