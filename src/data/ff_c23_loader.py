"""
FaceForensics++ C23 data loader.
Loads video frames as face crops for the visual liveness detection stream.
Supports real (original) vs. fake (Deepfakes, Face2Face, FaceSwap, etc.) binary classification.
Includes training augmentations and per-sample frame caching for speed.
"""

import csv
import hashlib
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transform(face_size: int = 224) -> transforms.Compose:
    """Augmented transform for training to reduce overfitting."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((face_size + 32, face_size + 32)),
        transforms.RandomCrop(face_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_eval_transform(face_size: int = 224) -> transforms.Compose:
    """Deterministic transform for validation / test."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((face_size, face_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FF_C23_Dataset(Dataset):
    """PyTorch dataset for FaceForensics++ C23 video frames."""

    def __init__(
        self,
        root: str,
        csv_dir: str,
        split: str = "train",
        num_frames: int = 8,
        face_size: int = 224,
        transform: Optional[transforms.Compose] = None,
        n_samples: Optional[int] = None,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        use_face_detection: bool = True,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ):
        self.root = Path(root)
        self.csv_dir = Path(csv_dir)
        self.num_frames = num_frames
        self.face_size = face_size
        self.use_face_detection = use_face_detection

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(face_size)
        else:
            self.transform = get_eval_transform(face_size)

        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/processed/frame_cache_v2")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._face_cascade = None

        self.samples: List[Dict] = []
        self._load_metadata()

        # Balanced sampling: equal real and fake
        if n_samples and len(self.samples) > n_samples:
            random.seed(seed)
            real = [s for s in self.samples if s["label"] == 0]
            fake = [s for s in self.samples if s["label"] == 1]
            random.shuffle(real)
            random.shuffle(fake)
            n_each = n_samples // 2
            self.samples = real[:n_each] + fake[:n_samples - n_each]
            random.seed(seed)
            random.shuffle(self.samples)

        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * (train_ratio + val_ratio))

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_val]
        else:
            selected = indices[n_val:]

        self.samples = [self.samples[i] for i in selected]
        self._preextract_all()

    def _load_metadata(self):
        csv_files = {
            "original": ("original.csv", 0),
            "Deepfakes": ("Deepfakes.csv", 1),
            "Face2Face": ("Face2Face.csv", 1),
            "FaceSwap": ("FaceSwap.csv", 1),
            "FaceShifter": ("FaceShifter.csv", 1),
            "NeuralTextures": ("NeuralTextures.csv", 1),
        }

        for category, (csv_name, label) in csv_files.items():
            csv_path = self.csv_dir / csv_name
            if not csv_path.exists():
                continue
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    video_path = self.root / row["File Path"]
                    if video_path.exists():
                        self.samples.append({
                            "path": str(video_path),
                            "label": label,
                            "category": category,
                            "frame_count": int(row["Frame Count"]),
                        })

    def _cache_key(self, video_path: str) -> str:
        tag = "fd" if self.use_face_detection else "cc"
        h = hashlib.md5(f"{video_path}_{self.num_frames}_{tag}".encode()).hexdigest()[:12]
        return h

    def _preextract_all(self):
        n_cached = 0
        n_extracted = 0
        for sample in self.samples:
            cache_path = self.cache_dir / f"{self._cache_key(sample['path'])}.pkl"
            if cache_path.exists():
                n_cached += 1
                continue

            frames = self._extract_frames_raw(sample["path"], sample["frame_count"])
            processed = [self._crop_face(f) for f in frames]

            with open(cache_path, "wb") as f:
                pickle.dump(processed, f, protocol=pickle.HIGHEST_PROTOCOL)
            n_extracted += 1

            if n_extracted % 20 == 0:
                print(f"    Extracted: {n_extracted}/{len(self.samples) - n_cached}")

        if n_extracted > 0:
            print(f"    Frame extraction: {n_extracted} new, {n_cached} cached")

    def _extract_frames_raw(self, video_path: str, frame_count: int) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8)] * self.num_frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = frame_count

        indices = np.linspace(0, max(total_frames - 1, 0), self.num_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.face_size, self.face_size, 3), dtype=np.uint8))
        return frames[:self.num_frames]

    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        if self.use_face_detection:
            if self._face_cascade is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            if len(faces) > 0:
                areas = [w * h for (_, _, w, h) in faces]
                x, y, w, h = faces[np.argmax(areas)]
                margin = int(0.25 * max(w, h))
                y1, y2 = max(0, y - margin), min(frame.shape[0], y + h + margin)
                x1, x2 = max(0, x - margin), min(frame.shape[1], x + w + margin)
                return frame[y1:y2, x1:x2]

        h, w = frame.shape[:2]
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return frame[y1:y1 + size, x1:x1 + size]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        cache_path = self.cache_dir / f"{self._cache_key(sample['path'])}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                face_crops = pickle.load(f)
        else:
            frames = self._extract_frames_raw(sample["path"], sample["frame_count"])
            face_crops = [self._crop_face(f) for f in frames]

        processed = [self.transform(crop) for crop in face_crops]
        video_tensor = torch.stack(processed)
        return video_tensor, sample["label"]


def build_ff_c23_datasets(
    data_config: dict, n_samples: Optional[int] = None
) -> Dict[str, FF_C23_Dataset]:
    cfg = data_config
    root = cfg["paths"]["ff_c23"]
    csv_dir = cfg["ff_c23"]["csv_dir"]
    num_frames = cfg["ff_c23"]["frame_extraction"]["num_frames"]
    face_size = cfg["ff_c23"]["frame_extraction"]["face_size"]
    seed = cfg["sampling"]["seed"]
    n = n_samples or cfg["sampling"]["n_per_dataset"]

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = FF_C23_Dataset(
            root=root, csv_dir=csv_dir, split=split,
            num_frames=num_frames, face_size=face_size,
            n_samples=n, seed=seed, use_face_detection=True,
        )
    return datasets
