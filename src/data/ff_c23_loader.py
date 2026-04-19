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

from src.data.audio_featurizer import CepstralFeaturizer
from src.data.video_audio import load_waveform_from_video


def load_ff_c23_metadata(root: Path, csv_dir: Path) -> List[Dict]:
    """Load all available video samples from FF++ CSVs (paths must exist on disk)."""
    samples: List[Dict] = []
    csv_files = {
        "original": ("original.csv", 0),
        "Deepfakes": ("Deepfakes.csv", 1),
        "Face2Face": ("Face2Face.csv", 1),
        "FaceSwap": ("FaceSwap.csv", 1),
        "FaceShifter": ("FaceShifter.csv", 1),
        "NeuralTextures": ("NeuralTextures.csv", 1),
    }
    for category, (csv_name, label) in csv_files.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = root / row["File Path"]
                if video_path.exists():
                    samples.append({
                        "path": str(video_path),
                        "label": label,
                        "category": category,
                        "frame_count": int(row["Frame Count"]),
                    })
    return samples


def balanced_sample_pool(
    samples: List[Dict], n_samples: Optional[int], seed: int
) -> List[Dict]:
    """Return a balanced real/fake subset (or full list if n_samples is None or larger than pool)."""
    if not samples:
        return []
    if not n_samples or len(samples) <= n_samples:
        return list(samples)
    random.seed(seed)
    real = [s for s in samples if s["label"] == 0]
    fake = [s for s in samples if s["label"] == 1]
    random.shuffle(real)
    random.shuffle(fake)
    n_each = n_samples // 2
    out = real[:n_each] + fake[: n_samples - n_each]
    random.seed(seed)
    random.shuffle(out)
    return out


def train_val_test_split_indices(
    n: int, seed: int, train_ratio: float, val_ratio: float
) -> Tuple[List[int], List[int], List[int]]:
    """Return index lists for train / val / test partitions."""
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_val]
    test_idx = indices[n_val:]
    return train_idx, val_idx, test_idx


def build_ff_samples_for_split(
    root: str,
    csv_dir: str,
    split: str,
    n_samples: Optional[int],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    explicit_samples: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build the sample list for one split (used by FF_C23_Dataset and k-fold)."""
    if explicit_samples is not None:
        return list(explicit_samples)
    pool = load_ff_c23_metadata(Path(root), Path(csv_dir))
    pool = balanced_sample_pool(pool, n_samples, seed)
    train_idx, val_idx, test_idx = train_val_test_split_indices(
        len(pool), seed, train_ratio, val_ratio
    )
    if split == "train":
        idx = train_idx
    elif split == "val":
        idx = val_idx
    else:
        idx = test_idx
    return [pool[i] for i in idx]


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
    """PyTorch dataset for FaceForensics++ C23 video frames.

    Set ``include_audio=True`` to also return cepstral features from the **same video file’s**
    soundtrack (time-aligned with the sampled face frames), for same-clip AV fusion.
    """

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
        face_detector: str = "haar",
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        explicit_samples: Optional[List[Dict]] = None,
        include_audio: bool = False,
        audio_cfg: Optional[Dict] = None,
    ):
        self.root = Path(root)
        self.csv_dir = Path(csv_dir)
        self.num_frames = num_frames
        self.face_size = face_size
        self.use_face_detection = use_face_detection
        self.face_detector = (face_detector or "haar").lower()
        self._face_cascade = None
        self._mp_detector = None

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transform(face_size)
        else:
            self.transform = get_eval_transform(face_size)

        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/processed/frame_cache_v2")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if explicit_samples is not None:
            self.samples = list(explicit_samples)
        else:
            self.samples = build_ff_samples_for_split(
                root=str(self.root),
                csv_dir=str(self.csv_dir),
                split=split,
                n_samples=n_samples,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                explicit_samples=None,
            )

        self.include_audio = bool(include_audio)
        self._audio_fea: Optional[CepstralFeaturizer] = None
        if self.include_audio:
            if not audio_cfg:
                raise ValueError("include_audio=True requires audio_cfg (e.g. asvspoof2019.audio from data_config).")
            self._audio_fea = CepstralFeaturizer(audio_cfg)

        self._preextract_all()

    def _cache_key(self, video_path: str) -> str:
        det = self.face_detector if self.use_face_detection else "cc"
        h = hashlib.md5(f"{video_path}_{self.num_frames}_{det}".encode()).hexdigest()[:12]
        return h

    def _audio_cache_key(self, video_path: str) -> str:
        assert self._audio_fea is not None
        raw = (
            f"{video_path}|{self._audio_fea.feature_type}|{self._audio_fea.max_length}|"
            f"{self._audio_fea.sample_rate}|{self._audio_fea.n_lfcc}|{self._audio_fea.n_fft}"
        )
        return hashlib.md5(raw.encode()).hexdigest()[:12]

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

        if self.include_audio and self._audio_fea is not None:
            n_au = 0
            n_auc = 0
            for sample in self.samples:
                ap = self.cache_dir / f"ava_{self._audio_cache_key(sample['path'])}.pt"
                if ap.exists():
                    n_auc += 1
                    continue
                try:
                    w, _ = load_waveform_from_video(
                        sample["path"],
                        self._audio_fea.sample_rate,
                        self._audio_fea.max_length,
                    )
                    feat = self._audio_fea(w)
                    torch.save(feat.cpu(), ap)
                    n_au += 1
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to extract audio from {sample['path']}: {e}\n"
                        "Install ffmpeg and ensure videos contain an audio track."
                    ) from e
                if n_au % 20 == 0:
                    print(f"    Audio features: {n_au} new, {n_auc} cached")
            if n_au > 0:
                print(f"    Audio extraction: {n_au} new, {n_auc} cached")

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

    def _crop_face_mediapipe(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            import mediapipe as mp
        except ImportError:
            return None

        if self._mp_detector is None:
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )

        h, w = frame.shape[:2]
        rgb = frame
        res = self._mp_detector.process(rgb)
        if not res.detections:
            return None
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = int(max(bbox.xmin * w, 0))
        y1 = int(max(bbox.ymin * h, 0))
        x2 = int(min((bbox.xmin + bbox.width) * w, w))
        y2 = int(min((bbox.ymin + bbox.height) * h, h))
        if x2 <= x1 or y2 <= y1:
            return None
        margin = int(0.2 * max(x2 - x1, y2 - y1))
        y1, y2 = max(0, y1 - margin), min(h, y2 + margin)
        x1, x2 = max(0, x1 - margin), min(w, x2 + margin)
        return frame[y1:y2, x1:x2]

    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        if self.use_face_detection and self.face_detector == "mediapipe":
            mp_crop = self._crop_face_mediapipe(frame)
            if mp_crop is not None and mp_crop.size > 0:
                return mp_crop

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
        if self.include_audio and self._audio_fea is not None:
            ap = self.cache_dir / f"ava_{self._audio_cache_key(sample['path'])}.pt"
            if ap.exists():
                audio_tensor = torch.load(ap, map_location="cpu", weights_only=False)
            else:
                w, _ = load_waveform_from_video(
                    sample["path"],
                    self._audio_fea.sample_rate,
                    self._audio_fea.max_length,
                )
                audio_tensor = self._audio_fea(w)
            return video_tensor, audio_tensor, sample["label"]
        return video_tensor, sample["label"]


def build_ff_c23_datasets(
    data_config: dict,
    n_samples: Optional[int] = None,
    face_detector: str = "haar",
) -> Dict[str, FF_C23_Dataset]:
    cfg = data_config
    root = cfg["paths"]["ff_c23"]
    csv_dir = cfg["ff_c23"]["csv_dir"]
    num_frames = cfg["ff_c23"]["frame_extraction"]["num_frames"]
    face_size = cfg["ff_c23"]["frame_extraction"]["face_size"]
    seed = cfg["sampling"]["seed"]
    n = n_samples if n_samples is not None else cfg["sampling"]["n_per_dataset"]

    datasets = {}
    for split in ["train", "val", "test"]:
        datasets[split] = FF_C23_Dataset(
            root=root,
            csv_dir=csv_dir,
            split=split,
            num_frames=num_frames,
            face_size=face_size,
            n_samples=n,
            seed=seed,
            use_face_detection=True,
            face_detector=face_detector,
        )
    return datasets
