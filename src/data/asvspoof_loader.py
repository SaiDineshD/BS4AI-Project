"""
ASVspoof 2019 Logical Access data loader.
Loads audio waveforms and extracts LFCC features for the audio liveness detection stream.
Supports bonafide vs. spoofed binary classification.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    """PyTorch dataset for ASVspoof 2019 LA audio samples."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_length: int = 64000,
        n_lfcc: int = 60,
        n_fft: int = 512,
        n_samples: Optional[int] = None,
        seed: int = 42,
        feature_type: str = "lfcc",
        n_mfcc: int = 60,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_lfcc = n_lfcc
        self.n_fft = n_fft
        self.feature_type = (feature_type or "lfcc").lower()
        self.n_mfcc = int(n_mfcc)
        hop = max(1, n_fft // 4)
        n_mels = max(64, self.n_mfcc * 2)
        self._mfcc_tfm = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop,
                "n_mels": n_mels,
                "center": False,
            },
        )

        self.samples: List[Dict] = []
        self._load_protocol(split)

        if n_samples and len(self.samples) > n_samples:
            random.seed(seed)
            bonafide = [s for s in self.samples if s["label"] == 0]
            spoofed = [s for s in self.samples if s["label"] == 1]
            n_each = n_samples // 2
            random.shuffle(bonafide)
            random.shuffle(spoofed)
            self.samples = bonafide[:n_each] + spoofed[: n_samples - n_each]
            random.shuffle(self.samples)

    def _load_protocol(self, split: str):
        """Parse ASVspoof protocol file to build sample list."""
        split_map = {
            "train": (
                "ASVspoof2019_LA_train/flac",
                "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
            ),
            "dev": (
                "ASVspoof2019_LA_dev/flac",
                "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            ),
            "eval": (
                "ASVspoof2019_LA_eval/flac",
                "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
            ),
        }

        if split not in split_map:
            raise ValueError(f"Unknown split: {split}. Choose from {list(split_map)}")

        audio_dir_rel, protocol_rel = split_map[split]
        audio_dir = self.root / audio_dir_rel
        protocol_path = self.root / protocol_rel

        with open(protocol_path) as f:
            for line in f:
                parts = line.strip().split()
                speaker_id = parts[0]
                audio_id = parts[1]
                system_id = parts[3]
                key = parts[4]  # bonafide or spoof

                audio_path = audio_dir / f"{audio_id}.flac"
                if audio_path.exists():
                    self.samples.append(
                        {
                            "path": str(audio_path),
                            "label": 0 if key == "bonafide" else 1,
                            "speaker_id": speaker_id,
                            "system_id": system_id,
                        }
                    )

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio waveform to fixed length."""
        data, sr = sf.read(path, dtype="float32")

        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)

        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(waveform.shape[1] * ratio)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=new_len, mode="linear", align_corners=False
            ).squeeze(0)

        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, : self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_len = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        return waveform

    def _extract_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract LFCC (Linear Frequency Cepstral Coefficients) features."""
        n_filters = self.n_lfcc * 2
        spec = torch.stft(
            waveform.squeeze(0),
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        power_spec = spec.abs().pow(2)

        n_freqs = power_spec.shape[0]
        linear_fb = torch.zeros(n_filters, n_freqs)
        freq_bins = np.linspace(0, n_freqs - 1, n_filters + 2, dtype=int)
        for i in range(n_filters):
            start, center, end = freq_bins[i], freq_bins[i + 1], freq_bins[i + 2]
            for j in range(start, center):
                if center != start:
                    linear_fb[i, j] = (j - start) / (center - start)
            for j in range(center, end):
                if end != center:
                    linear_fb[i, j] = (end - j) / (end - center)

        mel_energy = torch.matmul(linear_fb, power_spec)
        mel_energy = torch.clamp(mel_energy, min=1e-10)
        log_energy = torch.log(mel_energy)

        # DCT to get cepstral coefficients
        n = log_energy.shape[0]
        dct_matrix = torch.zeros(self.n_lfcc, n)
        for k in range(self.n_lfcc):
            for i in range(n):
                dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        dct_matrix *= np.sqrt(2.0 / n)

        lfcc = torch.matmul(dct_matrix, log_energy)
        return lfcc.unsqueeze(0)  # (1, n_lfcc, time)

    def _extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """MFCC spectrogram (Mel filterbank + log + DCT) via torchaudio."""
        w = waveform.squeeze(0)
        mfcc = self._mfcc_tfm(w)
        return mfcc.unsqueeze(0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        waveform = self._load_audio(sample["path"])
        if self.feature_type == "mfcc":
            feat = self._extract_mfcc(waveform)
        else:
            feat = self._extract_lfcc(waveform)
        return feat, sample["label"]


def asvspoof_kwargs_from_config(audio_cfg: dict) -> dict:
    """Keyword args for :class:`ASVspoofDataset` from ``data_config.yaml`` ``asvspoof2019.audio``."""
    return {
        "sample_rate": audio_cfg["sample_rate"],
        "max_length": audio_cfg["max_length"],
        "n_lfcc": audio_cfg["n_lfcc"],
        "n_fft": audio_cfg["n_fft"],
        "feature_type": audio_cfg.get("feature_type", "lfcc"),
        "n_mfcc": int(audio_cfg.get("n_mfcc", audio_cfg.get("n_lfcc", 60))),
    }


def build_asvspoof_datasets(
    data_config: dict, n_samples: Optional[int] = None
) -> Dict[str, ASVspoofDataset]:
    """Build train/dev/eval datasets from config."""
    cfg = data_config
    root = cfg["paths"]["asvspoof2019"]
    audio_cfg = cfg["asvspoof2019"]["audio"]
    seed = cfg["sampling"]["seed"]
    n = n_samples or cfg["sampling"]["n_per_dataset"]

    ak = asvspoof_kwargs_from_config(audio_cfg)
    datasets = {}
    for split in ["train", "dev", "eval"]:
        datasets[split] = ASVspoofDataset(
            root=root,
            split=split,
            n_samples=n,
            seed=seed,
            **ak,
        )
    return datasets
