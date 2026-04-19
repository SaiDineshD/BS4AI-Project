"""Shared LFCC / MFCC featurization for waveforms (ASVspoof + FF++ video audio)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torchaudio


class CepstralFeaturizer:
    """Match :class:`~src.data.asvspoof_loader.ASVspoofDataset` feature pipeline."""

    def __init__(self, audio_cfg: Dict[str, Any]):
        self.sample_rate = int(audio_cfg["sample_rate"])
        self.max_length = int(audio_cfg["max_length"])
        self.n_lfcc = int(audio_cfg["n_lfcc"])
        self.n_fft = int(audio_cfg["n_fft"])
        self.feature_type = (audio_cfg.get("feature_type") or "lfcc").lower()
        self.n_mfcc = int(audio_cfg.get("n_mfcc", audio_cfg.get("n_lfcc", 60)))
        hop = max(1, self.n_fft // 4)
        n_mels = max(64, self.n_mfcc * 2)
        self._mfcc_tfm = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": hop,
                "n_mels": n_mels,
                "center": False,
            },
        )

    def normalize_waveform(self, waveform: torch.Tensor, source_sr: int) -> torch.Tensor:
        """Resample to ``sample_rate`` and pad/crop to ``max_length`` samples."""
        w = waveform
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        sr = source_sr
        if sr != self.sample_rate:
            ratio = self.sample_rate / sr
            new_len = int(w.shape[1] * ratio)
            w = torch.nn.functional.interpolate(
                w.unsqueeze(0), size=new_len, mode="linear", align_corners=False
            ).squeeze(0)
        if w.shape[1] > self.max_length:
            w = w[:, : self.max_length]
        elif w.shape[1] < self.max_length:
            w = torch.nn.functional.pad(w, (0, self.max_length - w.shape[1]))
        return w

    def _extract_lfcc(self, waveform: torch.Tensor) -> torch.Tensor:
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
        n = log_energy.shape[0]
        dct_matrix = torch.zeros(self.n_lfcc, n)
        for k in range(self.n_lfcc):
            for i in range(n):
                dct_matrix[k, i] = np.cos(np.pi * k * (2 * i + 1) / (2 * n))
        dct_matrix *= np.sqrt(2.0 / n)
        lfcc = torch.matmul(dct_matrix, log_energy)
        return lfcc.unsqueeze(0)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return (1, n_coeff, time) feature map."""
        if self.feature_type == "mfcc":
            w = waveform.squeeze(0)
            mfcc = self._mfcc_tfm(w)
            return mfcc.unsqueeze(0)
        return self._extract_lfcc(waveform)
