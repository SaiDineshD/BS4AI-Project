"""Load mono waveform from video files (FaceForensics++ clips)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import torch
import torchaudio


def load_waveform_from_video(
    path: str,
    target_sr: int,
    max_length: int,
) -> Tuple[torch.Tensor, int]:
    """Load (1, T) float waveform at ``target_sr``, length capped/padded to ``max_length``.

    Tries ``torchaudio.load`` first; falls back to ffmpeg → temp WAV if needed.
    """
    path = str(path)
    w: torch.Tensor
    sr: int
    try:
        w, sr = torchaudio.load(path)
    except Exception:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-i",
                    path,
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sr),
                    "-f",
                    "wav",
                    tmp_path,
                ],
                check=True,
                capture_output=True,
            )
            w, sr = torchaudio.load(tmp_path)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    if w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if sr != target_sr:
        ratio = target_sr / sr
        new_len = int(w.shape[1] * ratio)
        w = torch.nn.functional.interpolate(
            w.unsqueeze(0), size=new_len, mode="linear", align_corners=False
        ).squeeze(0)
        sr = target_sr
    if w.shape[1] > max_length:
        w = w[:, :max_length]
    elif w.shape[1] < max_length:
        w = torch.nn.functional.pad(w, (0, max_length - w.shape[1]))
    return w, sr
