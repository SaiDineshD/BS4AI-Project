"""
Audio backbone for speech anti-spoofing / liveness detection.
Implements LCNN (Light CNN) architecture operating on LFCC spectrograms,
following the ASVspoof 2019 baseline approach.
"""

import torch
import torch.nn as nn


class MaxFeatureMap(nn.Module):
    """Max Feature Map activation (MFM).
    Splits channels in half and takes element-wise max, halving channel count.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        a, b = x[:, : c // 2], x[:, c // 2 :]
        return torch.max(a, b)


class LCNNBlock(nn.Module):
    """LCNN block: Conv2d -> BatchNorm -> MFM."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mfm(self.bn(self.conv(x)))


class AudioBackbone(nn.Module):
    """LCNN-based audio feature extractor for spoof detection.

    Input:  (B, 1, n_lfcc, T) — LFCC spectrogram
    Output: (B, embedding_dim) — audio embedding
    """

    def __init__(self, embedding_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            LCNNBlock(1, 32, 5, 2),
            nn.MaxPool2d(2, 2),
            LCNNBlock(32, 48, 3, 1),
            nn.MaxPool2d(2, 2),
            LCNNBlock(48, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            LCNNBlock(64, 32, 3, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Linear(32, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding without classification head.

        Args:
            x: (B, 1, n_lfcc, T) LFCC spectrogram
        Returns:
            (B, embedding_dim) audio feature
        """
        features = self.features(x)
        features = self.pool(features).view(features.size(0), -1)
        return self.embedding(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with classification.

        Args:
            x: (B, 1, n_lfcc, T) LFCC spectrogram
        Returns:
            (B, num_classes) logits
        """
        embedding = self.extract_features(x)
        return self.classifier(embedding)
