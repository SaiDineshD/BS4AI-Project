"""
Visual backbone for deepfake/liveness detection.
Uses a pre-trained ResNet18 with frozen early layers to prevent overfitting.
Temporal aggregation over video frames for video-level embedding.
"""

import torch
import torch.nn as nn
from torchvision import models


class VisualBackbone(nn.Module):
    """ResNet18-based visual feature extractor for face video frames.

    Input:  (B, T, C, H, W) — batch of T-frame face videos
    Output: (B, embedding_dim) — aggregated video-level embedding
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_layers: int = 6,
        dropout: float = 0.5,
    ):
        """
        Args:
            freeze_layers: number of ResNet children to freeze (0-8).
                          6 = freeze everything except layer4 and avgpool.
        """
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        children = list(resnet.children())
        self.feature_extractor = nn.Sequential(*children[:-1])
        resnet_out_dim = resnet.fc.in_features

        if pretrained and freeze_layers > 0:
            for i, child in enumerate(self.feature_extractor.children()):
                if i < freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(resnet_out_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding without classification head.

        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            (B, embedding_dim) aggregated feature
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        features = self.feature_extractor(x)
        features = features.view(B, T, -1)

        features = features.permute(0, 2, 1)
        features = self.temporal_pool(features).squeeze(-1)

        return self.embedding(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_features(x)
        return self.classifier(embedding)
