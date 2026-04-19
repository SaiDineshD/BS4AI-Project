"""
Visual backbone for deepfake/liveness detection.
Supports ResNet18 and EfficientNet-B0 with frozen early layers to limit overfitting.
Temporal aggregation over video frames for video-level embedding.
"""

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

BackboneName = Literal["resnet18", "resnet34", "efficientnet_b0"]


class VisualBackbone(nn.Module):
    """Visual feature extractor for face video frames.

    Input:  (B, T, C, H, W) — batch of T-frame face videos
    Output: (B, num_classes) logits from classifier head
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_layers: int = 6,
        dropout: float = 0.5,
        backbone: BackboneName = "resnet18",
    ):
        """
        Args:
            freeze_layers: ResNet: number of children in feature trunk to freeze (0–8).
                           EfficientNet-B0: number of leading blocks in .features to freeze.
            backbone: ``resnet18``, ``resnet34``, or ``efficientnet_b0``.
        """
        super().__init__()
        self.backbone_name = backbone
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        if backbone == "resnet18":
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

        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            resnet = models.resnet34(weights=weights)
            children = list(resnet.children())
            self.feature_extractor = nn.Sequential(*children[:-1])
            resnet_out_dim = resnet.fc.in_features

            if pretrained and freeze_layers > 0:
                for i, child in enumerate(self.feature_extractor.children()):
                    if i < freeze_layers:
                        for param in child.parameters():
                            param.requires_grad = False

        elif backbone == "efficientnet_b0":
            weights = (
                models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            enet = models.efficientnet_b0(weights=weights)
            self.feature_extractor = nn.Sequential(enet.features, enet.avgpool)
            self._feat_flatten = nn.Flatten()
            resnet_out_dim = enet.classifier[1].in_features

            if pretrained and freeze_layers > 0:
                for i, block in enumerate(enet.features.children()):
                    if i < freeze_layers:
                        for param in block.parameters():
                            param.requires_grad = False
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

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
        if self.backbone_name == "efficientnet_b0":
            features = self._feat_flatten(features)
        else:
            features = features.view(B * T, -1)

        features = features.view(B, T, -1)
        features = features.permute(0, 2, 1)
        features = self.temporal_pool(features).squeeze(-1)

        return self.embedding(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.extract_features(x)
        return self.classifier(embedding)


def build_visual_backbone(
    backbone: str = "resnet18",
    embedding_dim: int = 512,
    pretrained: bool = True,
    freeze_layers: int = 6,
    dropout: float = 0.5,
    num_classes: int = 2,
) -> VisualBackbone:
    """Factory used by training scripts and experiments."""
    b = backbone.lower().replace("-", "_")
    if b not in ("resnet18", "resnet34", "efficientnet_b0"):
        raise ValueError(f"Unsupported visual backbone: {backbone}")
    fl = freeze_layers
    if b == "efficientnet_b0" and fl > 8:
        fl = 8
    return VisualBackbone(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        num_classes=num_classes,
        freeze_layers=fl,
        dropout=dropout,
        backbone=b,  # type: ignore[arg-type]
    )
