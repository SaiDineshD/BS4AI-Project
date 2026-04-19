"""
Multi-modal fusion module.
Combines visual and audio embeddings for joint liveness/spoof detection.
Supports concatenation, attention-based fusion, and contrastive distance learning
(cross-modal consistency: pull genuine pairs, push attack / mismatch pairs).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Simple concatenation fusion: visual_emb || audio_emb -> MLP -> score."""

    def __init__(
        self,
        visual_dim: int = 512,
        audio_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, visual_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([visual_emb, audio_emb], dim=1)
        return self.classifier(combined)


class AttentionFusion(nn.Module):
    """Cross-modal attention fusion.
    Learns to weight visual and audio contributions based on their compatibility.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        audio_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, visual_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> torch.Tensor:
        v = self.visual_proj(visual_emb)
        a = self.audio_proj(audio_emb)

        combined = torch.cat([v, a], dim=1)
        weights = self.attention(combined)  # (B, 2)

        fused = weights[:, 0:1] * v + weights[:, 1:2] * a
        return self.classifier(fused)


class ContrastiveMetricFusion(nn.Module):
    """Fusion with L2-normalized projections and cosine distance.

    Trained with CE on logits plus a margin contrastive term on embedding distance:
    genuine pairs (y=0) minimize distance; spoof / mismatch (y=1) maximize distance
    past a margin (supervised contrastive on paired embeddings).
    """

    def __init__(
        self,
        visual_dim: int = 512,
        audio_dim: int = 512,
        proj_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)
        self.margin = 0.5
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2 + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self, visual_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v = F.normalize(self.visual_proj(visual_emb), dim=1)
        a = F.normalize(self.audio_proj(audio_emb), dim=1)
        cos = (v * a).sum(dim=1, keepdim=True)
        dist = 1.0 - cos.squeeze(1)
        h = torch.cat([v, a, cos], dim=1)
        logits = self.classifier(h)
        return logits, dist

    @staticmethod
    def contrastive_loss(dist: torch.Tensor, labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """Pairing loss: label 0 = aligned (low distance), label 1 = inconsistent (distance >= margin)."""
        y = labels.float()
        genuine = (1.0 - y) * dist.pow(2)
        attack = y * F.relu(margin - dist).pow(2)
        return (genuine + attack).mean()


def build_fusion(method: str = "concat", **kwargs) -> nn.Module:
    """Factory function to construct the requested fusion module."""
    methods = {
        "concat": ConcatFusion,
        "attention": AttentionFusion,
        "contrastive": ContrastiveMetricFusion,
    }
    if method not in methods:
        raise ValueError(f"Unknown fusion method: {method}. Choose from {list(methods)}")
    return methods[method](**kwargs)
