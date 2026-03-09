"""
Multi-modal liveness detector.
Combines visual (ResNet18) and audio (LCNN) backbones through a fusion module
to produce a unified liveness/spoof decision.
"""

import torch
import torch.nn as nn

from .audio_backbone import AudioBackbone
from .fusion import build_fusion
from .visual_backbone import VisualBackbone


class MultiModalLivenessDetector(nn.Module):
    """End-to-end multi-modal liveness detection model.

    Accepts video frames and audio spectrograms, extracts modality-specific
    embeddings, and fuses them for a binary live/spoof prediction.
    """

    def __init__(
        self,
        visual_embedding_dim: int = 512,
        audio_embedding_dim: int = 512,
        fusion_method: str = "concat",
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
        pretrained_visual: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        self.visual_backbone = VisualBackbone(
            embedding_dim=visual_embedding_dim,
            pretrained=pretrained_visual,
            num_classes=num_classes,
        )
        self.audio_backbone = AudioBackbone(
            embedding_dim=audio_embedding_dim,
            num_classes=num_classes,
        )
        self.fusion = build_fusion(
            method=fusion_method,
            visual_dim=visual_embedding_dim,
            audio_dim=audio_embedding_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            num_classes=num_classes,
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> dict:
        """
        Args:
            video: (B, T, C, H, W) face video frames
            audio: (B, 1, n_lfcc, time) LFCC spectrogram
        Returns:
            dict with keys:
                - fused_logits: (B, 2) from fusion module
                - visual_logits: (B, 2) from visual-only classifier
                - audio_logits: (B, 2) from audio-only classifier
                - visual_emb: (B, visual_embedding_dim)
                - audio_emb: (B, audio_embedding_dim)
        """
        visual_emb = self.visual_backbone.extract_features(video)
        audio_emb = self.audio_backbone.extract_features(audio)

        visual_logits = self.visual_backbone.classifier(visual_emb)
        audio_logits = self.audio_backbone.classifier(audio_emb)
        fused_logits = self.fusion(visual_emb, audio_emb)

        return {
            "fused_logits": fused_logits,
            "visual_logits": visual_logits,
            "audio_logits": audio_logits,
            "visual_emb": visual_emb,
            "audio_emb": audio_emb,
        }

    def forward_visual_only(self, video: torch.Tensor) -> torch.Tensor:
        """Visual-only inference path."""
        return self.visual_backbone(video)

    def forward_audio_only(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio-only inference path."""
        return self.audio_backbone(audio)
