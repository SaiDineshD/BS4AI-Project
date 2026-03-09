"""
Train the multi-modal fusion model.
Loads pre-trained visual and audio backbones, freezes them (optionally),
and trains only the fusion classifier on paired embeddings.
"""

import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.liveness_detector import MultiModalLivenessDetector
from src.training.trainer import Trainer


class PairedEmbeddingDataset(Dataset):
    """Wraps pre-extracted visual and audio embeddings for fusion training.

    In practice the visual and audio datasets are independent (different
    subjects/samples), so we pair them combinatorially with matching labels:
    real-visual + bonafide-audio = genuine, fake-visual + spoof-audio = attack.
    """

    def __init__(
        self,
        visual_embeddings: torch.Tensor,
        visual_labels: torch.Tensor,
        audio_embeddings: torch.Tensor,
        audio_labels: torch.Tensor,
    ):
        self.v_emb = visual_embeddings
        self.v_labels = visual_labels
        self.a_emb = audio_embeddings
        self.a_labels = audio_labels

        # Create paired indices: match by label
        self.pairs = []
        v_real = (visual_labels == 0).nonzero(as_tuple=True)[0]
        v_fake = (visual_labels == 1).nonzero(as_tuple=True)[0]
        a_real = (audio_labels == 0).nonzero(as_tuple=True)[0]
        a_fake = (audio_labels == 1).nonzero(as_tuple=True)[0]

        n_genuine = min(len(v_real), len(a_real))
        for i in range(n_genuine):
            self.pairs.append((v_real[i].item(), a_real[i].item(), 0))

        n_attack = min(len(v_fake), len(a_fake))
        for i in range(n_attack):
            self.pairs.append((v_fake[i].item(), a_fake[i].item(), 1))

        # Cross-modal mismatches are also attacks
        n_cross = min(len(v_real), len(a_fake), n_genuine // 2)
        for i in range(n_cross):
            self.pairs.append((v_real[i].item(), a_fake[i].item(), 1))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        vi, ai, label = self.pairs[idx]
        return self.v_emb[vi], self.a_emb[ai], label


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings from a pre-trained backbone."""
    model.eval()
    all_emb = []
    all_labels = []

    for batch in loader:
        inputs, labels = batch
        inputs = inputs.to(device)

        if modality == "visual":
            emb = model.extract_features(inputs)
        else:
            emb = model.extract_features(inputs)

        all_emb.append(emb.cpu())
        all_labels.append(labels)

    return torch.cat(all_emb), torch.cat(all_labels)


def fusion_forward(model, batch, device):
    """Custom forward function for fusion training."""
    v_emb, a_emb, labels = batch
    v_emb = v_emb.to(device)
    a_emb = a_emb.to(device)
    labels = labels.to(device)
    logits = model(v_emb, a_emb)
    return logits, labels


def train_fusion(
    visual_checkpoint: str,
    audio_checkpoint: str,
    visual_loader: DataLoader,
    audio_loader: DataLoader,
    model_config: dict,
    output_dir: str = "outputs/fusion",
    freeze_backbones: bool = True,
):
    """Train the fusion module on paired visual-audio embeddings.

    Args:
        visual_checkpoint: path to trained visual backbone checkpoint
        audio_checkpoint: path to trained audio backbone checkpoint
        visual_loader: DataLoader for visual validation data
        audio_loader: DataLoader for audio validation data
        model_config: model configuration dict
        output_dir: where to save fusion checkpoints
        freeze_backbones: if True, only train fusion layers
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.models.audio_backbone import AudioBackbone
    from src.models.visual_backbone import VisualBackbone

    visual_model = VisualBackbone(embedding_dim=model_config["visual"]["embedding_dim"])
    audio_model = AudioBackbone(embedding_dim=model_config["audio"]["embedding_dim"])

    v_ckpt = torch.load(visual_checkpoint, map_location=device, weights_only=False)
    visual_model.load_state_dict(v_ckpt["model_state_dict"])
    visual_model.to(device)

    a_ckpt = torch.load(audio_checkpoint, map_location=device, weights_only=False)
    audio_model.load_state_dict(a_ckpt["model_state_dict"])
    audio_model.to(device)

    print("Extracting visual embeddings...")
    v_emb, v_labels = extract_embeddings(visual_model, visual_loader, device, "visual")
    print("Extracting audio embeddings...")
    a_emb, a_labels = extract_embeddings(audio_model, audio_loader, device, "audio")

    # Split into train/val (80/20)
    n_pairs_train = int(min(len(v_emb), len(a_emb)) * 0.8)

    train_dataset = PairedEmbeddingDataset(
        v_emb[:n_pairs_train], v_labels[:n_pairs_train],
        a_emb[:n_pairs_train], a_labels[:n_pairs_train],
    )
    val_dataset = PairedEmbeddingDataset(
        v_emb[n_pairs_train:], v_labels[n_pairs_train:],
        a_emb[n_pairs_train:], a_labels[n_pairs_train:],
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    from src.models.fusion import build_fusion

    fusion_model = build_fusion(
        method=model_config["fusion"]["method"],
        visual_dim=model_config["visual"]["embedding_dim"],
        audio_dim=model_config["audio"]["embedding_dim"],
        hidden_dim=model_config["fusion"]["hidden_dim"],
        dropout=model_config["fusion"]["dropout"],
    )

    optimizer = torch.optim.Adam(
        fusion_model.parameters(),
        lr=model_config["training"]["lr"],
        weight_decay=model_config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        forward_fn=fusion_forward,
    )

    return trainer.train(epochs=model_config["training"]["epochs"], patience=10)
