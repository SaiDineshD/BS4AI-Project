"""
Train the visual (ResNet18) backbone on FaceForensics++ C23 data.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.ff_c23_loader import build_ff_c23_datasets
from src.data.sampling import load_data_config
from src.models.visual_backbone import VisualBackbone
from src.training.trainer import Trainer


def train_visual(data_config_path: str, model_config: dict, output_dir: str = "outputs/visual"):
    data_cfg = load_data_config(data_config_path)
    datasets = build_ff_c23_datasets(data_cfg)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=model_config["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=model_config["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisualBackbone(
        embedding_dim=model_config["visual"]["embedding_dim"],
        pretrained=model_config["visual"]["pretrained"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_config["training"]["lr"],
        weight_decay=model_config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        scheduler=scheduler,
    )

    return trainer.train(
        epochs=model_config["training"]["epochs"],
        patience=10,
    )
