"""
Train the audio (LCNN) backbone on ASVspoof 2019 LA data.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.asvspoof_loader import build_asvspoof_datasets
from src.data.sampling import load_data_config
from src.models.audio_backbone import AudioBackbone
from src.training.trainer import Trainer


def train_audio(data_config_path: str, model_config: dict, output_dir: str = "outputs/audio"):
    data_cfg = load_data_config(data_config_path)
    datasets = build_asvspoof_datasets(data_cfg)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=model_config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["dev"],
        batch_size=model_config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioBackbone(
        embedding_dim=model_config["audio"]["embedding_dim"],
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
    use_amp = model_config.get("training", {}).get("use_amp", False)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        scheduler=scheduler,
        use_amp=use_amp,
    )

    return trainer.train(
        epochs=model_config["training"]["epochs"],
        patience=10,
    )
