#!/usr/bin/env python3
"""
Run the full multi-modal liveness detection experiment.
Trains visual, audio, and fusion modules sequentially, then evaluates.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --mode visual   # visual only
    python scripts/run_experiment.py --mode audio    # audio only
    python scripts/run_experiment.py --mode fusion   # fusion only (requires pre-trained backbones)
    python scripts/run_experiment.py --mode all      # full pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.asvspoof_loader import build_asvspoof_datasets
from src.data.ff_c23_loader import build_ff_c23_datasets
from src.data.sampling import load_data_config
from src.evaluation.metrics import compute_all_metrics, metrics_to_jsonable
from src.training.train_audio import train_audio
from src.training.train_visual import train_visual
from src.utils.torch_device import get_torch_device


def load_model_config(path: str = "config/model_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(model, loader, device, forward_fn=None):
    """Evaluate a model and return metrics."""
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if forward_fn:
                logits, labels = forward_fn(model, batch, device)
            else:
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    return compute_all_metrics(all_labels, all_scores)


def run_visual(data_config_path, model_config, output_dir):
    """Train and evaluate visual backbone."""
    print("\n" + "=" * 60)
    print("VISUAL BACKBONE TRAINING (ResNet18 on FaceForensics++ C23)")
    print("=" * 60)

    history = train_visual(data_config_path, model_config, str(output_dir / "visual"))

    data_cfg = load_data_config(data_config_path)
    datasets = build_ff_c23_datasets(data_cfg)
    test_loader = DataLoader(datasets["test"], batch_size=model_config["training"]["batch_size"])

    from src.models.visual_backbone import build_visual_backbone

    device = get_torch_device()
    vb = model_config.get("visual", {}).get("backbone", "resnet18")
    model = build_visual_backbone(
        backbone=vb,
        embedding_dim=model_config["visual"]["embedding_dim"],
        pretrained=model_config["visual"]["pretrained"],
        freeze_layers=model_config["visual"].get("freeze_layers", 0),
    )
    ckpt = torch.load(output_dir / "visual" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"\nVisual Test Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(output_dir / "visual" / "test_metrics.json", "w") as f:
        json.dump(metrics_to_jsonable(test_metrics), f, indent=2)

    return history, test_metrics


def run_audio(data_config_path, model_config, output_dir):
    """Train and evaluate audio backbone."""
    print("\n" + "=" * 60)
    print("AUDIO BACKBONE TRAINING (LCNN on ASVspoof 2019 LA)")
    print("=" * 60)

    history = train_audio(data_config_path, model_config, str(output_dir / "audio"))

    data_cfg = load_data_config(data_config_path)
    datasets = build_asvspoof_datasets(data_cfg)
    test_loader = DataLoader(datasets["eval"], batch_size=model_config["training"]["batch_size"])

    from src.models.audio_backbone import AudioBackbone
    device = get_torch_device()
    model = AudioBackbone(embedding_dim=model_config["audio"]["embedding_dim"])
    ckpt = torch.load(output_dir / "audio" / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"\nAudio Test Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(output_dir / "audio" / "test_metrics.json", "w") as f:
        json.dump(metrics_to_jsonable(test_metrics), f, indent=2)

    return history, test_metrics


def run_fusion(data_config_path, model_config, output_dir):
    """Train and evaluate the fusion module."""
    print("\n" + "=" * 60)
    print("FUSION TRAINING (Visual + Audio)")
    print("=" * 60)

    from src.training.train_fusion import train_fusion

    data_cfg = load_data_config(data_config_path)
    visual_datasets = build_ff_c23_datasets(data_cfg)
    audio_datasets = build_asvspoof_datasets(data_cfg)

    visual_loader = DataLoader(visual_datasets["train"], batch_size=model_config["training"]["batch_size"])
    audio_loader = DataLoader(audio_datasets["train"], batch_size=model_config["training"]["batch_size"])

    visual_ckpt = str(output_dir / "visual" / "best_model.pt")
    audio_ckpt = str(output_dir / "audio" / "best_model.pt")

    if not Path(visual_ckpt).exists() or not Path(audio_ckpt).exists():
        print("ERROR: Pre-trained backbones not found. Run visual and audio training first.")
        return None, None

    history = train_fusion(
        visual_checkpoint=visual_ckpt,
        audio_checkpoint=audio_ckpt,
        visual_loader=visual_loader,
        audio_loader=audio_loader,
        model_config=model_config,
        output_dir=str(output_dir / "fusion"),
    )

    return history, {}


def main():
    parser = argparse.ArgumentParser(description="Multi-modal liveness detection experiment")
    parser.add_argument("--data-config", default="config/data_config.yaml")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--output", default="outputs")
    parser.add_argument(
        "--mode", default="all", choices=["visual", "audio", "fusion", "all"],
        help="Which component to train",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP for training (Trainer; no-op on CPU)",
    )
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    if args.amp:
        model_config.setdefault("training", {})["use_amp"] = True
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Multi-Modal Liveness Detection Experiment")
    print(f"Data config: {args.data_config}")
    print(f"Model config: {args.model_config}")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Device: {get_torch_device()}")

    start = time.time()
    results = {}

    if args.mode in ("visual", "all"):
        history, metrics = run_visual(args.data_config, model_config, output_dir)
        results["visual"] = {"history": history, "test_metrics": metrics}

    if args.mode in ("audio", "all"):
        history, metrics = run_audio(args.data_config, model_config, output_dir)
        results["audio"] = {"history": history, "test_metrics": metrics}

    if args.mode in ("fusion", "all"):
        history, metrics = run_fusion(args.data_config, model_config, output_dir)
        results["fusion"] = {"history": history}

    elapsed = time.time() - start
    print(f"\nTotal experiment time: {elapsed:.1f}s")

    summary = {}
    for component, data in results.items():
        if data.get("test_metrics"):
            summary[component] = data["test_metrics"]

    if summary:
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for comp, metrics in summary.items():
            print(f"\n{comp.upper()}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

        with open(output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
