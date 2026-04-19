#!/usr/bin/env python3
"""
FairFace genuine-only fairness evaluation for the trained visual backbone.

Uses the same score orientation as FF++ evaluation: optional threshold from
experiment_summary.json (visual_test_metrics.eer_threshold + scores_flipped),
or calibrates on the FF++ validation split, then reports FRR stratified by
race / gender / age on FairFace (all ground-truth labels are genuine = 0).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ff_c23_loader import FF_C23_Dataset
from src.data.fairface_loader import FairFaceImageDataset
from src.data.sampling import load_data_config
from src.evaluation.fairness_eval import evaluate_fairness
from src.evaluation.metrics import compute_all_metrics
from src.models.visual_backbone import build_visual_backbone
from src.utils.torch_device import get_torch_device


@torch.no_grad()
def collect_probs_visual(model, loader, device):
    model.eval()
    scores, labels = [], []
    for inputs, y in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def calibrate_on_ff_val(data_cfg: dict, model_config: dict, checkpoint: Path, device):
    """Return (eer_threshold, scores_flipped) from FF++ val split."""
    root = data_cfg["paths"]["ff_c23"]
    csv_dir = data_cfg["ff_c23"]["csv_dir"]
    face_size = data_cfg["ff_c23"]["frame_extraction"]["face_size"]
    nf = int(data_cfg["ff_c23"]["frame_extraction"]["num_frames"])
    n = data_cfg["sampling"].get("n_per_dataset")

    val_ds = FF_C23_Dataset(
        root=root,
        csv_dir=csv_dir,
        split="val",
        num_frames=nf,
        face_size=face_size,
        n_samples=n,
        seed=data_cfg["sampling"]["seed"],
        use_face_detection=True,
        face_detector="haar",
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    vb = model_config.get("visual", {}).get("backbone", "resnet18")
    model = build_visual_backbone(
        backbone=vb,
        embedding_dim=model_config["visual"]["embedding_dim"],
        pretrained=False,
        freeze_layers=0,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    raw, lbls = collect_probs_visual(model, loader, device)
    m = compute_all_metrics(lbls, raw)
    return float(m["eer_threshold"]), bool(m["scores_flipped"])


def apply_threshold(raw_scores: np.ndarray, threshold: float, scores_flipped: bool):
    if scores_flipped:
        s = 1.0 - raw_scores
    else:
        s = raw_scores
    preds = (s >= threshold).astype(int)
    return s, preds


def main():
    p = argparse.ArgumentParser(description="FairFace fairness eval (visual backbone)")
    p.add_argument("--data-config", default="config/data_config.yaml")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--visual-checkpoint", required=True, type=Path)
    p.add_argument(
        "--experiment-summary",
        type=Path,
        default=None,
        help="JSON with visual_test_metrics.eer_threshold and scores_flipped",
    )
    p.add_argument("--output-dir", type=Path, default=Path("outputs/fairness_eval"))
    p.add_argument("--threshold", type=float, default=None, help="Override EER threshold")
    p.add_argument("--no-flip", action="store_true", help="Ignore scores_flipped from summary")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap FairFace rows (default: fairface.max_samples in data_config.yaml)",
    )
    args = p.parse_args()

    data_cfg = load_data_config(str(args.data_config))
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    ff_cfg = data_cfg.get("fairface") or {}
    root = Path(data_cfg.get("paths", {}).get("fairface") or ff_cfg.get("root", "data/FairFace"))
    if not root.is_dir():
        sys.exit(
            f"FairFace root directory not found: {root.resolve()}\n"
            "Download FairFace and set paths.fairface in config/data_config.yaml."
        )

    csv_name = ff_cfg.get("train_csv", "fairface_label_train.csv")
    device = get_torch_device()

    scores_flipped = False
    threshold = args.threshold

    if threshold is None:
        if args.experiment_summary and args.experiment_summary.is_file():
            with open(args.experiment_summary) as f:
                summ = json.load(f)
            vm = summ.get("visual_test_metrics") or {}
            threshold = float(vm.get("eer_threshold", 0.5))
            scores_flipped = bool(vm.get("scores_flipped", False))
            if args.no_flip:
                scores_flipped = False
            print(f"Using threshold from summary: {threshold}, scores_flipped={scores_flipped}")
        else:
            print("Calibrating threshold on FF++ validation split...")
            threshold, scores_flipped = calibrate_on_ff_val(
                data_cfg, model_config, args.visual_checkpoint, device
            )
            print(f"Calibrated threshold: {threshold}, scores_flipped={scores_flipped}")

    max_samples = args.max_samples if args.max_samples is not None else ff_cfg.get("max_samples")
    ds = FairFaceImageDataset(
        root=str(root),
        csv_path=csv_name,
        face_size=int(ff_cfg.get("face_size", 224)),
        max_samples=max_samples,
        seed=int(data_cfg["sampling"]["seed"]),
        img_path_column=str(ff_cfg.get("img_path_column", "file")),
        race_column=str(ff_cfg.get("race_column", "race")),
        gender_column=str(ff_cfg.get("gender_column", "gender")),
        age_column=str(ff_cfg.get("age_column", "age")),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    vb = model_config.get("visual", {}).get("backbone", "resnet18")
    model = build_visual_backbone(
        backbone=vb,
        embedding_dim=model_config["visual"]["embedding_dim"],
        pretrained=False,
        freeze_layers=0,
    )
    ckpt = torch.load(args.visual_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    raw_scores, _ = collect_probs_visual(model, loader, device)
    _, preds = apply_threshold(raw_scores, threshold, scores_flipped)
    labels = np.zeros(len(preds), dtype=int)
    demographics = ds.meta.reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = evaluate_fairness(labels, preds, demographics, output_dir=str(args.output_dir))

    report = {
        "protocol": "fairface_genuine_only_frr",
        "n_samples": int(len(preds)),
        "threshold": threshold,
        "scores_flipped": scores_flipped,
        "checkpoint": str(args.visual_checkpoint),
        "summary": out["summary"],
    }
    with open(args.output_dir / "fairness_summary.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote {args.output_dir / 'fairness_summary.json'} and fairness plots.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
