#!/usr/bin/env python3
"""Post-hoc FRR mitigation: per-demographic thresholds targeting equal FRR (FairFace genuine-only)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fairface_loader import FairFaceImageDataset
from src.data.sampling import load_data_config
from src.evaluation.fairness_eval import compute_fairness_summary, stratified_fairness_metrics
from src.evaluation.fairness_mitigation import per_group_thresholds_equal_frr
from src.models.visual_backbone import build_visual_backbone


@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    scores = []
    for inputs, _ in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
    return np.concatenate(scores)


def main():
    p = argparse.ArgumentParser(description="Per-group threshold mitigation for FairFace FRR")
    p.add_argument("--data-config", default="config/data_config.yaml")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--visual-checkpoint", type=Path, required=True)
    p.add_argument("--target-frr", type=float, default=0.05, help="Target FRR per group (quantile)")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    data_cfg = load_data_config(args.data_config)
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)

    ff_cfg = data_cfg.get("fairface") or {}
    root = Path(data_cfg.get("paths", {}).get("fairface", "data/FairFace"))
    csv_name = ff_cfg.get("train_csv", "fairface_label_train.csv")

    from src.utils.torch_device import get_torch_device

    device = get_torch_device()
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

    ds = FairFaceImageDataset(
        root=str(root),
        csv_path=csv_name,
        face_size=int(ff_cfg.get("face_size", 224)),
        max_samples=args.max_samples,
        seed=int(data_cfg["sampling"]["seed"]),
        img_path_column=str(ff_cfg.get("img_path_column", "file")),
        race_column=str(ff_cfg.get("race_column", "race")),
        gender_column=str(ff_cfg.get("gender_column", "gender")),
        age_column=str(ff_cfg.get("age_column", "age")),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    raw_scores = collect_scores(model, loader, device)
    demo = ds.meta.reset_index(drop=True)

    # Global threshold from median score (placeholder) — report stratified FRR at score > 0.5
    labels = np.zeros(len(raw_scores), dtype=int)
    global_preds = (raw_scores >= 0.5).astype(int)
    metrics_before = stratified_fairness_metrics(labels, global_preds, demo)
    summary_before = compute_fairness_summary(metrics_before)

    mit = per_group_thresholds_equal_frr(
        raw_scores, demo, target_frr=args.target_frr
    )

    report = {
        "before_global_threshold_0.5": summary_before,
        "mitigation": mit,
        "note": "Per-group thresholds set at (1-target_frr) quantile of scores within each group "
        "(genuine-only protocol; reduces max-min FRR spread vs a single global threshold).",
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
