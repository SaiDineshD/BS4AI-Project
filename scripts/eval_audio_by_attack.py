#!/usr/bin/env python3
"""
Per–attack-type summary for a trained LCNN on ASVspoof 2019 LA.

Uses protocol ``system_id`` for spoof clips and reports n, mean oriented P(spoof),
and accuracy at the global EER threshold (from full eval set).
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.asvspoof_loader import ASVspoofDataset, asvspoof_kwargs_from_config
from src.data.sampling import load_data_config
from src.evaluation.metrics import compute_all_metrics, orient_scores_for_spoof
from src.models.audio_backbone import AudioBackbone
from src.utils.torch_device import get_torch_device


@torch.no_grad()
def collect_scores(model, ds, device, batch_size: int = 32):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores, labels, systems = [], [], []
    idx = 0
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        bs = int(y.shape[0])
        scores.append(p.cpu().numpy())
        labels.append(y.numpy())
        for _ in range(bs):
            systems.append(ds.samples[idx]["system_id"])
            idx += 1
    return np.concatenate(scores), np.concatenate(labels), np.array(systems)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-config", default="config/data_config.yaml")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="eval", choices=["train", "dev", "eval"])
    p.add_argument("--n-samples", type=int, default=0, help="0 = all samples in split")
    args = p.parse_args()

    data_cfg = load_data_config(args.data_config)
    root = data_cfg["paths"]["asvspoof2019"]
    acfg = data_cfg["asvspoof2019"]["audio"]
    n_eff = None if args.n_samples == 0 else args.n_samples

    ak = asvspoof_kwargs_from_config(acfg)
    ds = ASVspoofDataset(
        root=root,
        split=args.split,
        n_samples=n_eff,
        seed=data_cfg["sampling"]["seed"],
        **ak,
    )
    if len(ds) == 0:
        sys.exit("No ASVspoof samples loaded; check paths in data_config.yaml")

    device = get_torch_device()
    model = AudioBackbone(embedding_dim=512)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    raw_scores, labels, systems = collect_scores(model, ds, device)
    m = compute_all_metrics(labels, raw_scores)
    oriented_scores, _, _ = orient_scores_for_spoof(labels, raw_scores)
    thresh = float(m["eer_threshold"])
    preds = (oriented_scores >= thresh).astype(int)

    by_attack: dict = defaultdict(lambda: {"n": 0, "n_correct": 0, "score_sum": 0.0})

    for i in range(len(labels)):
        lab = int(labels[i])
        sid = str(systems[i])
        key = "bonafide" if lab == 0 else f"spoof:{sid}"
        by_attack[key]["n"] += 1
        by_attack[key]["score_sum"] += float(oriented_scores[i])
        if int(preds[i]) == lab:
            by_attack[key]["n_correct"] += 1

    rows = []
    for k in sorted(by_attack.keys()):
        v = by_attack[k]
        n = max(v["n"], 1)
        rows.append(
            {
                "group": k,
                "n": v["n"],
                "accuracy": v["n_correct"] / n,
                "mean_oriented_spoof_score": v["score_sum"] / n,
            }
        )

    out = {
        "split": args.split,
        "global_metrics": {
            "eer": float(m["eer"]),
            "auc": float(m["auc"]),
            "accuracy": float(m["accuracy"]),
            "eer_threshold": thresh,
            "scores_flipped": bool(m.get("scores_flipped", False)),
        },
        "per_group": rows,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
