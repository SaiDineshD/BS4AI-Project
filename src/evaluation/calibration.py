"""Threshold sweeps for binary spoof detection (oriented scores; label 1 = spoof)."""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from src.evaluation.metrics import orient_scores_for_spoof


def sweep_threshold_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 101,
    apply_orientation: bool = True,
) -> List[Dict[str, float]]:
    """Scan thresholds on oriented scores; return FRR, FAR, accuracy per threshold.

    FRR = P(predict spoof | genuine), FAR = P(predict genuine | spoof).
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if apply_orientation:
        scores, _, _ = orient_scores_for_spoof(labels, scores)

    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        thr = np.array([(lo + hi) / 2])
    else:
        thr = np.linspace(lo, hi, n_thresholds)

    gen = labels == 0
    spf = labels == 1
    n_g = max(int(gen.sum()), 1)
    n_s = max(int(spf.sum()), 1)

    rows: List[Dict[str, float]] = []
    for t in thr:
        pred = (scores >= t).astype(int)
        frr = float(((pred == 1) & gen).sum() / n_g)
        far = float(((pred == 0) & spf).sum() / n_s)
        acc = float(accuracy_score(labels, pred))
        rows.append({"threshold": float(t), "frr": frr, "far": far, "accuracy": acc})
    return rows


def best_threshold_min_max_frr_far(
    labels: np.ndarray,
    scores: np.ndarray,
    apply_orientation: bool = True,
) -> Optional[Dict[str, float]]:
    """Pick threshold minimizing max(FRR, FAR) on oriented scores."""
    rows = sweep_threshold_metrics(labels, scores, apply_orientation=apply_orientation)
    if not rows:
        return None
    best = min(rows, key=lambda r: max(r["frr"], r["far"]))
    out = dict(best)
    out["max_frr_far"] = max(best["frr"], best["far"])
    return out
