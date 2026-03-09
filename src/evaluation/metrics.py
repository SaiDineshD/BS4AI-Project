"""
Evaluation metrics for liveness/spoof detection.
Includes EER, min t-DCF, accuracy, and AUC — standard in ASVspoof evaluations.
"""

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding threshold.

    Args:
        labels: ground truth (0 = bonafide/real, 1 = spoof/fake)
        scores: model scores (higher = more likely spoof)
    Returns:
        (eer, threshold) tuple
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fpr - fnr)(x), 0.0, 1.0)
        thresh = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float(np.mean([fpr[idx], fnr[idx]]))
        thresh = float(thresholds[idx])

    return eer, thresh


def compute_min_tdcf(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    c_miss_spoof: float = 1.0,
    c_fa_spoof: float = 10.0,
    p_target: float = 0.9405,
) -> Tuple[float, float]:
    """Compute minimum tandem detection cost function (t-DCF).

    Simplified version following ASVspoof 2019 evaluation protocol.
    """
    all_scores = np.concatenate([bonafide_scores, spoof_scores])
    thresholds = np.sort(all_scores)

    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)

    min_tdcf = float("inf")
    best_thresh = thresholds[0]

    for thresh in thresholds:
        p_miss = np.sum(bonafide_scores > thresh) / max(n_bonafide, 1)
        p_fa = np.sum(spoof_scores <= thresh) / max(n_spoof, 1)

        tdcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        if tdcf < min_tdcf:
            min_tdcf = tdcf
            best_thresh = thresh

    # Normalize by default cost (no system)
    default_cost = min(c_miss * p_target, c_fa * (1 - p_target))
    if default_cost > 0:
        min_tdcf /= default_cost

    return min_tdcf, best_thresh


def compute_all_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = None
) -> Dict[str, float]:
    """Compute a full suite of evaluation metrics.

    Args:
        labels: binary ground truth
        scores: continuous model scores (higher = spoof)
        threshold: decision threshold; if None, uses EER threshold
    Returns:
        dict with eer, eer_threshold, min_tdcf, accuracy, auc
    """
    eer, eer_thresh = compute_eer(labels, scores)

    if threshold is None:
        threshold = eer_thresh

    predictions = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, predictions)

    try:
        auc_score = roc_auc_score(labels, scores)
    except ValueError:
        auc_score = 0.0

    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    min_tdcf = 0.0
    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        min_tdcf, _ = compute_min_tdcf(bonafide_scores, spoof_scores)

    return {
        "eer": eer,
        "eer_threshold": eer_thresh,
        "min_tdcf": min_tdcf,
        "accuracy": acc,
        "auc": auc_score,
    }
