"""
Evaluation metrics for liveness/spoof detection.
Includes EER, min t-DCF, accuracy, and AUC — standard in ASVspoof evaluations.

Scores are interpreted as "higher = more likely spoof (class 1)".
If the raw scores are inverted (AUC < 0.5), they are reflected to 1 - score
before all metrics. EER is additionally capped with min(eer, 1 - eer) so the
reported rate never exceeds 0.5 (50%).
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def orient_scores_for_spoof(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[np.ndarray, bool, float]:
    """Orient scores so higher means more spoof when both classes exist.

    Returns:
        oriented_scores: possibly reflected scores
        scores_flipped: True if 1 - scores was applied
        auc_raw: ROC-AUC on the original scores (before flip); 0.0 if undefined
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    auc_raw = 0.0
    scores_flipped = False

    if len(labels) == 0 or len(scores) == 0:
        return scores, False, 0.0

    uniq = np.unique(labels)
    if len(uniq) < 2:
        return scores, False, 0.0

    try:
        auc_raw = float(roc_auc_score(labels, scores))
    except ValueError:
        auc_raw = 0.0

    if auc_raw < 0.5:
        scores = 1.0 - scores
        scores_flipped = True

    return scores, scores_flipped, auc_raw


def symmetrize_eer(eer: float) -> float:
    """Ensure EER is in [0, 0.5] (equivalently EER% in [0, 50])."""
    e = float(np.clip(eer, 0.0, 1.0))
    return float(min(e, 1.0 - e))


def _compute_eer_from_curve(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    """EER and threshold from ROC; scores must already be spoof-oriented."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer = brentq(lambda x: interp1d(fpr, fpr - fnr)(x), 0.0, 1.0)
        thresh = float(interp1d(fpr, thresholds)(eer))
    except ValueError:
        idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float(np.mean([fpr[idx], fnr[idx]]))
        thresh = float(thresholds[idx])

    return eer, thresh


def compute_eer(
    labels: np.ndarray,
    scores: np.ndarray,
    apply_orientation: bool = True,
    apply_symmetry: bool = True,
) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the corresponding threshold.

    Args:
        labels: ground truth (0 = bonafide/real, 1 = spoof/fake)
        scores: model scores (conventionally higher = spoof; inverted if needed)
        apply_orientation: if True, reflect scores when AUC(raw) < 0.5
        apply_symmetry: if True, reported eer = min(eer, 1 - eer)

    Returns:
        (eer, threshold) — threshold operates on the **oriented** score scale
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    if apply_orientation:
        scores, _, _ = orient_scores_for_spoof(labels, scores)

    eer, thresh = _compute_eer_from_curve(labels, scores)
    if apply_symmetry:
        eer = symmetrize_eer(eer)

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
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    apply_orientation: bool = True,
    apply_symmetry: bool = True,
) -> Dict[str, Union[float, bool]]:
    """Compute a full suite of evaluation metrics on oriented scores.

    Args:
        labels: binary ground truth
        scores: continuous model scores (higher = spoof after orientation)
        threshold: decision threshold on **oriented** scale; if None, uses EER threshold
        apply_orientation: reflect scores when AUC(raw) < 0.5
        apply_symmetry: cap reported EER with min(eer, 1 - eer)

    Returns:
        dict with eer, eer_threshold, min_tdcf, accuracy, auc, auc_raw,
        scores_flipped, eer_before_symmetry
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    scores_flipped = False
    auc_raw = 0.0
    if apply_orientation:
        scores, scores_flipped, auc_raw = orient_scores_for_spoof(labels, scores)
    else:
        uniq = np.unique(labels)
        if len(uniq) >= 2:
            try:
                auc_raw = float(roc_auc_score(labels, scores))
            except ValueError:
                auc_raw = 0.0

    eer_raw, eer_thresh = _compute_eer_from_curve(labels, scores)
    eer = symmetrize_eer(eer_raw) if apply_symmetry else float(eer_raw)

    if threshold is None:
        threshold = eer_thresh

    predictions = (scores >= threshold).astype(int)
    acc = accuracy_score(labels, predictions)

    try:
        auc_score = float(roc_auc_score(labels, scores))
    except ValueError:
        auc_score = 0.0

    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]

    min_tdcf = 0.0
    if len(bonafide_scores) > 0 and len(spoof_scores) > 0:
        min_tdcf, _ = compute_min_tdcf(bonafide_scores, spoof_scores)

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_thresh),
        "min_tdcf": float(min_tdcf),
        "accuracy": float(acc),
        "auc": float(auc_score),
        "auc_raw": float(auc_raw),
        "scores_flipped": bool(scores_flipped),
        "eer_before_symmetry": float(eer_raw),
    }


def metrics_to_jsonable(m: Dict[str, Any]) -> Dict[str, Union[int, float, bool]]:
    """Serialize metric dicts for JSON (handles bool / numpy scalars)."""
    out: Dict[str, Union[int, float, bool]] = {}
    for k, v in m.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v  # type: ignore[assignment]
    return out
