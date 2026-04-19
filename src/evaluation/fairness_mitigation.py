"""Post-hoc fairness mitigation: per-group thresholds to reduce FRR disparity (genuine-only)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def per_group_thresholds_equal_frr(
    scores: np.ndarray,
    demographics: pd.DataFrame,
    attributes: Optional[List[str]] = None,
    target_frr: float = 0.05,
) -> Dict[str, Any]:
    """Choose per-group thresholds so genuine FRR ≈ ``target_frr`` (quantile-based).

    Assumes higher score = more likely spoof; genuine samples are audited with label 0.
    For each group, threshold t_g = quantile(scores_genuine, 1 - target_frr).

    Returns mapping group -> threshold and summary gap after mitigation.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if attributes is None:
        attributes = [c for c in ["race", "gender", "age"] if c in demographics.columns]

    out: Dict[str, Any] = {"target_frr": target_frr, "per_attribute": {}}

    for attr in attributes:
        if attr not in demographics.columns:
            continue
        groups = demographics[attr].astype(str).unique()
        thr_map: Dict[str, float] = {}
        frr_after: List[float] = []
        col = demographics[attr].astype(str).values
        for g in sorted(groups):
            mask = col == str(g)
            if mask.sum() == 0:
                continue
            s_g = scores[mask]
            # All FairFace rows are genuine-only in our protocol; treat all as genuine for FRR curve
            if len(s_g) < 5:
                thr_map[str(g)] = float(np.median(s_g))
                continue
            thr_map[str(g)] = float(np.quantile(s_g, 1.0 - target_frr))
            # realized FRR at this threshold: fraction of genuine above t
            frr_after.append(float((s_g >= thr_map[str(g)]).mean()))

        if frr_after:
            out["per_attribute"][attr] = {
                "thresholds": thr_map,
                "frr_max": max(frr_after),
                "frr_min": min(frr_after),
                "frr_gap": max(frr_after) - min(frr_after),
            }

    return out
