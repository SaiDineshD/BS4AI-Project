"""
Fairness evaluation using FairFace demographic annotations.
Reports False Rejection Rate (FRR) stratified by race, gender, and age group
to assess demographic bias in the liveness detection system.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_frr(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute False Rejection Rate — fraction of genuine samples misclassified as spoof."""
    genuine_mask = labels == 0
    if genuine_mask.sum() == 0:
        return 0.0
    false_rejects = ((predictions == 1) & genuine_mask).sum()
    return float(false_rejects / genuine_mask.sum())


def compute_far(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Compute False Acceptance Rate — fraction of spoof samples misclassified as genuine."""
    spoof_mask = labels == 1
    if spoof_mask.sum() == 0:
        return 0.0
    false_accepts = ((predictions == 0) & spoof_mask).sum()
    return float(false_accepts / spoof_mask.sum())


def stratified_fairness_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    demographics: pd.DataFrame,
    attributes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute FRR and FAR for each demographic subgroup.

    Args:
        labels: ground truth (0=genuine, 1=spoof)
        predictions: model predictions (0=genuine, 1=spoof)
        demographics: DataFrame with demographic attributes (race, gender, age, etc.)
        attributes: list of columns to stratify by; defaults to ["race", "gender", "age"]
    Returns:
        Nested dict: {attribute: {group: {"frr": ..., "far": ..., "n": ...}}}
    """
    if attributes is None:
        attributes = [c for c in ["race", "gender", "age"] if c in demographics.columns]

    results = {}
    for attr in attributes:
        if attr not in demographics.columns:
            continue

        groups = demographics[attr].unique()
        attr_results = {}

        for group in sorted(groups):
            mask = demographics[attr] == group
            if mask.sum() == 0:
                continue

            g_labels = labels[mask]
            g_preds = predictions[mask]

            attr_results[str(group)] = {
                "frr": compute_frr(g_labels, g_preds),
                "far": compute_far(g_labels, g_preds),
                "n": int(mask.sum()),
                "n_genuine": int((g_labels == 0).sum()),
                "n_spoof": int((g_labels == 1).sum()),
            }

        results[attr] = attr_results

    return results


def compute_fairness_summary(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Summarize fairness gaps for each attribute.

    Returns max, min, range (gap), and std of FRR across groups.
    """
    summary = {}
    for attr, groups in metrics.items():
        frrs = [g["frr"] for g in groups.values() if g["n_genuine"] > 0]
        if not frrs:
            continue
        summary[attr] = {
            "frr_max": max(frrs),
            "frr_min": min(frrs),
            "frr_gap": max(frrs) - min(frrs),
            "frr_std": float(np.std(frrs)),
            "frr_mean": float(np.mean(frrs)),
            "n_groups": len(frrs),
        }
    return summary


def plot_fairness(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Optional[str] = None,
    title: str = "Fairness Analysis: FRR by Demographic Group",
):
    """Generate bar plots of FRR across demographic groups."""
    n_attrs = len(metrics)
    if n_attrs == 0:
        return

    fig, axes = plt.subplots(1, n_attrs, figsize=(6 * n_attrs, 5))
    if n_attrs == 1:
        axes = [axes]

    for ax, (attr, groups) in zip(axes, metrics.items()):
        group_names = list(groups.keys())
        frrs = [groups[g]["frr"] * 100 for g in group_names]
        counts = [groups[g]["n_genuine"] for g in group_names]

        colors = sns.color_palette("husl", len(group_names))
        bars = ax.bar(group_names, frrs, color=colors, edgecolor="black", linewidth=0.5)

        for bar, n in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel(attr.capitalize())
        ax.set_ylabel("False Rejection Rate (%)")
        ax.set_title(f"FRR by {attr.capitalize()}")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_fairness(
    labels: np.ndarray,
    predictions: np.ndarray,
    demographics: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> Dict:
    """Run full fairness evaluation pipeline.

    Args:
        labels: ground truth (0=genuine, 1=spoof)
        predictions: binary predictions
        demographics: FairFace-style DataFrame with race/gender/age columns
        output_dir: optional directory to save plots and report
    Returns:
        dict with stratified metrics and summary
    """
    metrics = stratified_fairness_metrics(labels, predictions, demographics)
    summary = compute_fairness_summary(metrics)

    if output_dir:
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_fairness(metrics, str(out / "fairness_frr.png"))

        report_lines = ["Fairness Evaluation Report", "=" * 40]
        for attr, s in summary.items():
            report_lines.append(f"\n{attr.upper()}")
            report_lines.append(f"  FRR range: {s['frr_min']*100:.2f}% - {s['frr_max']*100:.2f}%")
            report_lines.append(f"  FRR gap:   {s['frr_gap']*100:.2f}%")
            report_lines.append(f"  FRR std:   {s['frr_std']*100:.2f}%")
            report_lines.append(f"  FRR mean:  {s['frr_mean']*100:.2f}%")

        with open(out / "fairness_report.txt", "w") as f:
            f.write("\n".join(report_lines))

    return {"stratified": metrics, "summary": summary}
