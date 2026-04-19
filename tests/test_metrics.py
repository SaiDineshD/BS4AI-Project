"""Tests for oriented EER, symmetry cap, and metric consistency."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_eer,
    orient_scores_for_spoof,
    symmetrize_eer,
)


def test_symmetrize_eer():
    assert symmetrize_eer(0.63) == pytest.approx(0.37)
    assert symmetrize_eer(0.44) == pytest.approx(0.44)
    assert symmetrize_eer(0.5) == pytest.approx(0.5)
    assert symmetrize_eer(0.0) == pytest.approx(0.0)


def test_inverted_scores_get_flipped():
    n = 200
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    # Higher score for class 0 -> inverted vs spoof-positive convention
    scores = np.concatenate(
        [np.linspace(0.75, 0.98, n // 2), np.linspace(0.02, 0.22, n // 2)]
    )
    oriented, flipped, auc_raw = orient_scores_for_spoof(labels, scores)
    assert flipped is True
    assert auc_raw < 0.5
    m = compute_all_metrics(labels, scores)
    assert m["scores_flipped"] is True
    assert m["auc"] >= 0.5
    assert m["eer"] <= 0.5 + 1e-6


def test_eer_never_above_half_after_orientation():
    np.random.seed(1)
    labels = np.array([0] * 120 + [1] * 120)
    scores = np.random.RandomState(1).rand(len(labels))
    eer, _ = compute_eer(labels, scores)
    assert eer <= 0.5 + 1e-5
    m = compute_all_metrics(labels, scores)
    assert m["eer"] <= 0.5 + 1e-5


def test_threshold_matches_oriented_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.85, 0.9, 0.88])
    m = compute_all_metrics(labels, scores)
    oriented, _, _ = orient_scores_for_spoof(labels, scores)
    preds = (oriented >= m["eer_threshold"]).astype(int)
    acc_from_manual = (preds == labels).mean()
    assert acc_from_manual == pytest.approx(m["accuracy"], abs=1e-6)


def test_compute_all_metrics_returns_aux_fields():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.8, 0.3, 0.7])
    m = compute_all_metrics(labels, scores)
    assert "auc_raw" in m
    assert "scores_flipped" in m
    assert "eer_before_symmetry" in m
