"""
test_exemplar_weighting.py — ExemplarKNN distance weighting (directive 1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.learners.exemplar_knn import ExemplarKNN  # noqa: E402


def _two_class():
    # concept 0 cluster near +x, concept 1 near -x, a few exemplars each.
    seed = np.array([
        [1.0, 0.0], [0.98, 0.02], [0.95, -0.05],
        [-1.0, 0.0], [-0.97, 0.03], [-0.96, -0.04],
    ])
    seed /= np.linalg.norm(seed, axis=1, keepdims=True)
    ids = np.array([0, 0, 0, 1, 1, 1])
    return seed, ids


def test_k1_weighted_equals_unweighted():
    seed, ids = _two_class()
    q = seed.copy()
    w = ExemplarKNN(2, k=1, weighted=True); w.fit(seed, ids)
    u = ExemplarKNN(2, k=1, weighted=False); u.fit(seed, ids)
    np.testing.assert_array_equal(w.predict(q), u.predict(q))


def test_weighted_and_unweighted_both_classify_clean_clusters():
    seed, ids = _two_class()
    for weighted in (True, False):
        m = ExemplarKNN(2, k=3, weighted=weighted); m.fit(seed, ids)
        np.testing.assert_array_equal(m.predict(seed), ids)


def test_weighting_changes_scores_not_just_predictions():
    seed, ids = _two_class()
    q = np.array([[0.6, 0.8]]); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w = ExemplarKNN(2, k=3, weighted=True); w.fit(seed, ids)
    u = ExemplarKNN(2, k=3, weighted=False); u.fit(seed, ids)
    # Unweighted score is an integer vote count per concept; weighted is a sum
    # of similarities — so the score vectors differ in scale/shape.
    assert not np.allclose(w.score(q), u.score(q))


def test_config_records_weighting():
    m = ExemplarKNN(2, k=5, weighted=False)
    cfg = m.config()
    assert cfg["k"] == 5 and cfg["weighted"] is False
