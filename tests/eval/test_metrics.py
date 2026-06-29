"""
test_metrics.py — per-episode metric definitions (plan §6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import metrics  # noqa: E402


def test_c1_naming_is_macro_averaged():
    # Concept 0 perfect (2/2), concept 1 half (1/2): macro = (1.0 + 0.5)/2 = 0.75,
    # which differs from micro = 3/4 only because of imbalance — here equal, but
    # the macro path is what we assert.
    preds = np.array([0, 0, 1, 0])
    true = np.array([0, 0, 1, 1])
    assert metrics.c1_naming(preds, true, n_concepts=2) == 0.75


def test_c1_naming_macro_differs_from_micro_under_imbalance():
    # concept 0 has 4 queries all correct; concept 1 has 1 query, wrong.
    preds = np.array([0, 0, 0, 0, 0])
    true = np.array([0, 0, 0, 0, 1])
    # micro = 4/5 = 0.8; macro = (1.0 + 0.0)/2 = 0.5
    assert metrics.c1_naming(preds, true, n_concepts=2) == 0.5


def test_margin_ignores_inf_padding():
    # top1=2.0, top2=0.5 among finite entries; -inf padding ignored.
    scores = np.array([[2.0, 0.5, -np.inf]])
    assert metrics.margin(scores) == 1.5


def test_oov_auroc_perfect_separation():
    in_vocab = np.array([0.9, 0.8, 0.85])
    oov = np.array([0.1, 0.2, 0.05])
    assert metrics.oov_auroc(in_vocab, oov) == 1.0


def test_nmi_prediction_category_identical_is_one():
    preds = np.array([0, 0, 1, 1, 2, 2])
    cats = np.array(["a", "a", "b", "b", "c", "c"])
    assert metrics.nmi_prediction_category(preds, cats) == 1.0


def test_nmi_prediction_category_independent_is_low():
    preds = np.array([0, 1, 0, 1, 0, 1])
    cats = np.array(["a", "a", "a", "b", "b", "b"])
    assert metrics.nmi_prediction_category(preds, cats) < 0.5


def test_prototype_failure_gap():
    assert metrics.prototype_failure_gap(0.9, 0.4) == 0.5
