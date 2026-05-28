"""
test_shannon.py — Unit tests for src/metrics/shannon.py
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metrics.shannon import (
    label_entropy,
    conditional_entropy,
    label_marginal,
    mutual_information,
    normalized_mi,
    normalized_conditional_entropy,
    grounding_threshold_round,
    distributions_from_ledger,
    compute_round_metrics,
)


# ─── label_entropy ────────────────────────────────────────────────────────────

def test_entropy_certain():
    """Single label with probability 1.0 → entropy = 0."""
    assert label_entropy({"a": 1.0}) == 0.0


def test_entropy_binary_uniform():
    """Two labels with equal probability → entropy = 1 bit."""
    assert abs(label_entropy({"a": 0.5, "b": 0.5}) - 1.0) < 1e-9


def test_entropy_ternary_uniform():
    """Three labels with equal probability → entropy ≈ 1.585 bits."""
    result = label_entropy({"a": 1/3, "b": 1/3, "c": 1/3})
    assert abs(result - math.log2(3)) < 1e-9


def test_entropy_zero_probability():
    """p=0 must be handled gracefully (0 * log2(0) = 0 by convention)."""
    result = label_entropy({"a": 1.0, "b": 0.0})
    assert result == 0.0


def test_entropy_empty():
    """Empty distribution → entropy = 0."""
    assert label_entropy({}) == 0.0


def test_entropy_non_uniform():
    """Asymmetric distribution → strictly between 0 and log2(n)."""
    h = label_entropy({"a": 0.9, "b": 0.1})
    assert 0 < h < 1.0


# ─── 3-agent entropy reference values ────────────────────────────────────────
# With 3 agents, per-image vote distributions take only 3 forms.

def test_entropy_3_unanimous():
    """3/3 agreement → H = 0."""
    assert label_entropy({"slithy": 1.0}) == 0.0


def test_entropy_3_split_2_1():
    """2/1 split → H ≈ 0.918 bits."""
    h = label_entropy({"slithy": 2/3, "mimsy": 1/3})
    assert abs(h - 0.9182958340544894) < 1e-9


def test_entropy_3_split_1_1_1():
    """1/1/1 split → H ≈ 1.585 bits."""
    h = label_entropy({"slithy": 1/3, "mimsy": 1/3, "vorpal": 1/3})
    assert abs(h - math.log2(3)) < 1e-9


# ─── conditional_entropy ─────────────────────────────────────────────────────

def test_conditional_entropy_all_certain():
    """All images have certain labels → H(label|image) = 0."""
    dists = [{"a": 1.0}, {"b": 1.0}, {"a": 1.0}]
    assert conditional_entropy(dists) == 0.0


def test_conditional_entropy_all_uncertain():
    """All images have maximal uncertainty → H(label|image) = log2(3)."""
    dists = [{"a": 1/3, "b": 1/3, "c": 1/3}] * 5
    result = conditional_entropy(dists)
    assert abs(result - math.log2(3)) < 1e-9


def test_conditional_entropy_empty():
    """Empty list → 0."""
    assert conditional_entropy([]) == 0.0


# ─── label_marginal ───────────────────────────────────────────────────────────

def test_marginal_uniform():
    """All images have the same distribution → marginal == that distribution."""
    dists = [{"a": 0.5, "b": 0.5}] * 4
    m = label_marginal(dists)
    assert abs(m["a"] - 0.5) < 1e-9
    assert abs(m["b"] - 0.5) < 1e-9


def test_marginal_collapse():
    """All agents agree on the same label → marginal has probability 1.0 for that label."""
    dists = [{"slithy": 1.0}, {"slithy": 1.0}]
    m = label_marginal(dists)
    assert abs(m["slithy"] - 1.0) < 1e-9


def test_marginal_never_uniform_assumption():
    """
    Marginal is computed empirically, not assumed uniform.
    A scenario where all images go to 'a' should NOT produce uniform marginal.
    """
    dists = [{"a": 1.0}, {"a": 1.0}, {"a": 1.0}]
    m = label_marginal(dists)
    assert abs(m.get("a", 0.0) - 1.0) < 1e-9
    # 'b' should not appear in marginal if never in any dist
    assert "b" not in m or abs(m["b"]) < 1e-12


def test_marginal_sums_to_one():
    """Marginal probabilities must sum to 1.0."""
    dists = [
        {"a": 2/3, "b": 1/3},
        {"a": 1/3, "b": 2/3},
        {"c": 1.0},
    ]
    m = label_marginal(dists)
    assert abs(sum(m.values()) - 1.0) < 1e-9


# ─── mutual_information ───────────────────────────────────────────────────────

def test_mi_no_division_by_zero_when_h_label_zero():
    """When H(label)=0, normalized_MI must return 0.0, not raise."""
    # All images, all agents agree on 'slithy' → collapse → H(label)=0
    dists = [{"slithy": 1.0}] * 10
    I, H_label, H_cond = mutual_information(dists)
    assert H_label == 0.0
    assert I == 0.0
    assert normalized_mi(I, H_label) == 0.0  # must not raise ZeroDivisionError


def test_mi_perfect_grounding():
    """
    When each image deterministically maps to a label (H_cond=0) and
    the marginal is non-degenerate → I = H(label) → normalized_MI = 1.0.
    """
    dists = [{"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0}]
    I, H_label, H_cond = mutual_information(dists)
    assert abs(H_cond) < 1e-9
    nmi = normalized_mi(I, H_label)
    assert abs(nmi - 1.0) < 1e-9


def test_mi_non_negative():
    """MI must be >= 0 for any distribution."""
    cases = [
        [{"a": 1/3, "b": 1/3, "c": 1/3}] * 5,
        [{"a": 0.5, "b": 0.5}] * 3,
        [{"a": 1.0}] * 4,
        [{"a": 2/3, "b": 1/3}, {"b": 2/3, "a": 1/3}],
    ]
    for dists in cases:
        I, _, _ = mutual_information(dists)
        assert I >= 0.0, f"MI was negative for {dists}"


def test_mi_zero_when_all_same():
    """
    If all images always receive the same distribution, H_cond = H_label
    → I = 0. (Knowing the image tells you nothing extra about the label.)
    """
    dists = [{"a": 0.5, "b": 0.5}] * 6
    I, _, _ = mutual_information(dists)
    assert abs(I) < 1e-9


# ─── normalized_conditional_entropy ──────────────────────────────────────────

def test_nce_max_entropy():
    """H(label|image) at maximum → normalized = 1.0."""
    max_h = math.log2(3)
    nce = normalized_conditional_entropy(max_h, n_labels=3)
    assert abs(nce - 1.0) < 1e-9


def test_nce_zero_entropy():
    """H(label|image) = 0 → normalized = 0.0."""
    assert normalized_conditional_entropy(0.0, n_labels=3) == 0.0


def test_nce_single_label():
    """n_labels=1 → avoids division by zero, returns 0."""
    assert normalized_conditional_entropy(0.0, n_labels=1) == 0.0


# ─── grounding_threshold_round ────────────────────────────────────────────────

def test_grounding_threshold_found():
    rounds = [
        {"round": 1, "normalized_mi": 0.2},
        {"round": 2, "normalized_mi": 0.4},
        {"round": 3, "normalized_mi": 0.6},
        {"round": 4, "normalized_mi": 0.8},
    ]
    assert grounding_threshold_round(rounds) == 3


def test_grounding_threshold_not_found():
    rounds = [
        {"round": i, "normalized_mi": 0.3}
        for i in range(1, 11)
    ]
    assert grounding_threshold_round(rounds) == -1


def test_grounding_threshold_first_round():
    rounds = [{"round": 1, "normalized_mi": 0.9}]
    assert grounding_threshold_round(rounds) == 1


# ─── distributions_from_ledger ────────────────────────────────────────────────

def _make_ledger_rows(round_num, pool, assignments):
    """
    assignments: list of (agent_id, image_id, predicted_label)
    """
    rows = []
    for agent_id, image_id, label in assignments:
        rows.append({
            "round":           round_num,
            "agent_id":        agent_id,
            "image_id":        image_id,
            "predicted_label": label,
            "pool":            pool,
            "true_category":   "frog",
            "confidence":      0.9,
            "margin":          0.3,
            "status":          "accepted",
        })
    return rows


def test_distributions_unanimous():
    """3 agents all pick 'slithy' for same image → distribution {slithy: 1.0}."""
    rows = _make_ledger_rows(1, "interaction", [
        ("agent_00", "frog_015", "slithy"),
        ("agent_01", "frog_015", "slithy"),
        ("agent_02", "frog_015", "slithy"),
    ])
    result = distributions_from_ledger(rows, n_agents=3, pool="interaction")
    assert 1 in result
    dists = result[1]
    assert len(dists) == 1
    assert abs(dists[0].get("slithy", 0) - 1.0) < 1e-9


def test_distributions_split():
    """2/1 split → distribution {slithy: 2/3, mimsy: 1/3}."""
    rows = _make_ledger_rows(1, "interaction", [
        ("agent_00", "frog_015", "slithy"),
        ("agent_01", "frog_015", "slithy"),
        ("agent_02", "frog_015", "mimsy"),
    ])
    result = distributions_from_ledger(rows, n_agents=3, pool="interaction")
    dists = result[1]
    assert len(dists) == 1
    d = dists[0]
    assert abs(d.get("slithy", 0) - 2/3) < 1e-9
    assert abs(d.get("mimsy",  0) - 1/3) < 1e-9


def test_distributions_pool_filter():
    """Pool filter isolates interaction vs held_out rows."""
    rows = (
        _make_ledger_rows(1, "interaction", [
            ("agent_00", "frog_015", "slithy"),
            ("agent_01", "frog_015", "slithy"),
            ("agent_02", "frog_015", "slithy"),
        ]) +
        _make_ledger_rows(1, "held_out", [
            ("agent_00", "frog_030", "mimsy"),
            ("agent_01", "frog_030", "mimsy"),
            ("agent_02", "frog_030", "mimsy"),
        ])
    )
    interact = distributions_from_ledger(rows, n_agents=3, pool="interaction")
    held     = distributions_from_ledger(rows, n_agents=3, pool="held_out")
    both     = distributions_from_ledger(rows, n_agents=3, pool=None)

    assert len(interact[1]) == 1   # only frog_015
    assert len(held[1])    == 1    # only frog_030
    assert len(both[1])    == 2    # both images
