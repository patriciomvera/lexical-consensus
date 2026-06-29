"""
test_stats.py — the statistics that the paper's claims rest on (plan §7).

If these are wrong, every CI and every H1 verdict downstream is wrong, so they
are tested directly: a known-different pair must come back `supported`, an
identical pair must not, and the effect-size sign must follow the direction of
the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval import stats  # noqa: E402


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(0.8, 0.05, size=40)
    ci = stats.bootstrap_ci(values, seed=1)
    assert ci.ci_low < ci.mean < ci.ci_high
    assert ci.n == 40


def test_bootstrap_ci_is_deterministic():
    values = np.linspace(0, 1, 30)
    a = stats.bootstrap_ci(values, seed=7)
    b = stats.bootstrap_ci(values, seed=7)
    assert (a.ci_low, a.ci_high, a.mean) == (b.ci_low, b.ci_high, b.mean)


def test_paired_test_detects_real_gap():
    """exemplar clearly > centroid on every episode => H1 supported."""
    rng = np.random.default_rng(0)
    centroid = rng.uniform(0.40, 0.55, size=30)
    exemplar = centroid + rng.uniform(0.15, 0.25, size=30)  # always higher
    res = stats.paired_test(exemplar, centroid, seed=2)
    assert res.mean_diff > 0
    assert res.excludes_zero
    assert res.wilcoxon_p < 0.05
    assert res.effect_size > 0
    assert res.supported


def test_paired_test_no_gap_not_supported():
    """No systematic difference => not supported (this is H2 on native)."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.9, 1.0, size=30)
    a = base + rng.normal(0, 0.001, size=30)
    b = base + rng.normal(0, 0.001, size=30)
    res = stats.paired_test(a, b, seed=3)
    assert not res.supported


def test_paired_test_is_directional():
    """A significant gap in the WRONG direction (a < b) must NOT be `supported`
    — the hypothesis is directional (a > b). This is the exp_007a native case:
    centroid marginally beats exemplar, significant under tight pairing, but it
    is NOT a prototype-failure dissociation."""
    rng = np.random.default_rng(4)
    b = rng.uniform(0.96, 0.98, size=30)
    a = b - rng.uniform(0.005, 0.012, size=30)        # a consistently LOWER
    res = stats.paired_test(a, b, seed=5)
    assert res.mean_diff < 0
    assert res.excludes_zero          # two-sided: it IS a real difference
    assert not res.supported          # directional: but not in the hypothesized way


def test_identical_arrays_are_degenerate_but_safe():
    x = np.full(20, 0.95)
    res = stats.paired_test(x, x, seed=0)
    assert res.mean_diff == 0.0
    assert res.wilcoxon_p == 1.0
    assert res.effect_size == 0.0
    assert not res.supported


def test_paired_bootstrap_preserves_pairing_shape_check():
    with pytest.raises(ValueError):
        stats.paired_bootstrap(np.zeros(5), np.zeros(6))
