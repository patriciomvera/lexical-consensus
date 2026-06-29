"""
stats.py
--------
Statistical aggregation — the module that answers the "cheap" Round-1
criticism with actual rigor (plan §7). Everything the experiment claims passes
through here: means with 95% bootstrap CIs, and PAIRED comparisons between
learners on the same episodes.

Functions
---------
bootstrap_ci      — mean + 95% CI for one metric over episodes (>= 1000 resamples).
paired_bootstrap  — CI of the per-episode difference d_i = a_i - b_i (e.g.
                    exemplar - centroid). Pairing is what gives H1 its power.
wilcoxon_signed   — Wilcoxon signed-rank p-value for the paired differences.
effect_size       — rank-biserial correlation for the paired test.

H1 is SUPPORTED iff (plan §1): on cross-cutting / disjunctive concepts the
prototype-failure gap has a 95% paired-bootstrap CI excluding 0 AND Wilcoxon
p < 0.05. These functions produce exactly those three numbers.

Determinism: every resample uses a passed-in seed (CLAUDE.md principle 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CIResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class PairedResult:
    """Result of a DIRECTIONAL paired comparison of arm `a` against arm `b`.

    The hypotheses exp_007 tests (H1, H3) are directional — the prototype-failure
    gap is supported only if the episodic arm is significantly GREATER than the
    prototype arm, not merely different. So `supported` requires a strictly
    positive CI (ci_low > 0), NOT just one that excludes zero. `excludes_zero`
    keeps the two-sided information for callers who want it (e.g. detecting that
    a baseline is significantly WORSE, which on native concepts shows up as a
    tiny but significant NEGATIVE gap — a non-dissociation, correctly not
    `supported`).
    """
    mean_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    effect_size: float          # rank-biserial
    n: int
    excludes_zero: bool         # two-sided: CI excludes 0 (either direction)
    supported: bool             # directional: ci_low > 0 AND wilcoxon_p < 0.05


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CIResult:
    """Percentile bootstrap mean + (1-alpha) CI over a 1-D array of per-episode
    metric values."""
    values = np.asarray(values, dtype=np.float64).ravel()
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap_ci: empty input.")
    if n == 1:
        v = float(values[0])
        return CIResult(mean=v, ci_low=v, ci_high=v, n=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return CIResult(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi), n=n)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CIResult:
    """Bootstrap CI of the paired difference mean(a - b). a and b are aligned
    per-episode (same split). Resample episode INDICES (not a and b
    independently) so the pairing is preserved."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired_bootstrap: shape mismatch {a.shape} vs {b.shape}.")
    return bootstrap_ci(a - b, n_resamples=n_resamples, alpha=alpha, seed=seed)


def wilcoxon_signed(a: np.ndarray, b: np.ndarray) -> float:
    """Wilcoxon signed-rank p-value for paired a vs b (scipy.stats.wilcoxon).

    Returns 1.0 (cannot reject) when every difference is zero, which would
    otherwise raise inside scipy — a degenerate but valid "no effect" outcome.
    """
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = a - b
    if np.allclose(d, 0.0):
        return 1.0
    # zero_method="wilcox" drops zero-differences (scipy default).
    return float(wilcoxon(a, b).pvalue)


def effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation effect size for the paired comparison.

    r_rb = (sum of positive signed ranks - sum of negative signed ranks) /
           (sum of all signed ranks). Range [-1, 1]; positive => a > b. Zero
           differences are excluded (standard for the matched-pairs rank-biserial).
    """
    from scipy.stats import rankdata

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = a - b
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    total = ranks.sum()
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    return float((r_pos - r_neg) / total)


def paired_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Run paired_bootstrap + wilcoxon_signed + effect_size and pack the H1
    decision (CI excludes 0 AND Wilcoxon p < 0.05) into one PairedResult. This
    is the function the analysis script calls per (concept_type, ways, shots)
    cell. `a` is the higher-expected arm (e.g. exemplar), `b` the baseline
    (e.g. centroid), so a positive mean_diff means the prototype-failure gap."""
    ci = paired_bootstrap(a, b, n_resamples=n_resamples, alpha=alpha, seed=seed)
    p = wilcoxon_signed(a, b)
    es = effect_size(a, b)
    excludes_zero = (ci.ci_low > 0.0) or (ci.ci_high < 0.0)
    # Directional: the gap is "supported" only when a is significantly GREATER
    # than b. A significant gap in the WRONG direction (b > a) is not a
    # prototype-failure dissociation and must not flip the flag.
    return PairedResult(
        mean_diff=ci.mean,
        ci_low=ci.ci_low,
        ci_high=ci.ci_high,
        wilcoxon_p=p,
        effect_size=es,
        n=ci.n,
        excludes_zero=excludes_zero,
        supported=(ci.ci_low > 0.0 and p < 0.05),
    )
