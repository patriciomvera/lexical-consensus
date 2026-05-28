"""
shannon.py — Information-theoretic metrics for the lexical consensus experiment.

Measures how much information a label assignment carries about the image it
labels, using Shannon's mutual information decomposition:

    I(image; label) = H(label) - H(label | image)

With 3 agents and 3 Carroll labels, the per-image entropy takes only three
discrete values:

    3/3 agreement:  H = 0.000 bits   (unanimous)
    2/1 split:      H ≈ 0.918 bits   (majority)
    1/1/1 split:    H ≈ 1.585 bits   (unresolved)

So mean entropy per round ≈ n_disagreements × 0.918 / n_images, where
n_disagreements counts images that are NOT unanimous.
"""

from __future__ import annotations

import math
from typing import Optional


# ─── Core entropy functions ───────────────────────────────────────────────────

def label_entropy(label_distribution: dict) -> float:
    """
    H(label) = -sum(p * log2(p))
    Handles p=0 gracefully: 0 * log2(0) is defined as 0.
    """
    h = 0.0
    for p in label_distribution.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def conditional_entropy(per_image_distributions: list[dict]) -> float:
    """
    H(label | image) = mean entropy across all images.
    Each dict in the list is {label: probability} for one image.
    """
    if not per_image_distributions:
        return 0.0
    entropies = [label_entropy(d) for d in per_image_distributions]
    return sum(entropies) / len(entropies)


def label_marginal(per_image_distributions: list[dict]) -> dict:
    """
    Compute the marginal label distribution empirically:
        P(label) = mean of P(label | image_i) over all images.

    NEVER assumes a uniform prior. If all agents collapse to one label,
    this returns {that_label: 1.0}, giving H(label) = 0 — which is correct:
    it means lexical collapse, not perfect grounding.
    """
    if not per_image_distributions:
        return {}

    all_labels: set[str] = set()
    for d in per_image_distributions:
        all_labels.update(d.keys())

    n = len(per_image_distributions)
    marginal: dict[str, float] = {lbl: 0.0 for lbl in all_labels}
    for d in per_image_distributions:
        for lbl in all_labels:
            marginal[lbl] += d.get(lbl, 0.0) / n

    # Re-normalize to guard against float64 accumulation error
    total = sum(marginal.values())
    if total > 0.0:
        marginal = {lbl: v / total for lbl, v in marginal.items()}

    return marginal


# ─── Mutual information ───────────────────────────────────────────────────────

def mutual_information(
    per_image_distributions: list[dict],
) -> tuple[float, float, float]:
    """
    I(image; label) = H(label) - H(label | image).

    H(label) is computed from the empirical marginal via label_marginal(),
    not from a uniform prior.

    Returns: (I, H_label, H_label_given_image)
    I is clamped to 0.0 to handle floating-point underflow.
    """
    marginal              = label_marginal(per_image_distributions)
    H_label               = label_entropy(marginal)
    H_label_given_image   = conditional_entropy(per_image_distributions)
    I                     = max(0.0, H_label - H_label_given_image)
    return I, H_label, H_label_given_image


def normalized_mi(I: float, H_label: float) -> float:
    """
    normalized_MI = I / H_label.
    Range: [0, 1] where 1 = perfect grounding, 0 = no information.
    Returns 0.0 when H_label == 0 (lexical collapse or single-label rounds).
    """
    if H_label == 0.0:
        return 0.0
    return I / H_label


def normalized_conditional_entropy(
    H_label_given_image: float,
    n_labels: int,
) -> float:
    """
    Normalized H(label | image) relative to the maximum possible entropy
    for a vocabulary of size n_labels.

    max_entropy = log2(n_labels)
    Allows cross-experiment comparison when vocabulary size differs.
    """
    if n_labels <= 1:
        return 0.0
    max_entropy = math.log2(n_labels)
    if max_entropy == 0.0:
        return 0.0
    return H_label_given_image / max_entropy


# ─── Round-level aggregates ───────────────────────────────────────────────────

def grounding_threshold_round(rounds_metrics: list[dict]) -> int:
    """
    First round where normalized_MI > 0.5, i.e.:
        I(image; label) > 0.5 * H(label)

    This is the formal grounding threshold: the label carries more than half
    the information available in H(label) about which image it came from.

    Returns -1 if the threshold is never reached.
    """
    for m in rounds_metrics:
        if m.get("normalized_mi", 0.0) > 0.5:
            return int(m["round"])
    return -1


def entropy_reduction_curve(rounds_metrics: list[dict]) -> list[dict]:
    """
    Select the per-round fields needed for plotting and reporting.
    Input dicts must contain all fields produced by compute_round_metrics().
    """
    keys = [
        "round", "H_label", "H_label_given_image",
        "mutual_information", "normalized_mi",
        "normalized_conditional_entropy",
    ]
    return [{k: m[k] for k in keys} for m in rounds_metrics]


# ─── Per-round computation from ledger data ───────────────────────────────────

def compute_round_metrics(
    round_num: int,
    per_image_distributions: list[dict],
    n_labels: int,
) -> dict:
    """
    Compute all Shannon metrics for one round.

    per_image_distributions: list of {label: probability} dicts, one per image.
    n_labels: vocabulary size (3 for Carroll experiment).

    Returns a flat dict suitable for JSON serialization and CSV export.
    """
    I, H_label, H_cond = mutual_information(per_image_distributions)
    nmi   = normalized_mi(I, H_label)
    nce   = normalized_conditional_entropy(H_cond, n_labels)

    return {
        "round":                          round_num,
        "H_label":                        round(H_label, 6),
        "H_label_given_image":            round(H_cond, 6),
        "mutual_information":             round(I, 6),
        "normalized_mi":                  round(nmi, 6),
        "normalized_conditional_entropy": round(nce, 6),
        "n_images":                       len(per_image_distributions),
    }


def distributions_from_ledger(
    ledger_rows: list[dict],
    n_agents: int,
    pool: Optional[str] = None,
) -> dict[int, list[dict]]:
    """
    Build per-round per-image label distributions from ledger_events rows.

    Args:
        ledger_rows: rows from ledger_events.csv (dicts)
        n_agents: number of agents (used as denominator for probabilities)
        pool: if given, filter to this pool ("interaction" | "held_out" | None = all)

    Returns: round -> list of {label: probability} dicts (one per image)
    """
    from collections import defaultdict

    if pool is not None:
        ledger_rows = [r for r in ledger_rows if r["pool"] == pool]

    # round -> image_id -> {label: count}
    counts: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for row in ledger_rows:
        r   = int(row["round"])
        img = row["image_id"]
        lbl = row["predicted_label"]
        counts[r][img][lbl] += 1

    result: dict[int, list[dict]] = {}
    for r, images in counts.items():
        dists: list[dict] = []
        for img_counts in images.values():
            total = sum(img_counts.values())
            if total == 0:
                continue
            dists.append({lbl: cnt / total for lbl, cnt in img_counts.items()})
        result[r] = dists

    return result


def compute_all_rounds(
    ledger_rows: list[dict],
    n_agents: int,
    n_labels: int,
    pool: Optional[str] = "interaction",
) -> list[dict]:
    """
    Compute Shannon metrics for every round in the ledger.
    Returns a list of dicts (one per round), sorted by round number.
    """
    round_dists = distributions_from_ledger(ledger_rows, n_agents, pool=pool)
    results: list[dict] = []
    for round_num in sorted(round_dists.keys()):
        dists = round_dists[round_num]
        m     = compute_round_metrics(round_num, dists, n_labels)
        results.append(m)
    return results
