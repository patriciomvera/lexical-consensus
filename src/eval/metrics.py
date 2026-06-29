"""
metrics.py
----------
Per-episode performance metrics (plan §6). Reuses the paper's existing
definitions (C1 / C2 / margin / AUROC) and adds the two exp_007-specific
quantities: the prototype-failure gap and the identity-cheating detector.

NOTE: distinct from src.metrics.shannon (exp_004). This module scores a single
(learner, episode) pair into the numbers that stats.py then aggregates.

Metrics
-------
c1_naming                — image -> label accuracy, MACRO-averaged over concepts.
c2_retrieval             — label -> image accuracy. For disjunctive concepts the
                           valid set X_ℓ is the UNION of member categories.
margin                   — paper eq. 13 stability margin from score vectors.
oov_auroc                — separability of in-vocabulary vs OOV queries by score.
prototype_failure_gap    — Acc(exemplar) - Acc(centroid). The H1 quantity;
                           returned per-episode so stats.py can pair it.
nmi_prediction_category  — NMI(prediction, object_category). The H4 identity-
                           cheating detector for cross-cutting concepts: high
                           accuracy + LOW value => genuine attribute acquisition.

All functions take already-computed score/prediction arrays so they stay
learner-agnostic and cheap to bootstrap.
"""

from __future__ import annotations

import numpy as np

from .concepts import Concept


def c1_naming(
    predictions: np.ndarray,
    true_concept_ids: np.ndarray,
    n_concepts: int,
) -> float:
    """Condition 1: macro-averaged naming accuracy.

    Mean of per-concept recall (not micro-accuracy) — required because
    disjunctive extensions are uneven, so micro-accuracy would over-weight the
    larger concepts (plan §3). Concepts with no query examples are skipped.
    """
    predictions = np.asarray(predictions)
    true_concept_ids = np.asarray(true_concept_ids)
    per_concept = []
    for c in range(n_concepts):
        mask = true_concept_ids == c
        if not mask.any():
            continue
        per_concept.append(float((predictions[mask] == c).mean()))
    return float(np.mean(per_concept)) if per_concept else 0.0


def c2_retrieval(
    learner,
    concept_local_id: int,
    valid_categories: set,
    candidate_hashes: list[str],
    candidate_embeddings: np.ndarray,
    candidate_categories: list[str],
) -> float:
    """Condition 2: label -> image accuracy for a SINGLE retrieval trial.

    The learner is handed the concept id and a candidate set; the retrieval is
    correct iff the chosen candidate's native category is in `valid_categories`
    (the valid set X_ℓ — the union of member categories for a disjunctive
    concept, the single category for a native one).

    Returns 1.0 / 0.0 for one trial; the runner averages over trials.
    """
    chosen = learner.retrieve(concept_local_id, candidate_embeddings)
    return 1.0 if candidate_categories[chosen] in valid_categories else 0.0


def margin(score_vectors: np.ndarray) -> float:
    """Mean top1-minus-top2 score margin (paper eq. 13). Stability proxy.

    -inf padding entries (concepts a learner structurally cannot emit) are
    ignored so the margin reflects only the live alternatives.
    """
    s = np.asarray(score_vectors, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    margins = []
    for row in s:
        finite = np.sort(row[np.isfinite(row)])[::-1]
        if len(finite) >= 2:
            margins.append(finite[0] - finite[1])
        elif len(finite) == 1:
            margins.append(finite[0])
    return float(np.mean(margins)) if margins else 0.0


def oov_auroc(
    in_vocab_scores: np.ndarray,
    oov_scores: np.ndarray,
) -> float:
    """AUROC separating in-vocabulary from OOV queries by max score. Run on the
    disjunctive AND cross-cutting sets, not just native (plan §6).

    Each input is a 1-D array of per-query MAX scores; in-vocab should score
    higher than OOV if the learner is properly grounded.
    """
    from sklearn.metrics import roc_auc_score

    in_vocab_scores = np.asarray(in_vocab_scores, dtype=np.float64).ravel()
    oov_scores = np.asarray(oov_scores, dtype=np.float64).ravel()
    if len(in_vocab_scores) == 0 or len(oov_scores) == 0:
        return float("nan")
    y = np.r_[np.ones_like(in_vocab_scores), np.zeros_like(oov_scores)]
    s = np.r_[in_vocab_scores, oov_scores]
    return float(roc_auc_score(y, s))


def prototype_failure_gap(acc_exemplar: float, acc_centroid: float) -> float:
    """The H1 quantity: Acc(exemplar) - Acc(centroid) for one episode. Positive
    => prototype fails where episodic holds. Aggregated with a PAIRED bootstrap
    CI and Wilcoxon in stats.py."""
    return acc_exemplar - acc_centroid


def nmi_prediction_category(
    predictions: np.ndarray,
    object_categories: np.ndarray,
) -> float:
    """H4 identity-cheating detector: NMI between a learner's predictions and
    the native object category. For a cross-cutting concept, high accuracy with
    LOW NMI means the learner tracks the attribute, not object identity."""
    from sklearn.metrics import normalized_mutual_info_score

    predictions = np.asarray(predictions)
    object_categories = np.asarray(object_categories)
    if len(predictions) == 0:
        return float("nan")
    return float(normalized_mutual_info_score(object_categories, predictions))
