"""
harness.py
----------
The bridge between an Episode (image hashes) and a Learner (embedding arrays):
resolve one episode to arrays, fit one learner, and score it into the per-episode
metric row that the ledger stores and stats.py aggregates.

Kept here (not in the experiment runner) for two reasons:
  1. It is learner- and dataset-agnostic, so 007a / 007b / 007c all reuse it.
  2. It makes the full pipeline unit-testable on synthetic embeddings, without
     touching DINOv2 — which is exactly where the "does the stats pipeline
     break" question gets answered cheaply (the user's point about finding
     breakage on the easy sub-experiment).

PAIRING NOTE: this module never samples. It consumes an Episode that was drawn
once and is shared across all learners, so every learner is scored on identical
support/query. Pairing is a property of the caller's loop order (episodes outer,
learners inner); harness.evaluate_learner_on_episode just honors it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import metrics
from .episodes import Episode


@dataclass(frozen=True)
class EpisodeArrays:
    """An Episode resolved to dense arrays in LOCAL concept-id space."""
    support_emb: np.ndarray            # (n_support, dim)
    support_ids: np.ndarray            # (n_support,)  local ids
    query_emb: np.ndarray              # (n_query, dim)
    query_ids: np.ndarray              # (n_query,)    local ids
    query_categories: np.ndarray       # (n_query,)    native object categories
    query_hashes: list[str]            # parallel to query_*
    n_ways: int


def resolve_episode_arrays(
    episode: Episode,
    embeddings: dict[str, np.ndarray],
    hash_to_category: dict[str, str],
) -> EpisodeArrays:
    """Turn an Episode's image hashes into stacked embedding arrays."""
    s_emb, s_ids = [], []
    for local_id, hashes in episode.support.items():
        for h in hashes:
            s_emb.append(embeddings[h]); s_ids.append(local_id)
    q_emb, q_ids, q_cat, q_hash = [], [], [], []
    for local_id, hashes in episode.query.items():
        for h in hashes:
            q_emb.append(embeddings[h]); q_ids.append(local_id)
            q_cat.append(hash_to_category[h]); q_hash.append(h)
    return EpisodeArrays(
        support_emb=np.asarray(s_emb, dtype=np.float64),
        support_ids=np.asarray(s_ids),
        query_emb=np.asarray(q_emb, dtype=np.float64),
        query_ids=np.asarray(q_ids),
        query_categories=np.asarray(q_cat),
        query_hashes=q_hash,
        n_ways=episode.n_ways,
    )


def evaluate_learner_on_episode(
    learner,
    episode: Episode,
    arrays: EpisodeArrays,
    *,
    n_retrieval_trials: int = 20,
    rng: np.random.Generator | None = None,
) -> dict:
    """Fit `learner` on the episode support and score it on the query.

    Returns one metric row:
        c1        — macro naming accuracy (Condition 1)
        c2        — macro retrieval accuracy (Condition 2, inverse grounding)
        margin    — mean top1-top2 score margin (WITHIN-learner diagnostic only;
                    score scales differ across learners, so never compare margin
                    across learners — use c1 / c2 / the gap for that)
        nmi_pred_category — NMI(prediction, object_category) (H4 detector)
        per_concept_c1 — {global concept label: per-concept recall} for this
                    episode, so accuracy can be plotted against each concept's
                    NS (the central H1 collapse figure)

    `rng` seeds the C2 candidate sampling; pass a per-(episode, learner) RNG so
    retrieval trials are reproducible without breaking the support/query pairing
    (which lives entirely in the Episode and is identical across learners).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    learner.fit(arrays.support_emb, arrays.support_ids)

    preds = np.asarray(learner.predict(arrays.query_emb))
    scores = learner.score(arrays.query_emb)

    c1 = metrics.c1_naming(preds, arrays.query_ids, arrays.n_ways)
    mar = metrics.margin(scores)
    nmi_pc = metrics.nmi_prediction_category(preds, arrays.query_categories)
    c2 = _c2_macro(learner, episode, arrays, n_retrieval_trials, rng)

    # Per-concept recall, keyed by GLOBAL label so it aggregates across episodes
    # (the same concept appears in many episodes under the same global label but
    # different local ids).
    per_concept_c1: dict[str, float] = {}
    for local_id in range(arrays.n_ways):
        mask = arrays.query_ids == local_id
        if mask.any():
            label = episode.concepts[local_id].label
            per_concept_c1[label] = float((preds[mask] == local_id).mean())

    return {"c1": c1, "c2": c2, "margin": mar, "nmi_pred_category": nmi_pc,
            "per_concept_c1": per_concept_c1}


def _c2_macro(
    learner,
    episode: Episode,
    arrays: EpisodeArrays,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """Condition 2 macro accuracy.

    For each concept, run `n_trials` retrieval trials. Each trial presents one
    positive (a query image of the target concept) plus one distractor drawn
    from each OTHER concept's query images; the learner must pick a candidate
    whose native category is in the target's valid set X_ℓ. Macro-averaged over
    concepts so uneven concept sizes don't bias the score (plan §6).
    """
    # Index query rows by local concept id for fast sampling.
    rows_by_concept: dict[int, np.ndarray] = {
        c: np.where(arrays.query_ids == c)[0] for c in range(arrays.n_ways)
    }
    per_concept = []
    for c in range(arrays.n_ways):
        valid_categories = set(episode.concepts[c].member_categories)
        others = [o for o in range(arrays.n_ways) if o != c]
        hits = 0
        for _ in range(n_trials):
            pos = rng.choice(rows_by_concept[c])
            distractors = [rng.choice(rows_by_concept[o]) for o in others]
            cand_rows = np.array([pos, *distractors])
            rng.shuffle(cand_rows)
            cand_emb = arrays.query_emb[cand_rows]
            cand_cat = [arrays.query_categories[r] for r in cand_rows]
            chosen = learner.retrieve(c, cand_emb)
            if cand_cat[chosen] in valid_categories:
                hits += 1
        per_concept.append(hits / n_trials)
    return float(np.mean(per_concept))
