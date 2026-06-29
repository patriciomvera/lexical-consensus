"""
pool_variants.py — C2 retrieval with multiple candidate pool constructions.

Pool types (spec §3):
    small  : 1 correct + 4 random from other episode concepts (5 total)
    medium : 1 correct + 9 random from other episode concepts (10 total)
    large  : 1 correct + 24 random from other episode concepts (25 total)
    hard   : 1 correct + 9 nearest-neighbor distractors — closest embeddings to
             the query image that are NOT from the correct concept's categories
    oov_25 : 1 correct + ~7 random from other concepts + ~2 OOV
    oov_50 : 1 correct + ~5 random from other concepts + ~5 OOV

All pools are seeded deterministically per (episode_id, local_id, trial_idx, pool_type)
so the positive selection is shared across pool types (same trial → same query image),
and distractor selection is independent per pool type.
"""

from __future__ import annotations

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from _shared import (
    POOL_SIZES, OOV_COUNTS, POOL_TYPE_SEED_OFFSET, N_RETRIEVAL_TRIALS,
)


def _nn_distractors(
    pos_emb_norm: np.ndarray,  # (dim,) already L2-normalised
    valid_cats: set,
    all_embs_norm: np.ndarray,  # (N, dim) pre-normalised
    all_cats: np.ndarray,       # (N,) str
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-n nearest images to pos_emb (by cosine sim) that are NOT in valid_cats.

    Both pos_emb_norm and all_embs_norm are expected to be L2-normalised so that
    dot products equal cosine similarity.
    """
    sims = all_embs_norm @ pos_emb_norm          # (N,)
    invalid = np.isin(all_cats, list(valid_cats))
    sims[invalid] = -np.inf
    top_idx = np.argsort(-sims)[:n]
    return all_embs_norm[top_idx], all_cats[top_idx]


def build_c2_pool(
    pos_emb: np.ndarray,           # (dim,) query/positive embedding
    pos_cat: str,                  # category of positive
    valid_cats: set,               # valid categories for the target concept
    pool_type: str,
    # Data sources:
    episode_other_embs: np.ndarray,   # (M, dim) - other episode query images (raw)
    episode_other_cats: np.ndarray,   # (M,) str - categories for above
    all_embs_norm: np.ndarray,        # (N, dim) - ALL cifar_all10 embeddings, pre-normalised
    all_cats: np.ndarray,             # (N,) str - categories for all_embs
    oov_mask: np.ndarray,             # (N,) bool - True if category not in any episode concept
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pool_embs (pool_size, dim), pool_cats (pool_size,)) — NOT shuffled.

    The positive is at index 0; caller shuffles before scoring to prevent
    position bias.
    """
    n_distractors = POOL_SIZES[pool_type] - 1

    if pool_type in ("small", "medium", "large"):
        if len(episode_other_embs) == 0:
            raise ValueError(f"[{pool_type}] No other-concept images for random pool.")
        idxs = rng.choice(len(episode_other_embs), size=n_distractors, replace=True)
        dist_embs = episode_other_embs[idxs]
        dist_cats = episode_other_cats[idxs]

    elif pool_type == "hard":
        pos_norm = pos_emb / (np.linalg.norm(pos_emb) + 1e-12)
        dist_embs, dist_cats = _nn_distractors(
            pos_norm, valid_cats, all_embs_norm, all_cats, n_distractors
        )
        # Convert back from normed to the original scale is fine — learners use
        # cosine similarity internally, but retrieve() just needs relative order.
        # We keep the normed embeddings since that's what all_embs_norm contains.

    elif pool_type in ("oov_25", "oov_50"):
        n_oov = OOV_COUNTS[pool_type]
        n_random = n_distractors - n_oov
        # Random distractors from other episode concepts
        if n_random > 0 and len(episode_other_embs) > 0:
            ridxs = rng.choice(len(episode_other_embs), size=n_random, replace=True)
            r_embs = episode_other_embs[ridxs]
            r_cats = episode_other_cats[ridxs]
        else:
            r_embs = np.zeros((0, pos_emb.shape[0]), dtype=pos_emb.dtype)
            r_cats = np.array([], dtype=str)
        # OOV distractors from categories not in any episode concept
        oov_embs_all = all_embs_norm[oov_mask]
        oov_cats_all = all_cats[oov_mask]
        if len(oov_embs_all) >= n_oov:
            oidxs = rng.choice(len(oov_embs_all), size=n_oov, replace=False)
        else:
            oidxs = rng.choice(len(oov_embs_all), size=n_oov, replace=True)
        o_embs = oov_embs_all[oidxs]
        o_cats = oov_cats_all[oidxs]
        # Combine
        if len(r_embs) > 0:
            dist_embs = np.vstack([r_embs, o_embs])
            dist_cats = np.concatenate([r_cats, o_cats])
        else:
            dist_embs = o_embs
            dist_cats = o_cats
    else:
        raise ValueError(f"Unknown pool_type '{pool_type}'.")

    pool_embs = np.vstack([pos_emb[None], dist_embs])
    pool_cats = np.concatenate([[pos_cat], dist_cats])
    return pool_embs, pool_cats


def evaluate_c2_all_pools(
    learner_dict: dict,         # name -> learner (already fitted)
    local_id: int,              # local concept id in this episode
    valid_cats: set,            # member_categories of the target concept
    q_rows: np.ndarray,         # indices into arrays.query_emb for this concept
    arrays,                     # EpisodeArrays
    all_embs_norm: np.ndarray,  # (N, dim) pre-normalised
    all_cats: np.ndarray,       # (N,) str
    oov_mask: np.ndarray,       # (N,) bool
    pool_types: list[str],
    n_trials: int,
    episode_id: int,
) -> dict[str, dict[str, float]]:
    """C2 accuracy per (pool_type, learner) for one concept in one episode.

    Returns {pool_type -> {learner_name -> mean_c2_over_trials}}.
    Positive (query image) is selected once per trial and is SHARED across pool
    types so the only thing that varies is the distractor set. This preserves
    interpretability: differences across pool types are purely a distractor effect.
    """
    # Index query images of OTHER concepts (for random distractors)
    other_rows = np.where(arrays.query_ids != local_id)[0]
    other_embs = arrays.query_emb[other_rows]
    other_cats = arrays.query_categories[other_rows]

    # Per-trial results accumulator
    per_trial: dict[str, dict[str, list]] = {
        pt: {name: [] for name in learner_dict} for pt in pool_types
    }

    for trial_idx in range(n_trials):
        # Positive selection: SAME seed for all pool types (ensures same positive
        # image across pool type variants within a trial — only distractors differ)
        pos_seed = (episode_id * 1_000_000 + local_id * 10_000 + trial_idx)
        pos_rng = np.random.default_rng(pos_seed)
        pos_row = int(pos_rng.choice(q_rows))
        pos_emb = arrays.query_emb[pos_row]
        pos_cat = str(arrays.query_categories[pos_row])

        for pt in pool_types:
            pt_seed = pos_seed + POOL_TYPE_SEED_OFFSET[pt]
            pt_rng = np.random.default_rng(pt_seed)

            pool_embs, pool_cats = build_c2_pool(
                pos_emb, pos_cat, valid_cats, pt,
                other_embs, other_cats,
                all_embs_norm, all_cats, oov_mask,
                pt_rng,
            )

            # Shuffle pool (positional bias prevention)
            shuffle_rng = np.random.default_rng(pt_seed + 7_777)
            perm = shuffle_rng.permutation(len(pool_embs))
            pool_embs = pool_embs[perm]
            pool_cats = pool_cats[perm]

            for name, learner in learner_dict.items():
                chosen = learner.retrieve(local_id, pool_embs)
                hit = 1.0 if pool_cats[chosen] in valid_cats else 0.0
                per_trial[pt][name].append(hit)

    return {
        pt: {name: float(np.mean(per_trial[pt][name])) for name in learner_dict}
        for pt in pool_types
    }


def evaluate_episode_all_pools(
    learner_dict: dict,         # name -> learner (already fitted)
    episode,                    # Episode
    arrays,                     # EpisodeArrays
    all_embs_norm: np.ndarray,
    all_cats: np.ndarray,
    pool_types: list[str],
    n_trials: int,
) -> dict[str, dict[str, float]]:
    """Macro-averaged C2 for all pool types and all learners over one episode.

    Returns {pool_type -> {learner_name -> macro_c2}}.
    """
    in_vocab_cats: set[str] = set()
    for c in episode.concepts:
        in_vocab_cats.update(c.member_categories)
    oov_mask = ~np.isin(all_cats, list(in_vocab_cats))

    # Accumulate per-concept C2 values, then macro-average
    per_concept_c2: dict[str, dict[str, list]] = {
        pt: {name: [] for name in learner_dict} for pt in pool_types
    }

    for local_id in range(episode.n_ways):
        valid_cats = set(episode.concepts[local_id].member_categories)
        q_rows = np.where(arrays.query_ids == local_id)[0]
        concept_c2 = evaluate_c2_all_pools(
            learner_dict, local_id, valid_cats, q_rows,
            arrays, all_embs_norm, all_cats, oov_mask,
            pool_types, n_trials, episode.episode_id,
        )
        for pt in pool_types:
            for name in learner_dict:
                per_concept_c2[pt][name].append(concept_c2[pt][name])

    return {
        pt: {name: float(np.mean(vals)) for name, vals in pt_d.items()}
        for pt, pt_d in per_concept_c2.items()
    }
