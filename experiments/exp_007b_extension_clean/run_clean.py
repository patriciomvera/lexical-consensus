"""
run_clean.py — exp_007b_extension_clean homogeneous re-run.

Runs C1 + C2 for all four tiers (native, near, mid, far) under the SAME seven
pool constructions, at fixed ways=3, shots=5, 30 episodes, master_seed=42.

Key design decisions vs the original extension:
  - All tiers re-run (no loading from prior ledgers); native and far are NOT
    loaded from exp_007/ledger.jsonl so pool conditions are homogeneous.
  - Pool types: std_nway (n_ways=3 candidates), small (5), medium (10),
    large (25), hard (10 NN distractors), oov_25 (10, ~2 OOV), oov_50 (10, ~5 OOV).
  - Positive selection seed is SHARED across pool types per trial; only the
    distractor seed varies.
  - Episodes outer, learners inner — pairing invariant preserved.

Output: results/exp_007b_extension_clean/ledger_c2_homogeneous.jsonl
        (one row per episode x learner x pool_type)

Usage:
    cd C:\\Lexical-Consensus
    python experiments/exp_007b_extension_clean/run_clean.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ─── Path bootstrap ────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]   # ext_clean -> experiments -> project root
_EXP007  = _PROJECT / "experiments" / "exp_007_concept_carving"
for _p in [str(_PROJECT), str(_EXP007)]:
    if _p not in sys.path:
        sys.path.append(_p)

import importlib.util as _ilu
_shared_spec = _ilu.spec_from_file_location("_shared_007", str(_EXP007 / "_shared.py"))
_shared007   = _ilu.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(_shared007)

load_or_build_embeddings = _shared007.load_or_build_embeddings
hash_to_category         = _shared007.hash_to_category

from src.eval.concepts import Concept, ConceptType
from src.eval.episodes  import generate_episodes
from src.eval.harness   import resolve_episode_arrays
from src.eval.metrics   import c1_naming, margin as _margin
from src.eval.stats     import bootstrap_ci

from src.learners.centroid      import CentroidLearner
from src.learners.kcentroid     import KCentroidLearner
from src.learners.exemplar_knn  import ExemplarKNN
from src.learners.logreg        import LogRegLearner
from src.learners.linsvm        import LinSVMLearner
from src.learners.random_baseline import RandomLearner

# ─── Configuration ────────────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT / "results" / "exp_007b_extension_clean"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WAYS              = 3
SHOTS             = 5
N_EPISODES        = 30
N_QUERY           = 50
N_RETRIEVAL_TRIALS = 20
MASTER_SEED       = 42
SUB               = "007b_ext_clean"

LEARNER_NAMES = [
    "centroid", "exemplar_knn_k3", "kcentroid_k2",
    "logreg", "linsvm", "random",
]

# Pool type -> (pool_size, distractor_rng_offset)
# Positive selection seed is shared (base seed); each pool type uses base + offset.
POOL_CONFIG = {
    "std_nway": (WAYS,  50_000),   # 1 pos + (n_ways-1) random from other concepts, one each
    "small":    (5,    100_000),   # 1 pos + 4 random from other episode query images
    "medium":   (10,   200_000),   # 1 pos + 9 random
    "large":    (25,   300_000),   # 1 pos + 24 random
    "hard":     (10,   400_000),   # 1 pos + 9 nearest-neighbor distractors
    "oov_25":   (10,   500_000),   # 1 pos + ~7 random + ~2 OOV
    "oov_50":   (10,   600_000),   # 1 pos + ~5 random + ~5 OOV
}
POOL_TYPES  = list(POOL_CONFIG.keys())
OOV_COUNTS  = {"oov_25": 2, "oov_50": 5}

# ─── Concept definitions ───────────────────────────────────────────────────────
# Fixed Carroll-vocabulary labels from the extension (do not re-select).
NATIVE_DEF = [   # (label, category)
    ("slithy",     "frog"),
    ("mimsy",      "horse"),
    ("vorpal",     "ship"),
    ("borogove",   "automobile"),
    ("tulgey",     "airplane"),
    ("frumious",   "bird"),
    ("manxome",    "cat"),
    ("galumphing", "deer"),
    ("uffish",     "dog"),
    ("jubjub",     "truck"),
]

NEAR_DEF = [   # (label, (cat_a, cat_b))
    ("tromble", ("cat",        "dog")),
    ("varble",  ("bird",       "deer")),
    ("glintz",  ("automobile", "truck")),
]

MID_DEF = [
    ("brumble", ("deer",     "dog")),
    ("strenk",  ("airplane", "ship")),
    ("worblix", ("bird",     "horse")),
]

FAR_DEF = [
    ("gleth", ("airplane", "cat")),
    ("mivor", ("frog",     "truck")),
    ("plonk", ("automobile", "deer")),
    ("quax",  ("horse",    "ship")),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_native_concepts(records) -> list[Concept]:
    h2c = hash_to_category(records)
    concepts = []
    for cid, (label, cat) in enumerate(NATIVE_DEF):
        hashes = tuple(r.img_hash for r in records if r.object_category == cat)
        concepts.append(Concept(
            concept_id=cid,
            label=label,
            concept_type=ConceptType.NATIVE,
            member_categories=(cat,),
            image_hashes=hashes,
            predicate_desc=f"native: {cat}",
        ))
    return concepts


def build_disjunctive_concepts(definitions: list, records, embeddings: dict) -> list[Concept]:
    h2c = hash_to_category(records)
    concepts = []
    for cid, (label, members) in enumerate(definitions):
        hashes = tuple(h for h in embeddings if h2c[h] in set(members))
        concepts.append(Concept(
            concept_id=cid,
            label=label,
            concept_type=ConceptType.DISJUNCTIVE,
            member_categories=tuple(members),
            image_hashes=hashes,
            predicate_desc=f"disjunctive: {{{members[0]}, {members[1]}}}",
        ))
    return concepts


def build_learner(name: str, n_concepts: int, *, seed: int):
    if name == "centroid":
        return CentroidLearner(n_concepts, seed=seed)
    if name == "exemplar_knn_k3":
        return ExemplarKNN(n_concepts, k=3, weighted=True, seed=seed)
    if name == "kcentroid_k2":
        return KCentroidLearner(n_concepts, k=2, seed=seed)
    if name == "logreg":
        return LogRegLearner(n_concepts, C=1.0, seed=seed)
    if name == "linsvm":
        return LinSVMLearner(n_concepts, C=1.0, seed=seed)
    if name == "random":
        return RandomLearner(n_concepts, seed=seed)
    raise ValueError(f"unknown learner '{name}'")


def build_all_embs_norm(embeddings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pre-normalised embedding matrix for NN-distractor search."""
    all_hashes = sorted(embeddings.keys())
    matrix = np.asarray([embeddings[h] for h in all_hashes], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    cats = np.array([h.rsplit("_", 1)[0] for h in all_hashes])
    return matrix, cats


def _nn_distractors(
    pos_norm: np.ndarray,
    valid_cats: set,
    all_embs_norm: np.ndarray,
    all_cats: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    sims = all_embs_norm @ pos_norm
    mask = np.isin(all_cats, list(valid_cats))
    sims[mask] = -np.inf
    top_idx = np.argsort(-sims)[:n]
    return all_embs_norm[top_idx], all_cats[top_idx]


def build_pool(
    pool_type: str,
    pos_emb: np.ndarray,
    pos_cat: str,
    valid_cats: set,
    other_rows_by_concept: dict,        # {concept_local_id: row_indices}
    all_other_rows: np.ndarray,         # all non-target-concept query rows
    arrays,
    all_embs_norm: np.ndarray,
    all_cats: np.ndarray,
    oov_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (pool_embs, pool_cats) for one trial. Positive is NOT yet included."""
    pool_size, _ = POOL_CONFIG[pool_type]
    n_dist = pool_size - 1

    if pool_type == "std_nway":
        # One image from each other concept (same semantics as harness._c2_macro)
        d_embs, d_cats = [], []
        for c_rows in other_rows_by_concept.values():
            r = int(rng.choice(c_rows))
            d_embs.append(arrays.query_emb[r])
            d_cats.append(str(arrays.query_categories[r]))
        dist_embs = np.asarray(d_embs)
        dist_cats = np.asarray(d_cats)

    elif pool_type in ("small", "medium", "large"):
        if len(all_other_rows) == 0:
            raise RuntimeError(f"[{pool_type}] no other-concept images available")
        idxs = rng.choice(len(all_other_rows), size=n_dist, replace=True)
        dist_embs = arrays.query_emb[all_other_rows[idxs]]
        dist_cats = arrays.query_categories[all_other_rows[idxs]]

    elif pool_type == "hard":
        pos_norm = pos_emb / (np.linalg.norm(pos_emb) + 1e-12)
        dist_embs, dist_cats = _nn_distractors(
            pos_norm.astype(np.float32), valid_cats,
            all_embs_norm, all_cats, n_dist,
        )

    elif pool_type in ("oov_25", "oov_50"):
        n_oov  = OOV_COUNTS[pool_type]
        n_rand = n_dist - n_oov
        if n_rand > 0 and len(all_other_rows) > 0:
            ridxs  = rng.choice(len(all_other_rows), size=n_rand, replace=True)
            r_embs = arrays.query_emb[all_other_rows[ridxs]]
            r_cats = arrays.query_categories[all_other_rows[ridxs]]
        else:
            r_embs = np.zeros((0, pos_emb.shape[0]), dtype=pos_emb.dtype)
            r_cats = np.array([], dtype=str)
        oov_embs_all = all_embs_norm[oov_mask]
        oov_cats_all = all_cats[oov_mask]
        replace_oov  = len(oov_embs_all) < n_oov
        oidxs  = rng.choice(len(oov_embs_all), size=n_oov, replace=replace_oov)
        o_embs = oov_embs_all[oidxs]
        o_cats = oov_cats_all[oidxs]
        if len(r_embs) > 0:
            dist_embs = np.vstack([r_embs, o_embs])
            dist_cats = np.concatenate([r_cats, o_cats])
        else:
            dist_embs = o_embs
            dist_cats = o_cats
    else:
        raise ValueError(f"unknown pool_type '{pool_type}'")

    pool_embs = np.vstack([pos_emb[None], dist_embs])
    pool_cats = np.concatenate([[pos_cat], dist_cats])
    return pool_embs, pool_cats


def evaluate_c2_all_pools(
    learner_dict: dict,
    episode,
    arrays,
    all_embs_norm: np.ndarray,
    all_cats: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Returns {pool_type: {learner_name: macro_c2}} for one episode."""
    # OOV mask: images whose category is not in any concept in this episode
    in_vocab = set()
    for c in episode.concepts:
        in_vocab.update(c.member_categories)
    oov_mask = ~np.isin(all_cats, list(in_vocab))

    rows_by_concept = {c: np.where(arrays.query_ids == c)[0]
                       for c in range(episode.n_ways)}

    # Accumulate per-concept c2 hits (averaged over trials)
    # {pool_type: {learner: [per-concept macro c2 values]}}
    per_concept_c2: dict[str, dict[str, list]] = {
        pt: {name: [] for name in learner_dict} for pt in POOL_TYPES
    }

    for local_id in range(episode.n_ways):
        valid_cats = set(episode.concepts[local_id].member_categories)
        pos_rows   = rows_by_concept[local_id]
        other_rows_by_concept = {
            c: rows_by_concept[c]
            for c in range(episode.n_ways) if c != local_id
        }
        all_other_rows = np.concatenate([
            rows_by_concept[c] for c in range(episode.n_ways) if c != local_id
        ])

        # Per-trial hits accumulator: {pool_type: {learner: int}}
        trial_hits: dict[str, dict[str, int]] = {
            pt: {name: 0 for name in learner_dict} for pt in POOL_TYPES
        }

        for trial_idx in range(N_RETRIEVAL_TRIALS):
            # Positive selection: same seed for all pool types (shared positive)
            pos_seed = (episode.episode_id * 1_000_000
                        + local_id         * 10_000
                        + trial_idx)
            pos_rng  = np.random.default_rng(pos_seed)
            pos_row  = int(pos_rng.choice(pos_rows))
            pos_emb  = arrays.query_emb[pos_row].copy()
            pos_cat  = str(arrays.query_categories[pos_row])

            for pt in POOL_TYPES:
                _, dist_offset = POOL_CONFIG[pt]
                dist_seed = pos_seed + dist_offset
                dist_rng  = np.random.default_rng(dist_seed)

                pool_embs, pool_cats = build_pool(
                    pt, pos_emb, pos_cat, valid_cats,
                    other_rows_by_concept, all_other_rows, arrays,
                    all_embs_norm, all_cats, oov_mask, dist_rng,
                )

                # Shuffle (positional bias prevention)
                shuf_rng = np.random.default_rng(dist_seed + 7_777)
                perm = shuf_rng.permutation(len(pool_embs))
                pool_embs = pool_embs[perm]
                pool_cats = pool_cats[perm]

                for name, learner in learner_dict.items():
                    chosen = learner.retrieve(local_id, pool_embs)
                    if pool_cats[chosen] in valid_cats:
                        trial_hits[pt][name] += 1

        for pt in POOL_TYPES:
            for name in learner_dict:
                per_concept_c2[pt][name].append(
                    trial_hits[pt][name] / N_RETRIEVAL_TRIALS
                )

    # Macro-average over concepts
    return {
        pt: {name: float(np.mean(per_concept_c2[pt][name]))
             for name in learner_dict}
        for pt in POOL_TYPES
    }


def compute_margin(score_vecs: np.ndarray) -> float:
    return float(_margin(score_vecs))


def per_concept_c1_dict(preds, arrays, episode) -> dict[str, float]:
    out = {}
    for local_id in range(episode.n_ways):
        mask = arrays.query_ids == local_id
        if mask.any():
            label = episode.concepts[local_id].label
            out[label] = float((preds[mask] == local_id).mean())
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[run_clean] loading embeddings ...")
    embeddings, records = load_or_build_embeddings("cifar_all10")
    h2c = hash_to_category(records)

    print("[run_clean] building normalized embedding matrix ...")
    all_embs_norm, all_cats_arr = build_all_embs_norm(embeddings)

    # Build concept lists for each tier
    native_concepts = build_native_concepts(records)
    near_concepts   = build_disjunctive_concepts(NEAR_DEF, records, embeddings)
    mid_concepts    = build_disjunctive_concepts(MID_DEF,  records, embeddings)
    far_concepts    = build_disjunctive_concepts(FAR_DEF,  records, embeddings)

    tiers = {
        "native": (native_concepts, "native"),
        "near":   (near_concepts,   "disjunctive"),
        "mid":    (mid_concepts,    "disjunctive"),
        "far":    (far_concepts,    "disjunctive"),
    }

    ledger_path = RESULTS_DIR / "ledger_c2_homogeneous.jsonl"
    n_total = len(tiers) * N_EPISODES * len(LEARNER_NAMES) * len(POOL_TYPES)
    print(f"[run_clean] writing {n_total} ledger rows -> {ledger_path.name}")

    with open(ledger_path, "w", encoding="utf-8") as fout:
        for tier_name, (concepts, concept_type) in tiers.items():
            print(f"\n[tier: {tier_name}] {len(concepts)} concepts")
            episodes = list(generate_episodes(
                concepts,
                n_episodes=N_EPISODES,
                n_ways=WAYS,
                n_shots=SHOTS,
                n_query_per_concept=N_QUERY,
                master_seed=MASTER_SEED,
                hash_to_category=h2c,
            ))

            for ep_idx, episode in enumerate(episodes):
                # Build arrays once; shared by all learners (pairing guarantee)
                arrays = resolve_episode_arrays(episode, embeddings, h2c)

                # Fit all learners on identical support
                learner_dict = {}
                for li, name in enumerate(LEARNER_NAMES):
                    seed = MASTER_SEED + episode.episode_id * 17 + li
                    lrn  = build_learner(name, episode.n_ways, seed=seed)
                    lrn.fit(arrays.support_emb, arrays.support_ids)
                    learner_dict[name] = lrn

                # C1 and margin (pool_type-independent)
                c1_by_learner     = {}
                margin_by_learner = {}
                pc1_by_learner    = {}
                for name, lrn in learner_dict.items():
                    preds  = lrn.predict(arrays.query_emb)
                    scores = lrn.score(arrays.query_emb)
                    c1_by_learner[name]     = c1_naming(preds, arrays.query_ids, episode.n_ways)
                    margin_by_learner[name] = compute_margin(scores)
                    pc1_by_learner[name]    = per_concept_c1_dict(preds, arrays, episode)

                # C2 for all pool types in one pass
                c2_by_pool = evaluate_c2_all_pools(
                    learner_dict, episode, arrays,
                    all_embs_norm, all_cats_arr,
                )

                # Write one row per (learner x pool_type)
                for pt in POOL_TYPES:
                    for name in LEARNER_NAMES:
                        row = {
                            "sub":          SUB,
                            "tier":         tier_name,
                            "concept_type": concept_type,
                            "learner":      name,
                            "ways":         episode.n_ways,
                            "shots":        episode.n_shots,
                            "episode_id":   episode.episode_id,
                            "pool_type":    pt,
                            "chance":       round(1.0 / episode.n_ways, 6),
                            "c1":           round(c1_by_learner[name], 6),
                            "c2":           round(c2_by_pool[pt][name], 6),
                            "margin":       round(margin_by_learner[name], 6),
                            "per_concept_c1": pc1_by_learner[name],
                        }
                        fout.write(json.dumps(row) + "\n")

                if (ep_idx + 1) % 10 == 0:
                    print(f"  episode {ep_idx + 1}/{N_EPISODES}")

    print(f"\n[run_clean] done. Ledger: {ledger_path}")


if __name__ == "__main__":
    main()
