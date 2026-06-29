"""
run_episodes.py — Phase 3 main compute script.

For each of the 100 sampled pairs from Phase 2, builds 3-way acquisition
episodes and evaluates 4 learners:
    - centroid         (prototype baseline)
    - exemplar_knn_k3  (instance-based, k=3)
    - logreg           (linear discriminative)
    - random           (chance floor)

Episode structure (pre-registered):
    - Ways: 3 (disjunctive target + 2 native contrasts)
    - Shots: 5 per concept
    - Query: 50 per concept
    - Episodes: 30 per pair (master_seed=42)

The DISJUNCTIVE concept (label="vorpal") is the union of the sampled pair
(class_u ∪ class_v). Two native contrast classes are sampled deterministically
per pair (excluded: the pair's superclasses). Carroll labels: vorpal/slithy/mimsy.

Episodes are OUTER, learners INNER — the same episode split is shared across
all learners, preserving the pairing invariant required by paired_bootstrap.

Outputs:
    <run_dir>/phase3/exp_008_c1_results.csv        — per-pair aggregated
    <run_dir>/phase3/exp_008_episode_results.csv   — per-episode (for paired tests)
    <run_dir>/phase3/raw/progress.jsonl            — one line per pair (checkpoint)

Usage:
    python src/phase3/run_episodes.py [--run-dir PATH] [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import (
    CIFAR100_CLASSES, CLASS_TO_SUPERCLASS, SUPERCLASSES,
    CARROLL_LABELS, get_run_dir, load_config, load_embeddings, build_episode_pool,
    PROJECT_ROOT,
)

sys.path.insert(0, str(PROJECT_ROOT))
from src.eval.concepts import Concept, ConceptType, ImageRecord
from src.eval.episodes import generate_episodes
from src.eval.harness import resolve_episode_arrays, evaluate_learner_on_episode
from src.eval.stats import bootstrap_ci
from src.learners.centroid import CentroidLearner
from src.learners.exemplar_knn import ExemplarKNN
from src.learners.logreg import LogRegLearner
from src.learners.random_baseline import RandomLearner


LEARNER_NAMES = ["centroid", "exemplar_knn_k3", "logreg", "random"]


def build_learner(name: str, n_ways: int, seed: int):
    if name == "centroid":
        return CentroidLearner(n_ways, seed=seed)
    if name == "exemplar_knn_k3":
        return ExemplarKNN(n_ways, k=3, weighted=True, seed=seed)
    if name == "logreg":
        return LogRegLearner(n_ways, C=1.0, seed=seed)
    if name == "random":
        return RandomLearner(n_ways, seed=seed)
    raise ValueError(f"Unknown learner: '{name}'")


def _get_contrast_classes(
    class_u: str,
    class_v: str,
    pair_id: int,
    base_seed: int,
) -> tuple[str, str]:
    """Sample 2 contrast classes, excluding the pair's superclasses."""
    sc_u = CLASS_TO_SUPERCLASS.get(class_u)
    sc_v = CLASS_TO_SUPERCLASS.get(class_v)
    excluded_sc = {sc for sc in (sc_u, sc_v) if sc is not None}

    candidates = [
        cls for cls in CIFAR100_CLASSES
        if cls not in (class_u, class_v)
        and CLASS_TO_SUPERCLASS.get(cls) not in excluded_sc
    ]
    if len(candidates) < 2:
        raise RuntimeError(
            f"Not enough contrast candidates for pair ({class_u}, {class_v}): "
            f"only {len(candidates)} available after excluding superclasses."
        )

    rng = np.random.default_rng(base_seed + pair_id)
    chosen = rng.choice(len(candidates), size=2, replace=False)
    return candidates[chosen[0]], candidates[chosen[1]]


def _build_concepts(
    class_u: str,
    class_v: str,
    contrast1: str,
    contrast2: str,
    embeddings_pool: dict[str, np.ndarray],
    h2c: dict[str, str],
) -> list[Concept]:
    """Build the 3 Concept objects for one episode set:
        concept 0: disjunctive (class_u ∪ class_v)  → "vorpal"
        concept 1: native (contrast1)                → "slithy"
        concept 2: native (contrast2)                → "mimsy"
    """
    def hashes_for(cls: str) -> tuple[str, ...]:
        return tuple(h for h, c in h2c.items() if c == cls)

    disjunctive_hashes = hashes_for(class_u) + hashes_for(class_v)
    native1_hashes     = hashes_for(contrast1)
    native2_hashes     = hashes_for(contrast2)

    if not disjunctive_hashes:
        raise RuntimeError(f"No hashes found for disjunctive pair ({class_u}, {class_v})")
    if not native1_hashes:
        raise RuntimeError(f"No hashes found for contrast class '{contrast1}'")
    if not native2_hashes:
        raise RuntimeError(f"No hashes found for contrast class '{contrast2}'")

    return [
        Concept(
            concept_id=0,
            label=CARROLL_LABELS[0],        # "vorpal"
            concept_type=ConceptType.DISJUNCTIVE,
            member_categories=(class_u, class_v),
            image_hashes=disjunctive_hashes,
            predicate_desc=f"disjunctive: {class_u} ∪ {class_v}",
        ),
        Concept(
            concept_id=1,
            label=CARROLL_LABELS[1],        # "slithy"
            concept_type=ConceptType.NATIVE,
            member_categories=(contrast1,),
            image_hashes=native1_hashes,
            predicate_desc=f"native: {contrast1}",
        ),
        Concept(
            concept_id=2,
            label=CARROLL_LABELS[2],        # "mimsy"
            concept_type=ConceptType.NATIVE,
            member_categories=(contrast2,),
            image_hashes=native2_hashes,
            predicate_desc=f"native: {contrast2}",
        ),
    ]


def _run_pair(
    pair: dict,
    embeddings: dict[str, np.ndarray],
    h2c: dict[str, str],
    config: dict,
    contrast_base_seed: int,
) -> tuple[list[dict], list[dict]]:
    """Run all episodes for one sampled pair.

    Returns:
        episode_rows:  list of per-episode result dicts
        agg_rows:      list of per-(pair, learner) aggregated result dicts
    """
    sample_id = pair["sample_id"]
    class_u   = pair["class_u"]
    class_v   = pair["class_v"]
    quadrant  = pair["quadrant"]
    d_perc    = float(pair["d_perc"])
    d_sem     = float(pair["d_sem"])

    contrast1, contrast2 = _get_contrast_classes(
        class_u, class_v, pair_id=int(sample_id),
        base_seed=contrast_base_seed,
    )

    concepts = _build_concepts(class_u, class_v, contrast1, contrast2, embeddings, h2c)

    p3 = config["phase3"]
    episodes = list(generate_episodes(
        concepts,
        n_episodes=p3["n_episodes"],
        n_ways=p3["n_ways"],
        n_shots=p3["n_shots"],
        n_query_per_concept=p3["n_query_per_concept"],
        master_seed=p3["master_seed"],
        hash_to_category=h2c,
    ))

    # Collect per-episode results, keyed by learner (for aggregation)
    per_learner_c1: dict[str, list[float]] = {ln: [] for ln in LEARNER_NAMES}
    per_learner_margin: dict[str, list[float]] = {ln: [] for ln in LEARNER_NAMES}
    episode_rows: list[dict] = []

    for ep in episodes:
        arrays = resolve_episode_arrays(ep, embeddings, h2c)
        for lname in LEARNER_NAMES:
            learner = build_learner(lname, n_ways=p3["n_ways"], seed=ep.episode_id)
            m = evaluate_learner_on_episode(
                learner, ep, arrays,
                rng=np.random.default_rng(20_000 + ep.episode_id),
            )
            per_learner_c1[lname].append(m["c1"])
            per_learner_margin[lname].append(m["margin"])
            episode_rows.append({
                "sample_id":  sample_id,
                "class_u":    class_u,
                "class_v":    class_v,
                "contrast1":  contrast1,
                "contrast2":  contrast2,
                "quadrant":   quadrant,
                "d_perc":     round(d_perc, 6),
                "d_sem":      round(d_sem, 6),
                "learner":    lname,
                "episode_id": ep.episode_id,
                "c1":         round(m["c1"], 4),
                "margin":     round(m["margin"], 4),
            })

    # Aggregate per learner
    agg_rows: list[dict] = []
    n_boot = config["phase4"]["bootstrap_resamples"]
    alpha  = config["phase4"]["alpha"]
    for lname in LEARNER_NAMES:
        c1_vals = np.array(per_learner_c1[lname])
        ci = bootstrap_ci(c1_vals, n_resamples=n_boot, alpha=alpha, seed=42)
        agg_rows.append({
            "sample_id":    sample_id,
            "class_u":      class_u,
            "class_v":      class_v,
            "contrast1":    contrast1,
            "contrast2":    contrast2,
            "quadrant":     quadrant,
            "d_perc":       round(d_perc, 6),
            "d_sem":        round(d_sem, 6),
            "learner":      lname,
            "c1_accuracy":  round(ci.mean, 4),
            "c1_ci_low":    round(ci.ci_low, 4),
            "c1_ci_high":   round(ci.ci_high, 4),
            "mean_margin":  round(float(np.mean(per_learner_margin[lname])), 4),
        })

    return episode_rows, agg_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 3 — run acquisition episodes"
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already in progress.jsonl")
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    config  = load_config()

    pairs_path = run_dir / "phase2" / "sampled_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing: {pairs_path}. Run sample_pairs.py first.")

    pairs_df = pd.read_csv(pairs_path)
    n_pairs  = len(pairs_df)
    print(f"[episodes] {n_pairs} pairs to evaluate")

    print("[episodes] loading embedding pool ...")
    emb, labels, class_names = load_embeddings()
    embeddings, h2c = build_episode_pool(
        emb, labels, class_names,
        pool_per_class=config["dataset"]["episode_pool_per_class"],
        seed=42,
    )

    progress_path = run_dir / "phase3" / "raw" / "progress.jsonl"
    done_ids: set[int] = set()
    if args.resume and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["sample_id"])
        print(f"[episodes] resuming — {len(done_ids)} pairs already done")

    contrast_base_seed = config["phase3"]["contrast_base_seed"]
    all_episode_rows: list[dict] = []
    all_agg_rows: list[dict] = []

    for row_idx, pair in pairs_df.iterrows():
        sid = int(pair["sample_id"])
        if sid in done_ids:
            continue

        print(f"  [{sid:>3}/{n_pairs-1}] ({pair['class_u']}, {pair['class_v']}) "
              f"Q={pair['quadrant']}", end=" ... ", flush=True)

        ep_rows, agg_rows = _run_pair(
            pair.to_dict(), embeddings, h2c, config, contrast_base_seed,
        )
        all_episode_rows.extend(ep_rows)
        all_agg_rows.extend(agg_rows)

        # Checkpoint
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sample_id": sid, "n_episode_rows": len(ep_rows)}) + "\n")

        c1_centroid = next(r["c1_accuracy"] for r in agg_rows if r["learner"] == "centroid")
        print(f"centroid C1={c1_centroid:.3f}")

    # Write outputs
    ep_df  = pd.DataFrame(all_episode_rows)
    agg_df = pd.DataFrame(all_agg_rows)

    ep_path  = run_dir / "phase3" / "exp_008_episode_results.csv"
    agg_path = run_dir / "phase3" / "exp_008_c1_results.csv"
    ep_df.to_csv(ep_path,  index=False)
    agg_df.to_csv(agg_path, index=False)
    print(f"\n[episodes] episode results  -> {ep_path}  ({len(ep_df)} rows)")
    print(f"[episodes] aggregated C1    -> {agg_path}  ({len(agg_df)} rows)")
    print(f"\nNext step:\n  python src/phase4/regression_analysis.py")


if __name__ == "__main__":
    main()
