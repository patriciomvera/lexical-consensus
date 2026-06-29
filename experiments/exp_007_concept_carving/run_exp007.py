"""
run_exp007.py — condition loop -> ledger + descriptors (plan §8, §12).

Drives the full grid for one sub-experiment. Loop order is EPISODES OUTER,
LEARNERS INNER: a single Episode is drawn once and every learner is fitted and
scored on its byte-identical support/query split. That ordering is what makes
the paired bootstrap / Wilcoxon in stats.py valid (see src.eval.episodes).

Outputs (plan §12), written under results/exp_007/:
    ledger.jsonl     — one row per (sub, concept_type, learner, ways, shots,
                       episode): c1, c2, margin, nmi_pred_category, chance.
    descriptors.csv  — NS_silhouette, compactness, centroid_lands_outside,
                       NMI_native per concept.

Aggregation into aggregate.csv / paired_tests.csv and the figures is done
separately by analyze_exp007.py, so a long run and its analysis re-run
independently.

Usage:
    python run_exp007.py --sub 007a
    python run_exp007.py --sub 007b --controls   # (007b/007c not yet wired)
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from _shared import (  # noqa: E402  (path injected in _shared)
    LEARNERS, WAYS, SHOTS, N_EPISODES, N_QUERY_PER_CONCEPT, MASTER_SEED,
    KCENTROID_K, KNN_K, N_DISJUNCTIVE_CONCEPTS, DISJUNCTION_ARITY, RESULTS_DIR,
    load_or_build_embeddings, native_category_to_label, hash_to_category,
    inter_category_distance_matrix, ALL_CIFAR_CATEGORIES,
)

from src.eval import descriptors, harness                       # noqa: E402
from src.eval.concepts import (                                  # noqa: E402
    build_native_concepts, build_disjunctive_concepts, chance_level, ConceptType,
)
from src.eval.episodes import generate_episodes                  # noqa: E402
from src.learners.centroid import CentroidLearner                # noqa: E402
from src.learners.kcentroid import KCentroidLearner              # noqa: E402
from src.learners.exemplar_knn import ExemplarKNN                # noqa: E402
from src.learners.logreg import LogRegLearner                    # noqa: E402
from src.learners.linsvm import LinSVMLearner                    # noqa: E402
from src.learners.random_baseline import RandomLearner           # noqa: E402

LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"
DESCRIPTORS_PATH = RESULTS_DIR / "descriptors.csv"

# 007a representative hyperparameters (bare names).
KCENTROID_K_DEFAULT = 2
KNN_K_DEFAULT = 3

# 007b roster (directives 1-2): exemplar k in {1,3,5}, kcentroid K in {1,2,3}
# (K=1 collapses to centroid — a sanity check; K=2 = disjunction arity is the
# theory-motivated anchor). Variant names carry the hyperparameter so the
# paired-test machinery (which groups by the 'learner' string) keeps each
# variant separate.
ROSTER_007B = (
    ["centroid"]
    + [f"kcentroid_k{k}" for k in KCENTROID_K]
    + [f"exemplar_knn_k{k}" for k in KNN_K]
    + ["logreg", "linsvm", "random"]
)


def build_learner(name: str, n_concepts: int, *, seed: int):
    """Instantiate a learner by name. Accepts bare 007a names and the 007b
    swept variants 'kcentroid_k{K}' / 'exemplar_knn_k{K}'."""
    if name == "centroid":
        return CentroidLearner(n_concepts, seed=seed)
    if name == "kcentroid":
        return KCentroidLearner(n_concepts, k=KCENTROID_K_DEFAULT, seed=seed)
    if name.startswith("kcentroid_k"):
        return KCentroidLearner(n_concepts, k=int(name.rsplit("k", 1)[1]), seed=seed)
    if name == "exemplar_knn":
        return ExemplarKNN(n_concepts, k=KNN_K_DEFAULT, weighted=True, seed=seed)
    if name.startswith("exemplar_knn_k"):
        return ExemplarKNN(n_concepts, k=int(name.rsplit("k", 1)[1]), weighted=True, seed=seed)
    if name == "logreg":
        return LogRegLearner(n_concepts, C=1.0, seed=seed)
    if name == "linsvm":
        return LinSVMLearner(n_concepts, C=1.0, seed=seed)
    if name == "random":
        return RandomLearner(n_concepts, seed=seed)
    raise ValueError(f"unknown learner '{name}'")


def _compute_descriptors(concepts, embeddings, h2c, *, sub: str) -> list[dict]:
    """NS / compactness / centroid_lands_outside / NMI_native per concept.

    NMI_native is set-level: for a NATIVE set it is 1.0 (label == rename); for a
    DISJUNCTIVE set it is < 1 (each concept merges two categories) — the
    "partial NMI_native" the plan expects (§5)."""
    all_hashes = list(embeddings.keys())
    all_concept_ids = np.array([
        next((c.concept_id for c in concepts if h2c[h] in c.member_categories), -1)
        for h in all_hashes
    ])
    all_categories = np.array([h2c[h] for h in all_hashes])
    nmi = descriptors.nmi_native(all_concept_ids, all_categories)
    rows = []
    for c in concepts:
        members = [h for h in all_hashes if h2c[h] in c.member_categories]
        others = [h for h in all_hashes if h not in set(members)]
        ext = np.asarray([embeddings[h] for h in members], dtype=np.float64)
        oth = np.asarray([embeddings[h] for h in others], dtype=np.float64)
        rows.append({
            "sub": sub,
            "concept_id": c.concept_id,
            "label": c.label,
            "concept_type": c.concept_type.value,
            "members": "|".join(c.member_categories),
            "ns_silhouette": round(descriptors.ns_silhouette(ext, oth), 4),
            "compactness": round(descriptors.compactness(ext), 4),
            "centroid_lands_outside": descriptors.centroid_lands_outside(ext, oth),
            "nmi_native_setlevel": round(nmi, 4),
        })
    return rows


def _merge_csv_by_sub(new_rows: list[dict], path, sub: str) -> None:
    """Idempotent CSV write keyed by sub: drop prior rows for this sub, keep the
    rest, append new (so 007a and 007b descriptors coexist in one file)."""
    import csv
    kept = []
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r.get("sub") != sub:
                kept.append(r)
    all_rows = kept + new_rows
    fields = list(new_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"  saved {path.name} ({len(new_rows)} rows for {sub}, {len(kept)} retained)")


def _rewrite_ledger_dropping_sub(new_rows: list[dict], sub: str) -> None:
    """Idempotent: drop any prior rows for this sub, keep the rest, append new."""
    kept = []
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sub") != sub:
                kept.append(row)
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        for row in kept + new_rows:
            f.write(json.dumps(row) + "\n")
    print(f"  saved {LEDGER_PATH.name} ({len(new_rows)} new rows for {sub}, "
          f"{len(kept)} retained)")


def _eval_grid(sub, concept_type, concepts, embeddings, h2c, learners, ways_list) -> list[dict]:
    """Shared condition loop: episodes OUTER, learners INNER (preserves the
    paired-episode invariant). Returns ledger rows. The per-concept C1 dict is
    kept un-rounded (it's a nested map); scalar metrics are rounded."""
    rows: list[dict] = []
    for ways in ways_list:
        if ways > len(concepts):
            print(f"  [skip] {concept_type.value} ways={ways} > concepts ({len(concepts)})")
            continue
        chance = chance_level(concept_type, ways)
        for shots in SHOTS:
            episodes = generate_episodes(
                concepts, n_episodes=N_EPISODES, n_ways=ways, n_shots=shots,
                n_query_per_concept=N_QUERY_PER_CONCEPT, master_seed=MASTER_SEED,
                hash_to_category=h2c,
            )
            for ep in episodes:
                arrays = harness.resolve_episode_arrays(ep, embeddings, h2c)
                for name in learners:
                    learner = build_learner(name, ways, seed=ep.episode_id)
                    m = harness.evaluate_learner_on_episode(
                        learner, ep, arrays,
                        rng=np.random.default_rng(10_000 + ep.episode_id),
                    )
                    per_concept = m.pop("per_concept_c1")
                    rows.append({
                        "sub": sub,
                        "concept_type": concept_type.value,
                        "learner": name,
                        "ways": ways,
                        "shots": shots,
                        "episode_id": ep.episode_id,
                        "chance": round(chance, 4),
                        **{k: round(v, 4) for k, v in m.items()},
                        "per_concept_c1": {k: round(v, 4) for k, v in per_concept.items()},
                    })
            print(f"  {concept_type.value:11s} ways={ways} shots={shots:>2}: "
                  f"{len(learners) * N_EPISODES} rows")
    return rows


def run_007a() -> list[dict]:
    """NATIVE (5 categories). Establishes H2; first fully-rigorous table."""
    embeddings, records = load_or_build_embeddings("cifar_native")
    h2c = hash_to_category(records)
    concepts = build_native_concepts(records, native_category_to_label())
    _merge_csv_by_sub(_compute_descriptors(concepts, embeddings, h2c, sub="007a"),
                      DESCRIPTORS_PATH, "007a")
    return _eval_grid("007a", ConceptType.NATIVE, concepts, embeddings, h2c, LEARNERS, WAYS)


def run_007b() -> list[dict]:
    """DISJUNCTIVE (primary H1/H3 payoff). Runs BOTH the native anchor (10
    categories) and the disjunctive set (5 disjoint perceptually-distant pairs)
    so the Accuracy-vs-NS collapse curve is produced from one run, with the
    swept learner roster (kcentroid K∈{1,2,3}, exemplar k∈{1,3,5})."""
    embeddings, records = load_or_build_embeddings("cifar_all10")
    h2c = hash_to_category(records)

    # Native anchor over all 10 categories (high-NS end of the curve).
    native = build_native_concepts(records, native_category_to_label(ALL_CIFAR_CATEGORIES))

    # Disjunctive concepts: distance-driven disjoint matching (directives 4-5).
    dist, order = inter_category_distance_matrix(embeddings, records)
    disjunctive = build_disjunctive_concepts(
        records, embeddings, n_concepts=N_DISJUNCTIVE_CONCEPTS,
        distance_matrix=dist, category_order=order, arity=DISJUNCTION_ARITY,
        seed=MASTER_SEED,
    )

    # Descriptors first, and PRINT the native-vs-disjunctive comparison so the
    # selection (directive 5: disjunctive NS/compactness below native) is visible.
    nat_desc = _compute_descriptors(native, embeddings, h2c, sub="007b")
    dis_desc = _compute_descriptors(disjunctive, embeddings, h2c, sub="007b")
    _print_selection_table(nat_desc, dis_desc)
    _merge_csv_by_sub(nat_desc + dis_desc, DESCRIPTORS_PATH, "007b")

    rows = _eval_grid("007b", ConceptType.NATIVE, native, embeddings, h2c, ROSTER_007B, WAYS)
    rows += _eval_grid("007b", ConceptType.DISJUNCTIVE, disjunctive, embeddings, h2c,
                       ROSTER_007B, WAYS)
    return rows


def _print_selection_table(nat_desc, dis_desc) -> None:
    nat_ns = sorted(r["ns_silhouette"] for r in nat_desc)
    nat_median = float(np.median(nat_ns))
    nat_comp_median = float(np.median([r["compactness"] for r in nat_desc]))
    print("\n  -- disjunctive selection (directive 5) --")
    print(f"  native NS:          min={nat_ns[0]:.3f}  median={nat_median:.3f}  max={nat_ns[-1]:.3f}")
    print(f"  native compactness: median={nat_comp_median:.3f}")
    print(f"  reference for carving = native MEDIAN (min is skewed by degenerate "
          f"native categories, e.g. bird)")
    print(f"  {'concept':>8} {'members':>22} {'NS':>7} {'compact':>8} {'cen_out':>8}")
    for r in dis_desc:
        below = "ok" if r["ns_silhouette"] < nat_median else "ABOVE median!"
        print(f"  {r['label']:>8} {r['members']:>22} {r['ns_silhouette']:>7.3f} "
              f"{r['compactness']:>8.3f} {str(r['centroid_lands_outside']):>8}   "
              f"NS<median: {below}")
    print()


def run_sub_experiment(sub: str, *, with_controls: bool = False) -> None:
    if sub == "007a":
        rows = run_007a()
    elif sub == "007b":
        rows = run_007b()
    elif sub == "007c":
        raise NotImplementedError("007c needs the synthetic shapes set (plan §9).")
    else:
        raise ValueError(f"unknown sub '{sub}'")
    _rewrite_ledger_dropping_sub(rows, sub)


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_007 concept-carving runner")
    parser.add_argument("--sub", choices=["007a", "007b", "007c"], required=True)
    parser.add_argument("--controls", action="store_true",
                        help="also run falsification controls on the same episodes")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_sub_experiment(args.sub, with_controls=args.controls)


if __name__ == "__main__":
    main()
