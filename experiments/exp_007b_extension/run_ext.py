"""
run_ext.py — exp_007b_extension main runner.

Execution order (spec §11):
    1. Compute inter_member_distance for all CIFAR-10 category pairs using the
       native centroid matrix from the cifar_all10 embedding cache.
    2. Run the concept gate for all NEAR and MID candidates. Save concept_gate.csv.
    3. Extract far-disjunctive (gleth, mivor, plonk, quax) and native results
       from results/exp_007/ledger.jsonl — do NOT re-run.
    4. Run the episode grid for NEAR and MID tiers with all pool variants.
    5. Write ledger.jsonl and descriptors.csv.

Usage:
    cd experiments/exp_007b_extension
    python run_ext.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# ─── Path bootstrap ───────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]   # experiments/exp_007b_extension -> experiments -> project root
_PARENT_EXP = _PROJECT / "experiments" / "exp_007_concept_carving"
# _HERE MUST be first so `from _shared import` resolves to THIS experiment's _shared.py,
# not the parent's. PROJECT and PARENT_EXP are appended (not inserted at front) so
# they remain searchable but don't shadow the extension's own modules.
if str(_HERE) in sys.path:
    sys.path.remove(str(_HERE))
sys.path.insert(0, str(_HERE))
for p in [str(_PROJECT), str(_PARENT_EXP)]:
    if p not in sys.path:
        sys.path.append(p)

from _shared import (  # noqa: E402
    RESULTS_DIR, EXP007_RESULTS,
    FAR_THRESHOLD, FAR_MEDIAN, NATIVE_MEDIAN, GATE_NMI_MAX,
    CLEAN_FAR_LABELS,
    POOL_TYPES, PRIMARY_POOL,
    WAYS, SHOTS, N_EPISODES, N_QUERY_PER_CONCEPT, N_RETRIEVAL_TRIALS,
    MASTER_SEED, ROSTER, REFERENCE_LEARNER,
    NEAR_CANDIDATES, MID_CANDIDATES,
    NEAR_TOKENS, MID_TOKENS,
    build_native_centroid_dict,
)
from pool_variants import evaluate_episode_all_pools  # noqa: E402

# Import from parent experiment's _shared.py (added to sys.path above as _PARENT_EXP)
# We import via importlib to avoid ambiguity with this experiment's own _shared.py.
import importlib.util as _ilu
_p007_spec = _ilu.spec_from_file_location(
    "_shared_007", str(_PARENT_EXP / "_shared.py")
)
_p007 = _ilu.module_from_spec(_p007_spec)
_p007_spec.loader.exec_module(_p007)
load_or_build_embeddings = _p007.load_or_build_embeddings
hash_to_category = _p007.hash_to_category
from src.eval import descriptors as desc_mod, harness  # noqa: E402
from src.eval.concepts import (                         # noqa: E402
    Concept, ConceptType, ImageRecord, chance_level,
)
from src.eval.episodes import generate_episodes         # noqa: E402

from src.learners.centroid import CentroidLearner        # noqa: E402
from src.learners.kcentroid import KCentroidLearner      # noqa: E402
from src.learners.exemplar_knn import ExemplarKNN        # noqa: E402
from src.learners.logreg import LogRegLearner            # noqa: E402
from src.learners.linsvm import LinSVMLearner            # noqa: E402
from src.learners.random_baseline import RandomLearner   # noqa: E402

_KCENTROID_K_DEFAULT = 2
_KNN_K_DEFAULT = 3


def build_learner(name: str, n_concepts: int, *, seed: int):
    """Instantiate a learner by name (same logic as parent run_exp007.py)."""
    if name == "centroid":
        return CentroidLearner(n_concepts, seed=seed)
    if name == "kcentroid":
        return KCentroidLearner(n_concepts, k=_KCENTROID_K_DEFAULT, seed=seed)
    if name.startswith("kcentroid_k"):
        return KCentroidLearner(n_concepts, k=int(name.rsplit("k", 1)[1]), seed=seed)
    if name == "exemplar_knn":
        return ExemplarKNN(n_concepts, k=_KNN_K_DEFAULT, weighted=True, seed=seed)
    if name.startswith("exemplar_knn_k"):
        return ExemplarKNN(n_concepts, k=int(name.rsplit("k", 1)[1]), weighted=True, seed=seed)
    if name == "logreg":
        return LogRegLearner(n_concepts, C=1.0, seed=seed)
    if name == "linsvm":
        return LinSVMLearner(n_concepts, C=1.0, seed=seed)
    if name == "random":
        return RandomLearner(n_concepts, seed=seed)
    raise ValueError(f"unknown learner '{name}'")


# ─── Output paths ─────────────────────────────────────────────────────────────
GATE_PATH       = RESULTS_DIR / "concept_gate.csv"
DESCRIPTORS_PATH = RESULTS_DIR / "descriptors.csv"
LEDGER_PATH     = RESULTS_DIR / "ledger.jsonl"


# ─── Step 1 helpers — centroid matrix ─────────────────────────────────────────

def compute_centroid_dict(embeddings, records):
    return build_native_centroid_dict(embeddings, records)


# ─── Step 2 — concept gate ────────────────────────────────────────────────────

def _concept_ns_nmi(cat_a: str, cat_b: str, embeddings: dict, h2c: dict):
    """Compute NS silhouette and NMI_native for the union concept {cat_a, cat_b}."""
    all_hashes = list(embeddings.keys())
    members = [h for h in all_hashes if h2c[h] in (cat_a, cat_b)]
    others  = [h for h in all_hashes if h not in set(members)]
    ext = np.asarray([embeddings[h] for h in members], dtype=np.float64)
    oth = np.asarray([embeddings[h] for h in others], dtype=np.float64)
    ns = desc_mod.ns_silhouette(ext, oth)

    # NMI_native: concept_id = 0 for members, 1 for others
    all_concept_ids  = np.array([0 if h2c[h] in (cat_a, cat_b) else 1 for h in all_hashes])
    all_native_cats  = np.array([h2c[h] for h in all_hashes])
    nmi = desc_mod.nmi_native(all_concept_ids, all_native_cats)
    return ns, nmi


def run_concept_gate(
    candidates: list[tuple],   # (cat_a, cat_b, approx_dist)
    tier: str,                 # "near" or "mid"
    centroid_dict: dict,
    embeddings: dict,
    h2c: dict,
    *,
    dist_lo: float,
    dist_hi: float,
) -> tuple[list[dict], list[dict]]:
    """Evaluate gate for all candidates.

    Returns (gate_rows, passing_rows).
    gate_rows goes to concept_gate.csv; passing_rows are used for concept building.
    Conditions:
        1. dist_lo <= inter_member_distance < dist_hi
        2. ns_silhouette < NATIVE_MEDIAN
        3. nmi_native < GATE_NMI_MAX
    """
    from src.eval.descriptors import inter_member_distance

    gate_rows, passing = [], []
    for cat_a, cat_b, _ in candidates:
        imd = inter_member_distance((cat_a, cat_b), centroid_dict)
        ns, nmi = _concept_ns_nmi(cat_a, cat_b, embeddings, h2c)

        cond1 = dist_lo <= imd < dist_hi
        cond2 = ns < NATIVE_MEDIAN
        cond3 = nmi < GATE_NMI_MAX
        passes = cond1 and cond2 and cond3

        reasons = []
        if not cond1:
            reasons.append(f"imd={imd:.4f} outside [{dist_lo},{dist_hi})")
        if not cond2:
            reasons.append(f"ns={ns:.4f}>={NATIVE_MEDIAN}")
        if not cond3:
            reasons.append(f"nmi={nmi:.4f}>={GATE_NMI_MAX}")

        row = {
            "tier": tier,
            "cat_a": cat_a, "cat_b": cat_b,
            "inter_member_distance": round(imd, 4),
            "ns_silhouette": round(ns, 4),
            "nmi_native": round(nmi, 4),
            "passes_gate": passes,
            "exclusion_reason": "; ".join(reasons) if reasons else "",
        }
        gate_rows.append(row)
        if passes:
            passing.append(row)
        cond_str = "PASS" if passes else "FAIL"
        print(f"    [{cond_str}] {cat_a}+{cat_b:12s} imd={imd:.4f} ns={ns:.4f} nmi={nmi:.4f}"
              + (f"  — {'; '.join(reasons)}" if reasons else ""))

    return gate_rows, passing


def _select_disjoint_concepts(
    passing_rows: list[dict],
    tokens: list[str],
    embeddings: dict,
    h2c: dict,
    tier: str,
    target_n: int = 5,
) -> list[Concept]:
    """Greedily select a PAIRWISE-DISJOINT subset from passing gate rows.

    Ordered by inter_member_distance ascending (nearest first, to maximise
    number of concepts in the near tier where most pairs are very close).
    Returns as many disjoint concepts as possible up to target_n.
    """
    from src.eval.descriptors import inter_member_distance as imd_fn

    concept_type = ConceptType.DISJUNCTIVE
    sorted_rows = sorted(passing_rows, key=lambda r: r["inter_member_distance"])

    used_cats: set[str] = set()
    chosen: list[dict] = []
    for row in sorted_rows:
        a, b = row["cat_a"], row["cat_b"]
        if a in used_cats or b in used_cats:
            continue
        used_cats.update((a, b))
        chosen.append(row)
        if len(chosen) == target_n:
            break

    concepts: list[Concept] = []
    for cid, row in enumerate(chosen):
        a, b = row["cat_a"], row["cat_b"]
        hashes = tuple(h for h in embeddings if h2c[h] in (a, b))
        label = tokens[cid % len(tokens)]
        concepts.append(Concept(
            concept_id=cid,
            label=label,
            concept_type=concept_type,
            member_categories=(a, b),
            image_hashes=hashes,
            predicate_desc=(
                f"disjunctive ({tier}): {{{a}, {b}}} "
                f"imd={row['inter_member_distance']:.3f} "
                f"ns={row['ns_silhouette']:.3f}"
            ),
        ))
        print(f"    concept[{cid}] {label}: {{{a}, {b}}} imd={row['inter_member_distance']:.3f}")

    return concepts


# ─── Step 3 — load exp_007b results ──────────────────────────────────────────

def load_007b_ledger_rows(tiers: list[str]) -> list[dict]:
    """Load ledger rows from exp_007/ledger.jsonl for the given concept labels.

    tiers = "far" loads the 4 clean far-disjunctive concepts (CLEAN_FAR_LABELS);
    tiers = "native" loads the native anchor rows (sub=007b, concept_type=native).
    """
    path = EXP007_RESULTS / "ledger.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"exp_007 ledger not found at {path}")

    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("sub") != "007b":
            continue
        ctype = r.get("concept_type", "")
        if "far" in tiers and ctype == "disjunctive":
            # Keep only clean far-disjunctive (exclude zorb)
            pc = r.get("per_concept_c1", {})
            labels_in_row = set(pc.keys())
            if labels_in_row & CLEAN_FAR_LABELS:
                loaded.append(r)
            elif not labels_in_row:
                # Row may not have per_concept_c1 filtering available; keep all disjunctive
                loaded.append(r)
        if "native" in tiers and ctype == "native":
            loaded.append(r)
    return loaded


def load_007b_descriptors() -> list[dict]:
    path = EXP007_RESULTS / "descriptors.csv"
    if not path.exists():
        return []
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("sub") == "007b":
            rows.append(r)
    return rows


# ─── Step 4 — episode grid for near/mid ──────────────────────────────────────

def _build_all_embs_norm(embeddings: dict, h2c: dict):
    """Build a pre-normalised embedding matrix for all cifar_all10 images.

    Returns (all_embs_norm (N, dim), all_cats (N,)).
    The normalised matrix is used by the NN-distractor logic in pool_variants.py.
    """
    all_hashes = sorted(embeddings.keys())
    matrix = np.asarray([embeddings[h] for h in all_hashes], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms
    cats = np.array([h2c[h] for h in all_hashes])
    return matrix, cats


def _run_tier_grid(
    tier_label: str,
    concepts: list[Concept],
    embeddings: dict,
    h2c: dict,
    all_embs_norm: np.ndarray,
    all_cats: np.ndarray,
) -> list[dict]:
    """Condition loop for one tier. Returns ledger rows."""
    rows: list[dict] = []

    for ways in WAYS:
        if ways > len(concepts):
            print(f"    [skip] {tier_label} ways={ways} — only {len(concepts)} concepts")
            continue
        chance = chance_level(ConceptType.DISJUNCTIVE, ways)

        for shots in SHOTS:
            need = shots + N_QUERY_PER_CONCEPT
            # Check pool size
            for c in concepts:
                if len(c.image_hashes) < need:
                    print(f"    [warn] {c.label}: only {len(c.image_hashes)} images "
                          f"but need {need} — skip shots={shots}")
                    break
            else:
                pass  # all concepts have enough images

            try:
                episodes = list(generate_episodes(
                    concepts, n_episodes=N_EPISODES, n_ways=ways, n_shots=shots,
                    n_query_per_concept=N_QUERY_PER_CONCEPT,
                    master_seed=MASTER_SEED, hash_to_category=h2c,
                ))
            except ValueError as e:
                print(f"    [skip] {tier_label} ways={ways} shots={shots}: {e}")
                continue

            for ep in episodes:
                ep_arrays = harness.resolve_episode_arrays(ep, embeddings, h2c)

                # Fit ALL learners ONCE (episodes outer, learners inner)
                learner_dict = {}
                for name in ROSTER:
                    lrn = build_learner(name, ways, seed=ep.episode_id)
                    lrn.fit(ep_arrays.support_emb, ep_arrays.support_ids)
                    learner_dict[name] = lrn

                # C1 and margin (pool-type independent)
                preds = {}
                scores = {}
                for name, lrn in learner_dict.items():
                    p = np.asarray(lrn.predict(ep_arrays.query_emb))
                    s = lrn.score(ep_arrays.query_emb)
                    preds[name] = p
                    scores[name] = s

                from src.eval import metrics as met_mod
                c1_by_learner = {
                    name: met_mod.c1_naming(preds[name], ep_arrays.query_ids, ways)
                    for name in ROSTER
                }
                margin_by_learner = {
                    name: met_mod.margin(scores[name]) for name in ROSTER
                }
                per_concept_c1_by_learner = {}
                for name in ROSTER:
                    pcc = {}
                    for lid in range(ways):
                        mask = ep_arrays.query_ids == lid
                        if mask.any():
                            label = ep.concepts[lid].label
                            pcc[label] = float((preds[name][mask] == lid).mean())
                    per_concept_c1_by_learner[name] = pcc

                # C2 for all pool types
                c2_all = evaluate_episode_all_pools(
                    learner_dict, ep, ep_arrays,
                    all_embs_norm, all_cats,
                    POOL_TYPES, N_RETRIEVAL_TRIALS,
                )

                for pool_type in POOL_TYPES:
                    for name in ROSTER:
                        rows.append({
                            "sub":          "007b_ext",
                            "tier":         tier_label,
                            "concept_type": "disjunctive",
                            "learner":      name,
                            "ways":         ways,
                            "shots":        shots,
                            "episode_id":   ep.episode_id,
                            "pool_type":    pool_type,
                            "chance":       round(chance, 4),
                            "c1":           round(c1_by_learner[name], 4),
                            "c2":           round(c2_all[pool_type][name], 4),
                            "margin":       round(margin_by_learner[name], 4),
                            "per_concept_c1": {
                                k: round(v, 4)
                                for k, v in per_concept_c1_by_learner[name].items()
                            },
                        })

            n_rows = len(ROSTER) * len(POOL_TYPES) * N_EPISODES
            print(f"    {tier_label:6s} ways={ways} shots={shots:>2}: "
                  f"{n_rows} rows written")
    return rows


# ─── Descriptor helpers ───────────────────────────────────────────────────────

def _build_concept_descriptors(
    concepts: list[Concept],
    embeddings: dict,
    h2c: dict,
    centroid_dict: dict,
    tier: str,
) -> list[dict]:
    from src.eval.descriptors import (
        ns_silhouette, compactness, centroid_lands_outside,
        nmi_native, inter_member_distance as imd_fn,
    )
    all_hashes = list(embeddings.keys())
    rows = []
    for c in concepts:
        members = [h for h in all_hashes if h2c[h] in c.member_categories]
        others  = [h for h in all_hashes if h not in set(members)]
        ext = np.asarray([embeddings[h] for h in members], dtype=np.float64)
        oth = np.asarray([embeddings[h] for h in others], dtype=np.float64)

        all_ids = np.array([0 if h2c[h] in c.member_categories else 1 for h in all_hashes])
        all_native = np.array([h2c[h] for h in all_hashes])

        rows.append({
            "sub":                  "007b_ext",
            "tier":                 tier,
            "concept_id":           c.concept_id,
            "label":                c.label,
            "concept_type":         c.concept_type.value,
            "members":              "|".join(c.member_categories),
            "inter_member_distance": round(imd_fn(c.member_categories, centroid_dict), 4),
            "ns_silhouette":        round(ns_silhouette(ext, oth), 4),
            "compactness":          round(compactness(ext), 4),
            "centroid_lands_outside": centroid_lands_outside(ext, oth),
            "nmi_native":           round(nmi_native(all_ids, all_native), 4),
        })
    return rows


def _append_imd_to_007b_descriptors(rows_007b: list[dict], centroid_dict: dict) -> list[dict]:
    """Enrich loaded exp_007b descriptor rows with inter_member_distance."""
    from src.eval.descriptors import inter_member_distance as imd_fn
    out = []
    for r in rows_007b:
        r2 = dict(r)
        members_str = r2.get("members", "")
        if "|" in members_str:
            cats = tuple(members_str.split("|"))
            try:
                r2["inter_member_distance"] = round(imd_fn(cats, centroid_dict), 4)
            except KeyError:
                r2["inter_member_distance"] = ""
        else:
            r2["inter_member_distance"] = ""  # native: single category, undefined
        out.append(r2)
    return out


# ─── CSV / JSONL writers ──────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [skip] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path.name} ({len(rows)} rows)")


def _save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  saved {path.name} ({len(rows)} rows)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== exp_007b_extension — C2 Robustness & Disjunctive Gradient ===\n")

    # ── Step 1: Load embeddings and build centroid matrix ────────────────────
    print("Step 1 — loading cifar_all10 embeddings and building centroid matrix …")
    embeddings, records = load_or_build_embeddings("cifar_all10")
    h2c = hash_to_category(records)
    centroid_dict = compute_centroid_dict(embeddings, records)
    print(f"  loaded {len(embeddings)} embeddings, {len(centroid_dict)} centroids\n")

    # ── Step 2: Run concept gate ─────────────────────────────────────────────
    print("Step 2 — concept gate …")
    print("  NEAR candidates (d < FAR_THRESHOLD={:.4f}):".format(FAR_THRESHOLD))
    near_gate_rows, near_passing = run_concept_gate(
        NEAR_CANDIDATES, "near", centroid_dict, embeddings, h2c,
        dist_lo=0.0, dist_hi=FAR_THRESHOLD,
    )
    print(f"  -> {len(near_passing)} / {len(NEAR_CANDIDATES)} pass\n")

    print("  MID candidates ({:.4f} <= d < {:.4f}):".format(FAR_THRESHOLD, FAR_MEDIAN))
    mid_gate_rows, mid_passing = run_concept_gate(
        MID_CANDIDATES, "mid", centroid_dict, embeddings, h2c,
        dist_lo=FAR_THRESHOLD, dist_hi=FAR_MEDIAN,
    )
    print(f"  ->{len(mid_passing)} / {len(MID_CANDIDATES)} pass\n")

    all_gate_rows = near_gate_rows + mid_gate_rows
    _save_csv(all_gate_rows, GATE_PATH)

    # Build concept objects from passing gate rows
    print("  Building near-disjunctive concept pool …")
    near_concepts = _select_disjoint_concepts(
        near_passing, NEAR_TOKENS, embeddings, h2c, "near"
    )
    print(f"  ->{len(near_concepts)} near-disjunctive concepts\n")

    print("  Building mid-disjunctive concept pool …")
    mid_concepts = _select_disjoint_concepts(
        mid_passing, MID_TOKENS, embeddings, h2c, "mid"
    )
    print(f"  ->{len(mid_concepts)} mid-disjunctive concepts\n")

    # ── Step 3: Load far-disjunctive and native from exp_007b ───────────────
    print("Step 3 — loading far-disjunctive and native results from exp_007b …")
    far_native_rows = load_007b_ledger_rows(["far", "native"])
    print(f"  loaded {len(far_native_rows)} rows from exp_007 ledger")
    desc_007b = load_007b_descriptors()
    print(f"  loaded {len(desc_007b)} descriptor rows from exp_007\n")

    # ── Step 4: Run episode grid for near + mid ───────────────────────────────
    print("Step 4 — building pre-normalised embedding matrix for NN pool …")
    all_embs_norm, all_cats = _build_all_embs_norm(embeddings, h2c)
    print(f"  matrix shape: {all_embs_norm.shape}\n")

    ledger_rows: list[dict] = []

    if near_concepts:
        print("Step 4a — episode grid for NEAR tier …")
        ledger_rows += _run_tier_grid(
            "near", near_concepts, embeddings, h2c, all_embs_norm, all_cats
        )
    else:
        print("  [skip] no near-disjunctive concepts passed the gate")

    if mid_concepts:
        print("\nStep 4b — episode grid for MID tier …")
        ledger_rows += _run_tier_grid(
            "mid", mid_concepts, embeddings, h2c, all_embs_norm, all_cats
        )
    else:
        print("  [skip] no mid-disjunctive concepts passed the gate")

    # ── Step 5: Save outputs ────────────────────────────────────────────────
    print("\nStep 5 — saving outputs …")

    # Ledger: extension rows only (far/native loaded separately by analyze_ext.py)
    _save_jsonl(ledger_rows, LEDGER_PATH)

    # Descriptors: near + mid (new) with inter_member_distance; far + native enriched
    ext_desc_rows: list[dict] = []
    for tier, concepts in [("near", near_concepts), ("mid", mid_concepts)]:
        ext_desc_rows += _build_concept_descriptors(
            concepts, embeddings, h2c, centroid_dict, tier
        )
    enriched_007b = _append_imd_to_007b_descriptors(desc_007b, centroid_dict)
    all_desc = ext_desc_rows + enriched_007b
    _save_csv(all_desc, DESCRIPTORS_PATH)

    print("\n=== run_ext.py complete ===")
    print(f"  Ledger rows (near+mid): {len(ledger_rows)}")
    print(f"  Descriptors: {len(all_desc)}")
    print(f"  Results dir: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
