"""
_shared.py — shared config, data loading, and embedding cache for exp_007.

exp_007 attacks the deep Round-1 criticism (the learner is a pointer to a
pre-existing DINOv2 cluster, not a concept acquirer) by asking whether any
lexical mechanism can acquire concepts whose extension does NOT coincide with a
native cluster. See the experiment plan and README.md for the full rationale.

The experiment is a 2-D grid (plan §2):
    Axis A — concept type:  NATIVE / DISJUNCTIVE / CROSS_CUTTING [/ SYNTHETIC_NOVEL]
    Axis B — learner:       centroid / kcentroid / exemplar_knn / logreg / linsvm / random

Sub-experiments (execution order, plan §11):
    007a  NATIVE        — replicate Round 1 with full stats + all learners (H2).
    007b  DISJUNCTIVE   — 5+ unions spanning an NS range; the central figure +
                          paired exemplar-vs-centroid test (H1, H3). No new data.
    007c  CROSS_CUTTING — synthetic shapes×colors; substrate diagnostic first,
                          then the H4 identity-cheating detector.

This module owns: paths, the experiment grid constants, CIFAR-10 / synthetic
loading, and the build-once embedding cache (.npy). All learning is CPU over
cached embeddings (plan §10); DINOv2 is touched only to fill the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR  = PROJECT_ROOT / "results" / "exp_007"
CACHE_DIR    = PROJECT_ROOT / ".cache"
EMB_CACHE_DIR = CACHE_DIR / "embeddings_exp007"   # <dataset>.npy + <dataset>_index.json


# ─── Experiment grid (plan §2, §7) ────────────────────────────────────────────

# 007a roster. 007b expands this (per-learner sweeps) in build_007b_roster().
LEARNERS = ["centroid", "kcentroid", "exemplar_knn", "logreg", "linsvm", "random"]

# Sweeps. Start small (3-way) and widen where balance allows (plan §7).
WAYS   = [3, 5]
SHOTS  = [1, 3, 5, 10, 20]
N_EPISODES          = 30        # >= 30 per (concept_type × learner × ways × shots)
N_QUERY_PER_CONCEPT = 50        # 50–100
MASTER_SEED         = 42

# Per-learner hyperparameter sweeps (plan §2; refined for 007b per the
# directives: exemplar k in {1,3,5}, kcentroid K in {1,2,3} — K=arity=2 is the
# theory-motivated anchor, K=1 collapses to centroid as a sanity check).
KNN_K       = [1, 3, 5]
KCENTROID_K = [1, 2, 3]

# Disjunctive set construction (plan §3): >= 5 concepts spanning an NS range.
# Arity 2 (unions of two categories) so a 5-pair matching partitions all 10
# CIFAR categories into pairwise-DISJOINT concepts — required so no image
# belongs to two concepts in an episode (which would make labels ambiguous).
N_DISJUNCTIVE_CONCEPTS = 5
DISJUNCTION_ARITY      = 2


# ─── CIFAR inventory ──────────────────────────────────────────────────────────
# All 10 categories mapped to the 10 Carroll words. The first 5 (frog ... airplane)
# match exp_007a / exp_006 so the native anchor is continuous; the other 5 are
# added for the disjunctive construction (directive 4: cache all 10 before
# building disjunctive concepts).
ALL_CIFAR_CATEGORIES = {
    "frog":       {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse":      {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":       {"cifar_idx": 8, "carroll_label": "vorpal"},
    "automobile": {"cifar_idx": 1, "carroll_label": "borogove"},
    "airplane":   {"cifar_idx": 0, "carroll_label": "tulgey"},
    "bird":       {"cifar_idx": 2, "carroll_label": "frumious"},
    "cat":        {"cifar_idx": 3, "carroll_label": "manxome"},
    "deer":       {"cifar_idx": 4, "carroll_label": "galumphing"},
    "dog":        {"cifar_idx": 5, "carroll_label": "uffish"},
    "truck":      {"cifar_idx": 9, "carroll_label": "jubjub"},
}
# 007a's NATIVE subset (kept for back-compat / the cifar_native cache).
NATIVE_CATEGORIES = {k: ALL_CIFAR_CATEGORIES[k]
                     for k in ("frog", "horse", "ship", "automobile", "airplane")}

N_IMAGES_PER_CATEGORY = 200   # pool per category; >> n_shots + n_query so
                              # episodes resample without exhausting it.

DATASET_CATEGORIES = {
    "cifar_native": NATIVE_CATEGORIES,    # 5 categories (007a)
    "cifar_all10":  ALL_CIFAR_CATEGORIES,  # 10 categories (007b)
}


def native_category_to_label(categories: dict | None = None) -> dict[str, str]:
    """category -> Carroll label, insertion order = concept_id order."""
    categories = categories if categories is not None else NATIVE_CATEGORIES
    return {cat: info["carroll_label"] for cat, info in categories.items()}


# ─── Embedding cache (build once, reuse across every condition) ───────────────

def load_or_build_embeddings(dataset: str = "cifar_native"):
    """Return (embeddings: dict[img_hash -> (dim,) np.ndarray], records).

    `records` is a list of src.eval.concepts.ImageRecord. If the cache exists it
    is loaded (no torch import); otherwise PerceptionLayer encodes the dataset
    once and the result is persisted as a single .npz (matrix + hashes +
    categories). `dataset` selects the category set (cifar_native = 5,
    cifar_all10 = 10). Mirrors exp_006.encode_all_images but caches so
    subsequent CPU-only runs are instant.
    """
    from src.eval.concepts import ImageRecord

    if dataset not in DATASET_CATEGORIES:
        raise ValueError(f"unknown dataset '{dataset}'. Known: {list(DATASET_CATEGORIES)}")
    categories = DATASET_CATEGORIES[dataset]

    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EMB_CACHE_DIR / f"{dataset}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        hashes = [str(h) for h in data["hashes"]]
        cats = [str(c) for c in data["categories"]]
        matrix = data["emb"]
        embeddings = {h: matrix[i] for i, h in enumerate(hashes)}
        records = [ImageRecord(img_hash=h, object_category=c) for h, c in zip(hashes, cats)]
        print(f"[cache] loaded {len(hashes)} embeddings from {cache_path.name}")
        return embeddings, records

    # ── Build path (the only step that needs torch / DINOv2) ──
    import torchvision
    from src.agents.perception import PerceptionLayer

    print(f"[cache] building {dataset} embeddings (DINOv2, CPU) ...")
    ds = torchvision.datasets.CIFAR10(
        root=str(CACHE_DIR / "cifar10"), train=True, download=False, transform=None,
    )
    rng = np.random.default_rng(MASTER_SEED)
    perception = PerceptionLayer(device="cpu")

    hashes, cats, vecs = [], [], []
    for cat, info in categories.items():
        idxs = [i for i, (_, lbl) in enumerate(ds) if lbl == info["cifar_idx"]]
        chosen = rng.choice(idxs, size=N_IMAGES_PER_CATEGORY, replace=False)
        for j, gi in enumerate(chosen):
            h = f"{cat}_{j:03d}"
            hashes.append(h)
            cats.append(cat)
            vecs.append(perception.encode(ds[int(gi)][0]))
        print(f"  [{cat:11s}] {N_IMAGES_PER_CATEGORY} encoded")

    matrix = np.asarray(vecs, dtype=np.float32)
    np.savez(cache_path, emb=matrix, hashes=np.array(hashes), categories=np.array(cats))
    print(f"[cache] saved {len(hashes)} embeddings -> {cache_path.name}")

    embeddings = {h: matrix[i] for i, h in enumerate(hashes)}
    records = [ImageRecord(img_hash=h, object_category=c) for h, c in zip(hashes, cats)]
    return embeddings, records


def hash_to_category(records) -> dict[str, str]:
    """img_hash -> native object category, from a record list."""
    return {r.img_hash: r.object_category for r in records}


def inter_category_distance_matrix(embeddings: dict, records):
    """Mean-centroid cosine distance between native categories (reuses the
    exp_000 measurement) — feeds concepts.build_disjunctive_concepts so unions
    span near->far (plan §3). Returns (matrix (C,C), category_order list)."""
    cats = sorted({r.object_category for r in records})
    h2c = hash_to_category(records)
    centroids = []
    for cat in cats:
        vecs = np.asarray([embeddings[h] for h in embeddings if h2c[h] == cat], dtype=np.float64)
        m = vecs.mean(axis=0)
        n = np.linalg.norm(m)
        centroids.append(m / n if n > 0 else m)
    C = np.asarray(centroids)
    dist = 1.0 - (C @ C.T)
    return dist, cats
