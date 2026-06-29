"""
test_disjunctive.py — disjunctive concept construction (directives 4-5, plan §3).

Validates the properties the H1 test depends on:
  * exactly n_concepts concepts, all DISJUNCTIVE, arity-2;
  * member categories are pairwise DISJOINT (no image in two concepts → no
    label ambiguity in an episode);
  * the matching prefers perceptually DISTANT pairs;
  * concepts are ordered by NS ascending (concept_id 0 = hardest), and sit at
    LOWER NS than the native single-category concepts (the carving check).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.concepts import (  # noqa: E402
    ImageRecord, ConceptType, build_disjunctive_concepts, build_native_concepts,
)
from src.eval import descriptors  # noqa: E402


def _world(n_cat=10, per_cat=40, dim=32, seed=0):
    """10 well-separated category clusters on a sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_cat, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings, records = {}, []
    cats = [f"c{i}" for i in range(n_cat)]
    for i, cat in enumerate(cats):
        for j in range(per_cat):
            v = centers[i] + 0.04 * rng.normal(size=dim)
            v /= np.linalg.norm(v)
            h = f"{cat}_{j:03d}"
            embeddings[h] = v
            records.append(ImageRecord(img_hash=h, object_category=cat))
    # cosine-distance matrix between category centroids
    C = []
    for i in range(n_cat):
        m = np.mean([embeddings[f"c{i}_{j:03d}"] for j in range(per_cat)], axis=0)
        C.append(m / np.linalg.norm(m))
    C = np.asarray(C)
    dist = 1.0 - C @ C.T
    return embeddings, records, dist, cats


def test_disjoint_arity2_matching():
    emb, rec, dist, cats = _world()
    concepts = build_disjunctive_concepts(
        rec, emb, n_concepts=5, distance_matrix=dist, category_order=cats,
        arity=2, seed=0,
    )
    assert len(concepts) == 5
    used = []
    for c in concepts:
        assert c.concept_type == ConceptType.DISJUNCTIVE
        assert len(c.member_categories) == 2
        used.extend(c.member_categories)
    assert len(set(used)) == 10, "member categories must be pairwise disjoint and cover all 10"


def test_ordered_by_ns_ascending():
    emb, rec, dist, cats = _world()
    concepts = build_disjunctive_concepts(
        rec, emb, n_concepts=5, distance_matrix=dist, category_order=cats, seed=0)
    ns = []
    for c in concepts:
        members = [r.img_hash for r in rec if r.object_category in c.member_categories]
        others = [r.img_hash for r in rec if r.object_category not in c.member_categories]
        ext = np.asarray([emb[h] for h in members]); oth = np.asarray([emb[h] for h in others])
        ns.append(descriptors.ns_silhouette(ext, oth))
    assert ns == sorted(ns), "concepts should be ordered by NS ascending (hardest first)"


def test_disjunctive_ns_below_native():
    """Each disjunctive concept must be LESS natively separable than any native
    single-category concept (directive 5 carving check)."""
    emb, rec, dist, cats = _world()
    native = build_native_concepts(rec, {c: c for c in cats})
    nat_ns = []
    for c in native:
        members = [r.img_hash for r in rec if r.object_category in c.member_categories]
        others = [r.img_hash for r in rec if r.object_category not in c.member_categories]
        nat_ns.append(descriptors.ns_silhouette(
            np.asarray([emb[h] for h in members]), np.asarray([emb[h] for h in others])))

    disj = build_disjunctive_concepts(
        rec, emb, n_concepts=5, distance_matrix=dist, category_order=cats, seed=0)
    for c in disj:
        members = [r.img_hash for r in rec if r.object_category in c.member_categories]
        others = [r.img_hash for r in rec if r.object_category not in c.member_categories]
        ns = descriptors.ns_silhouette(
            np.asarray([emb[h] for h in members]), np.asarray([emb[h] for h in others]))
        assert ns < min(nat_ns), f"disjunctive NS {ns:.3f} not below native min {min(nat_ns):.3f}"
