"""
test_pipeline_synthetic.py — full exp_007 pipeline on synthetic embeddings.

This is the cheap dress rehearsal the user asked for: run the entire stack
(concepts -> paired episodes -> all six learners -> harness -> metrics -> paired
stats) on synthetic, perfectly separable "native" clusters, BEFORE spending
DINOv2 time. It validates two things at once:

  1. Nothing in the stats / episode / harness wiring breaks end to end.
  2. The H2 expectation holds on native-style concepts: separable clusters are
     named near-perfectly by every non-chance learner, and there is NO
     prototype-failure dissociation (exemplar - centroid gap ~ 0, paired test
     NOT supported). If H2 failed here, the H1 test on disjunctive concepts
     would be uninterpretable.

Separable Gaussian blobs on a hypersphere stand in for native CIFAR categories;
the real 007a swaps these for cached DINOv2 embeddings via the same harness.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.concepts import Concept, ConceptType            # noqa: E402
from src.eval.episodes import generate_episodes                # noqa: E402
from src.eval import harness, stats                            # noqa: E402
from src.learners import LEARNER_REGISTRY                       # noqa: E402


def _instantiate(name, n_concepts, seed):
    module_path, cls_name = LEARNER_REGISTRY[name].split(":")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)(n_concepts, seed=seed)


def _make_world(n_categories=5, per_cat=80, dim=48, seed=0):
    """Separable unit-norm Gaussian clusters: embeddings + records + concepts."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_categories, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings, hash_to_category, records = {}, {}, []
    for c in range(n_categories):
        cat = f"cat{c}"
        for j in range(per_cat):
            v = centers[c] + 0.05 * rng.normal(size=dim)   # tight, well separated
            v /= np.linalg.norm(v)
            h = f"{cat}_{j:03d}"
            embeddings[h] = v
            hash_to_category[h] = cat
            records.append((h, cat))
    cat_to_label = {f"cat{c}": f"word{c}" for c in range(n_categories)}
    concepts = [
        Concept(
            concept_id=i, label=lbl, concept_type=ConceptType.NATIVE,
            member_categories=(cat,),
            image_hashes=tuple(h for h, cc in records if cc == cat),
        )
        for i, (cat, lbl) in enumerate(cat_to_label.items())
    ]
    return embeddings, hash_to_category, concepts


def test_full_pipeline_native_h2():
    embeddings, hash_to_category, concepts = _make_world()
    learner_names = list(LEARNER_REGISTRY)

    # Per-learner accumulators, indexed identically so they stay PAIRED.
    c1 = {name: [] for name in learner_names}

    episodes = list(generate_episodes(
        concepts, n_episodes=15, n_ways=3, n_shots=5,
        n_query_per_concept=20, master_seed=42,
        hash_to_category=hash_to_category,
    ))

    for ep in episodes:
        arrays = harness.resolve_episode_arrays(ep, embeddings, hash_to_category)
        for name in learner_names:                       # learners INNER => paired
            learner = _instantiate(name, ep.n_ways, seed=ep.episode_id)
            row = harness.evaluate_learner_on_episode(
                learner, ep, arrays,
                n_retrieval_trials=10,
                rng=np.random.default_rng(1000 + ep.episode_id),
            )
            c1[name].append(row["c1"])

    # Every non-chance learner names separable clusters near-perfectly.
    for name in learner_names:
        if name == "random":
            continue
        assert np.mean(c1[name]) > 0.95, f"{name} C1={np.mean(c1[name]):.3f}"

    # random sits near 1/3 chance.
    assert 0.15 < np.mean(c1["random"]) < 0.55

    # H2: no prototype-failure dissociation on native concepts.
    gap = stats.paired_test(np.array(c1["exemplar_knn"]), np.array(c1["centroid"]), seed=0)
    assert not gap.supported, f"unexpected dissociation on native: {gap}"
