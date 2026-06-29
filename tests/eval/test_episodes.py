"""
test_episodes.py — the pairing + determinism invariants (plan §7).

The single most important property: the SAME (master_seed, episode_id) yields a
byte-identical Episode no matter how many times it is generated or how many
learners consume it. Without that, the paired bootstrap and Wilcoxon in stats.py
compare mismatched splits and are invalid. test_pairing_identical_across_passes
is the guard the reviewer's concern maps onto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.concepts import Concept, ConceptType  # noqa: E402
from src.eval.episodes import generate_episodes      # noqa: E402


def _toy_concepts(n=4, pool=60):
    """n native-style concepts, each with `pool` image hashes in one category."""
    return [
        Concept(
            concept_id=i,
            label=f"w{i}",
            concept_type=ConceptType.NATIVE,
            member_categories=(f"cat{i}",),
            image_hashes=tuple(f"cat{i}_{j:03d}" for j in range(pool)),
        )
        for i in range(n)
    ]


def test_support_query_disjoint():
    eps = list(generate_episodes(
        _toy_concepts(), n_episodes=5, n_ways=3, n_shots=5,
        n_query_per_concept=10, master_seed=42,
    ))
    for ep in eps:
        for c in range(ep.n_ways):
            s, q = set(ep.support[c]), set(ep.query[c])
            assert not (s & q), "support and query overlap"
            assert len(s) == 5 and len(q) == 10


def test_pairing_identical_across_passes():
    """Generating the episodes twice (as two different learners would each
    trigger) yields identical splits — the pairing guarantee."""
    def draw():
        return list(generate_episodes(
            _toy_concepts(), n_episodes=8, n_ways=3, n_shots=5,
            n_query_per_concept=10, master_seed=123,
        ))

    a, b = draw(), draw()
    for ea, eb in zip(a, b):
        assert ea.support == eb.support
        assert ea.query == eb.query
        assert tuple(c.label for c in ea.concepts) == tuple(c.label for c in eb.concepts)


def test_different_episode_ids_differ():
    eps = list(generate_episodes(
        _toy_concepts(), n_episodes=5, n_ways=3, n_shots=5,
        n_query_per_concept=10, master_seed=42,
    ))
    # Not all episodes should be identical splits (sanity that the seed varies).
    supports = [tuple(sorted(ep.support.items())) for ep in eps]
    assert len(set(supports)) > 1


def test_raises_when_pool_too_small():
    with pytest.raises(ValueError):
        list(generate_episodes(
            _toy_concepts(pool=8), n_episodes=1, n_ways=3, n_shots=5,
            n_query_per_concept=10, master_seed=0,   # needs 15 > 8
        ))


def test_raises_when_not_enough_concepts():
    with pytest.raises(ValueError):
        list(generate_episodes(
            _toy_concepts(n=2), n_episodes=1, n_ways=3, n_shots=2,
            n_query_per_concept=5, master_seed=0,
        ))
