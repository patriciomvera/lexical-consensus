"""
episodes.py
-----------
Seeded N-way K-shot episode generator (plan §7).

An EPISODE is one evaluation trial: from the pool of images for a set of
concepts, draw K support (seed) examples per concept and a disjoint query set,
under a logged random seed. The statistical backbone of exp_007 is >= 30
episodes per (concept_type × learner × ways × shots), each a fresh seeded
split, so every metric gets a mean ± 95% bootstrap CI.

PAIRING — THE INVARIANT THE STATS DEPEND ON
-------------------------------------------
A single `Episode` object is produced ONCE and handed to EVERY learner. Learners
are the INNER loop; episodes the OUTER loop. There is no per-learner sampling.
This is what makes stats.paired_bootstrap / stats.wilcoxon_signed valid: the
exemplar arm and the centroid arm see the byte-identical support and query
splits, so the per-episode difference d_i = Acc(exemplar)_i − Acc(centroid)_i is
a genuine matched pair. Sampling independently per learner would silently
invalidate the entire H1 test. (Enforced by test_episode_pairing.)

Determinism is non-negotiable (CLAUDE.md principle 3): a given (master_seed,
episode_index) reproduces exactly the same concept choice and support/query
indices, regardless of how many learners later consume the episode.

Concept ids inside an episode are LOCAL (0..n_ways-1), assigned in the order
the episode chose its concepts. Learners only ever see local ids, so they cannot
exploit global label semantics and score vectors are always dense width n_ways.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .concepts import Concept


@dataclass(frozen=True)
class Episode:
    """One N-way K-shot trial.

    All fields needed to reproduce, log, and score the trial are here (plan §12
    ledger row). `concepts[local_id]` is the global Concept behind local id
    `local_id`; metrics use its `member_categories` for the C2 valid set.
    """
    episode_id: int
    seed: int
    n_ways: int
    n_shots: int
    concepts: tuple[Concept, ...]                 # indexed by local id 0..n_ways-1
    support: dict[int, tuple[str, ...]]           # local_id -> support image hashes
    query: dict[int, tuple[str, ...]]             # local_id -> query image hashes (disjoint)


def _episode_rng(master_seed: int, episode_id: int) -> np.random.Generator:
    """Deterministic per-episode RNG. Independent of learner consumption and of
    how many episodes were drawn before it."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, episode_id]))


def _balanced_sample(
    pool: list[str],
    n: int,
    rng: np.random.Generator,
    *,
    hash_to_category: Optional[dict[str, str]] = None,
    member_categories: tuple[str, ...] = (),
) -> list[str]:
    """Sample `n` hashes from `pool` without replacement.

    For a disjunctive concept (len(member_categories) > 1 and a category map is
    supplied) the draw is balanced across member categories so macro-averaged
    accuracy is not dominated by the larger member (plan §3 balance rule). For a
    native concept (single member) this is a plain uniform draw.
    """
    if hash_to_category is None or len(member_categories) <= 1:
        return list(rng.choice(pool, size=n, replace=False))

    by_cat: dict[str, list[str]] = {c: [] for c in member_categories}
    for h in pool:
        cat = hash_to_category.get(h)
        if cat in by_cat:
            by_cat[cat].append(h)

    # Largest-remainder allocation of n across the member categories.
    base = n // len(member_categories)
    rem = n - base * len(member_categories)
    chosen: list[str] = []
    cats = list(member_categories)
    rng.shuffle(cats)
    for i, cat in enumerate(cats):
        take = base + (1 if i < rem else 0)
        take = min(take, len(by_cat[cat]))
        chosen.extend(rng.choice(by_cat[cat], size=take, replace=False))
    # If a category was short, top up uniformly from whatever remains.
    if len(chosen) < n:
        remaining = [h for h in pool if h not in set(chosen)]
        chosen.extend(rng.choice(remaining, size=n - len(chosen), replace=False))
    return chosen


def generate_episodes(
    concepts: list[Concept],
    *,
    n_episodes: int,
    n_ways: int,
    n_shots: int,
    n_query_per_concept: int,
    master_seed: int,
    hash_to_category: Optional[dict[str, str]] = None,
) -> Iterator[Episode]:
    """Yield `n_episodes` seeded N-way K-shot episodes.

    Contract:
      * Support and query are DISJOINT within an episode.
      * Per-episode RNG derives deterministically from (master_seed, episode_id)
        so runs are reproducible and resumable, and identical no matter how many
        learners later consume the episode (the pairing guarantee).
      * When len(concepts) > n_ways, a fresh n_ways subset is drawn per episode.
      * For disjunctive concepts, support/query are balanced across member
        categories (plan §3) when `hash_to_category` is supplied.

    Plan sweeps: shots K in {1,3,5,10,20}, ways N in {3,5} (10 where balance
    allows), query 50-100 per concept.
    """
    if len(concepts) < n_ways:
        raise ValueError(
            f"generate_episodes: need >= n_ways={n_ways} concepts, got {len(concepts)}."
        )
    need = n_shots + n_query_per_concept

    for episode_id in range(n_episodes):
        rng = _episode_rng(master_seed, episode_id)

        chosen_idx = rng.choice(len(concepts), size=n_ways, replace=False)
        chosen = tuple(concepts[i] for i in chosen_idx)

        support: dict[int, tuple[str, ...]] = {}
        query: dict[int, tuple[str, ...]] = {}
        for local_id, concept in enumerate(chosen):
            pool = list(concept.image_hashes)
            if len(pool) < need:
                raise ValueError(
                    f"Concept '{concept.label}' has {len(pool)} images but the "
                    f"episode needs {need} (={n_shots} shot + "
                    f"{n_query_per_concept} query)."
                )
            picked = _balanced_sample(
                pool, need, rng,
                hash_to_category=hash_to_category,
                member_categories=concept.member_categories,
            )
            support[local_id] = tuple(picked[:n_shots])
            query[local_id] = tuple(picked[n_shots:need])

        yield Episode(
            episode_id=episode_id,
            seed=master_seed,
            n_ways=n_ways,
            n_shots=n_shots,
            concepts=chosen,
            support=support,
            query=query,
        )
