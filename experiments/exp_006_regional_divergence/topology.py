"""
topology.py — communication topology per regime.

Each regime defines, for each round, which (agent set, image set) pairs
form a consensus group. A consensus group is the unit that votes
together: every agent in the group classifies every image in the group,
and majority within the group determines ACCEPT / REJECT.

The three regimes:

  VERVET   — one consensus group per round containing ALL agents and
             ALL interaction images (both clusters' subsets merged).
             Equivalent to a single fully-connected 6-agent vote.
  RAVEN    — two intra-cluster groups per round (each cluster votes on
             its OWN interaction images) PLUS, on every 5th round, an
             extra all-agents group on the merged interaction pool to
             represent the periodic bridge.
  LATIN    — two intra-cluster groups per round, no bridge ever.

The interaction pools are cluster-specific because the input
distributions are cluster-specific by design — that is the central
manipulation of exp_006.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_006_regional_divergence._shared import (  # noqa: E402
    CLUSTERS, REGIME_CONFIG, ALL_AGENT_IDS,
    WorldPools,
)


@dataclass(frozen=True)
class ConsensusGroup:
    """One consensus event: who votes, on which images, labelled by scope."""
    scope:        str               # "cluster_1" | "cluster_2" | "bridge"
    agent_ids:    tuple[str, ...]
    image_hashes: tuple[str, ...]


def consensus_groups_for_round(
    regime:    str,
    round_num: int,
    pools:     WorldPools,
) -> list[ConsensusGroup]:
    """
    Return all consensus groups that run this round under this regime.

    For raven, the periodic bridge runs IN ADDITION to (not instead of) the
    intra-cluster groups — the intuition is that local cliques keep talking
    every day, and once every five days the two flocks meet at the same
    waterhole and exchange names. Centroid updates from the bridge round
    affect every agent that participated.
    """
    cfg = REGIME_CONFIG[regime]
    bridge_n = cfg["bridge_every_n_rounds"]

    if regime == "vervet":
        # One global consensus group — all agents on the merged interaction pool.
        merged = tuple(sorted(
            set(pools.cluster_pools["cluster_1"].interaction_hashes)
            | set(pools.cluster_pools["cluster_2"].interaction_hashes)
        ))
        return [ConsensusGroup(
            scope="bridge",
            agent_ids=tuple(ALL_AGENT_IDS),
            image_hashes=merged,
        )]

    # raven and latin both have intra-cluster groups every round.
    groups: list[ConsensusGroup] = []
    for cluster_id, agent_ids in CLUSTERS.items():
        cluster_imgs = tuple(sorted(pools.cluster_pools[cluster_id].interaction_hashes))
        groups.append(ConsensusGroup(
            scope=cluster_id,
            agent_ids=tuple(agent_ids),
            image_hashes=cluster_imgs,
        ))

    # Raven adds an inter-cluster bridge on rounds divisible by bridge_n.
    # Latin has bridge_n=0 and therefore never adds one.
    if bridge_n > 0 and round_num % bridge_n == 0:
        merged = tuple(sorted(
            set(pools.cluster_pools["cluster_1"].interaction_hashes)
            | set(pools.cluster_pools["cluster_2"].interaction_hashes)
        ))
        groups.append(ConsensusGroup(
            scope="bridge",
            agent_ids=tuple(ALL_AGENT_IDS),
            image_hashes=merged,
        ))

    return groups


def is_bridge_round(regime: str, round_num: int) -> bool:
    """Convenience for logging / plotting — does this round contain a bridge group?"""
    cfg = REGIME_CONFIG[regime]
    n = cfg["bridge_every_n_rounds"]
    if n == 0:
        return False
    return (round_num % n) == 0
