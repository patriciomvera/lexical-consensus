"""
_shared.py — shared constants, data loading, and config for exp_006.

exp_006 tests whether systematically different input distributions across
isolated agent sub-populations produce regional lexical divergence — the
computational analogue of Latin vulgar splitting into the Romance
languages.

Given exp_005 and exp_005b both found that DINOv2 geometric consistency
dominates inter-agent alignment when agents share both encoder AND data,
exp_006 manipulates the data side: each cluster of agents sees a
systematically different subset of the visual world, and the question is
whether that — combined with network isolation and transmission noise —
is enough to overcome the shared perceptual anchor.

Three regimes (the independent variable):

  VERVET  — fully connected, σ=0.0.   Expected: no divergence.
  RAVEN   — clustered, weak bridge every 5th round, σ=0.05.
             Expected: partial divergence pulled back periodically.
  LATIN   — isolated clusters (no inter-cluster after seeding), σ=0.10.
             Expected: full monotonic divergence.

Six agents in two clusters (0,1,2 / 3,4,5). Each cluster has its own
disjoint 25-image slice per category; a separate 30-image shared
held-out pool is used to test cross-cluster generalization. The
LexicalAdapter from exp_005b is NOT used — exp_005b showed it does
not act as an attractor under the regularization regime explored
there, so exp_006 returns to the simpler frozen-DINOv2 + centroid
architecture from exp_003.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torchvision  # noqa: E402  (used by load_all_images)


# ─── Carroll vocabulary (unchanged across the project) ────────────────────────

CATEGORIES = {
    "frog":  {"cifar_idx": 6, "carroll_label": "slithy"},
    "horse": {"cifar_idx": 7, "carroll_label": "mimsy"},
    "ship":  {"cifar_idx": 8, "carroll_label": "vorpal"},
}
CARROLL_LABELS = ["slithy", "mimsy", "vorpal"]
CAT_TO_CARROLL = {k: v["carroll_label"] for k, v in CATEGORIES.items()}
CARROLL_TO_CAT = {v["carroll_label"]: k for k, v in CATEGORIES.items()}


# ─── Image distribution ───────────────────────────────────────────────────────

# 80 images per category — partitioned into two 25-image cluster slices and a
# 30-image shared held-out pool. The held-out images are unseen by either
# cluster during training and are the only ground on which we can compare
# how the two clusters generalize.
N_TOTAL_PER_CATEGORY = 80

CLUSTER_IMAGE_SLICES = {
    "cluster_1": (0, 25),
    "cluster_2": (25, 50),
}
HELD_OUT_SLICE = (50, 80)

# Within a cluster: 5 seeds × 3 agents = 15, leaves 10 for interaction.
N_AGENTS_PER_CLUSTER = 3
N_SEEDS_PER_AGENT    = 5

# Per-agent seed slices INSIDE a cluster's 25-image subset.
AGENT_SEED_SLICES_PER_CLUSTER = [(0, 5), (5, 10), (10, 15)]
INTERACTION_SLICE_PER_CLUSTER = (15, 25)


# ─── Cluster + regime config ─────────────────────────────────────────────────

# 6 agents, two clusters of three. agent_ids deliberately keep their global
# 00–05 numbering so logs and Neo4j queries stay coherent across regimes.
CLUSTERS = {
    "cluster_1": ["agent_00", "agent_01", "agent_02"],
    "cluster_2": ["agent_03", "agent_04", "agent_05"],
}
ALL_AGENT_IDS = CLUSTERS["cluster_1"] + CLUSTERS["cluster_2"]

REGIMES = ["vervet", "raven", "latin"]

REGIME_CONFIG = {
    # σ is per-element Gaussian noise added to the centroid at classification
    # time, after which the noisy centroid is renormalized to unit length.
    # bridge_every_n_rounds = N means inter-cluster consensus happens on
    # rounds divisible by N (so 5 → rounds 5, 10, 15, 20, 25). 0 means
    # never. A regime with no bridge AND only one cluster would be a
    # vervet-style fully-connected setup; we therefore use bridge=1
    # for vervet (every round connects the two clusters).
    "vervet": {"sigma": 0.0,  "bridge_every_n_rounds": 1, "description": "fully connected, zero noise"},
    "raven":  {"sigma": 0.05, "bridge_every_n_rounds": 5, "description": "clustered, weak bridge every 5 rounds"},
    "latin":  {"sigma": 0.10, "bridge_every_n_rounds": 0, "description": "isolated clusters, no bridge after seeding"},
}

MAX_ROUNDS           = 25
CONFIDENCE_THRESHOLD = 0.30
RANDOM_SEED          = 42

SUCCESS_CRITERIA = {
    "within_cluster_agreement_min":     0.90,   # each cluster must still agree internally
    "shared_held_out_accuracy_min":     0.80,   # cross-cluster test set baseline
    "divergence_signal_threshold":      0.005,  # latin > vervet by at least this much
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_all_images() -> dict[str, list]:
    """
    Load N_TOTAL_PER_CATEGORY=80 images per category from CIFAR-10.

    Same RNG strategy as exp_003 (seed 42) but a larger draw to leave room
    for the shared held-out pool. Returns cat_name -> list of 80 PIL Images.
    """
    cache_dir = PROJECT_ROOT / ".cache" / "cifar10"
    dataset = torchvision.datasets.CIFAR10(
        root=str(cache_dir), train=True, download=False, transform=None,
    )
    rng = np.random.default_rng(RANDOM_SEED)
    all_images: dict[str, list] = {}
    for cat_name, info in CATEGORIES.items():
        cifar_idx = info["cifar_idx"]
        indices   = [i for i, (_, lbl) in enumerate(dataset) if lbl == cifar_idx]
        chosen    = rng.choice(indices, size=N_TOTAL_PER_CATEGORY, replace=False)
        all_images[cat_name] = [dataset[int(i)][0] for i in chosen]
        print(f"  [{cat_name:5s}] {N_TOTAL_PER_CATEGORY} images loaded")
    return all_images


def encode_all_images(all_images: dict[str, list], perception) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Encode every image with DINOv2 once and stash by hash. Encoded once,
    consumed by every regime. ground_truth maps img_hash -> Carroll label
    (the unambiguous-category label; cross-cluster tests use this).
    """
    embeddings:   dict[str, np.ndarray] = {}
    ground_truth: dict[str, str]        = {}
    n_total = N_TOTAL_PER_CATEGORY * len(CATEGORIES)
    done    = 0
    print(f"\n[Encoding] {n_total} images with DINOv2-small ...")
    for cat_name, imgs in all_images.items():
        label = CAT_TO_CARROLL[cat_name]
        for idx, img in enumerate(imgs):
            h = f"{cat_name}_{idx:03d}"
            embeddings[h]   = perception.encode(img)
            ground_truth[h] = label
            done += 1
            if done % 60 == 0 or done == n_total:
                print(f"  {done}/{n_total}")
    return embeddings, ground_truth


# ─── Pool / seed assignment ───────────────────────────────────────────────────

@dataclass
class ClusterPool:
    """Image hashes for one cluster's seed and interaction pools."""
    cluster_id:        str
    agent_seed_hashes: dict[str, dict[str, str]]   # agent_id -> {img_hash: cat_name}
    interaction_hashes: dict[str, str]             # img_hash -> cat_name


@dataclass
class WorldPools:
    """All pools needed for one experiment run, independent of regime."""
    cluster_pools:   dict[str, ClusterPool]   # cluster_id -> ClusterPool
    held_out_hashes: dict[str, str]           # img_hash -> cat_name (shared)


def build_pools() -> WorldPools:
    """
    Construct per-cluster seed + interaction pools and a single shared
    held-out pool, using the cluster image slices declared above.

    img_hash format mirrors exp_003: "{cat_name}_{idx:03d}" where idx is
    the global per-category index (0..N_TOTAL_PER_CATEGORY-1). This means
    cluster 1's frog seeds are "frog_000".."frog_004" and cluster 2's
    are "frog_025".."frog_029".
    """
    cluster_pools: dict[str, ClusterPool] = {}
    for cluster_id, agent_ids in CLUSTERS.items():
        c_start, c_end = CLUSTER_IMAGE_SLICES[cluster_id]
        agent_seed_hashes: dict[str, dict[str, str]] = {aid: {} for aid in agent_ids}
        interaction_hashes: dict[str, str] = {}

        for cat_name in CATEGORIES:
            # Seed slices are positions WITHIN the cluster's 25-image subset.
            for agent_idx, (s_start, s_end) in enumerate(AGENT_SEED_SLICES_PER_CLUSTER):
                agent_id = agent_ids[agent_idx]
                for offset in range(s_start, s_end):
                    h = f"{cat_name}_{c_start + offset:03d}"
                    agent_seed_hashes[agent_id][h] = cat_name
            # Interaction slice within cluster's subset
            i_start, i_end = INTERACTION_SLICE_PER_CLUSTER
            for offset in range(i_start, i_end):
                h = f"{cat_name}_{c_start + offset:03d}"
                interaction_hashes[h] = cat_name

        cluster_pools[cluster_id] = ClusterPool(
            cluster_id=cluster_id,
            agent_seed_hashes=agent_seed_hashes,
            interaction_hashes=interaction_hashes,
        )

    held_out_hashes: dict[str, str] = {}
    h_start, h_end = HELD_OUT_SLICE
    for cat_name in CATEGORIES:
        for idx in range(h_start, h_end):
            held_out_hashes[f"{cat_name}_{idx:03d}"] = cat_name

    return WorldPools(cluster_pools=cluster_pools, held_out_hashes=held_out_hashes)


def agent_cluster(agent_id: str) -> str:
    """Map agent_id -> cluster_id. Raises KeyError on an unknown agent."""
    for cluster_id, ids in CLUSTERS.items():
        if agent_id in ids:
            return cluster_id
    raise KeyError(agent_id)


# ─── CSV helper (shared with the rest of the module) ─────────────────────────

def write_csv(rows: list[dict], path: Path, fieldnames: Optional[list[str]] = None) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    fns = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")
