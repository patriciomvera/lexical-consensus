"""
run.py — exp_006_regional_divergence main runner.

Runs all three regimes (vervet / raven / latin) sequentially with the
same encoded data, the same cluster-specific seed assignments, and
identical centroid-accumulation rules. The independent variables across
regimes are network topology (which agents talk to which) and
transmission noise σ.

Architecture decision (recorded in CLAUDE.md after exp_005b):
the LexicalAdapter is NOT used here. Centroids live in raw DINOv2
space. Divergence — if it emerges — must be attributable to input
distribution + topology + noise, not to a per-agent learned
projection.

Output layout (under results/exp_006_regional_divergence/):
  config.yaml
  within_cluster_metrics.csv
  between_cluster_metrics.csv
  centroid_vectors.npz                  — every (regime, agent, label, round)
  cluster_centroid_vectors.npz          — cluster-collective centroids
  ledger_events.csv                     — assignment-level log
  divergence_summary.json
  (plots produced by metrics_and_plots.py)

Run:
    python -m experiments.exp_006_regional_divergence.run
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.perception import PerceptionLayer  # noqa: E402

from experiments.exp_006_regional_divergence._shared import (  # noqa: E402
    ALL_AGENT_IDS, CLUSTERS, REGIMES, REGIME_CONFIG,
    CARROLL_LABELS, CAT_TO_CARROLL,
    CLUSTER_IMAGE_SLICES, HELD_OUT_SLICE,
    AGENT_SEED_SLICES_PER_CLUSTER, INTERACTION_SLICE_PER_CLUSTER,
    MAX_ROUNDS, CONFIDENCE_THRESHOLD, RANDOM_SEED, SUCCESS_CRITERIA,
    load_all_images, encode_all_images, build_pools, agent_cluster,
    write_csv,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_006_regional_divergence"


# ─── Per-agent state ──────────────────────────────────────────────────────────

class AgentState:
    """
    One agent's mutable state for one regime.

    centroids holds the agent's mean embedding per label, computed from
    accepted_by_label (a set of image hashes per label). Both are
    rebuilt after each consensus round.

    The agent only stores raw DINOv2 embeddings (via the shared encoder
    cache that lives in run.py / metrics_and_plots.py). Per the
    architectural decision from exp_005b, there is no per-agent
    trainable projection in exp_006.
    """
    __slots__ = ("agent_id", "cluster_id", "accepted_by_label", "centroids")

    def __init__(self, agent_id: str):
        self.agent_id   = agent_id
        self.cluster_id = agent_cluster(agent_id)
        self.accepted_by_label: dict[str, set[str]] = {}
        self.centroids:         dict[str, np.ndarray] = {}


def seed_agent_state(state: AgentState, seed_hashes: dict[str, str]) -> None:
    """Inject this agent's tutor seeds (img_hash -> cat_name) into accepted_by_label."""
    for img_hash, cat_name in seed_hashes.items():
        label = CAT_TO_CARROLL[cat_name]
        state.accepted_by_label.setdefault(label, set()).add(img_hash)


def rebuild_centroids(
    state: AgentState,
    embeddings: dict[str, np.ndarray],
) -> None:
    """centroid_label = normalize(mean(x_i for i in accepted_by_label[label]))."""
    state.centroids = {}
    for label, img_hashes in state.accepted_by_label.items():
        if not img_hashes:
            continue
        X = np.stack([embeddings[h] for h in img_hashes], axis=0)
        m = X.mean(axis=0)
        n = float(np.linalg.norm(m))
        if n > 1e-9:
            state.centroids[label] = (m / n).astype(np.float32)


# ─── Stubs for the round loop (filled in commit 2) ────────────────────────────

def run_regime(
    regime: str,
    pools,                       # WorldPools
    embeddings: dict[str, np.ndarray],
    ground_truth: dict[str, str],
    rng: np.random.Generator,
) -> dict:
    """
    Placeholder for the full round loop. Commit 1 only constructs the
    AgentState objects and seeds them; the round loop, noise injection,
    consensus, divergence metrics, and Neo4j logging are added in
    subsequent commits.
    """
    print(f"\n[{regime}] init — sigma={REGIME_CONFIG[regime]['sigma']}, "
          f"bridge_every={REGIME_CONFIG[regime]['bridge_every_n_rounds']}")
    states = {aid: AgentState(aid) for aid in ALL_AGENT_IDS}
    for cluster_id, cluster_pool in pools.cluster_pools.items():
        for agent_id, seed_hashes in cluster_pool.agent_seed_hashes.items():
            seed_agent_state(states[agent_id], seed_hashes)
            rebuild_centroids(states[agent_id], embeddings)
            print(f"  {agent_id} ({cluster_id}): "
                  f"labels={sorted(states[agent_id].accepted_by_label)} "
                  f"({sum(len(v) for v in states[agent_id].accepted_by_label.values())} seeds)")
    return {
        "regime":             regime,
        "states":              states,
        "round_rows":          [],
        "within_metric_rows":  [],
        "between_metric_rows": [],
        "ledger_rows":         [],
        "centroid_vectors":    {},
        "cluster_centroids":   {},
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":     "exp_006_regional_divergence",
        "date":           datetime.now().strftime("%Y-%m-%d"),
        "encoder":        "facebook/dinov2-small (frozen)",
        "adapter":        "none (per exp_005b decision)",
        "dataset":        "CIFAR-10",
        "label_assignment": CAT_TO_CARROLL,
        "n_agents":       len(ALL_AGENT_IDS),
        "clusters":       {k: list(v) for k, v in CLUSTERS.items()},
        "cluster_image_slices":            {k: list(v) for k, v in CLUSTER_IMAGE_SLICES.items()},
        "held_out_slice":                  list(HELD_OUT_SLICE),
        "agent_seed_slices_per_cluster":   [list(s) for s in AGENT_SEED_SLICES_PER_CLUSTER],
        "interaction_slice_per_cluster":   list(INTERACTION_SLICE_PER_CLUSTER),
        "regimes":        REGIME_CONFIG,
        "max_rounds":     MAX_ROUNDS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "random_seed":    RANDOM_SEED,
        "success_criteria": SUCCESS_CRITERIA,
    }
    with open(RESULTS_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("[Config] Written.")

    perception = PerceptionLayer(device="cpu")
    print("\n[Data] Loading CIFAR-10 ...")
    all_images = load_all_images()
    embeddings, ground_truth = encode_all_images(all_images, perception)

    pools = build_pools()
    rng   = np.random.default_rng(RANDOM_SEED)

    # Commit 1 only initializes each regime — round loop arrives in commit 2.
    bundles = {}
    for regime in REGIMES:
        bundles[regime] = run_regime(regime, pools, embeddings, ground_truth, rng)

    print("\n[Commit 1] Setup complete. Round loop arrives in commit 2.")


if __name__ == "__main__":
    run()
