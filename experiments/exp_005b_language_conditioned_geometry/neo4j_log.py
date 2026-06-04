"""
neo4j_log.py — exp_005b: log Adapter nodes, Centroid nodes, and
PROJECTED_PROXIMITY_TO edges to Neo4j AuraDB.

This is a separate runnable module — it depends on the artifacts written by
run.py and metrics_and_plots.py, and on a working Neo4j connection (via
.env credentials). It does NOT need to be re-run when only metrics or
plots change.

Schema additions per spec:

  (:Adapter {
      adapter_id,                       — "{condition}_{agent_id}_{round}"
      experiment_id, condition, agent_id, round,
      frobenius_distance_to_identity,
      frobenius_drift_from_initial,
      vector_path                       — pointer to adapter_weights_*.npz
  })

  (:Agent)-[:HAS_ADAPTER {round, condition}]->(:Adapter)

  (:Centroid)-[:PROJECTED_PROXIMITY_TO {
      round, label, condition, cosine_distance, space, experiment_id
  }]->(:Centroid)

For namespacing, exp_005b uses one experiment_id per condition:
    exp_005b_frozen / exp_005b_consensus / exp_005b_random

That keeps exp_005b nodes separable from the existing exp_003/exp_005 graph
and lets a single Cypher query target one or all conditions cleanly.

Usage:
    python -m experiments.exp_005b_language_conditioned_geometry.neo4j_log
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.neo4j_logger import Neo4jLogger  # noqa: E402
from experiments.exp_005b_language_conditioned_geometry._shared import (  # noqa: E402
    AGENT_IDS, CARROLL_LABELS, CONDITIONS, MAX_ROUNDS,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_005b_language_conditioned_geometry"
AGENT_PAIRS = list(combinations(AGENT_IDS, 2))


# ─── Loaders ──────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_drift() -> dict[tuple, dict]:
    """(condition, agent_id, round) -> drift row."""
    rows = read_csv(RESULTS_DIR / "adapter_drift.csv")
    return {
        (r["condition"], r["agent_id"], int(r["round"])): r
        for r in rows
    }


def load_inter_agent(path: Path) -> dict[tuple, dict]:
    """
    Returns (condition, label, round) -> row with pair distances
    {d_00_01, d_00_02, d_01_02, mean_pairwise_distance}.
    """
    rows = read_csv(path)
    return {
        (r["condition"], r["label"], int(r["round"])): r
        for r in rows
    }


# ─── Experiment seeding (Experiment, Agent, Label nodes) ─────────────────────

def seed_experiment_metadata(logger: Neo4jLogger, condition: str) -> None:
    """
    Create one :Experiment, three :Agent nodes, and three :Label nodes for
    this condition. Reuses neo4j_logger helpers so the existing Cypher
    constraints (MERGE on (label, experiment_id) etc.) are honored.
    """
    experiment_id = f"exp_005b_{condition}"
    logger.log_experiment(
        experiment_id=experiment_id,
        date="2026-06-04",
        encoder="facebook/dinov2-small + LexicalAdapter",
        dataset="CIFAR-10 (frog/horse/ship)",
        n_agents=len(AGENT_IDS),
        n_rounds=MAX_ROUNDS,
        n_labels=len(CARROLL_LABELS),
    )
    for idx, aid in enumerate(AGENT_IDS):
        logger.log_agent(
            agent_id=aid,
            experiment_id=experiment_id,
            n_seeds=5 * len(CARROLL_LABELS),
            seed_indices=[5 * idx + i for i in range(5)],
        )
    for label in CARROLL_LABELS:
        logger.log_label(
            name=label,
            carroll_source="Jabberwocky",
            experiment_id=experiment_id,
        )


# ─── Centroid + Adapter nodes ─────────────────────────────────────────────────

def log_centroids_for_condition(
    logger: Neo4jLogger,
    condition: str,
    proj_vecs: dict[str, np.ndarray],
) -> None:
    """
    One :Centroid node per (condition, agent, label, round). Stores the
    centroid metadata; the actual vector lives in projected_centroid_vectors.npz
    referenced via vector_path.
    """
    experiment_id = f"exp_005b_{condition}"
    rows: list[dict] = []
    for k, vec in proj_vecs.items():
        parts = k.split("_")
        cond, agent_id, label, round_str = parts[0], f"{parts[1]}_{parts[2]}", parts[3], parts[4]
        if cond != condition:
            continue
        round_num = int(round_str)
        rows.append({
            "centroid_id":         f"{experiment_id}_{agent_id}_{label}_{round_num}",
            "experiment_id":       experiment_id,
            "agent_id":            agent_id,
            "label":               label,
            "round":               round_num,
            # The Centroid schema from earlier experiments expects these
            # numerical metadata fields. We populate them with sensible
            # placeholders so downstream queries that join Centroid with
            # PROJECTED_PROXIMITY_TO don't trip over missing properties.
            "cohesion":            1.0,
            "vector_norm":         1.0,
            "shift_from_previous": 0.0,
            "distance_to_initial": 0.0,
            "n_points":            -1,
            "vector_path":         f"projected_centroid_vectors.npz:{k}",
        })
    logger.log_centroids_batch(rows)


def log_adapters_for_condition(
    logger: Neo4jLogger,
    condition: str,
    drift_by_key: dict[tuple, dict],
) -> None:
    """
    One :Adapter node per (condition, agent, round) and a
    (:Agent)-[:HAS_ADAPTER]->(:Adapter) edge per node.

    The Adapter only stores the two frobenius scalars + a pointer; the W
    matrix itself lives in adapter_weights_final.npz (or _initial.npz at
    round 0). Round-specific weights would be too large to persist for
    every round — the drift scalars are the salient signal.
    """
    experiment_id = f"exp_005b_{condition}"

    rows: list[dict] = []
    for aid in AGENT_IDS:
        for round_num in range(1, MAX_ROUNDS + 1):
            drift_row = drift_by_key.get((condition, aid, round_num))
            if drift_row is None:
                continue
            rows.append({
                "adapter_id":     f"{condition}_{aid}_{round_num}",
                "experiment_id":  experiment_id,
                "condition":      condition,
                "agent_id":       aid,
                "round":          round_num,
                "frobenius_distance_to_identity": float(drift_row["frobenius_distance_to_identity"]),
                "frobenius_drift_from_initial":   float(drift_row["frobenius_drift_from_initial"]),
                # Round 0 weights are in the initial npz; for any later
                # round, the user must run.py the round in question to
                # reproduce W_round. We point to the final npz as the
                # canonical "saved" weight set; future ablations can
                # extend run.py to save every round if needed.
                "vector_path":    (
                    "adapter_weights_initial.npz" if round_num == 1
                    else "adapter_weights_final.npz"
                ) + f":{condition}_{aid}",
            })

    if not rows:
        return

    query = (
        "UNWIND $rows AS row "
        "MATCH (a:Agent {agent_id: row.agent_id, experiment_id: row.experiment_id}) "
        "MERGE (ad:Adapter {adapter_id: row.adapter_id}) "
        "SET ad += row "
        "MERGE (a)-[r:HAS_ADAPTER {round: row.round, condition: row.condition}]->(ad)"
    )
    chunk = 500
    for i in range(0, len(rows), chunk):
        logger._run(query, rows=rows[i:i + chunk])
    print(f"  Logged {len(rows)} Adapter nodes + HAS_ADAPTER edges for {condition}.")


# ─── PROJECTED_PROXIMITY_TO edges ─────────────────────────────────────────────

def log_proximity_edges(
    logger: Neo4jLogger,
    condition: str,
    space: str,                                # "projected" | "raw"
    dist_by_key: dict[tuple, dict],
) -> None:
    """
    For every (condition, label, round) row, create three undirected-ish
    PROJECTED_PROXIMITY_TO edges (one per agent pair). We always direct
    the edge from the lower agent id to the higher one to avoid duplicates.

    All edges carry `space` ("projected" or "raw") so a single Cypher
    query can isolate one space at a time. The edge name is the same in
    both spaces; that follows the spec wording, which says the
    PROJECTED_PROXIMITY_TO relationship has a `space` property rather
    than a separate relationship type per space.
    """
    experiment_id = f"exp_005b_{condition}"

    rows: list[dict] = []
    for (cond, label, round_num), drow in dist_by_key.items():
        if cond != condition:
            continue
        for a, b in AGENT_PAIRS:
            key = f"d_{a[-2:]}_{b[-2:]}"
            if key not in drow:
                continue
            try:
                d = float(drow[key])
            except (TypeError, ValueError):
                continue
            rows.append({
                "centroid_id_a": f"{experiment_id}_{a}_{label}_{round_num}",
                "centroid_id_b": f"{experiment_id}_{b}_{label}_{round_num}",
                "round":         round_num,
                "label":         label,
                "condition":     condition,
                "cosine_distance": d,
                "space":         space,
                "experiment_id": experiment_id,
            })

    if not rows:
        return

    query = (
        "UNWIND $rows AS row "
        "MATCH (c1:Centroid {centroid_id: row.centroid_id_a}) "
        "MATCH (c2:Centroid {centroid_id: row.centroid_id_b}) "
        "MERGE (c1)-[p:PROJECTED_PROXIMITY_TO {"
        "    round: row.round, label: row.label, "
        "    space: row.space, experiment_id: row.experiment_id"
        "}]->(c2) "
        "SET p.cosine_distance = row.cosine_distance, "
        "    p.condition       = row.condition"
    )
    chunk = 500
    for i in range(0, len(rows), chunk):
        logger._run(query, rows=rows[i:i + chunk])
    print(f"  Logged {len(rows)} PROJECTED_PROXIMITY_TO edges ({space}) for {condition}.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] {RESULTS_DIR} missing — run run.py + metrics_and_plots.py first.")
        sys.exit(1)

    print("[Load] CSVs and centroid vectors ...")
    drift_by_key = load_drift()
    proj_dists   = load_inter_agent(RESULTS_DIR / "inter_agent_distances_projected.csv")
    raw_dists    = load_inter_agent(RESULTS_DIR / "inter_agent_distances_raw.csv")
    proj_vecs    = dict(np.load(RESULTS_DIR / "projected_centroid_vectors.npz"))
    print(f"  drift_rows={len(drift_by_key)}, proj_dists={len(proj_dists)}, "
          f"raw_dists={len(raw_dists)}, proj_vecs={len(proj_vecs)}")

    with Neo4jLogger() as logger:
        for cond in CONDITIONS:
            print(f"\n[Neo4j] Seeding metadata for condition: {cond}")
            seed_experiment_metadata(logger, cond)

            print(f"[Neo4j] Centroids for {cond} ...")
            log_centroids_for_condition(logger, cond, proj_vecs)

            print(f"[Neo4j] Adapters for {cond} ...")
            log_adapters_for_condition(logger, cond, drift_by_key)

            print(f"[Neo4j] Proximity edges (projected) for {cond} ...")
            log_proximity_edges(logger, cond, "projected", proj_dists)
            print(f"[Neo4j] Proximity edges (raw) for {cond} ...")
            log_proximity_edges(logger, cond, "raw", raw_dists)

    print("\n[Done] exp_005b Neo4j upload complete.")


if __name__ == "__main__":
    run()
