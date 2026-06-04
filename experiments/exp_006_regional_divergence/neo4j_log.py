"""
neo4j_log.py — exp_006: push the regional cluster structure to Neo4j.

Schema additions:

  (:Cluster {cluster_id, regime, experiment_id})
  (:Cluster)-[:CONTAINS]->(:Agent)
  (:Cluster)-[:DIVERGENCE_TO {
      round, label, regime,
      between_cluster_distance,
      experiment_id
  }]->(:Cluster)

One experiment_id per regime (exp_006_{vervet|raven|latin}) so a single
Cypher query can pick any regime cleanly without depending on the
others' presence.

This is the data backbone for the eventual exp_008 interactive
visualization of regional lexical drift; the heavy artifacts (raw
centroid vectors, full ledger) stay on disk and are pointed to from
node properties only when needed.

Run AFTER run.py + metrics_and_plots.py:
    python -m experiments.exp_006_regional_divergence.neo4j_log
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.neo4j_logger import Neo4jLogger  # noqa: E402
from experiments.exp_006_regional_divergence._shared import (  # noqa: E402
    REGIMES, REGIME_CONFIG, CLUSTERS, CARROLL_LABELS, MAX_ROUNDS,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_006_regional_divergence"


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─── Experiment + Agent + Cluster nodes ──────────────────────────────────────

def seed_experiment_metadata(logger: Neo4jLogger, regime: str) -> None:
    """Create :Experiment, :Agent, :Label, and :Cluster nodes for this regime."""
    experiment_id = f"exp_006_{regime}"
    logger.log_experiment(
        experiment_id=experiment_id,
        date="2026-06-04",
        encoder="facebook/dinov2-small (frozen)",
        dataset="CIFAR-10 (cluster-specific image subsets)",
        n_agents=len(CLUSTERS["cluster_1"]) + len(CLUSTERS["cluster_2"]),
        n_rounds=MAX_ROUNDS,
        n_labels=len(CARROLL_LABELS),
    )
    # Agents — straightforward map; each agent's seed_indices is its
    # cluster's slice plus its within-cluster offset.
    for cluster_id, agent_ids in CLUSTERS.items():
        for agent_id in agent_ids:
            logger.log_agent(
                agent_id=agent_id,
                experiment_id=experiment_id,
                n_seeds=5 * len(CARROLL_LABELS),
                seed_indices=[],  # full slice already captured in config.yaml
            )
    for label in CARROLL_LABELS:
        logger.log_label(
            name=label,
            carroll_source="Jabberwocky",
            experiment_id=experiment_id,
        )

    # Cluster nodes + CONTAINS edges
    for cluster_id, agent_ids in CLUSTERS.items():
        logger._run(
            "MERGE (c:Cluster {cluster_id: $cid, experiment_id: $eid}) "
            "SET c.regime = $regime",
            cid=cluster_id, eid=experiment_id, regime=regime,
        )
        for agent_id in agent_ids:
            logger._run(
                "MATCH (c:Cluster {cluster_id: $cid, experiment_id: $eid}) "
                "MATCH (a:Agent {agent_id: $aid, experiment_id: $eid}) "
                "MERGE (c)-[:CONTAINS]->(a)",
                cid=cluster_id, eid=experiment_id, aid=agent_id,
            )


# ─── DIVERGENCE_TO edges ──────────────────────────────────────────────────────

def log_divergence_edges(logger: Neo4jLogger, regime: str, between_rows: list[dict]) -> None:
    """
    For every (regime, round, label) emit one DIVERGENCE_TO edge from
    cluster_1 to cluster_2 carrying the between_cluster_distance.

    We use MERGE-with-properties on (round, label) so each edge is
    unique; SET applies the distance value. The edge is directed only
    in the schema sense — cosine distance is symmetric, so any
    downstream query can ignore the direction.
    """
    experiment_id = f"exp_006_{regime}"
    rows = [
        {
            "src_cluster":              "cluster_1",
            "dst_cluster":              "cluster_2",
            "round":                    int(r["round"]),
            "label":                    r["label"],
            "regime":                   regime,
            "between_cluster_distance": float(r["between_cluster_distance"]),
            "experiment_id":            experiment_id,
        }
        for r in between_rows if r["regime"] == regime
    ]
    if not rows:
        return

    query = (
        "UNWIND $rows AS row "
        "MATCH (c1:Cluster {cluster_id: row.src_cluster, experiment_id: row.experiment_id}) "
        "MATCH (c2:Cluster {cluster_id: row.dst_cluster, experiment_id: row.experiment_id}) "
        "MERGE (c1)-[d:DIVERGENCE_TO {round: row.round, label: row.label, "
        "                              experiment_id: row.experiment_id}]->(c2) "
        "SET d.between_cluster_distance = row.between_cluster_distance, "
        "    d.regime                    = row.regime"
    )
    chunk = 500
    for i in range(0, len(rows), chunk):
        logger._run(query, rows=rows[i:i + chunk])
    print(f"  Logged {len(rows)} DIVERGENCE_TO edges for {regime}.")


# ─── Query the divergence trajectory ─────────────────────────────────────────

DIVERGENCE_TRAJECTORY_QUERY = """
MATCH (c1:Cluster)-[d:DIVERGENCE_TO]->(c2:Cluster)
WHERE d.experiment_id STARTS WITH 'exp_006_'
RETURN d.regime AS regime,
       d.label  AS label,
       d.round  AS round,
       d.between_cluster_distance AS between_cluster_distance
ORDER BY d.regime, d.label, d.round
"""


def run_queries(logger: Neo4jLogger) -> None:
    print("\n[Queries]")
    print("  divergence trajectory across all regimes ...")
    rows = logger.run_query(DIVERGENCE_TRAJECTORY_QUERY)
    out = RESULTS_DIR / "neo4j_query_divergence_trajectory.csv"
    if not rows:
        out.write_text("(no rows)\n", encoding="utf-8")
        return
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] {RESULTS_DIR} missing — run run.py first.")
        sys.exit(1)

    between_rows = read_csv(RESULTS_DIR / "between_cluster_metrics.csv")
    print(f"[Load] between_cluster_metrics.csv: {len(between_rows)} rows")

    with Neo4jLogger() as logger:
        for regime in REGIMES:
            print(f"\n[Neo4j] Seeding metadata for {regime} ...")
            seed_experiment_metadata(logger, regime)
            print(f"[Neo4j] Divergence edges for {regime} ...")
            log_divergence_edges(logger, regime, between_rows)
        run_queries(logger)

    print("\n[Done] exp_006 Neo4j upload complete.")


if __name__ == "__main__":
    run()
