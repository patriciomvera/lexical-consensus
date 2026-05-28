"""
replay_exp003.py — Replay exp_003a events into Neo4j.

Reads ledger_events.csv and centroid_drift.csv from the exp_003a results
directory and populates the Neo4j AuraDB graph. Then runs four diagnostic
Cypher queries and saves results to results/exp_004_neo4j_shannon/.

Usage:
    python -m src.graph.replay_exp003
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.neo4j_logger import Neo4jLogger  # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────

EXP_003A_DIR  = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"
EXP_004_DIR   = PROJECT_ROOT / "results" / "exp_004_neo4j_shannon"
LEDGER_CSV    = EXP_003A_DIR / "ledger_events.csv"
DRIFT_CSV     = EXP_003A_DIR / "centroid_drift.csv"
CONFIG_YAML   = EXP_003A_DIR / "config.yaml"

EXPERIMENT_ID = "exp_003a"
CARROLL_LABELS = ["slithy", "mimsy", "vorpal"]
CARROLL_SOURCE = "Lewis Carroll, Jabberwocky (1871)"


# ─── CSV helpers ─────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no results)\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path.name}")


# ─── Pre-flight check ─────────────────────────────────────────────────────────

def verify_inputs() -> None:
    missing = [p for p in (LEDGER_CSV, DRIFT_CSV) if not p.exists()]
    if missing:
        print("[ERROR] Required files not found:")
        for p in missing:
            print(f"  {p}")
        print("Re-run exp_003a to generate them.")
        sys.exit(1)
    print("[Check] ledger_events.csv and centroid_drift.csv — OK")


# ─── Schema population ────────────────────────────────────────────────────────

def log_schema(logger: Neo4jLogger, ledger_rows: list[dict]) -> None:
    """Log Experiment, Agent, Label, and Image nodes."""

    config = {}
    if CONFIG_YAML.exists():
        with open(CONFIG_YAML, encoding="utf-8") as f:
            config = yaml.safe_load(f)

    n_rounds = max(int(r["round"]) for r in ledger_rows)

    print("  Logging Experiment node...")
    logger.log_experiment(
        experiment_id=EXPERIMENT_ID,
        date=config.get("date", "2026-05-28"),
        encoder=config.get("encoder", "facebook/dinov2-small"),
        dataset=config.get("dataset", "CIFAR-10"),
        n_agents=int(config.get("n_agents", 3)),
        n_rounds=n_rounds,
        n_labels=len(CARROLL_LABELS),
    )

    print("  Logging Agent nodes...")
    agent_seed_slices = config.get("agent_seed_slices", {
        "agent_00": [0, 5],
        "agent_01": [5, 10],
        "agent_02": [10, 15],
    })
    for agent_id, slices in agent_seed_slices.items():
        logger.log_agent(
            agent_id=agent_id,
            experiment_id=EXPERIMENT_ID,
            n_seeds=int(slices[1] - slices[0]) * 3,  # per-category × 3 categories
            seed_indices=slices,
        )

    print("  Logging Label nodes...")
    for name in CARROLL_LABELS:
        logger.log_label(name=name, carroll_source=CARROLL_SOURCE, experiment_id=EXPERIMENT_ID)

    print("  Logging Image nodes...")
    seen_images: set[str] = set()
    for row in ledger_rows:
        img = row["image_id"]
        if img not in seen_images:
            logger.log_image(
                image_id=img,
                true_category=row["true_category"],
                pool=row["pool"],
                experiment_id=EXPERIMENT_ID,
            )
            seen_images.add(img)
    print(f"    {len(seen_images)} images logged.")


def log_assignments(logger: Neo4jLogger, ledger_rows: list[dict]) -> None:
    """Build Assignment nodes from ledger_events rows."""
    print(f"  Building {len(ledger_rows)} assignment records...")
    assignments = []
    for row in ledger_rows:
        assignment_id = (
            f"{EXPERIMENT_ID}_{row['round']}_{row['agent_id']}_{row['image_id']}"
        )
        assignments.append({
            "assignment_id":   assignment_id,
            "experiment_id":   EXPERIMENT_ID,
            "round":           int(row["round"]),
            "agent_id":        row["agent_id"],
            "image_id":        row["image_id"],
            "label":           row["predicted_label"],
            "confidence":      float(row["confidence"]),
            "margin":          float(row["margin"]),
            "status":          row["status"],
            "pool":            row["pool"],
        })
    logger.log_assignments_batch(assignments)


def log_centroids(logger: Neo4jLogger, drift_rows: list[dict]) -> None:
    """Build Centroid nodes from centroid_drift rows."""
    print(f"  Building {len(drift_rows)} centroid records...")
    centroids = []
    for row in drift_rows:
        centroid_id = (
            f"{EXPERIMENT_ID}_{row['agent_id']}_{row['label']}_{row['round']}"
        )
        centroids.append({
            "centroid_id":         centroid_id,
            "experiment_id":       EXPERIMENT_ID,
            "agent_id":            row["agent_id"],
            "label":               row["label"],
            "round":               int(row["round"]),
            "cohesion":            float(row["cohesion"]),
            "vector_norm":         float(row["vector_norm"]),
            "shift_from_previous": float(row["shift_from_previous"]),
            "distance_to_initial": float(row["distance_to_initial"]),
            "n_points":            int(row["n_points"]),
        })
    logger.log_centroids_batch(centroids)


def log_consensus_edges(logger: Neo4jLogger, ledger_rows: list[dict]) -> None:
    """
    Create CONSENSUS_WITH relationships for every (round, image) where
    two or more agents agreed on the same label.

    Accepted agents are those with status='accepted' (their label matched
    the majority). For unanimous agreement, all three pairs get the edge.
    For majority (2/3), one pair gets the edge.
    """
    from collections import defaultdict

    # group by (round, image_id, predicted_label) → list of accepted agent_ids
    accepted: dict[tuple, list[str]] = defaultdict(list)
    for row in ledger_rows:
        if row["status"] == "accepted":
            key = (int(row["round"]), row["image_id"], row["predicted_label"])
            accepted[key].append(row["agent_id"])

    print(f"  Logging consensus edges...")
    n_edges = 0
    for (round_num, image_id, label), agent_ids in accepted.items():
        if len(agent_ids) < 2:
            continue
        agreement_type = "unanimous" if len(agent_ids) == 3 else "majority"
        for a, b in combinations(sorted(agent_ids), 2):
            logger.log_consensus(
                agent_id_a=a,
                agent_id_b=b,
                image_id=image_id,
                label=label,
                round_num=round_num,
                agreement_type=agreement_type,
                experiment_id=EXPERIMENT_ID,
            )
            n_edges += 1
    print(f"    {n_edges} CONSENSUS_WITH edges logged.")


# ─── Cypher queries ────────────────────────────────────────────────────────────
#
# Each query is run twice: once filtered to the interaction pool and once to
# held_out. Queries 1 and 2 directly filter by asgn.pool. Queries 3 and 4 are
# centroid-level and have no natural pool dimension; they produce identical
# results for both pool suffixes — the duplication is kept for consistency and
# to make the pool dimension explicit in every output file name.

QUERY_1 = """
MATCH (a:Agent)-[:MADE]->(asgn:Assignment)-[:USES_LABEL]->(l:Label)
WHERE asgn.experiment_id = 'exp_003a'
  AND asgn.pool = $pool
RETURN asgn.round AS round, l.name AS label,
       count(asgn) AS n_assignments,
       avg(asgn.confidence) AS mean_confidence,
       avg(asgn.margin) AS mean_margin
ORDER BY asgn.round, l.name
"""

QUERY_2 = """
MATCH (asgn:Assignment)-[:USES_LABEL]->(l:Label)
WHERE asgn.experiment_id = 'exp_003a'
  AND asgn.status = 'accepted'
  AND asgn.pool = $pool
WITH asgn.round AS round,
     l.name AS label,
     asgn.image_id AS image_id,
     count(DISTINCT asgn.agent_id) AS n_agents
WHERE n_agents = 3
WITH label, image_id, min(round) AS first_unanimous_round
RETURN label, image_id, first_unanimous_round
ORDER BY label, first_unanimous_round
"""

QUERY_3 = """
MATCH (a:Agent)-[:HAS_CENTROID]->(c:Centroid)
WHERE c.experiment_id = 'exp_003a'
RETURN a.agent_id AS agent_id, c.label AS label, c.round AS round,
       c.cohesion AS cohesion,
       c.shift_from_previous AS shift_from_previous,
       c.distance_to_initial AS distance_to_initial
ORDER BY a.agent_id, c.label, c.round
"""

QUERY_4 = """
MATCH (a:Agent)-[:HAS_CENTROID]->(c:Centroid)
WHERE c.experiment_id = 'exp_003a'
WITH a.agent_id AS agent, c.label AS label,
     avg(c.shift_from_previous) AS mean_shift
RETURN agent, label, mean_shift
ORDER BY label, mean_shift ASC
"""

POOLS = ["interaction", "held_out"]


def run_queries(logger: Neo4jLogger) -> None:
    EXP_004_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Queries]")

    for pool in POOLS:
        print(f"  Pool: {pool}")

        print(f"    Query 1: label assignments per round...")
        rows = logger.run_query(QUERY_1, pool=pool)
        write_csv(rows, EXP_004_DIR / f"query_1_label_assignments_{pool}.csv")

        print(f"    Query 2: first unanimous round per label per image...")
        rows = logger.run_query(QUERY_2, pool=pool)
        write_csv(rows, EXP_004_DIR / f"query_2_first_unanimous_{pool}.csv")

        # Q3/Q4 are centroid-level — pool-agnostic. Run once per pool label
        # for naming consistency; results are identical across pools.
        print(f"    Query 3: centroid stability trajectory...")
        rows = logger.run_query(QUERY_3)
        write_csv(rows, EXP_004_DIR / f"query_3_centroid_trajectory_{pool}.csv")

        print(f"    Query 4: most stable agent per label...")
        rows = logger.run_query(QUERY_4)
        write_csv(rows, EXP_004_DIR / f"query_4_most_stable_agent_{pool}.csv")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    verify_inputs()

    print("\n[Load] Reading CSVs...")
    ledger_rows = read_csv(LEDGER_CSV)
    drift_rows  = read_csv(DRIFT_CSV)
    print(f"  ledger_events: {len(ledger_rows)} rows")
    print(f"  centroid_drift: {len(drift_rows)} rows")

    with Neo4jLogger() as logger:
        print("\n[Clear] Removing previous exp_003a nodes...")
        logger.clear_experiment(EXPERIMENT_ID)

        print("\n[Schema] Logging nodes...")
        log_schema(logger, ledger_rows)

        print("\n[Assignments] Logging assignment events...")
        log_assignments(logger, ledger_rows)

        print("\n[Centroids] Logging centroid trajectories...")
        log_centroids(logger, drift_rows)

        print("\n[Consensus] Logging consensus relationships...")
        log_consensus_edges(logger, ledger_rows)

        run_queries(logger)

    print(f"\n[Done] Results in: {EXP_004_DIR}")


if __name__ == "__main__":
    run()
