"""
update_neo4j.py — exp_005 Part D: add PROXIMITY_TO edges and centroid
proximity properties to the Neo4j AuraDB graph.

Adds to the existing graph:
  - (:Centroid)-[:PROXIMITY_TO {round, label, cosine_distance, condition,
    experiment_id}]->(:Centroid)  for every cross-agent same-label pair
  - Updates existing exp_003a Centroid nodes with mean_pairwise_distance,
    distance_to_agent_00/01/02, vector_hash, vector_path
  - Creates exp_003b Centroid nodes (rounds 1-6) with same properties

Runs three Cypher queries (Q5, Q6, Q7) and saves results as CSV.

Usage:
    python -m experiments.exp_005_centroid_alignment.update_neo4j
"""

from __future__ import annotations

import csv
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.neo4j_logger import Neo4jLogger  # noqa: E402

EXP_003A_DRIFT = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"  / "centroid_drift.csv"
EXP_003B_DRIFT = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline" / "centroid_drift.csv"
OUT_DIR        = PROJECT_ROOT / "results" / "exp_005_centroid_alignment"
NPZ_PATH       = OUT_DIR / "centroid_vectors.npz"

AGENT_IDS      = ["agent_00", "agent_01", "agent_02"]
AGENT_PAIRS    = list(combinations(AGENT_IDS, 2))
CARROLL_LABELS = ["slithy", "mimsy", "vorpal"]

CONDITION_MAP  = {
    "exp_003a": "003a_feedback",
    "exp_003b": "003b_no_feedback",
}


# ─── CSV helpers ──────────────────────────────────────────────────────────────

def read_drift_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no results)\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")


# ─── Vector helpers ────────────────────────────────────────────────────────────

def vector_hash(vec: np.ndarray) -> str:
    return hashlib.md5(vec.astype(np.float32).tobytes()).hexdigest()[:16]


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.dot(u, v)))


# ─── Scalar proximity properties ──────────────────────────────────────────────

def compute_proximity_props(
    vectors: dict[str, np.ndarray],
    experiment_id: str,
    rounds: list[int],
) -> dict[tuple, dict]:
    """
    For each (agent_id, label, round), compute:
      - distance_to_agent_00/01/02
      - mean_pairwise_distance
      - vector_hash
      - vector_path (key in .npz)

    Returns: (agent_id, label, round) -> property dict
    """
    props: dict[tuple, dict] = {}

    for label in CARROLL_LABELS:
        for round_num in rounds:
            agent_vecs: dict[str, np.ndarray | None] = {}
            for agent_id in AGENT_IDS:
                k = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                agent_vecs[agent_id] = vectors.get(k)

            # Pairwise distances
            pair_dists: dict[tuple, float] = {}
            for a, b in AGENT_PAIRS:
                va, vb = agent_vecs.get(a), agent_vecs.get(b)
                if va is not None and vb is not None:
                    pair_dists[(a, b)] = cosine_distance(va, vb)

            for agent_id in AGENT_IDS:
                vec = agent_vecs.get(agent_id)
                if vec is None:
                    continue

                # Distance from this agent to each other agent (ordered)
                dist_to: dict[str, float] = {}
                for other in AGENT_IDS:
                    if other == agent_id:
                        dist_to[f"distance_to_{other}"] = 0.0
                        continue
                    pair_key = (min(agent_id, other), max(agent_id, other))
                    d = pair_dists.get(pair_key, float("nan"))
                    dist_to[f"distance_to_{other}"] = round(d, 8)

                valid_dists = [v for k, v in pair_dists.items()
                               if agent_id in k and v == v]  # filter nan
                mpd = round(sum(valid_dists) / len(valid_dists), 8) if valid_dists else float("nan")

                npz_key = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                props[(agent_id, label, round_num)] = {
                    "mean_pairwise_distance": mpd,
                    **dist_to,
                    "vector_hash": vector_hash(vec),
                    "vector_path": f"centroid_vectors.npz:{npz_key}",
                }

    return props


# ─── Neo4j operations ─────────────────────────────────────────────────────────

def add_003b_centroids(logger: Neo4jLogger, drift_rows: list[dict], rounds: list[int]) -> None:
    """
    Create exp_003b Centroid nodes and Agent nodes for rounds in `rounds`.
    Reuses centroid_drift.csv scalars. Agent nodes created standalone (no
    full experiment schema needed for proximity queries).
    """
    experiment_id = "exp_003b"

    # Create minimal Agent nodes for 003b (needed for HAS_CENTROID relationship)
    print("  Creating 003b Agent nodes...")
    for agent_id in AGENT_IDS:
        logger._run(
            "MERGE (a:Agent {agent_id: $aid, experiment_id: $eid})",
            aid=agent_id, eid=experiment_id,
        )

    # Build centroid records for requested rounds only
    drift_by_key = {
        (r["agent_id"], r["label"], int(r["round"])): r
        for r in drift_rows
    }
    # 003b is flat — use round 1 scalars for all rounds
    drift_r1 = {
        (r["agent_id"], r["label"]): r
        for r in drift_rows
        if int(r["round"]) == 1
    }

    centroids: list[dict] = []
    for agent_id in AGENT_IDS:
        for label in CARROLL_LABELS:
            base = drift_r1.get((agent_id, label))
            if base is None:
                continue
            for round_num in rounds:
                centroid_id = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                centroids.append({
                    "centroid_id":         centroid_id,
                    "experiment_id":       experiment_id,
                    "agent_id":            agent_id,
                    "label":               label,
                    "round":               round_num,
                    "cohesion":            float(base["cohesion"]),
                    "vector_norm":         float(base["vector_norm"]),
                    "shift_from_previous": 0.0,
                    "distance_to_initial": 0.0,
                    "n_points":            int(base["n_points"]),
                })

    logger.log_centroids_batch(centroids)


def update_centroid_properties(
    logger: Neo4jLogger,
    experiment_id: str,
    props: dict[tuple, dict],
) -> None:
    """
    SET new scalar properties on existing Centroid nodes.
    Batched via UNWIND.
    """
    rows = []
    for (agent_id, label, round_num), p in props.items():
        centroid_id = f"{experiment_id}_{agent_id}_{label}_{round_num}"
        rows.append({
            "centroid_id":            centroid_id,
            "mean_pairwise_distance": p["mean_pairwise_distance"],
            "distance_to_agent_00":   p.get("distance_to_agent_00", float("nan")),
            "distance_to_agent_01":   p.get("distance_to_agent_01", float("nan")),
            "distance_to_agent_02":   p.get("distance_to_agent_02", float("nan")),
            "vector_hash":            p["vector_hash"],
            "vector_path":            p["vector_path"],
        })

    query = (
        "UNWIND $rows AS row "
        "MATCH (c:Centroid {centroid_id: row.centroid_id}) "
        "SET c.mean_pairwise_distance = row.mean_pairwise_distance, "
        "    c.distance_to_agent_00   = row.distance_to_agent_00, "
        "    c.distance_to_agent_01   = row.distance_to_agent_01, "
        "    c.distance_to_agent_02   = row.distance_to_agent_02, "
        "    c.vector_hash            = row.vector_hash, "
        "    c.vector_path            = row.vector_path"
    )
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        logger._run(query, rows=rows[i: i + chunk_size])
    print(f"  Updated {len(rows)} Centroid nodes with proximity properties.")


def add_proximity_edges(
    logger: Neo4jLogger,
    vectors: dict[str, np.ndarray],
    experiment_id: str,
    rounds: list[int],
) -> None:
    """
    Create PROXIMITY_TO edges between centroid pairs of different agents
    with the same label and round within this experiment.
    """
    condition = CONDITION_MAP[experiment_id]
    rows = []
    for label in CARROLL_LABELS:
        for round_num in rounds:
            for agent_a, agent_b in AGENT_PAIRS:
                ka = f"{experiment_id}_{agent_a}_{label}_{round_num}"
                kb = f"{experiment_id}_{agent_b}_{label}_{round_num}"
                va = vectors.get(ka)
                vb = vectors.get(kb)
                if va is None or vb is None:
                    continue
                d = cosine_distance(va, vb)
                cid_a = f"{experiment_id}_{agent_a}_{label}_{round_num}"
                cid_b = f"{experiment_id}_{agent_b}_{label}_{round_num}"
                rows.append({
                    "centroid_id_a":  cid_a,
                    "centroid_id_b":  cid_b,
                    "round":          round_num,
                    "label":          label,
                    "cosine_distance": round(d, 8),
                    "condition":      condition,
                    "experiment_id":  experiment_id,
                })

    query = (
        "UNWIND $rows AS row "
        "MATCH (c1:Centroid {centroid_id: row.centroid_id_a}) "
        "MATCH (c2:Centroid {centroid_id: row.centroid_id_b}) "
        "MERGE (c1)-[p:PROXIMITY_TO {"
        "    round: row.round, label: row.label, experiment_id: row.experiment_id"
        "}]->(c2) "
        "SET p.cosine_distance = row.cosine_distance, "
        "    p.condition       = row.condition"
    )
    chunk_size = 500
    for i in range(0, len(rows), chunk_size):
        logger._run(query, rows=rows[i: i + chunk_size])
    print(f"  {len(rows)} PROXIMITY_TO edges for {experiment_id}.")


# ─── Cypher queries ────────────────────────────────────────────────────────────

QUERY_5 = """
MATCH (c1:Centroid)-[p:PROXIMITY_TO]->(c2:Centroid)
WHERE p.experiment_id = 'exp_003a'
  AND c1.agent_id < c2.agent_id
RETURN c1.label AS label, p.round AS round,
       c1.agent_id AS agent_a, c2.agent_id AS agent_b,
       p.cosine_distance AS cosine_distance
ORDER BY c1.label, p.round
"""

QUERY_6 = """
MATCH (c1:Centroid)-[p:PROXIMITY_TO]->(c2:Centroid)
WHERE p.experiment_id IN ['exp_003a', 'exp_003b']
  AND c1.agent_id < c2.agent_id
RETURN p.condition AS condition, c1.label AS label, p.round AS round,
       avg(p.cosine_distance) AS mean_pairwise_distance
ORDER BY p.condition, c1.label, p.round
"""

QUERY_7 = """
MATCH (c1:Centroid)-[p:PROXIMITY_TO]->(c2:Centroid)
WHERE c1.agent_id < c2.agent_id
WITH p.condition AS condition,
     c1.label AS label,
     max(p.round) AS final_round,
     avg(p.cosine_distance) AS mean_dist
RETURN label,
  max(CASE WHEN condition='003b_no_feedback' THEN mean_dist END)
  - max(CASE WHEN condition='003a_feedback' THEN mean_dist END)
  AS alignment_gain
ORDER BY label
"""


def run_queries(logger: Neo4jLogger) -> None:
    print("\n[Queries]")

    print("  Query 5: centroid proximity trajectory (003a)...")
    rows = logger.run_query(QUERY_5)
    write_csv(rows, OUT_DIR / "query_5_proximity_trajectory.csv")

    print("  Query 6: mean pairwise distance per condition/label/round...")
    rows = logger.run_query(QUERY_6)
    write_csv(rows, OUT_DIR / "query_6_mean_pairwise.csv")

    print("  Query 7: alignment gain per label...")
    rows = logger.run_query(QUERY_7)
    write_csv(rows, OUT_DIR / "query_7_alignment_gain.csv")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NPZ_PATH.exists():
        print(f"[ERROR] Missing: {NPZ_PATH}")
        print("  Run reconstruct_centroids.py first.")
        sys.exit(1)

    print("[Load] centroid_vectors.npz...")
    data    = np.load(NPZ_PATH)
    vectors: dict[str, np.ndarray] = dict(data)
    print(f"  {len(vectors)} vectors loaded.")

    rounds_a = sorted({int(k.split("_")[-1]) for k in vectors if k.startswith("exp_003a")})
    print(f"  003a rounds: {rounds_a}")

    # Load drift CSVs for scalar properties
    drift_a = read_drift_csv(EXP_003A_DRIFT)
    drift_b = read_drift_csv(EXP_003B_DRIFT)

    # Compute proximity properties
    print("\n[Compute] Proximity properties...")
    props_a = compute_proximity_props(vectors, "exp_003a", rounds_a)
    props_b = compute_proximity_props(vectors, "exp_003b", rounds_a)
    print(f"  003a: {len(props_a)} (agent, label, round) properties")
    print(f"  003b: {len(props_b)} (agent, label, round) properties")

    with Neo4jLogger() as logger:
        # 1. Add 003b Centroid nodes (rounds matching 003a for comparison)
        print("\n[Neo4j] Adding exp_003b Centroid nodes...")
        add_003b_centroids(logger, drift_b, rounds_a)

        # 2. Update 003a Centroid nodes with new properties
        print("\n[Neo4j] Updating exp_003a Centroid nodes...")
        update_centroid_properties(logger, "exp_003a", props_a)

        # 3. Update 003b Centroid nodes with new properties
        print("\n[Neo4j] Updating exp_003b Centroid nodes...")
        update_centroid_properties(logger, "exp_003b", props_b)

        # 4. Add PROXIMITY_TO edges for both experiments
        print("\n[Neo4j] Adding PROXIMITY_TO edges...")
        add_proximity_edges(logger, vectors, "exp_003a", rounds_a)
        add_proximity_edges(logger, vectors, "exp_003b", rounds_a)

        # 5. Run queries
        run_queries(logger)

    print(f"\n[Done] Results in: {OUT_DIR}")


if __name__ == "__main__":
    run()
