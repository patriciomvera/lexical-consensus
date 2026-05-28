"""
neo4j_logger.py — Graph database logger for the Lexical Consensus experiment.

Writes every label assignment, consensus event, and centroid trajectory from
an experiment run into a Neo4j AuraDB instance as a queryable property graph.

Credentials are loaded from .env (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).
NEVER hardcode credentials in this file.

Graph schema overview:
    (:Experiment)-[:HAS_AGENT]->(:Agent)
    (:Experiment)-[:HAS_LABEL]->(:Label)
    (:Experiment)-[:HAS_IMAGE]->(:Image)
    (:Agent)-[:MADE]->(:Assignment)-[:USES_LABEL]->(:Label)
                                   -[:APPLIES_TO]->(:Image)
    (:Agent)-[:HAS_CENTROID {round}]->(:Centroid)
    (:Agent)-[:CONSENSUS_WITH {round, image_id, label, agreement_type}]->(:Agent)

WHY Assignment as a node:
    An assignment is a ternary fact (Agent × Label × Image). Modeling it as a
    node allows querying across all three dimensions without ambiguity.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


_ENV_LOADED = False

def _load_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
        _ENV_LOADED = True


class Neo4jLogger:
    """
    Manages a connection to Neo4j AuraDB and provides structured logging
    methods for experiment artifacts.

    Usage:
        logger = Neo4jLogger()
        logger.connect()
        logger.log_experiment(...)
        logger.close()

    Or use as a context manager:
        with Neo4jLogger() as logger:
            logger.log_experiment(...)
    """

    def __init__(self) -> None:
        _load_env()
        self._uri      = os.environ["NEO4J_URI"]
        self._username = os.environ["NEO4J_USERNAME"]
        self._password = os.environ["NEO4J_PASSWORD"]
        self._driver   = None

    # ─── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        self._driver.verify_connectivity()
        print("[Neo4j] Connected.")

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            print("[Neo4j] Connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    # ─── Safe delete ─────────────────────────────────────────────────────────

    def clear_experiment(self, experiment_id: str) -> None:
        """
        Delete all nodes (and their relationships) that belong to this
        experiment. Safe: only nodes with a matching experiment_id are touched.
        Runs in batches to avoid memory pressure on large datasets.
        """
        print(f"[Neo4j] Clearing experiment '{experiment_id}'...")
        query = (
            "MATCH (n {experiment_id: $eid}) "
            "CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 500 ROWS"
        )
        with self._driver.session() as session:
            session.run(query, eid=experiment_id)
        print(f"[Neo4j] Cleared.")

    # ─── Node logging ─────────────────────────────────────────────────────────

    def log_experiment(
        self,
        experiment_id: str,
        date: str,
        encoder: str,
        dataset: str,
        n_agents: int,
        n_rounds: int,
        n_labels: int,
    ) -> None:
        query = (
            "MERGE (e:Experiment {experiment_id: $experiment_id}) "
            "SET e.date = $date, "
            "    e.encoder = $encoder, "
            "    e.dataset = $dataset, "
            "    e.n_agents = $n_agents, "
            "    e.n_rounds = $n_rounds, "
            "    e.n_labels = $n_labels"
        )
        self._run(query, experiment_id=experiment_id, date=date, encoder=encoder,
                  dataset=dataset, n_agents=n_agents, n_rounds=n_rounds, n_labels=n_labels)

    def log_agent(
        self,
        agent_id: str,
        experiment_id: str,
        n_seeds: int,
        seed_indices: list,
    ) -> None:
        query = (
            "MATCH (e:Experiment {experiment_id: $experiment_id}) "
            "MERGE (a:Agent {agent_id: $agent_id, experiment_id: $experiment_id}) "
            "SET a.n_seeds = $n_seeds, a.seed_indices = $seed_indices "
            "MERGE (e)-[:HAS_AGENT]->(a)"
        )
        self._run(query, agent_id=agent_id, experiment_id=experiment_id,
                  n_seeds=n_seeds, seed_indices=seed_indices)

    def log_label(
        self,
        name: str,
        carroll_source: str,
        experiment_id: str,
    ) -> None:
        query = (
            "MATCH (e:Experiment {experiment_id: $experiment_id}) "
            "MERGE (l:Label {name: $name, experiment_id: $experiment_id}) "
            "SET l.carroll_source = $carroll_source "
            "MERGE (e)-[:HAS_LABEL]->(l)"
        )
        self._run(query, name=name, carroll_source=carroll_source, experiment_id=experiment_id)

    def log_image(
        self,
        image_id: str,
        true_category: str,
        pool: str,
        experiment_id: str,
    ) -> None:
        query = (
            "MATCH (e:Experiment {experiment_id: $experiment_id}) "
            "MERGE (i:Image {image_id: $image_id, experiment_id: $experiment_id}) "
            "SET i.true_category = $true_category, i.pool = $pool "
            "MERGE (e)-[:HAS_IMAGE]->(i)"
        )
        self._run(query, image_id=image_id, true_category=true_category,
                  pool=pool, experiment_id=experiment_id)

    def log_assignment(self, assignment: dict) -> None:
        """
        Log a single label assignment event.

        Expected keys: assignment_id, experiment_id, round, agent_id, image_id,
                       label, confidence, margin, status, pool
        """
        query = (
            "MATCH (a:Agent {agent_id: $agent_id, experiment_id: $experiment_id}) "
            "MATCH (l:Label {name: $label, experiment_id: $experiment_id}) "
            "MATCH (i:Image {image_id: $image_id, experiment_id: $experiment_id}) "
            "MERGE (asgn:Assignment {assignment_id: $assignment_id}) "
            "SET asgn.experiment_id = $experiment_id, "
            "    asgn.round = $round, "
            "    asgn.agent_id = $agent_id, "
            "    asgn.image_id = $image_id, "
            "    asgn.label = $label, "
            "    asgn.confidence = $confidence, "
            "    asgn.margin = $margin, "
            "    asgn.status = $status, "
            "    asgn.pool = $pool "
            "MERGE (a)-[:MADE]->(asgn) "
            "MERGE (asgn)-[:USES_LABEL]->(l) "
            "MERGE (asgn)-[:APPLIES_TO]->(i)"
        )
        self._run(query, **assignment)

    def log_assignments_batch(self, assignments: list[dict]) -> None:
        """
        Bulk-insert assignments using UNWIND for efficiency.
        All assignments must belong to the same experiment.
        """
        if not assignments:
            return
        experiment_id = assignments[0]["experiment_id"]
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:Agent {agent_id: row.agent_id, experiment_id: $experiment_id}) "
            "MATCH (l:Label {name: row.label, experiment_id: $experiment_id}) "
            "MATCH (i:Image {image_id: row.image_id, experiment_id: $experiment_id}) "
            "MERGE (asgn:Assignment {assignment_id: row.assignment_id}) "
            "SET asgn += row "
            "MERGE (a)-[:MADE]->(asgn) "
            "MERGE (asgn)-[:USES_LABEL]->(l) "
            "MERGE (asgn)-[:APPLIES_TO]->(i)"
        )
        # Process in chunks to avoid transaction size limits
        chunk_size = 500
        for i in range(0, len(assignments), chunk_size):
            chunk = assignments[i : i + chunk_size]
            self._run(query, rows=chunk, experiment_id=experiment_id)
        print(f"  [Neo4j] Logged {len(assignments)} assignments.")

    def log_centroid(self, centroid: dict) -> None:
        """
        Expected keys: centroid_id, experiment_id, agent_id, label, round,
                       cohesion, vector_norm, shift_from_previous,
                       distance_to_initial, n_points
        """
        query = (
            "MATCH (a:Agent {agent_id: $agent_id, experiment_id: $experiment_id}) "
            "MERGE (c:Centroid {centroid_id: $centroid_id}) "
            "SET c.experiment_id = $experiment_id, "
            "    c.agent_id = $agent_id, "
            "    c.label = $label, "
            "    c.round = $round, "
            "    c.cohesion = $cohesion, "
            "    c.vector_norm = $vector_norm, "
            "    c.shift_from_previous = $shift_from_previous, "
            "    c.distance_to_initial = $distance_to_initial, "
            "    c.n_points = $n_points "
            "MERGE (a)-[:HAS_CENTROID {round: $round}]->(c)"
        )
        self._run(query, **centroid)

    def log_centroids_batch(self, centroids: list[dict]) -> None:
        """Bulk-insert centroids using UNWIND."""
        if not centroids:
            return
        experiment_id = centroids[0]["experiment_id"]
        query = (
            "UNWIND $rows AS row "
            "MATCH (a:Agent {agent_id: row.agent_id, experiment_id: $experiment_id}) "
            "MERGE (c:Centroid {centroid_id: row.centroid_id}) "
            "SET c += row "
            "MERGE (a)-[:HAS_CENTROID {round: row.round}]->(c)"
        )
        chunk_size = 500
        for i in range(0, len(centroids), chunk_size):
            chunk = centroids[i : i + chunk_size]
            self._run(query, rows=chunk, experiment_id=experiment_id)
        print(f"  [Neo4j] Logged {len(centroids)} centroids.")

    def log_consensus(
        self,
        agent_id_a: str,
        agent_id_b: str,
        image_id: str,
        label: str,
        round_num: int,
        agreement_type: str,    # "majority" | "unanimous"
        experiment_id: str,
    ) -> None:
        """
        Create a CONSENSUS_WITH relationship between two agents who agreed
        on the same label for a specific image in a given round.

        image_id is REQUIRED per schema — consensus is always grounded in a
        specific perceptual event, not an abstract inter-agent alignment.
        """
        query = (
            "MATCH (a:Agent {agent_id: $agent_id_a, experiment_id: $experiment_id}) "
            "MATCH (b:Agent {agent_id: $agent_id_b, experiment_id: $experiment_id}) "
            "MERGE (a)-[r:CONSENSUS_WITH {"
            "    round: $round_num, "
            "    image_id: $image_id, "
            "    label: $label, "
            "    experiment_id: $experiment_id"
            "}]->(b) "
            "SET r.agreement_type = $agreement_type"
        )
        self._run(
            query,
            agent_id_a=agent_id_a,
            agent_id_b=agent_id_b,
            image_id=image_id,
            label=label,
            round_num=round_num,
            agreement_type=agreement_type,
            experiment_id=experiment_id,
        )

    # ─── Query execution ─────────────────────────────────────────────────────

    def run_query(self, cypher: str, **params) -> list[dict]:
        """Execute a read query and return results as a list of flat dicts."""
        with self._driver.session() as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]

    # ─── Private ─────────────────────────────────────────────────────────────

    def _run(self, query: str, **params) -> None:
        with self._driver.session() as session:
            session.run(query, **params)
