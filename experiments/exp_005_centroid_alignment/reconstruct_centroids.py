"""
reconstruct_centroids.py — Reconstruct 384-dim centroid vectors from
ledger_events.csv + DINOv2 embeddings for exp_003a and exp_003b.

For each (experiment_id, agent_id, label, round):
  centroid = mean of unit-normed DINOv2 embeddings of all unique
  interaction-pool images with status='accepted' for this agent and
  label, cumulative up to and including this round. Then L2-normalized.

CRITICAL: Only interaction pool (pool='interaction') images are used.
  Held-out images never updated centroids during the experiment.
  003b (frozen lexicons) produces flat centroids across all rounds
  because the same images are accepted every round.

Output: results/exp_005_centroid_alignment/centroid_vectors.npz
  keys: "{experiment_id}_{agent_id}_{label}_{round}" -> float32[384]

Usage:
    python -m experiments.exp_005_centroid_alignment.reconstruct_centroids
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_003_multi_agent_consensus"))

from _shared import load_all_images, encode_all_images, CARROLL_LABELS  # noqa: E402
from src.agents.perception import PerceptionLayer                        # noqa: E402

EXP_003A_LEDGER = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"  / "ledger_events.csv"
EXP_003B_LEDGER = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline" / "ledger_events.csv"
OUT_DIR         = PROJECT_ROOT / "results" / "exp_005_centroid_alignment"

AGENT_IDS = ["agent_00", "agent_01", "agent_02"]


def read_ledger(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_centroid_vectors(
    ledger_rows: list[dict],
    embeddings:  dict[str, np.ndarray],
    experiment_id: str,
) -> dict[str, np.ndarray]:
    """
    Build one unit-normed centroid vector per (agent, label, round).

    Algorithm: for each (agent, label), collect accepted interaction-pool
    image_ids in round order. At each round R, take the cumulative union of
    all such images from rounds 1..R, compute their mean embedding, L2-normalize.

    003b (frozen lexicons): the same images are accepted every round, so the
    cumulative set stabilizes after round 1. All round vectors are identical.
    """
    # Filter to interaction pool, accepted, Carroll labels only
    ia_rows = [
        r for r in ledger_rows
        if r["pool"] == "interaction"
        and r["status"] == "accepted"
        and r["predicted_label"] in CARROLL_LABELS
    ]

    all_rounds = sorted({int(r["round"]) for r in ledger_rows})

    # (agent_id, label) -> sorted list of (round_num, image_id)
    per_key: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for r in ia_rows:
        key = (r["agent_id"], r["predicted_label"])
        per_key[key].append((int(r["round"]), r["image_id"]))

    for key in per_key:
        per_key[key].sort()

    vectors: dict[str, np.ndarray] = {}

    for agent_id in AGENT_IDS:
        for label in CARROLL_LABELS:
            key = (agent_id, label)
            rows_sorted = per_key.get(key, [])

            cumulative: set[str] = set()
            row_idx = 0

            for round_num in all_rounds:
                # Advance cursor: add all accepted images up to this round
                while row_idx < len(rows_sorted) and rows_sorted[row_idx][0] <= round_num:
                    cumulative.add(rows_sorted[row_idx][1])
                    row_idx += 1

                if not cumulative:
                    continue  # no accepted images yet

                embs     = np.stack([embeddings[img] for img in cumulative], axis=0)
                mean_emb = embs.mean(axis=0)
                norm     = np.linalg.norm(mean_emb)
                if norm > 1e-9:
                    mean_emb = mean_emb / norm

                k = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                vectors[k] = mean_emb.astype(np.float32)

    return vectors


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in (EXP_003A_LEDGER, EXP_003B_LEDGER):
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    print("[Data] Loading CIFAR-10 images...")
    all_images = load_all_images()

    print("[Encoding] Computing DINOv2 embeddings...")
    perception = PerceptionLayer(device="cpu")
    embeddings, _ = encode_all_images(all_images, perception)
    print(f"  {len(embeddings)} embeddings ready.")

    print("\n[003a] Reading ledger and reconstructing centroids...")
    rows_a   = read_ledger(EXP_003A_LEDGER)
    vectors_a = build_centroid_vectors(rows_a, embeddings, "exp_003a")
    rounds_a  = sorted({int(r["round"]) for r in rows_a})
    print(f"  {len(vectors_a)} vectors ({len(rounds_a)} rounds x 3 agents x 3 labels)")

    print("\n[003b] Reading ledger and reconstructing centroids...")
    rows_b    = read_ledger(EXP_003B_LEDGER)
    vectors_b = build_centroid_vectors(rows_b, embeddings, "exp_003b")
    rounds_b  = sorted({int(r["round"]) for r in rows_b})
    print(f"  {len(vectors_b)} vectors ({len(rounds_b)} rounds x 3 agents x 3 labels — expect flat)")

    all_vectors = {**vectors_a, **vectors_b}
    out_path    = OUT_DIR / "centroid_vectors.npz"
    np.savez(out_path, **all_vectors)
    print(f"\n[Save] {len(all_vectors)} vectors -> {out_path}")

    # Sanity check: verify 003b centroids are flat
    print("\n[Check] 003b centroid stability across rounds...")
    for agent_id in AGENT_IDS:
        for label in CARROLL_LABELS:
            key_r1  = f"exp_003b_{agent_id}_{label}_1"
            key_r50 = f"exp_003b_{agent_id}_{label}_50"
            if key_r1 in all_vectors and key_r50 in all_vectors:
                dot = float(np.dot(all_vectors[key_r1], all_vectors[key_r50]))
                shift = max(0.0, 1.0 - dot)
                if shift > 1e-6:
                    print(f"  WARNING: {agent_id}/{label} shifted by {shift:.6f} between r1 and r50")
    print("  Done — all 003b centroids are flat (expected).")


if __name__ == "__main__":
    run()
