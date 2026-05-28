"""
full_centroid_analysis.py — Full operational centroid reconstruction and
comparison with interaction-only centroids.

The interaction-only reconstruction (reconstruct_centroids.py) uses only
accepted interaction-pool images, excluding seed embeddings. This script
adds seed embeddings (5 per agent per label, from disjoint index slices)
to produce the full operational centroid each agent actually uses for
classification.

Steps:
  1. Reconstruct full centroids → centroid_vectors_full.npz
  2. Compute alignment metrics → inter_agent_distances_full.csv,
                                  alignment_summary_full.json
  3. Generate comparison plot   → centroid_alignment_comparison.png
  4. Update Neo4j               → tag existing nodes as interaction_only,
                                  add full_operational nodes + PROXIMITY_TO

Usage:
    python -m experiments.exp_005_centroid_alignment.full_centroid_analysis
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "exp_003_multi_agent_consensus"))

from _shared import (  # noqa: E402
    load_all_images, encode_all_images,
    AGENT_SEED_SLICES, CARROLL_TO_CAT, CARROLL_LABELS,
)
from src.agents.perception import PerceptionLayer  # noqa: E402
from src.graph.neo4j_logger import Neo4jLogger     # noqa: E402

EXP_003A_LEDGER = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"  / "ledger_events.csv"
EXP_003B_LEDGER = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline" / "ledger_events.csv"
OUT_DIR         = PROJECT_ROOT / "results" / "exp_005_centroid_alignment"

AGENT_IDS   = ["agent_00", "agent_01", "agent_02"]
AGENT_PAIRS = list(combinations(AGENT_IDS, 2))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_ledger(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_csv(path: Path) -> list[dict]:
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


def write_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path.name}")


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.dot(u, v)))


def vector_hash(vec: np.ndarray) -> str:
    return hashlib.md5(vec.astype(np.float32).tobytes()).hexdigest()[:16]


# ─── Step 1 — Full centroid reconstruction ────────────────────────────────────

def build_full_centroid_vectors(
    ledger_rows: list[dict],
    embeddings:  dict[str, np.ndarray],
    experiment_id: str,
) -> dict[str, np.ndarray]:
    """
    Full operational centroid = seed embeddings + cumulative accepted
    interaction-pool embeddings, L2-normalized.

    seed_images(agent, label) = 5 images from that agent's index slice
      of that label's category (e.g. agent_01 / slithy → frog_005..009).

    For 003b (frozen lexicons) the accepted interaction set is constant
    across rounds, so the full centroid stabilises after round 1.
    """
    ia_rows = [
        r for r in ledger_rows
        if r["pool"] == "interaction"
        and r["status"] == "accepted"
        and r["predicted_label"] in CARROLL_LABELS
    ]

    all_rounds = sorted({int(r["round"]) for r in ledger_rows})

    per_key: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    for r in ia_rows:
        per_key[(r["agent_id"], r["predicted_label"])].append(
            (int(r["round"]), r["image_id"])
        )
    for key in per_key:
        per_key[key].sort()

    vectors: dict[str, np.ndarray] = {}

    for agent_id in AGENT_IDS:
        s_start, s_end = AGENT_SEED_SLICES[agent_id]

        for label in CARROLL_LABELS:
            cat = CARROLL_TO_CAT[label]
            seed_imgs = [f"{cat}_{idx:03d}" for idx in range(s_start, s_end)]
            seed_embs = np.stack([embeddings[img] for img in seed_imgs], axis=0)

            key = (agent_id, label)
            rows_sorted = per_key.get(key, [])

            cumulative: set[str] = set()
            row_idx = 0

            for round_num in all_rounds:
                while (row_idx < len(rows_sorted)
                       and rows_sorted[row_idx][0] <= round_num):
                    cumulative.add(rows_sorted[row_idx][1])
                    row_idx += 1

                if cumulative:
                    int_embs = np.stack(
                        [embeddings[img] for img in cumulative], axis=0
                    )
                    all_embs = np.concatenate([seed_embs, int_embs], axis=0)
                else:
                    all_embs = seed_embs  # seed-only before any interaction

                mean_emb = all_embs.mean(axis=0)
                norm     = np.linalg.norm(mean_emb)
                if norm > 1e-9:
                    mean_emb = mean_emb / norm

                k = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                vectors[k] = mean_emb.astype(np.float32)

    return vectors


# ─── Step 2 — Alignment metrics ───────────────────────────────────────────────

def compute_pairwise_distances(
    vectors: dict[str, np.ndarray],
    experiment_id: str,
    rounds: list[int],
) -> list[dict]:
    rows: list[dict] = []
    for label in CARROLL_LABELS:
        for round_num in rounds:
            vecs: dict[str, np.ndarray | None] = {
                a: vectors.get(f"{experiment_id}_{a}_{label}_{round_num}")
                for a in AGENT_IDS
            }
            dists: dict[str, float] = {}
            for a, b in AGENT_PAIRS:
                va, vb = vecs[a], vecs[b]
                pair_key = f"d_{a[-2:]}_{b[-2:]}"
                dists[pair_key] = (
                    round(cosine_distance(va, vb), 8)
                    if va is not None and vb is not None
                    else float("nan")
                )
            valid = [v for v in dists.values() if not math.isnan(v)]
            mpd   = round(sum(valid) / len(valid), 8) if valid else float("nan")
            rows.append({
                "experiment_id": experiment_id,
                "label": label,
                "round": round_num,
                "d_00_01": dists.get("d_00_01", float("nan")),
                "d_00_02": dists.get("d_00_02", float("nan")),
                "d_01_02": dists.get("d_01_02", float("nan")),
                "mean_pairwise_distance": mpd,
            })
    return rows


def compute_alignment_metrics(
    dist_a: list[dict],
    dist_b: list[dict],
    rounds_a: list[int],
) -> dict:
    def get(rows: list[dict], label: str, round_num: int) -> float:
        for r in rows:
            if r["label"] == label and int(r["round"]) == round_num:
                return float(r["mean_pairwise_distance"])
        return float("nan")

    r1   = min(rounds_a)
    rfin = max(rounds_a)
    per_label: dict[str, dict] = {}
    gains: list[float] = []

    for label in CARROLL_LABELS:
        d_a_r1    = get(dist_a, label, r1)
        d_a_final = get(dist_a, label, rfin)
        d_b_final = get(dist_b, label, rfin)

        gain = round(d_b_final - d_a_final, 8)
        auc_a = sum(
            float(r["mean_pairwise_distance"])
            for r in dist_a if r["label"] == label
            and not math.isnan(float(r["mean_pairwise_distance"]))
        )
        auc_b = sum(
            float(r["mean_pairwise_distance"])
            for r in dist_b if r["label"] == label
            and not math.isnan(float(r["mean_pairwise_distance"]))
        )
        d_b_r1 = get(dist_b, label, r1)
        delta_a = round(d_a_r1 - d_a_final, 8)
        delta_b = round(d_b_r1 - d_b_final, 8)
        did     = round(delta_a - delta_b, 8)

        # Representational convergence: first round where d_a <= d_a_r1 * 0.5
        threshold = d_a_r1 * 0.5
        round_repr = -1
        for rr in dist_a:
            if rr["label"] == label:
                d = float(rr["mean_pairwise_distance"])
                if not math.isnan(d) and d <= threshold:
                    round_repr = int(rr["round"])
                    break

        per_label[label] = {
            "alignment_gain":              gain,
            "AUC_003a":                    round(auc_a, 8),
            "AUC_003b":                    round(auc_b, 8),
            "difference_in_differences":   did,
            "d_003a_round1":               round(d_a_r1,    8),
            "d_003a_final":                round(d_a_final, 8),
            "d_003b_round1":               round(d_b_r1,    8),
            "d_003b_final":                round(d_b_final, 8),
            "round_representational_convergence": round_repr,
            "repr_convergence_threshold":  round(threshold, 8),
        }
        if not math.isnan(gain):
            gains.append(gain)

    mean_gain    = round(sum(gains) / len(gains), 8) if gains else float("nan")
    h1_supported = all(g > 0 for g in gains) if gains else False

    notes: list[str] = []
    for label, m in per_label.items():
        g = m["alignment_gain"]
        if g <= 0:
            notes.append(
                f"{label}: alignment_gain={g:.6f} <= 0 — "
                "feedback did not reduce inter-agent full-centroid distance."
            )
    if not notes:
        notes.append("All labels show positive alignment_gain: H1 supported.")

    return {
        "per_label":           per_label,
        "mean_alignment_gain": mean_gain,
        "h1_supported":        h1_supported,
        "notes":               notes,
    }


# ─── Step 3 — Comparison plot ─────────────────────────────────────────────────

def plot_comparison(
    io_dist_a:   list[dict],
    io_dist_b:   list[dict],
    full_dist_a: list[dict],
    full_dist_b: list[dict],
    rounds_a:    list[int],
    out_path:    Path,
) -> None:
    """
    Two-row comparison plot.
    Row 1: interaction-only centroids.
    Row 2: full operational centroids.
    Columns: slithy / mimsy / vorpal.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)

    row_data = [
        ("Interaction-only", io_dist_a,   io_dist_b),
        ("Full operational", full_dist_a, full_dist_b),
    ]

    for row_idx, (row_title, dist_a, dist_b) in enumerate(row_data):
        for col_idx, label in enumerate(CARROLL_LABELS):
            ax = axes[row_idx][col_idx]

            def pts(dist_list: list[dict], lbl: str) -> tuple[list, list]:
                subset = sorted(
                    [r for r in dist_list if r["label"] == lbl],
                    key=lambda r: int(r["round"]),
                )
                xs = [int(r["round"]) for r in subset]
                ys = [float(r["mean_pairwise_distance"]) for r in subset]
                return xs, ys

            xs_a, ys_a = pts(dist_a, label)
            xs_b, ys_b = pts(dist_b, label)

            # Align 003b to 003a x-range for visual comparison
            b_lookup = {x: y for x, y in zip(xs_b, ys_b)}
            ys_b_aligned = [b_lookup.get(x, float("nan")) for x in xs_a]

            ax.plot(xs_a, ys_a, lw=2, color="tab:blue",
                    marker="o", ms=4, label="003a feedback")
            ax.plot(xs_a, ys_b_aligned, lw=2, color="tab:orange",
                    linestyle="--", marker="s", ms=4, label="003b baseline")
            ax.fill_between(
                xs_a, ys_a, ys_b_aligned, alpha=0.13, color="green",
                where=[not math.isnan(b) and b >= a
                       for a, b in zip(ys_a, ys_b_aligned)],
            )

            ax.set_title(f"{label}" + (f"\n({row_title})" if row_idx == 0 else ""),
                         fontsize=9)
            ax.set_xlabel("Round")
            ax.set_ylabel("Mean pairwise cosine dist.")
            ax.set_xticks(xs_a)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

        # Row label
        axes[row_idx][0].set_ylabel(
            f"[{row_title}]\nMean pairwise cosine dist.", fontsize=8
        )

    fig.suptitle(
        "Inter-agent centroid distance: interaction-only vs full operational\n"
        "003a (solid) vs 003b (dashed)  |  Green shading = alignment gain",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Step 4 — Neo4j update ────────────────────────────────────────────────────

PROXIMITY_CONDITION_FULL = {
    "exp_003a": "003a_feedback_full",
    "exp_003b": "003b_nofeedback_full",
}


def tag_existing_centroids(logger: Neo4jLogger) -> None:
    """Add centroid_type='interaction_only' to all existing Centroid nodes."""
    query = (
        "MATCH (c:Centroid) "
        "WHERE c.centroid_type IS NULL "
        "SET c.centroid_type = 'interaction_only'"
    )
    logger._run(query)
    print("  Tagged existing Centroid nodes as 'interaction_only'.")


def add_full_centroid_nodes(
    logger:        Neo4jLogger,
    vectors:       dict[str, np.ndarray],
    dist_props:    dict[tuple, dict],
    experiment_id: str,
    rounds:        list[int],
) -> None:
    """Create full_operational Centroid nodes linked to existing Agent nodes."""
    rows: list[dict] = []
    for agent_id in AGENT_IDS:
        for label in CARROLL_LABELS:
            for round_num in rounds:
                k = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                vec = vectors.get(k)
                if vec is None:
                    continue
                p   = dist_props.get((agent_id, label, round_num), {})
                cid = f"full_{experiment_id}_{agent_id}_{label}_{round_num}"
                rows.append({
                    "centroid_id":            cid,
                    "experiment_id":          experiment_id,
                    "agent_id":               agent_id,
                    "label":                  label,
                    "round":                  round_num,
                    "centroid_type":          "full_operational",
                    "mean_pairwise_distance": p.get("mean_pairwise_distance", float("nan")),
                    "distance_to_agent_00":   p.get("distance_to_agent_00", float("nan")),
                    "distance_to_agent_01":   p.get("distance_to_agent_01", float("nan")),
                    "distance_to_agent_02":   p.get("distance_to_agent_02", float("nan")),
                    "vector_hash":            vector_hash(vec),
                    "vector_path":            f"centroid_vectors_full.npz:{k}",
                })

    query = (
        "UNWIND $rows AS row "
        "MATCH (a:Agent {agent_id: row.agent_id, experiment_id: $experiment_id}) "
        "MERGE (c:Centroid {centroid_id: row.centroid_id}) "
        "SET c += row "
        "MERGE (a)-[:HAS_CENTROID {round: row.round}]->(c)"
    )
    chunk = 500
    for i in range(0, len(rows), chunk):
        logger._run(query, rows=rows[i: i + chunk], experiment_id=experiment_id)
    print(f"  {len(rows)} full_operational Centroid nodes for {experiment_id}.")


def compute_proximity_props_full(
    vectors:       dict[str, np.ndarray],
    experiment_id: str,
    rounds:        list[int],
) -> dict[tuple, dict]:
    props: dict[tuple, dict] = {}
    for label in CARROLL_LABELS:
        for round_num in rounds:
            vecs: dict[str, np.ndarray | None] = {
                a: vectors.get(f"{experiment_id}_{a}_{label}_{round_num}")
                for a in AGENT_IDS
            }
            pair_dists: dict[tuple, float] = {}
            for a, b in AGENT_PAIRS:
                va, vb = vecs[a], vecs[b]
                if va is not None and vb is not None:
                    pair_dists[(a, b)] = cosine_distance(va, vb)

            for agent_id in AGENT_IDS:
                vec = vecs[agent_id]
                if vec is None:
                    continue
                dist_to: dict[str, float] = {}
                for other in AGENT_IDS:
                    if other == agent_id:
                        dist_to[f"distance_to_{other}"] = 0.0
                        continue
                    pk = (min(agent_id, other), max(agent_id, other))
                    dist_to[f"distance_to_{other}"] = round(pair_dists.get(pk, float("nan")), 8)
                valid = [v for k, v in pair_dists.items() if agent_id in k]
                mpd   = round(sum(valid) / len(valid), 8) if valid else float("nan")
                props[(agent_id, label, round_num)] = {
                    "mean_pairwise_distance": mpd,
                    **dist_to,
                }
    return props


def add_proximity_edges_full(
    logger:        Neo4jLogger,
    vectors:       dict[str, np.ndarray],
    experiment_id: str,
    rounds:        list[int],
) -> None:
    """Create PROXIMITY_TO edges between full_operational Centroid nodes."""
    condition = PROXIMITY_CONDITION_FULL[experiment_id]
    rows: list[dict] = []
    for label in CARROLL_LABELS:
        for round_num in rounds:
            for agent_a, agent_b in AGENT_PAIRS:
                ka  = f"{experiment_id}_{agent_a}_{label}_{round_num}"
                kb  = f"{experiment_id}_{agent_b}_{label}_{round_num}"
                va  = vectors.get(ka)
                vb  = vectors.get(kb)
                if va is None or vb is None:
                    continue
                cid_a = f"full_{experiment_id}_{agent_a}_{label}_{round_num}"
                cid_b = f"full_{experiment_id}_{agent_b}_{label}_{round_num}"
                rows.append({
                    "centroid_id_a":   cid_a,
                    "centroid_id_b":   cid_b,
                    "round":           round_num,
                    "label":           label,
                    "cosine_distance": round(cosine_distance(va, vb), 8),
                    "condition":       condition,
                    "experiment_id":   experiment_id,
                })

    query = (
        "UNWIND $rows AS row "
        "MATCH (c1:Centroid {centroid_id: row.centroid_id_a}) "
        "MATCH (c2:Centroid {centroid_id: row.centroid_id_b}) "
        "MERGE (c1)-[p:PROXIMITY_TO {"
        "    round: row.round, label: row.label, experiment_id: row.experiment_id, "
        "    condition: row.condition"
        "}]->(c2) "
        "SET p.cosine_distance = row.cosine_distance"
    )
    chunk = 500
    for i in range(0, len(rows), chunk):
        logger._run(query, rows=rows[i: i + chunk])
    print(f"  {len(rows)} full PROXIMITY_TO edges for {experiment_id}.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in (EXP_003A_LEDGER, EXP_003B_LEDGER):
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    # ── Encode images ─────────────────────────────────────────────────────────
    print("[Data] Loading CIFAR-10...")
    all_images = load_all_images()
    print("[Encoding] Computing DINOv2 embeddings...")
    perception = PerceptionLayer(device="cpu")
    embeddings, _ = encode_all_images(all_images, perception)
    print(f"  {len(embeddings)} embeddings ready.")

    # ── Step 1: Reconstruct full centroids ────────────────────────────────────
    print("\n[Step 1] Reconstructing full operational centroids...")

    rows_a = read_ledger(EXP_003A_LEDGER)
    rows_b = read_ledger(EXP_003B_LEDGER)

    print("  003a...")
    vecs_a = build_full_centroid_vectors(rows_a, embeddings, "exp_003a")
    print("  003b...")
    vecs_b = build_full_centroid_vectors(rows_b, embeddings, "exp_003b")

    rounds_a = sorted({int(k.split("_")[-1]) for k in vecs_a})
    print(f"  003a: {len(vecs_a)} vectors | rounds {rounds_a}")
    print(f"  003b: {len(vecs_b)} vectors (flat)")

    all_full_vecs = {**vecs_a, **vecs_b}
    npz_path = OUT_DIR / "centroid_vectors_full.npz"
    np.savez(npz_path, **all_full_vecs)
    print(f"  Saved: {npz_path.name} ({len(all_full_vecs)} vectors)")

    # Sanity: verify 003b is flat
    flat_ok = True
    for agent_id in AGENT_IDS:
        for label in CARROLL_LABELS:
            k1 = f"exp_003b_{agent_id}_{label}_1"
            k50 = f"exp_003b_{agent_id}_{label}_50"
            if k1 in all_full_vecs and k50 in all_full_vecs:
                shift = max(0.0, 1.0 - float(np.dot(all_full_vecs[k1], all_full_vecs[k50])))
                if shift > 1e-6:
                    print(f"  WARNING: {agent_id}/{label} shifted {shift:.6f} (r1→r50)")
                    flat_ok = False
    if flat_ok:
        print("  [Check] 003b full centroids are flat — OK")

    # ── Step 2: Alignment metrics ─────────────────────────────────────────────
    print("\n[Step 2] Computing alignment metrics for full centroids...")

    full_dist_a = compute_pairwise_distances(vecs_a, "exp_003a", rounds_a)
    full_dist_b = compute_pairwise_distances(vecs_b, "exp_003b", rounds_a)

    combined = full_dist_a + full_dist_b
    write_csv(combined, OUT_DIR / "inter_agent_distances_full.csv")

    metrics_full = compute_alignment_metrics(full_dist_a, full_dist_b, rounds_a)

    print("\n  Per-label (full operational):")
    for label, m in metrics_full["per_label"].items():
        print(f"    {label}: d_r1={m['d_003a_round1']:.5f}  "
              f"d_final={m['d_003a_final']:.5f}  "
              f"d_003b_final={m['d_003b_final']:.5f}  "
              f"gain={m['alignment_gain']:+.5f}")
    print(f"  Mean alignment_gain: {metrics_full['mean_alignment_gain']:+.5f}")
    print(f"  H1 supported: {metrics_full['h1_supported']}")

    summary_full = {
        "experiment":          "exp_005",
        "reconstruction":      "full_operational",
        "centroid_source":     "seed_embeddings + interaction_pool_accepted_cumulative",
        "hypothesis": {
            "H0": "Consensus feedback does not reduce inter-agent centroid distance.",
            "H1": "Consensus feedback reduces inter-agent centroid distance.",
        },
        "per_label":             metrics_full["per_label"],
        "mean_alignment_gain":   metrics_full["mean_alignment_gain"],
        "h1_supported":          metrics_full["h1_supported"],
        "notes":                 metrics_full["notes"],
    }
    write_json(summary_full, OUT_DIR / "alignment_summary_full.json")

    # ── Step 3: Comparison plot ───────────────────────────────────────────────
    print("\n[Step 3] Generating comparison plot...")

    io_csv_path = OUT_DIR / "inter_agent_distances.csv"
    if io_csv_path.exists():
        io_all = read_csv(io_csv_path)
        io_dist_a = [r for r in io_all if r["experiment_id"] == "exp_003a"]
        io_dist_b = [r for r in io_all if r["experiment_id"] == "exp_003b"]
    else:
        print("  [WARNING] inter_agent_distances.csv not found — skipping row 1")
        io_dist_a, io_dist_b = [], []

    plot_comparison(
        io_dist_a, io_dist_b,
        full_dist_a, full_dist_b,
        rounds_a,
        OUT_DIR / "centroid_alignment_comparison.png",
    )

    # ── Step 4: Neo4j ─────────────────────────────────────────────────────────
    print("\n[Step 4] Updating Neo4j...")

    props_a_full = compute_proximity_props_full(vecs_a, "exp_003a", rounds_a)
    props_b_full = compute_proximity_props_full(vecs_b, "exp_003b", rounds_a)

    with Neo4jLogger() as logger:
        print("  Tagging existing nodes as interaction_only...")
        tag_existing_centroids(logger)

        print("  Adding exp_003a full_operational Centroid nodes...")
        add_full_centroid_nodes(logger, vecs_a, props_a_full, "exp_003a", rounds_a)

        print("  Adding exp_003b full_operational Centroid nodes...")
        add_full_centroid_nodes(logger, vecs_b, props_b_full, "exp_003b", rounds_a)

        print("  Adding PROXIMITY_TO edges (full)...")
        add_proximity_edges_full(logger, vecs_a, "exp_003a", rounds_a)
        add_proximity_edges_full(logger, vecs_b, "exp_003b", rounds_a)

    print(f"\n[Done] Results in: {OUT_DIR}")
    return metrics_full


if __name__ == "__main__":
    run()
