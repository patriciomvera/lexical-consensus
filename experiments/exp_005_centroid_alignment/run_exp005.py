"""
run_exp005.py — exp_005 Parts A, B, C: centroid alignment metrics and
temporal analysis for the Sapir-Whorf analogue experiment.

Loads centroid_vectors.npz (produced by reconstruct_centroids.py) and
computes inter-agent pairwise centroid distances per label per round for
both exp_003a (feedback) and exp_003b (no-feedback baseline).

Parts A: Alignment metrics — pairwise distances, alignment_gain, AUC, DiD
Parts B: Hypothesis test — H0 vs H1 per label
Parts C: Temporal ordering — lexical vs representational convergence

Usage:
    python -m experiments.exp_005_centroid_alignment.run_exp005
"""

from __future__ import annotations

import csv
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

EXP_003A_DIR = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"
EXP_003B_DIR = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline"
OUT_DIR      = PROJECT_ROOT / "results" / "exp_005_centroid_alignment"

AGENT_IDS     = ["agent_00", "agent_01", "agent_02"]
AGENT_PAIRS   = list(combinations(AGENT_IDS, 2))
CARROLL_LABELS = ["slithy", "mimsy", "vorpal"]
PAIR_KEYS      = [f"{a}_{b}" for a, b in AGENT_PAIRS]  # 00_01, 00_02, 01_02


# ─── Helpers ──────────────────────────────────────────────────────────────────

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cos(u, v). Both vectors assumed unit-normed."""
    return float(max(0.0, 1.0 - np.dot(u, v)))


def read_ledger(path: Path) -> list[dict]:
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


# ─── Distance computation ──────────────────────────────────────────────────────

def compute_pairwise_distances(
    vectors: dict[str, np.ndarray],
    experiment_id: str,
    rounds: list[int],
) -> list[dict]:
    """
    Compute pairwise cosine distances between all agent pairs for each
    (label, round) in this experiment.

    Returns a list of flat dicts suitable for CSV export.
    """
    rows: list[dict] = []
    for label in CARROLL_LABELS:
        for round_num in rounds:
            agent_vecs: dict[str, np.ndarray | None] = {}
            for agent_id in AGENT_IDS:
                k = f"{experiment_id}_{agent_id}_{label}_{round_num}"
                agent_vecs[agent_id] = vectors.get(k)

            dists: dict[str, float] = {}
            for a, b in AGENT_PAIRS:
                va, vb = agent_vecs.get(a), agent_vecs.get(b)
                if va is not None and vb is not None:
                    dists[f"d_{a[-2:]}_{b[-2:]}"] = round(cosine_distance(va, vb), 8)
                else:
                    dists[f"d_{a[-2:]}_{b[-2:]}"] = float("nan")

            valid = [v for v in dists.values() if not math.isnan(v)]
            mean_d = round(sum(valid) / len(valid), 8) if valid else float("nan")

            rows.append({
                "experiment_id":   experiment_id,
                "label":           label,
                "round":           round_num,
                "d_00_01":         dists.get("d_00_01", float("nan")),
                "d_00_02":         dists.get("d_00_02", float("nan")),
                "d_01_02":         dists.get("d_01_02", float("nan")),
                "mean_pairwise_distance": mean_d,
            })
    return rows


# ─── Lexical convergence round ────────────────────────────────────────────────

def compute_lexical_consensus_rounds(
    ledger_rows: list[dict],
) -> dict[str, int]:
    """
    For each Carroll label, find the first round where all 3 agents assign
    that label unanimously to every interaction-pool image of the matching
    category. Returns -1 if never reached.

    Mapping: slithy=frog, mimsy=horse, vorpal=ship
    """
    cat_to_label = {"frog": "slithy", "horse": "mimsy", "ship": "vorpal"}
    label_to_cat = {v: k for k, v in cat_to_label.items()}

    # Interaction pool only
    ia = [r for r in ledger_rows if r["pool"] == "interaction"]

    # (round, image_id) -> {agent_id: predicted_label}
    from collections import defaultdict
    vote_map: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in ia:
        vote_map[(int(r["round"]), r["image_id"])][r["agent_id"]] = r["predicted_label"]

    all_rounds = sorted({int(r["round"]) for r in ia})

    # Interaction images per category
    cat_images: dict[str, set[str]] = defaultdict(set)
    for r in ia:
        cat_images[r["true_category"]].add(r["image_id"])

    result: dict[str, int] = {}
    for label in CARROLL_LABELS:
        cat = label_to_cat[label]
        images = cat_images.get(cat, set())
        found = -1
        for round_num in all_rounds:
            all_unanimous = True
            for img in images:
                votes = vote_map.get((round_num, img), {})
                labels_assigned = set(votes.values())
                if len(labels_assigned) != 1 or list(labels_assigned)[0] != label:
                    all_unanimous = False
                    break
            if all_unanimous and images:
                found = round_num
                break
        result[label] = found

    return result


# ─── Alignment metrics ─────────────────────────────────────────────────────────

def auc_distance_curve(dist_rows: list[dict], label: str) -> float:
    """Sum of mean_pairwise_distance over all rounds for a given label."""
    values = [
        r["mean_pairwise_distance"]
        for r in dist_rows
        if r["label"] == label and not math.isnan(r["mean_pairwise_distance"])
    ]
    return round(sum(values), 8)


def alignment_metrics(
    dist_a: list[dict],
    dist_b: list[dict],
    rounds_a: list[int],
    rounds_b: list[int],
    lexical_rounds: dict[str, int],
) -> dict:
    """
    Compute per-label alignment metrics.

    alignment_gain(label) = mean_pairwise_distance_003b_final
                           - mean_pairwise_distance_003a_final
    If positive: feedback produced greater representational alignment.

    difference_in_differences(label):
      (d_003a_round1 - d_003a_final) - (d_003b_round1 - d_003b_final)

    round_representational_convergence: first round where
      mean_pairwise_distance <= initial_distance * 0.5
    """
    def get_dist(rows: list[dict], label: str, round_num: int) -> float:
        for r in rows:
            if r["label"] == label and r["round"] == round_num:
                return r["mean_pairwise_distance"]
        return float("nan")

    results: dict[str, dict] = {}
    gains: list[float] = []

    for label in CARROLL_LABELS:
        r1_a   = min(rounds_a)
        rfin_a = max(rounds_a)
        r1_b   = min(rounds_b)
        rfin_b = max(rounds_b)

        d_a_r1    = get_dist(dist_a, label, r1_a)
        d_a_final = get_dist(dist_a, label, rfin_a)
        d_b_r1    = get_dist(dist_b, label, r1_b)
        d_b_final = get_dist(dist_b, label, rfin_b)

        alignment_gain = round(d_b_final - d_a_final, 8) if not math.isnan(d_b_final + d_a_final) else float("nan")

        # Difference in differences
        delta_a = (d_a_r1 - d_a_final) if not math.isnan(d_a_r1 + d_a_final) else float("nan")
        delta_b = (d_b_r1 - d_b_final) if not math.isnan(d_b_r1 + d_b_final) else float("nan")
        did = round(delta_a - delta_b, 8) if not math.isnan(delta_a + delta_b) else float("nan")

        # AUC
        auc_a = auc_distance_curve(dist_a, label)
        auc_b = auc_distance_curve(dist_b, label)

        # Representational convergence: first round where d_003a <= d_a_r1 * 0.5
        threshold_repr = d_a_r1 * 0.5 if not math.isnan(d_a_r1) else float("nan")
        round_repr = -1
        for r in dist_a:
            if r["label"] == label and not math.isnan(r["mean_pairwise_distance"]):
                if r["mean_pairwise_distance"] <= threshold_repr:
                    round_repr = int(r["round"])
                    break

        lex_round  = lexical_rounds.get(label, -1)
        repr_round = round_repr

        if lex_round == -1 and repr_round == -1:
            convergence_order = "neither_reached"
        elif lex_round == -1:
            convergence_order = "representational_first"
        elif repr_round == -1:
            convergence_order = "lexical_first"
        elif repr_round < lex_round:
            convergence_order = "representational_first"
        elif lex_round < repr_round:
            convergence_order = "lexical_first"
        else:
            convergence_order = "simultaneous"

        results[label] = {
            "alignment_gain":                      alignment_gain,
            "AUC_003a":                            auc_a,
            "AUC_003b":                            auc_b,
            "difference_in_differences":           did,
            "d_003a_round1":                       round(d_a_r1,    8),
            "d_003a_final":                        round(d_a_final, 8),
            "d_003b_round1":                       round(d_b_r1,    8),
            "d_003b_final":                        round(d_b_final, 8),
            "round_lexical_consensus":             lex_round,
            "round_representational_convergence":  repr_round,
            "repr_convergence_threshold":          round(threshold_repr, 8) if not math.isnan(threshold_repr) else -1,
            "convergence_order":                   convergence_order,
        }
        if not math.isnan(alignment_gain):
            gains.append(alignment_gain)

    mean_gain = round(sum(gains) / len(gains), 8) if gains else float("nan")
    h1_supported = all(g > 0 for g in gains) if gains else False

    notes: list[str] = []
    for label, m in results.items():
        if m["alignment_gain"] <= 0:
            notes.append(
                f"{label}: alignment_gain={m['alignment_gain']:.6f} <= 0 — "
                "feedback did not reduce inter-agent distance for this label."
            )
    if not notes:
        notes.append("All labels show positive alignment_gain: H1 supported.")

    return {
        "per_label":         results,
        "mean_alignment_gain": mean_gain,
        "h1_supported":      h1_supported,
        "notes":             notes,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_alignment_curve(
    dist_a: list[dict],
    dist_b: list[dict],
    rounds_a: list[int],
    metrics: dict,
    out_path: Path,
) -> None:
    """Three-subplot alignment curve: one per Carroll label."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    colors = {"003a": "tab:blue", "003b": "tab:orange"}

    for ax, label in zip(axes, CARROLL_LABELS):
        rows_a_lbl = sorted(
            [r for r in dist_a if r["label"] == label], key=lambda r: r["round"]
        )
        rows_b_lbl = sorted(
            [r for r in dist_b if r["label"] == label], key=lambda r: r["round"]
        )

        xs_a  = [r["round"] for r in rows_a_lbl]
        ys_a  = [r["mean_pairwise_distance"] for r in rows_a_lbl]
        # For 003b, use rounds matching 003a's x-range for visual comparison
        b_at_a_rounds = {r["round"]: r["mean_pairwise_distance"] for r in rows_b_lbl}
        ys_b = [b_at_a_rounds.get(rn, float("nan")) for rn in xs_a]

        ax.plot(xs_a, ys_a, lw=2, color=colors["003a"],
                marker="o", ms=5, label="003a (feedback)")
        ax.plot(xs_a, ys_b, lw=2, color=colors["003b"],
                marker="s", ms=5, linestyle="--", label="003b (baseline)")

        # Shade alignment gain region (between the two curves)
        ax.fill_between(xs_a, ys_a, ys_b, alpha=0.15, color="green",
                        where=[b >= a for a, b in zip(ys_a, ys_b)],
                        label="alignment gain")

        # Vertical markers
        m = metrics["per_label"][label]
        lex_r  = m["round_lexical_consensus"]
        repr_r = m["round_representational_convergence"]
        if lex_r > 0:
            ax.axvline(lex_r, color="tab:blue", linestyle=":", lw=1.5,
                       alpha=0.8, label=f"lexical consensus r{lex_r}")
        if repr_r > 0:
            ax.axvline(repr_r, color="tab:green", linestyle="-.", lw=1.5,
                       alpha=0.8, label=f"repr. conv. r{repr_r}")

        gain = m["alignment_gain"]
        ax.set_title(
            f"{label}\nalignment_gain={gain:+.5f}",
            fontsize=10,
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean pairwise cosine distance")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xticks(xs_a)

    fig.suptitle(
        "Inter-agent centroid distance: 003a (feedback) vs 003b (baseline)\n"
        "Positive alignment_gain = feedback reduced representational distance",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_alignment_gain_bar(metrics: dict, out_path: Path) -> None:
    """Bar chart of alignment_gain per label."""
    labels = CARROLL_LABELS
    gains  = [metrics["per_label"][lbl]["alignment_gain"] for lbl in labels]
    colors = ["tab:green" if g > 0 else "tab:red" for g in gains]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, gains, color=colors, width=0.5, edgecolor="black", lw=0.8)
    ax.axhline(0, color="black", lw=1.2)

    for bar, val in zip(bars, gains):
        yoff = 0.0002 if val >= 0 else -0.0002
        va   = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + yoff,
                f"{val:+.5f}", ha="center", va=va, fontsize=9, fontweight="bold")

    mean_g = metrics["mean_alignment_gain"]
    ax.axhline(mean_g, color="gray", linestyle="--", lw=1,
               label=f"mean gain = {mean_g:+.5f}")
    ax.set_title(
        "Alignment gain per Carroll label\n"
        "(003b_final − 003a_final mean pairwise distance)"
    )
    ax.set_ylabel("Alignment gain (cosine distance)")
    ax.set_xlabel("Carroll label")
    ax.legend(fontsize=9)
    h1 = metrics["h1_supported"]
    verdict = "H1 SUPPORTED" if h1 else "H0 NOT REJECTED"
    ax.text(0.98, 0.02, verdict, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            color="darkgreen" if h1 else "darkred", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_pca_trajectories(
    vectors: dict[str, np.ndarray],
    rounds_a: list[int],
    out_path: Path,
) -> None:
    """
    PCA projection of 384-dim centroid vectors to 2D.
    Two subplots: 003a trajectory (round 1 -> final) and 003b (single point).
    Color by label, marker by agent.
    """
    label_colors = {"slithy": "tab:blue", "mimsy": "tab:orange", "vorpal": "tab:green"}
    agent_markers = {"agent_00": "o", "agent_01": "s", "agent_02": "^"}
    agent_labels  = {"agent_00": "agent 00", "agent_01": "agent 01", "agent_02": "agent 02"}

    # Collect all vectors for PCA fitting
    all_vecs: list[np.ndarray] = []
    all_meta: list[tuple] = []  # (exp, agent, label, round)

    for exp_id, rounds in [("exp_003a", rounds_a), ("exp_003b", [1])]:
        for agent_id in AGENT_IDS:
            for label in CARROLL_LABELS:
                for round_num in rounds:
                    k = f"{exp_id}_{agent_id}_{label}_{round_num}"
                    if k in vectors:
                        all_vecs.append(vectors[k])
                        all_meta.append((exp_id, agent_id, label, round_num))

    if not all_vecs:
        print("  [PCA] No vectors found — skipping PCA plot.")
        return

    X = np.stack(all_vecs, axis=0)
    pca = PCA(n_components=2)
    X2  = pca.fit_transform(X)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (exp_id, title, exp_rounds) in zip(axes, [
        ("exp_003a", "003a — feedback (rounds 1→6)", rounds_a),
        ("exp_003b", "003b — baseline (round 1, frozen)",  [1]),
    ]):
        # Plot per (agent, label)
        for agent_id in AGENT_IDS:
            for label in CARROLL_LABELS:
                pts = [
                    (X2[i], meta[3])
                    for i, meta in enumerate(all_meta)
                    if meta[0] == exp_id and meta[1] == agent_id and meta[2] == label
                ]
                if not pts:
                    continue
                pts.sort(key=lambda x: x[1])  # sort by round
                coords = np.array([p[0] for p in pts])
                rnds   = [p[1] for p in pts]

                color  = label_colors[label]
                marker = agent_markers[agent_id]

                if exp_id == "exp_003a" and len(coords) > 1:
                    # Draw trajectory line
                    ax.plot(coords[:, 0], coords[:, 1],
                            color=color, lw=1.2, alpha=0.6)
                    # Start and end markers
                    ax.scatter(coords[0, 0], coords[0, 1],
                               color=color, marker=marker, s=60, zorder=4, alpha=0.7)
                    ax.scatter(coords[-1, 0], coords[-1, 1],
                               color=color, marker=marker, s=120, zorder=5, edgecolors="black", lw=0.8)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1],
                               color=color, marker=marker, s=80, zorder=4, alpha=0.8)

        # Legend patches
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=label_colors[lbl], lw=2, label=lbl)
            for lbl in CARROLL_LABELS
        ] + [
            Line2D([0], [0], color="gray", marker=agent_markers[aid], lw=0,
                   markersize=8, label=agent_labels[aid])
            for aid in AGENT_IDS
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        "PCA projection of 384-dim centroid vectors\n"
        "Larger dots = final round | Arrows = trajectory direction",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    npz_path = OUT_DIR / "centroid_vectors.npz"
    if not npz_path.exists():
        print(f"[ERROR] Missing: {npz_path}")
        print("  Run reconstruct_centroids.py first.")
        sys.exit(1)

    print("[Load] centroid_vectors.npz...")
    data = np.load(npz_path)
    vectors: dict[str, np.ndarray] = dict(data)
    print(f"  {len(vectors)} vectors loaded.")

    # Determine rounds from keys
    rounds_a = sorted({int(k.split("_")[-1]) for k in vectors if k.startswith("exp_003a")})
    rounds_b = sorted({int(k.split("_")[-1]) for k in vectors if k.startswith("exp_003b")})
    print(f"  003a rounds: {rounds_a}")
    print(f"  003b rounds: {len(rounds_b)} (first={rounds_b[0]}, last={rounds_b[-1]})")

    # Compute pairwise distances — use rounds_a x-range for both (for comparison)
    print("\n[Distances] Computing pairwise distances...")
    dist_a = compute_pairwise_distances(vectors, "exp_003a", rounds_a)
    dist_b = compute_pairwise_distances(vectors, "exp_003b", rounds_a)  # same x-range

    # Write inter_agent_distances.csv (combined)
    combined_dists = dist_a + dist_b
    write_csv(combined_dists, OUT_DIR / "inter_agent_distances.csv")

    # Load ledger for lexical convergence
    print("\n[Lexical] Computing lexical consensus rounds...")
    ledger_a = read_ledger(EXP_003A_DIR / "ledger_events.csv")
    lex_rounds = compute_lexical_consensus_rounds(ledger_a)
    print(f"  Lexical consensus rounds per label: {lex_rounds}")

    # Alignment metrics (Parts A, B, C)
    print("\n[Metrics] Computing alignment metrics...")
    metrics = alignment_metrics(dist_a, dist_b, rounds_a, rounds_a, lex_rounds)

    print(f"\n  Per-label results:")
    for label, m in metrics["per_label"].items():
        print(f"    {label}: gain={m['alignment_gain']:+.6f}  "
              f"lex_r={m['round_lexical_consensus']}  "
              f"repr_r={m['round_representational_convergence']}  "
              f"order={m['convergence_order']}")
    print(f"\n  Mean alignment_gain: {metrics['mean_alignment_gain']:+.6f}")
    print(f"  H1 supported: {metrics['h1_supported']}")

    # Write alignment_summary.json
    summary = {
        "experiment":        "exp_005",
        "condition_a":       "exp_003a_with_feedback",
        "condition_b":       "exp_003b_no_feedback_baseline",
        "n_rounds_a":        len(rounds_a),
        "n_rounds_b":        len(rounds_b),
        "centroid_source":   "interaction_pool_accepted_cumulative",
        "hypothesis": {
            "H0": "Consensus feedback does not reduce inter-agent centroid distance relative to the no-feedback baseline.",
            "H1": "Consensus feedback reduces inter-agent centroid distance relative to the no-feedback baseline.",
        },
        "per_label":             metrics["per_label"],
        "mean_alignment_gain":   metrics["mean_alignment_gain"],
        "h1_supported":          metrics["h1_supported"],
        "notes":                 metrics["notes"],
        "3_agent_pairs": [f"{a[-2:]}_{b[-2:]}" for a, b in [("agent_00","agent_01"),("agent_00","agent_02"),("agent_01","agent_02")]],
    }
    write_json(summary, OUT_DIR / "alignment_summary.json")

    # Plots
    print("\n[Plots]")
    plot_alignment_curve(
        dist_a, dist_b, rounds_a, metrics,
        OUT_DIR / "centroid_alignment_curve.png",
    )
    plot_alignment_gain_bar(metrics, OUT_DIR / "alignment_gain_by_label.png")
    plot_pca_trajectories(vectors, rounds_a, OUT_DIR / "pca_centroid_trajectories.png")

    print(f"\n[Done] Results in: {OUT_DIR}")


if __name__ == "__main__":
    run()
