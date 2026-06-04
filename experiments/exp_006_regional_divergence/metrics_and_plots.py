"""
metrics_and_plots.py — exp_006: compute divergence summary and the three
prescribed plots.

Run AFTER run.py:
    python -m experiments.exp_006_regional_divergence.run
    python -m experiments.exp_006_regional_divergence.metrics_and_plots

Inputs (under results/exp_006_regional_divergence/):
  round_log_by_regime.csv
  within_cluster_metrics.csv
  between_cluster_metrics.csv
  centroid_vectors.npz
  cluster_centroid_vectors.npz

Outputs:
  divergence_summary.json
  divergence_curves.png
  divergence_comparison.png
  cluster_centroid_pca.png
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_006_regional_divergence._shared import (  # noqa: E402
    REGIMES, REGIME_CONFIG, CLUSTERS, CARROLL_LABELS, SUCCESS_CRITERIA,
)
from experiments.exp_006_regional_divergence.topology import is_bridge_round  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_006_regional_divergence"

REGIME_COLOR = {
    "vervet": "tab:green",
    "raven":  "tab:orange",
    "latin":  "tab:red",
}
LABEL_MARKER = {"slithy": "o", "mimsy": "s", "vorpal": "^"}


# ─── IO ───────────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path.name}")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {path.name}")


# ─── Summary builder ─────────────────────────────────────────────────────────

def linear_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope of ys vs xs. Returns 0 if there's no variation in x."""
    if len(xs) < 2:
        return 0.0
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(((x_centered * (y - y.mean())).sum()) / denom)


def build_divergence_summary(
    between_rows: list[dict],
    within_rows:  list[dict],
    round_rows:   list[dict],
) -> dict:
    """
    Per regime × label, compute:
      final_between_cluster_distance
      divergence_growth_rate (slope over rounds)
      within_cluster_distance_final (mean across both clusters)
      net_divergence (between - within at final round)
      held_out_accuracy_per_cluster (final round, both clusters)
    Plus a regime-level rollup and the success-criteria evaluation.
    """
    # Index helpers
    by_rl_btw: dict[tuple, list[dict]] = defaultdict(list)
    for r in between_rows:
        by_rl_btw[(r["regime"], r["label"])].append(r)
    by_rcl_win: dict[tuple, list[dict]] = defaultdict(list)
    for r in within_rows:
        by_rcl_win[(r["regime"], r["cluster_id"], r["label"])].append(r)
    by_r_round: dict[str, list[dict]] = defaultdict(list)
    for r in round_rows:
        by_r_round[r["regime"]].append(r)

    summary: dict = {
        "experiment":          "exp_006_regional_divergence",
        "spec_question":       ("Do isolated agent clusters with different input "
                                "distributions diverge lexically, or does shared "
                                "DINOv2 perception anchor them together?"),
        "regimes":             {k: v for k, v in REGIME_CONFIG.items()},
        "clusters":            {k: list(v) for k, v in CLUSTERS.items()},
        "labels":              list(CARROLL_LABELS),
        "success_criteria":    dict(SUCCESS_CRITERIA),
        "per_regime":          {},
    }

    for regime in REGIMES:
        rrounds = sorted(by_r_round[regime], key=lambda r: int(r["round"]))
        if not rrounds:
            continue
        last = rrounds[-1]
        per_label: dict[str, dict] = {}

        for label in CARROLL_LABELS:
            btw_rows = sorted(by_rl_btw[(regime, label)], key=lambda r: int(r["round"]))
            if not btw_rows:
                continue
            xs = [int(r["round"]) for r in btw_rows]
            ys = [float(r["between_cluster_distance"]) for r in btw_rows]
            slope = linear_slope(xs, ys)
            final_btw = ys[-1]
            # Mean within across both clusters at final round
            w1 = by_rcl_win[(regime, "cluster_1", label)]
            w2 = by_rcl_win[(regime, "cluster_2", label)]
            w1_final = float(sorted(w1, key=lambda r: int(r["round"]))[-1]["within_cluster_distance"]) if w1 else 0.0
            w2_final = float(sorted(w2, key=lambda r: int(r["round"]))[-1]["within_cluster_distance"]) if w2 else 0.0
            within_final = (w1_final + w2_final) / 2.0
            per_label[label] = {
                "final_between_cluster_distance": round(final_btw, 8),
                "divergence_growth_rate":         round(slope, 8),
                "within_cluster_distance_final":  round(within_final, 8),
                "net_divergence":                 round(final_btw - within_final, 8),
            }

        per_cluster_acc = {
            "cluster_1": float(last["per_cluster_acc_cluster_1"]),
            "cluster_2": float(last["per_cluster_acc_cluster_2"]),
        }
        shared_acc = float(last["shared_held_out_consensus_accuracy"])
        btw_agree  = float(last["between_cluster_label_agreement"])

        # Mean final between-cluster distance across all labels (regime rollup)
        if per_label:
            mean_final_btw = sum(v["final_between_cluster_distance"] for v in per_label.values()) / len(per_label)
            mean_growth    = sum(v["divergence_growth_rate"]         for v in per_label.values()) / len(per_label)
        else:
            mean_final_btw = 0.0
            mean_growth    = 0.0

        # Within-cluster agreement check: per cluster, accuracy >= success criterion
        within_acc_ok = all(a >= SUCCESS_CRITERIA["within_cluster_agreement_min"] for a in per_cluster_acc.values())
        shared_ok     = shared_acc >= SUCCESS_CRITERIA["shared_held_out_accuracy_min"]

        summary["per_regime"][regime] = {
            "description":                  REGIME_CONFIG[regime]["description"],
            "sigma":                        REGIME_CONFIG[regime]["sigma"],
            "bridge_every_n_rounds":        REGIME_CONFIG[regime]["bridge_every_n_rounds"],
            "per_label":                    per_label,
            "mean_final_between_cluster_distance": round(mean_final_btw, 8),
            "mean_divergence_growth_rate":  round(mean_growth, 8),
            "held_out_accuracy_per_cluster": per_cluster_acc,
            "shared_held_out_consensus_accuracy": round(shared_acc, 4),
            "between_cluster_label_agreement_final": round(btw_agree, 4),
            "within_cluster_accuracy_passes": within_acc_ok,
            "shared_held_out_passes":         shared_ok,
        }

    # Cross-regime "is there a divergence signal" check
    vervet_btw = summary["per_regime"].get("vervet", {}).get("mean_final_between_cluster_distance", 0.0)
    latin_btw  = summary["per_regime"].get("latin",  {}).get("mean_final_between_cluster_distance", 0.0)
    summary["divergence_signal"] = {
        "latin_minus_vervet_final_between_cluster":  round(latin_btw - vervet_btw, 8),
        "exceeds_threshold":                          (latin_btw - vervet_btw) >= SUCCESS_CRITERIA["divergence_signal_threshold"],
        "interpretation": (
            "If the difference exceeds the success threshold, isolation + noise + "
            "input-distribution differences produced a measurable divergence over "
            "shared perception. If it does not, shared DINOv2 perception anchored "
            "the clusters together despite isolation."
        ),
    }

    return summary


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_divergence_curves(between_rows: list[dict], out_path: Path) -> None:
    """Three subplots (one per regime). Each plots between-cluster distance
    per label over rounds, with bridge rounds marked on raven."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    for ax, regime in zip(axes, REGIMES):
        for label in CARROLL_LABELS:
            rows = sorted(
                [r for r in between_rows if r["regime"] == regime and r["label"] == label],
                key=lambda r: int(r["round"]),
            )
            if not rows:
                continue
            xs = [int(r["round"]) for r in rows]
            ys = [float(r["between_cluster_distance"]) for r in rows]
            ax.plot(xs, ys, lw=2, marker=LABEL_MARKER[label], ms=5, label=label)
        # Mark bridge rounds on raven
        if regime == "raven":
            for round_num in range(1, 26):
                if is_bridge_round(regime, round_num):
                    ax.axvline(round_num, color="black", lw=0.5, linestyle=":", alpha=0.5)
        ax.set_xlabel("Round")
        ax.set_ylabel("Between-cluster centroid distance")
        ax.set_title(f"{regime} — {REGIME_CONFIG[regime]['description']}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Between-cluster centroid distance per label per round\n"
        "Lower = clusters agree on the centroid for that Carroll label. Dotted vertical = raven bridge round.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_divergence_comparison(round_rows: list[dict], out_path: Path) -> None:
    """All three regimes on one axis: mean between-cluster distance over rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for regime in REGIMES:
        rows = sorted(
            [r for r in round_rows if r["regime"] == regime],
            key=lambda r: int(r["round"]),
        )
        xs = [int(r["round"]) for r in rows]
        ys = [float(r["mean_between_cluster_distance"]) for r in rows]
        ax.plot(xs, ys, lw=2.5, marker="o", ms=4, color=REGIME_COLOR[regime],
                label=f"{regime} (σ={REGIME_CONFIG[regime]['sigma']}, bridge={REGIME_CONFIG[regime]['bridge_every_n_rounds']})")
    # Mark raven bridge rounds
    for round_num in range(1, 26):
        if is_bridge_round("raven", round_num):
            ax.axvline(round_num, color=REGIME_COLOR["raven"], lw=0.5, linestyle=":", alpha=0.4)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean between-cluster centroid distance")
    ax.set_title(
        "Regime comparison — mean between-cluster distance over rounds\n"
        "Tests whether isolation + noise produces lexical divergence over shared DINOv2 perception."
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cluster_centroid_pca(out_path: Path) -> None:
    """
    PCA projection of all cluster-collective centroids. One scatter point
    per (regime, cluster, label, round). Color by cluster, marker by label.
    Shape per regime via subplots.
    """
    npz_path = RESULTS_DIR / "cluster_centroid_vectors.npz"
    data = np.load(npz_path)

    # Parse keys: f"{regime}_{cluster_id}_{label}_{round}"
    vectors:    list[np.ndarray] = []
    meta:       list[dict] = []
    for k in data.files:
        # cluster_id contains an underscore (e.g. "cluster_1"), so parse carefully.
        # Format: {regime}_cluster_{cluster_idx}_{label}_{round}
        parts = k.split("_")
        regime    = parts[0]
        cluster_id = f"{parts[1]}_{parts[2]}"
        label     = parts[3]
        round_num = int(parts[4])
        vectors.append(data[k])
        meta.append({"regime": regime, "cluster_id": cluster_id, "label": label, "round": round_num})

    if not vectors:
        print("  [PCA] No vectors to plot.")
        return

    X = np.stack(vectors, axis=0)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, sharex=True)
    cluster_colors = {"cluster_1": "tab:blue", "cluster_2": "tab:red"}

    for ax, regime in zip(axes, REGIMES):
        for i, m in enumerate(meta):
            if m["regime"] != regime:
                continue
            ax.scatter(
                Z[i, 0], Z[i, 1],
                color=cluster_colors[m["cluster_id"]],
                marker=LABEL_MARKER[m["label"]],
                s=40, alpha=0.6, edgecolors="none",
            )
        ax.set_xlabel(f"PC1 ({var1:.1f}%)")
        ax.set_ylabel(f"PC2 ({var2:.1f}%)")
        ax.set_title(f"{regime} — {REGIME_CONFIG[regime]['description']}", fontsize=10)
        ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cluster_colors["cluster_1"], markersize=10, label="cluster_1"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cluster_colors["cluster_2"], markersize=10, label="cluster_2"),
    ] + [
        Line2D([0], [0], marker=LABEL_MARKER[lbl], color="gray", lw=0, markersize=10, label=lbl)
        for lbl in CARROLL_LABELS
    ]
    axes[-1].legend(handles=legend, loc="best", fontsize=8)
    fig.suptitle(
        "Cluster-collective centroids in 2D PCA across all rounds\n"
        "If clusters occupy separate regions of label space → divergence. If interleaved → shared anchor.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Query CSV for Neo4j prep ─────────────────────────────────────────────────

def build_divergence_trajectory_csv(between_rows: list[dict]) -> list[dict]:
    """
    Flattened CSV with the columns most useful for downstream Neo4j load
    and the eventual exp_008 visualization: per (regime, label, round)
    the between_cluster_distance and a flag for bridge rounds.
    """
    rows: list[dict] = []
    for r in between_rows:
        rows.append({
            "regime":                   r["regime"],
            "round":                    int(r["round"]),
            "label":                    r["label"],
            "between_cluster_distance": float(r["between_cluster_distance"]),
            "is_bridge_round":          r["is_bridge_round"],
        })
    return sorted(rows, key=lambda r: (r["regime"], r["label"], r["round"]))


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] {RESULTS_DIR} missing — run run.py first.")
        sys.exit(1)

    print("[Load] Inputs from run.py ...")
    round_rows   = read_csv(RESULTS_DIR / "round_log_by_regime.csv")
    within_rows  = read_csv(RESULTS_DIR / "within_cluster_metrics.csv")
    between_rows = read_csv(RESULTS_DIR / "between_cluster_metrics.csv")
    print(f"  round_rows={len(round_rows)}, within_rows={len(within_rows)}, "
          f"between_rows={len(between_rows)}")

    print("\n[Summary]")
    summary = build_divergence_summary(between_rows, within_rows, round_rows)
    write_json(summary, RESULTS_DIR / "divergence_summary.json")

    print("\n[Trajectory CSV]")
    traj_rows = build_divergence_trajectory_csv(between_rows)
    write_csv(traj_rows, RESULTS_DIR / "query_divergence_trajectory.csv")

    print("\n[Plots]")
    plot_divergence_curves(between_rows, RESULTS_DIR / "divergence_curves.png")
    plot_divergence_comparison(round_rows, RESULTS_DIR / "divergence_comparison.png")
    plot_cluster_centroid_pca(RESULTS_DIR / "cluster_centroid_pca.png")

    print(f"\n[Done] Metrics + plots in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
