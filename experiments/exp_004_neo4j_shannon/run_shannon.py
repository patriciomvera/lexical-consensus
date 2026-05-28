"""
run_shannon.py — Compute Shannon metrics for exp_003a and exp_003b.

Reads ledger_events.csv from both experiment results directories, computes
mutual information, conditional entropy, and normalized MI per round, then
writes JSON summary files and a combined entropy curve plot.

Usage:
    python -m experiments.exp_004_neo4j_shannon.run_shannon
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.shannon import (  # noqa: E402
    compute_all_rounds,
    grounding_threshold_round,
    entropy_reduction_curve,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

EXP_003A_DIR = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"
EXP_003B_DIR = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline"
OUT_DIR      = PROJECT_ROOT / "results" / "exp_004_neo4j_shannon"

N_AGENTS  = 3
N_LABELS  = 3    # slithy, mimsy, vorpal
POOL      = "interaction"   # primary pool for Shannon analysis


# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_ledger(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(data: object, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path.name}")


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_entropy_curve(
    rounds_a: list[dict],
    rounds_b: list[dict],
    threshold_a: int,
    threshold_b: int,
    out_path: Path,
) -> None:
    """
    Dual-axis plot:
      Primary y-axis: H(label|image) per round for 003a and 003b
      Secondary y-axis: normalized_MI per round

    Vertical line at the grounding threshold round (003a).

    With 3 agents the entropy takes only three values:
      0.000 (unanimous), 0.918 (2/1 split), 1.585 (1/1/1 split)
    The dashed reference lines mark these discrete levels.
    """
    rounds_a_nums = [m["round"]              for m in rounds_a]
    h_cond_a      = [m["H_label_given_image"] for m in rounds_a]
    nmi_a         = [m["normalized_mi"]       for m in rounds_a]

    rounds_b_nums = [m["round"]              for m in rounds_b]
    h_cond_b      = [m["H_label_given_image"] for m in rounds_b]
    nmi_b         = [m["normalized_mi"]       for m in rounds_b]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # H(label|image)
    ax1.plot(rounds_a_nums, h_cond_a, lw=2, color="tab:blue",
             label="003a H(label|image) — with feedback")
    ax1.plot(rounds_b_nums, h_cond_b, lw=2, color="tab:orange", linestyle="--",
             label="003b H(label|image) — no feedback")

    # 3-agent discrete entropy reference lines
    ax1.axhline(0.9182958, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    ax1.text(rounds_a_nums[-1] * 1.01, 0.9182958, "2/1 split", fontsize=7,
             color="gray", va="center")
    ax1.axhline(1.5849625, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    ax1.text(rounds_a_nums[-1] * 1.01, 1.5849625, "1/1/1 split", fontsize=7,
             color="gray", va="center")

    # normalized MI
    ax2.plot(rounds_a_nums, nmi_a, lw=1.5, color="tab:purple", linestyle="-.",
             label="003a normalized MI")
    ax2.plot(rounds_b_nums, nmi_b, lw=1.5, color="tab:red", linestyle=":",
             label="003b normalized MI")
    ax2.axhline(0.5, color="purple", linestyle=":", lw=0.8, alpha=0.6)
    ax2.text(0.5, 0.52, "grounding threshold 0.5", fontsize=7,
             color="purple", transform=ax2.get_yaxis_transform())

    # Grounding threshold vertical marker
    if threshold_a > 0:
        ax1.axvline(threshold_a, color="tab:blue", linestyle="--", lw=1.2,
                    alpha=0.6, label=f"003a threshold @ round {threshold_a}")
    if threshold_b > 0:
        ax1.axvline(threshold_b, color="tab:orange", linestyle="--", lw=1.2,
                    alpha=0.6, label=f"003b threshold @ round {threshold_b}")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("H(label | image) (bits)", color="black")
    ax2.set_ylabel("Normalized MI", color="tab:purple")
    ax2.set_ylim(-0.05, 1.1)
    ax1.set_ylim(-0.05, 1.8)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title("Shannon metrics — exp_003a (feedback) vs exp_003b (baseline)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify inputs
    ledger_a_path = EXP_003A_DIR / "ledger_events.csv"
    ledger_b_path = EXP_003B_DIR / "ledger_events.csv"

    for p in (ledger_a_path, ledger_b_path):
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            print("Re-run the corresponding experiment first.")
            sys.exit(1)

    print("[Load] Reading ledger events...")
    rows_a = read_ledger(ledger_a_path)
    rows_b = read_ledger(ledger_b_path)
    print(f"  003a: {len(rows_a)} rows")
    print(f"  003b: {len(rows_b)} rows")

    print("\n[Shannon] Computing metrics — exp_003a (interaction pool)...")
    rounds_a = compute_all_rounds(rows_a, N_AGENTS, N_LABELS, pool=POOL)

    print("[Shannon] Computing metrics — exp_003b (interaction pool)...")
    rounds_b = compute_all_rounds(rows_b, N_AGENTS, N_LABELS, pool=POOL)

    threshold_a = grounding_threshold_round(rounds_a)
    threshold_b = grounding_threshold_round(rounds_b)
    print(f"  003a grounding threshold round: {threshold_a}")
    print(f"  003b grounding threshold round: {threshold_b}")

    # Full per-round curves
    curve_a = entropy_reduction_curve(rounds_a)
    curve_b = entropy_reduction_curve(rounds_b)

    # Summary JSON — 003a
    last_a = rounds_a[-1]
    summary_a = {
        "experiment":              "exp_003a",
        "condition":               "with_feedback",
        "pool":                    POOL,
        "n_rounds":                len(rounds_a),
        "n_agents":                N_AGENTS,
        "n_labels":                N_LABELS,
        "grounding_threshold_round": threshold_a,
        "final_H_label":           last_a["H_label"],
        "final_H_label_given_image": last_a["H_label_given_image"],
        "final_mutual_information":  last_a["mutual_information"],
        "final_normalized_mi":       last_a["normalized_mi"],
        "final_normalized_ce":       last_a["normalized_conditional_entropy"],
        "round_1_normalized_mi":   rounds_a[0]["normalized_mi"] if rounds_a else None,
        "interpretation": {
            "3_agent_entropy_reference": {
                "unanimous_3_3": 0.0,
                "majority_2_1":  0.9182958340544894,
                "unresolved_1_1_1": 1.5849625007211563,
            },
            "n_disagreements_formula": (
                "n_disagreements ≈ mean_entropy × n_images / 0.9182958"
            ),
            "entropy_curve":     curve_a,
        },
    }

    # Summary JSON — 003b
    last_b = rounds_b[-1]

    # For 003b (100 rounds all identical), report round 1 as representative
    rep_b = rounds_b[0] if rounds_b else {}
    summary_b = {
        "experiment":              "exp_003b",
        "condition":               "no_feedback_baseline",
        "pool":                    POOL,
        "n_rounds":                len(rounds_b),
        "n_agents":                N_AGENTS,
        "n_labels":                N_LABELS,
        "grounding_threshold_round": threshold_b,
        "final_H_label":           last_b["H_label"],
        "final_H_label_given_image": last_b["H_label_given_image"],
        "final_mutual_information":  last_b["mutual_information"],
        "final_normalized_mi":       last_b["normalized_mi"],
        "final_normalized_ce":       last_b["normalized_conditional_entropy"],
        "round_1_normalized_mi":   rounds_b[0]["normalized_mi"] if rounds_b else None,
        "delta_vs_003a": {
            "delta_H_label_given_image": round(
                last_b["H_label_given_image"] - last_a["H_label_given_image"], 6
            ),
            "delta_mutual_information": round(
                last_b["mutual_information"] - last_a["mutual_information"], 6
            ),
            "delta_normalized_mi": round(
                last_b["normalized_mi"] - last_a["normalized_mi"], 6
            ),
        },
        "interpretation": {
            "3_agent_entropy_reference": {
                "unanimous_3_3":    0.0,
                "majority_2_1":     0.9182958340544894,
                "unresolved_1_1_1": 1.5849625007211563,
            },
            "n_disagreements_formula": (
                "n_disagreements ≈ mean_entropy × n_images / 0.9182958"
            ),
            "entropy_curve": curve_b[:5],  # first 5 rounds (rest are identical)
        },
    }

    print("\n[Write] Saving JSON files...")
    write_json(summary_a, OUT_DIR / "shannon_metrics_003a.json")
    write_json(summary_b, OUT_DIR / "shannon_metrics_003b.json")

    print("\n[Plot] Generating entropy curve...")
    # For 003b, only plot first round (it's flat)
    plot_b_rounds = rounds_b[:1] if len(set(r["H_label_given_image"] for r in rounds_b)) == 1 else rounds_b
    plot_entropy_curve(
        rounds_a, plot_b_rounds,
        threshold_a, threshold_b,
        OUT_DIR / "entropy_curve.png",
    )

    print(f"\n[Done] Shannon metrics written to: {OUT_DIR}")


if __name__ == "__main__":
    run()
