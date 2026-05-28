"""
run_shannon.py — Compute Shannon metrics for exp_003a and exp_003b.

Reads ledger_events.csv from both experiment results directories, computes
mutual information, conditional entropy, and normalized MI per round, then
writes JSON summary files and a combined entropy curve plot.

Two variants are computed and reported for each experiment:
  - including_uncertain: UNCERTAIN treated as a fourth effective symbol
  - carroll_only: UNCERTAIN votes dropped and distributions renormalized;
    images where all agents said UNCERTAIN are excluded entirely

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
    compute_all_rounds_carroll_only,
    grounding_threshold_round,
    entropy_reduction_curve,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

EXP_003A_DIR = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"
EXP_003B_DIR = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline"
OUT_DIR      = PROJECT_ROOT / "results" / "exp_004_neo4j_shannon"

N_AGENTS        = 3
N_CARROLL_LABELS = 3    # slithy, mimsy, vorpal
N_EFF_SYMBOLS   = 4    # slithy, mimsy, vorpal, UNCERTAIN
POOL            = "interaction"

UNCERTAIN_NOTE = (
    "UNCERTAIN is emitted when agent confidence < threshold (0.3). "
    "It is treated as a fourth symbol in Shannon calculations, "
    "which explains H(label) > log2(3) = 1.585. This is "
    "intentional — it represents genuine labeling uncertainty, "
    "not a bug."
)

SYMBOL_DECLARATION = {
    "n_carroll_labels":  N_CARROLL_LABELS,
    "n_effective_symbols": N_EFF_SYMBOLS,
    "effective_symbols": ["slithy", "mimsy", "vorpal", "UNCERTAIN"],
    "note":              UNCERTAIN_NOTE,
}


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
    rounds_a:       list[dict],
    rounds_b:       list[dict],
    carroll_a:      list[dict],
    carroll_b:      list[dict],
    threshold_a:    int,
    threshold_b:    int,
    out_path:       Path,
) -> None:
    """
    Dual-axis plot showing four entropy curves plus normalized MI.

    Primary y-axis (left):  H(label|image) — including-UNCERTAIN (solid)
                                           — Carroll-only (dotted)
    Secondary y-axis (right): normalized MI — including-UNCERTAIN (dash-dot)

    003a = blue family, 003b = orange family.
    Vertical line at grounding threshold round for each condition.

    With 3 agents the entropy takes only three discrete values:
      0.000 (unanimous), 0.918 (2/1 split), 1.585 (1/1/1 split)
    """
    def _xs(rounds): return [m["round"] for m in rounds]
    def _ys(rounds, key): return [m[key] for m in rounds]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # H(label|image) — including UNCERTAIN
    ax1.plot(_xs(rounds_a), _ys(rounds_a, "H_label_given_image"),
             lw=2, color="tab:blue",
             label="003a incl. UNCERTAIN — feedback")
    ax1.plot(_xs(rounds_b), _ys(rounds_b, "H_label_given_image"),
             lw=2, color="tab:orange", linestyle="--",
             label="003b incl. UNCERTAIN — baseline")

    # H(label|image) — Carroll-only
    ax1.plot(_xs(carroll_a), _ys(carroll_a, "H_label_given_image"),
             lw=1.5, color="tab:blue", linestyle=":",
             label="003a Carroll-only — feedback")
    ax1.plot(_xs(carroll_b), _ys(carroll_b, "H_label_given_image"),
             lw=1.5, color="tab:orange", linestyle=":",
             label="003b Carroll-only — baseline")

    # 3-agent discrete entropy reference lines
    ax1.axhline(0.9182958, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    ax1.text(_xs(rounds_a)[-1] * 1.01, 0.9182958, "2/1 split",
             fontsize=7, color="gray", va="center")
    ax1.axhline(1.5849625, color="gray", linestyle=":", lw=0.8, alpha=0.5)
    ax1.text(_xs(rounds_a)[-1] * 1.01, 1.5849625, "1/1/1 split",
             fontsize=7, color="gray", va="center")

    # normalized MI (including UNCERTAIN) — secondary axis
    ax2.plot(_xs(rounds_a), _ys(rounds_a, "normalized_mi"),
             lw=1.5, color="tab:purple", linestyle="-.",
             label="003a normalized MI")
    ax2.plot(_xs(rounds_b), _ys(rounds_b, "normalized_mi"),
             lw=1.5, color="tab:red", linestyle=":",
             label="003b normalized MI")
    ax2.axhline(0.5, color="purple", linestyle=":", lw=0.8, alpha=0.6)
    ax2.text(0.5, 0.52, "grounding threshold 0.5", fontsize=7,
             color="purple", transform=ax2.get_yaxis_transform())

    # Grounding threshold vertical markers
    if threshold_a > 0:
        ax1.axvline(threshold_a, color="tab:blue", linestyle="--", lw=1.2,
                    alpha=0.6, label=f"003a threshold @ round {threshold_a}")
    if threshold_b > 0 and threshold_b != threshold_a:
        ax1.axvline(threshold_b, color="tab:orange", linestyle="--", lw=1.2,
                    alpha=0.6, label=f"003b threshold @ round {threshold_b}")

    ax1.set_xlabel("Round")
    ax1.set_ylabel("H(label | image)  (bits)", color="black")
    ax2.set_ylabel("Normalized MI", color="tab:purple")
    ax2.set_ylim(-0.05, 1.1)
    ax1.set_ylim(-0.05, 1.9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=7.5, ncol=2)

    ax1.set_title(
        "Shannon metrics — exp_003a (feedback) vs exp_003b (baseline)\n"
        "Solid/dashed: including UNCERTAIN  |  Dotted: Carroll-only"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── JSON builders ────────────────────────────────────────────────────────────

def _shannon_block(rounds: list[dict]) -> dict:
    """Extract the key Shannon metrics from a rounds list into a summary block."""
    if not rounds:
        return {}
    last = rounds[-1]
    first = rounds[0]
    threshold = grounding_threshold_round(rounds)
    return {
        "grounding_threshold_round": threshold,
        "round_1_normalized_mi":     first["normalized_mi"],
        "final_H_label":             last["H_label"],
        "final_H_label_given_image": last["H_label_given_image"],
        "final_mutual_information":  last["mutual_information"],
        "final_normalized_mi":       last["normalized_mi"],
        "final_normalized_ce":       last["normalized_conditional_entropy"],
        "entropy_curve":             entropy_reduction_curve(rounds),
    }


def _carroll_block(rounds: list[dict]) -> dict:
    """Extract the final-round Carroll-only metrics into a compact block."""
    if not rounds:
        return {}
    last = rounds[-1]
    return {
        "H_label":                   last["H_label"],
        "H_label_given_image":       last["H_label_given_image"],
        "mutual_information":        last["mutual_information"],
        "normalized_mi":             last["normalized_mi"],
        "n_images_included":         last["n_images_included"],
        "n_images_excluded_uncertain": last["n_images_excluded_uncertain"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ledger_a_path = EXP_003A_DIR / "ledger_events.csv"
    ledger_b_path = EXP_003B_DIR / "ledger_events.csv"

    for p in (ledger_a_path, ledger_b_path):
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)

    print("[Load] Reading ledger events...")
    rows_a = read_ledger(ledger_a_path)
    rows_b = read_ledger(ledger_b_path)
    print(f"  003a: {len(rows_a)} rows | 003b: {len(rows_b)} rows")

    # ── Including-UNCERTAIN metrics ───────────────────────────────────────────
    print("\n[Shannon] Including-UNCERTAIN metrics...")
    rounds_a = compute_all_rounds(rows_a, N_AGENTS, N_EFF_SYMBOLS, pool=POOL)
    rounds_b = compute_all_rounds(rows_b, N_AGENTS, N_EFF_SYMBOLS, pool=POOL)

    # ── Carroll-only metrics ──────────────────────────────────────────────────
    print("[Shannon] Carroll-only metrics (UNCERTAIN excluded)...")
    carroll_a, _ = compute_all_rounds_carroll_only(rows_a, N_AGENTS, N_CARROLL_LABELS, pool=POOL)
    carroll_b, _ = compute_all_rounds_carroll_only(rows_b, N_AGENTS, N_CARROLL_LABELS, pool=POOL)

    # ── Grounding thresholds ──────────────────────────────────────────────────
    thresh_a = grounding_threshold_round(rounds_a)
    thresh_b = grounding_threshold_round(rounds_b)
    thresh_ca = grounding_threshold_round(carroll_a)
    thresh_cb = grounding_threshold_round(carroll_b)
    print(f"  003a thresholds: incl={thresh_a}  carroll={thresh_ca}")
    print(f"  003b thresholds: incl={thresh_b}  carroll={thresh_cb}")

    # ── Build JSON — 003a ─────────────────────────────────────────────────────
    last_a = rounds_a[-1]
    last_b = rounds_b[-1]

    summary_a = {
        "experiment": "exp_003a",
        "condition":  "with_feedback",
        "pool":       POOL,
        "n_rounds":   len(rounds_a),
        "n_agents":   N_AGENTS,
        **SYMBOL_DECLARATION,
        "shannon_including_uncertain": _shannon_block(rounds_a),
        "shannon_carroll_only":        _carroll_block(carroll_a),
        "interpretation": {
            "3_agent_entropy_reference": {
                "unanimous_3_3":    0.0,
                "majority_2_1":     0.9182958340544894,
                "unresolved_1_1_1": 1.5849625007211563,
            },
            "n_disagreements_formula": (
                "n_disagreements ≈ mean_entropy × n_images / 0.9182958"
            ),
        },
    }

    # ── Build JSON — 003b ─────────────────────────────────────────────────────
    summary_b = {
        "experiment": "exp_003b",
        "condition":  "no_feedback_baseline",
        "pool":       POOL,
        "n_rounds":   len(rounds_b),
        "n_agents":   N_AGENTS,
        **SYMBOL_DECLARATION,
        "shannon_including_uncertain": _shannon_block(rounds_b),
        "shannon_carroll_only":        _carroll_block(carroll_b),
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
        },
    }

    print("\n[Write] Saving JSON files...")
    write_json(summary_a, OUT_DIR / "shannon_metrics_003a.json")
    write_json(summary_b, OUT_DIR / "shannon_metrics_003b.json")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[Plot] Generating entropy curve (both variants)...")
    # For 003b, plot only round 1 (100 rounds are identical)
    is_flat_b = len(set(r["H_label_given_image"] for r in rounds_b)) == 1
    plot_rounds_b  = rounds_b[:1]  if is_flat_b else rounds_b
    plot_carroll_b = carroll_b[:1] if is_flat_b else carroll_b

    plot_entropy_curve(
        rounds_a, plot_rounds_b,
        carroll_a, plot_carroll_b,
        thresh_a, thresh_b,
        OUT_DIR / "entropy_curve.png",
    )

    print(f"\n[Done] Shannon metrics written to: {OUT_DIR}")


if __name__ == "__main__":
    run()
