"""
compare.py — EXP_003 Comparison: feedback vs. no-feedback.

Reads round_by_round_metrics.csv from exp_003a and exp_003b, generates
side-by-side comparison plots, and writes the summary_report.md that
answers the central question:

    Did consensus feedback actually accelerate agreement and improve
    accuracy beyond what partial shared perception alone would produce?

Run AFTER both run_003a.py and run_003b.py have completed.

Usage:
    python -m experiments.exp_003_multi_agent_consensus.compare
"""

from __future__ import annotations

import csv
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results"

DIR_003A = RESULTS_DIR / "exp_003a_consensus_feedback"
DIR_003B = RESULTS_DIR / "exp_003b_no_feedback_baseline"
DIR_COMP = RESULTS_DIR / "exp_003_comparison"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    """Load a round_by_round_metrics.csv and return as list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}\n"
                                f"Run run_003a.py and run_003b.py first.")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = int(v) if "." not in v and v.lstrip("-").isdigit() else float(v)
                except (ValueError, AttributeError):
                    converted[k] = v
            rows.append(converted)
    return rows


# ─── Comparison plots ─────────────────────────────────────────────────────────

def plot_comparison_convergence(
    rows_a: list[dict],
    rows_b: list[dict],
    out_path: Path,
) -> None:
    """
    Majority and unanimous agreement per round, 003a vs 003b.
    """
    rounds_a = [r["round"]               for r in rows_a]
    maj_a    = [r["majority_agreement"]  for r in rows_a]
    unan_a   = [r["unanimous_agreement"] for r in rows_a]

    rounds_b = [r["round"]               for r in rows_b]
    maj_b    = [r["majority_agreement"]  for r in rows_b]
    unan_b   = [r["unanimous_agreement"] for r in rows_b]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, rounds, maj, unan, label, color in [
        (axes[0], rounds_a, maj_a, unan_a, "003a — with feedback", "tab:blue"),
        (axes[1], rounds_b, maj_b, unan_b, "003b — no feedback",   "tab:orange"),
    ]:
        ax.plot(rounds, maj,  label="Majority (≥ 2/3)", lw=2, color=color)
        ax.plot(rounds, unan, label="Unanimous (3/3)",  lw=2, color=color, linestyle="--")
        ax.axhline(0.90, color="gray", linestyle=":", lw=1, label="Target 0.90")
        ax.set_xlabel("Round")
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Fraction of images")
    fig.suptitle("Convergence comparison — inter-agent agreement (interaction pool)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_comparison_accuracy(
    rows_a: list[dict],
    rows_b: list[dict],
    out_path: Path,
) -> None:
    """
    Held-out consensus accuracy per round, 003a vs 003b on same axes.
    """
    rounds_a = [r["round"]                      for r in rows_a]
    acc_a    = [r["held_out_consensus_accuracy"] for r in rows_a]

    rounds_b = [r["round"]                      for r in rows_b]
    acc_b    = [r["held_out_consensus_accuracy"] for r in rows_b]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds_a, acc_a, label="003a (with feedback)",  lw=2.5, color="tab:blue")
    ax.plot(rounds_b, acc_b, label="003b (no feedback)",    lw=2.5, color="tab:orange", linestyle="--")
    ax.axhline(0.90, color="gray", linestyle=":", lw=1, label="Target 0.90")
    ax.set_xlabel("Round")
    ax.set_ylabel("Consensus accuracy (held-out)")
    ax.set_title("Held-out consensus accuracy: 003a vs 003b")
    ax.set_ylim(-0.02, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_comparison_entropy(
    rows_a: list[dict],
    rows_b: list[dict],
    out_path: Path,
) -> None:
    rounds_a = [r["round"]        for r in rows_a]
    ent_a    = [r["mean_entropy"] for r in rows_a]

    rounds_b = [r["round"]        for r in rows_b]
    ent_b    = [r["mean_entropy"] for r in rows_b]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds_a, ent_a, label="003a (with feedback)", lw=2.5, color="tab:blue")
    ax.plot(rounds_b, ent_b, label="003b (no feedback)",   lw=2.5, color="tab:orange", linestyle="--")
    ax.axhline(0.0, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean entropy (bits)")
    ax.set_title("Label distribution entropy: 003a vs 003b")
    all_vals = ent_a + ent_b
    max_h    = max(all_vals) if all_vals else 1.585
    ax.set_ylim(-0.05, max_h * 1.15 + 0.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Summary report ───────────────────────────────────────────────────────────

def write_summary_report(
    rows_a: list[dict],
    rows_b: list[dict],
    out_path: Path,
) -> None:
    """
    The key comparison document. Answers: did feedback help?
    """
    last_a  = rows_a[-1]
    last_b  = rows_b[-1]
    first_a = rows_a[0]
    first_b = rows_b[0]

    n_rounds_a = last_a["round"]
    n_rounds_b = last_b["round"]

    # Agreement at round 1 vs final
    delta_maj_a  = last_a["majority_agreement"]  - first_a["majority_agreement"]
    delta_unan_a = last_a["unanimous_agreement"] - first_a["unanimous_agreement"]
    delta_acc_a  = last_a["held_out_consensus_accuracy"] - first_a["held_out_consensus_accuracy"]

    # Difference between 003a and 003b at same round
    common_rounds = min(n_rounds_a, n_rounds_b)
    rows_a_trunc  = rows_a[:common_rounds]
    rows_b_trunc  = rows_b[:common_rounds]

    # Improvement of 003a over 003b (final values)
    gain_maj  = last_a["majority_agreement"]  - last_b["majority_agreement"]
    gain_unan = last_a["unanimous_agreement"] - last_b["unanimous_agreement"]
    gain_acc  = last_a["held_out_consensus_accuracy"] - last_b["held_out_consensus_accuracy"]
    gain_ent  = last_b["mean_entropy"] - last_a["mean_entropy"]   # lower entropy in 003a = gain

    # Convergence speed: first round where 003a majority >= 0.90
    round_90_a = next(
        (r["round"] for r in rows_a if r["majority_agreement"] >= 0.90),
        None
    )
    round_90_b = next(
        (r["round"] for r in rows_b if r["majority_agreement"] >= 0.90),
        None
    )

    # Per-agent accuracy comparison
    agent_ids = [f"agent_{i:02d}" for i in range(3)]
    agent_rows = "\n    ".join(
        f"| {aid} | {last_b.get(f'per_agent_accuracy_{aid}', 0.0):.4f} | "
        f"{last_a.get(f'per_agent_accuracy_{aid}', 0.0):.4f} | "
        f"{last_a.get(f'per_agent_accuracy_{aid}', 0.0) - last_b.get(f'per_agent_accuracy_{aid}', 0.0):+.4f} |"
        for aid in agent_ids
    )

    # Verdict on the central question
    feedback_helped_acc     = gain_acc  > 0.01
    feedback_helped_maj     = gain_maj  > 0.01
    feedback_reduced_entropy = gain_ent > 0.001

    feedback_verdict = (
        "YES — feedback demonstrably improved agreement and/or accuracy"
        if (feedback_helped_acc or feedback_helped_maj)
        else "UNCLEAR — feedback produced marginal or no improvement over baseline"
    )
    if gain_acc < -0.01:
        feedback_verdict = "NO — feedback did not help; baseline performed comparably or better"

    report = textwrap.dedent(f"""\
    # Experiment 003 — Comparison Summary: Feedback vs. No-Feedback

    **Central question:** Did consensus feedback actually accelerate agreement
    and improve accuracy beyond what partial shared perception alone produces?

    **Verdict:** {feedback_verdict}

    ---

    ## Experimental Setup

    Both conditions: 3 agents, shared DINOv2-small, 3 Carroll labels
    (frog→slithy, horse→mimsy, ship→vorpal).

    Seed split: agent_00=[0:5], agent_01=[5:10], agent_02=[10:15] per category.
    Interaction pool: [15:30] (45 images). Held-out pool: [30:50] (60 images).

    | | 003a (feedback) | 003b (no feedback) |
    |---|---|---|
    | Feedback rule | ACCEPTED → centroid update | None (lexicons frozen) |
    | Rounds run | {n_rounds_a} | {n_rounds_b} |

    ---

    ## Key Metrics at Final Round

    | Metric | 003b baseline | 003a feedback | Gain |
    |---|---|---|---|
    | Majority agreement (interaction) | {last_b['majority_agreement']:.4f} | {last_a['majority_agreement']:.4f} | {gain_maj:+.4f} |
    | Unanimous agreement (interaction) | {last_b['unanimous_agreement']:.4f} | {last_a['unanimous_agreement']:.4f} | {gain_unan:+.4f} |
    | Held-out consensus accuracy | {last_b['held_out_consensus_accuracy']:.4f} | {last_a['held_out_consensus_accuracy']:.4f} | {gain_acc:+.4f} |
    | Mean entropy (bits) | {last_b['mean_entropy']:.4f} | {last_a['mean_entropy']:.4f} | {-gain_ent:+.4f} |
    | Label coverage | {int(last_b['label_coverage'])} | {int(last_a['label_coverage'])} | — |

    ---

    ## Per-Agent Accuracy (held-out, final round)

    | Agent | 003b baseline | 003a feedback | Gain |
    |---|---|---|---|
    {agent_rows}
    | Consensus | {last_b['held_out_consensus_accuracy']:.4f} | {last_a['held_out_consensus_accuracy']:.4f} | {gain_acc:+.4f} |

    ---

    ## Convergence Speed

    First round where majority agreement >= 0.90 (interaction pool):
    - 003a (with feedback):  {'round ' + str(round_90_a) if round_90_a else 'never reached'}
    - 003b (no feedback):    {'round ' + str(round_90_b) if round_90_b else 'never reached'}

    ---

    ## Trajectory of 003a Over Rounds

    | Metric | Round 1 | Final | Delta |
    |---|---|---|---|
    | Majority agreement | {first_a['majority_agreement']:.4f} | {last_a['majority_agreement']:.4f} | {delta_maj_a:+.4f} |
    | Unanimous agreement | {first_a['unanimous_agreement']:.4f} | {last_a['unanimous_agreement']:.4f} | {delta_unan_a:+.4f} |
    | Held-out consensus acc. | {first_a['held_out_consensus_accuracy']:.4f} | {last_a['held_out_consensus_accuracy']:.4f} | {delta_acc_a:+.4f} |
    | Mean entropy | {first_a['mean_entropy']:.4f} | {last_a['mean_entropy']:.4f} | {last_a['mean_entropy'] - first_a['mean_entropy']:+.4f} |

    ---

    ## Interpretation

    The baseline (003b) reveals the level of agreement achievable through
    shared DINOv2 perception alone, with no communication between agents.
    Since all agents use the same frozen encoder and 5-seed training sets
    of similar quality (exp_001a: accuracy=1.000 with 5 seeds), the baseline
    is expected to start high.

    The feedback condition (003a) tests whether the consensus mechanism
    adds value on top of this strong perceptual baseline. Key indicators:

    1. **Feedback helps if** 003a shows higher agreement or accuracy than
       003b, or if 003a converges faster to 0.90+ thresholds.

    2. **Feedback helps if** 003a entropy decreases over rounds while
       003b entropy stays flat.

    3. **Collective vs. individual**: if consensus accuracy > per-agent
       accuracy in 003a but not 003b, the feedback loop is producing
       emergent collective improvement.

    Gain in consensus accuracy: {gain_acc:+.4f}
    Entropy reduction from feedback: {gain_ent:+.4f} bits

    ---

    ## Implications for the Paper

    If baseline agreement (003b) is already high (> 0.90), this is not a
    null result — it is a positive structural finding: DINOv2's embedding
    space is geometrically consistent enough that independent agents
    naturally converge on the same Carroll label assignments from disjoint
    seed sets. The feedback mechanism then provides refinement on top of
    an already-functional shared vocabulary.

    If feedback shows additional gain (003a > 003b), this demonstrates
    that the consensus mechanism provides value beyond shared perception —
    the distributed protocol actively improves vocabulary alignment.

    Both outcomes are interpretable and publishable. Document accordingly
    in the paper's multi-agent results section.

    ---

    *Generated from:*
    - `results/exp_003a_consensus_feedback/round_by_round_metrics.csv`
    - `results/exp_003b_no_feedback_baseline/round_by_round_metrics.csv`
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    DIR_COMP.mkdir(parents=True, exist_ok=True)

    print("[Loading] Reading round-by-round metrics...")
    rows_a = load_csv(DIR_003A / "round_by_round_metrics.csv")
    rows_b = load_csv(DIR_003B / "round_by_round_metrics.csv")

    print(f"  003a: {len(rows_a)} rounds | 003b: {len(rows_b)} rounds")

    print("\n[Plots]")
    plot_comparison_convergence(rows_a, rows_b, DIR_COMP / "comparison_convergence.png")
    plot_comparison_accuracy(rows_a, rows_b,    DIR_COMP / "comparison_accuracy.png")
    plot_comparison_entropy(rows_a, rows_b,     DIR_COMP / "comparison_entropy.png")

    print("\n[Report]")
    write_summary_report(rows_a, rows_b, DIR_COMP / "summary_report.md")

    print(f"\n[Done] Comparison artifacts in: {DIR_COMP}")


if __name__ == "__main__":
    run()
