"""
run_003b.py — EXP_003B: Multi-agent consensus WITHOUT feedback (baseline).

Control condition. Identical setup to exp_003a: same agents, same seeds,
same interaction and held-out pools. The ONLY difference: agents receive
NO consensus feedback, so lexicons are frozen after the initial seed phase.

Purpose: isolate how much agreement arises from shared DINOv2 perception
alone, before attributing improvement to the feedback mechanism in 003a.

Because lexicons never update in this condition, round metrics are
deterministic (every round is identical). The flat curves are the baseline.

Usage:
    python -m experiments.exp_003_multi_agent_consensus.run_003b
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import yaml

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

from _shared import (  # noqa: E402
    CATEGORIES, AGENT_SEED_SLICES, N_AGENTS, RANDOM_SEED,
    CARROLL_LABELS, CAT_TO_CARROLL, INTERACTION_SLICE, HELD_OUT_SLICE,
    CONFIDENCE_THRESHOLD, MAX_ROUNDS, SUCCESS_CRITERIA,
    load_all_images, encode_all_images, build_pools,
    compute_consensus, compute_interaction_metrics,
    label_pool_detailed, collect_ledger_events,
    snapshot_centroids_rich, compute_centroid_drift_rich,
    write_metrics_csv, write_ledger_events_csv, write_drift_rich_csv,
    plot_convergence_curve, plot_entropy_curve, plot_per_agent_accuracy,
)
from src.agents.learner_agent import LearnerAgent  # noqa: E402
from src.agents.perception import PerceptionLayer   # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_003b_no_feedback_baseline"


# ─── Seeding (identical to 003a) ─────────────────────────────────────────────

def seed_agents(agents, agent_seed_hashes, embeddings) -> None:
    """
    Assign tutor seed images to each agent via direct lexicon injection.
    After this step, lexicons are permanently frozen (no feedback in 003b).
    """
    for agent in agents:
        seed_map = agent_seed_hashes[agent.agent_id]
        for img_hash, cat_name in seed_map.items():
            agent.lexicon.assign(embeddings[img_hash], CAT_TO_CARROLL[cat_name], round_number=0)
        print(f"  {agent.agent_id}: labels={agent.lexicon.get_labels()}, "
              f"seeds={len(seed_map)} images [FROZEN after this]")


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(
    config: dict,
    metrics_rows: list[dict],
    results_dir: Path,
) -> None:
    if not metrics_rows:
        return

    last      = metrics_rows[-1]
    first     = metrics_rows[0]
    n_rounds  = last["round"]
    agent_ids = [f"agent_{i:02d}" for i in range(N_AGENTS)]

    # Drift across rounds (max - min) to verify flatness
    acc_range = max(r["held_out_consensus_accuracy"] for r in metrics_rows) - \
                min(r["held_out_consensus_accuracy"] for r in metrics_rows)
    is_flat = acc_range < 0.001

    agent_acc_rows = "\n    ".join(
        f"| {aid} | {first.get(f'per_agent_accuracy_{aid}', 0.0):.4f} | "
        f"{last.get(f'per_agent_accuracy_{aid}', 0.0):.4f} |"
        for aid in agent_ids
    )

    report = textwrap.dedent(f"""\
    # Experiment 003b — No-Feedback Baseline

    **Date:** {config['date']}
    **Encoder:** {config['encoder']}
    **Dataset:** {config['dataset']}
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Feedback rule:** NONE — lexicons frozen after seed phase

    **Seed split (identical to 003a):**
    - agent_00: indices [0:5]   per category
    - agent_01: indices [5:10]  per category
    - agent_02: indices [10:15] per category

    **Interaction pool:** [15:30] per category (45 images)
    **Held-out pool:**    [30:50] per category (60 images)
    **Total rounds run:** {n_rounds}

    ---

    ## Round-by-round stability

    Lexicons do not update after seeding, so metrics should be identical
    across all rounds. Observed accuracy range across rounds:
    **{acc_range:.4f}** ({'flat as expected' if is_flat else 'unexpected variation'})

    ---

    ## Baseline Metrics (constant across all rounds)

    | Metric | Value |
    |---|---|
    | Majority agreement (interaction) | {first['majority_agreement']:.4f} |
    | Unanimous agreement (interaction) | {first['unanimous_agreement']:.4f} |
    | Consensus accuracy (interaction) | {first['consensus_accuracy']:.4f} |
    | Held-out inter-agent agreement | {first['inter_agent_agreement']:.4f} |
    | Held-out consensus accuracy | {first['held_out_consensus_accuracy']:.4f} |
    | Label coverage | {first['label_coverage']} |
    | Mean entropy | {first['mean_entropy']:.4f} bits |

    ---

    ## Per-Agent Accuracy — Held-out

    | Agent | Round 1 | Final |
    |---|---|---|
    {agent_acc_rows}
    | Consensus | {first['held_out_consensus_accuracy']:.4f} | {last['held_out_consensus_accuracy']:.4f} |

    ---

    ## Interpretation

    These are the agreement and accuracy levels achievable by three agents
    with disjoint 5-seed training sets and shared DINOv2 perception, with
    NO communication or feedback mechanism.

    Any improvement seen in 003a above these values is attributable to the
    consensus feedback loop, not to the shared perceptual substrate alone.

    Full comparison: results/exp_003_comparison/summary_report.md
    """)

    out = results_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"  Saved: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":        "exp_003b_no_feedback_baseline",
        "date":              datetime.now().strftime("%Y-%m-%d"),
        "encoder":           "facebook/dinov2-small",
        "dataset":           "CIFAR-10",
        "label_assignment":  CAT_TO_CARROLL,
        "agent_seed_slices": {k: list(v) for k, v in AGENT_SEED_SLICES.items()},
        "interaction_slice": list(INTERACTION_SLICE),
        "held_out_slice":    list(HELD_OUT_SLICE),
        "n_agents":          N_AGENTS,
        "max_rounds":        MAX_ROUNDS,
        "feedback":          "none",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "random_seed":       RANDOM_SEED,
        "note":              "Lexicons frozen after seed phase. All rounds are deterministic.",
    }
    with open(RESULTS_DIR / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print("[Config] Written.")

    # Load and encode all images upfront
    print("\n[Data] Loading CIFAR-10...")
    all_images = load_all_images()

    perception = PerceptionLayer(device="cpu")
    embeddings, ground_truth = encode_all_images(all_images, perception)

    interaction_hashes, held_out_hashes, agent_seed_hashes = build_pools()

    # Create agents sharing the same (already-loaded) perception layer
    agents = [LearnerAgent(f"agent_{i:02d}", perception=perception) for i in range(N_AGENTS)]

    print("\n[Seeding] Assigning tutor seed images (lexicons will be frozen)...")
    seed_agents(agents, agent_seed_hashes, embeddings)

    # Round loop — no feedback, no stopping criterion
    print(f"\n[Rounds] Running {MAX_ROUNDS} rounds (no feedback, deterministic)")
    print("-" * 75)

    metrics_rows:  list[dict] = []
    drift_records: list[dict] = []
    ledger_events: list[dict] = []
    stable_counts: dict       = {h: (None, 0) for h in interaction_hashes}
    initial_snap              = snapshot_centroids_rich(agents)
    prev_snap_rich            = initial_snap

    for round_num in range(1, MAX_ROUNDS + 1):
        for agent in agents:
            agent.advance_round()

        # Steps 1-2: label interaction pool (read-only)
        detailed_interaction = label_pool_detailed(agents, interaction_hashes, embeddings)
        agent_labels = {
            aid: {h: d["label"] for h, d in det.items()}
            for aid, det in detailed_interaction.items()
        }

        # Step 3: compute consensus for logging only — no feedback sent
        consensus = compute_consensus(agent_labels)
        ledger_events.extend(collect_ledger_events(
            "exp_003b", round_num, detailed_interaction, consensus,
            interaction_hashes, "interaction",
        ))

        # Steps 4-5: skipped — no feedback, no lexicon update

        # Step 6: evaluate held-out with details (no update)
        detailed_held = label_pool_detailed(agents, held_out_hashes, embeddings)
        agent_labels_held = {
            aid: {h: d["label"] for h, d in det.items()}
            for aid, det in detailed_held.items()
        }
        ho_consensus = compute_consensus(agent_labels_held)
        ledger_events.extend(collect_ledger_events(
            "exp_003b", round_num, detailed_held, ho_consensus,
            held_out_hashes, "held_out",
        ))

        n_held_out    = len(held_out_hashes)
        n_majority_ho = sum(1 for c in ho_consensus.values() if c["majority_label"] is not None)
        majority_correct_ho = sum(
            1 for img_hash, c in ho_consensus.items()
            if c["majority_label"] is not None
            and c["majority_label"] == ground_truth.get(img_hash)
        )
        per_agent_acc = {
            aid: round(
                sum(1 for h, lbl in labels.items() if lbl == ground_truth.get(h)) / n_held_out,
                4,
            )
            for aid, labels in agent_labels_held.items()
        }
        held_out = {
            "per_agent_accuracy":          per_agent_acc,
            "held_out_consensus_accuracy": round(
                majority_correct_ho / n_majority_ho if n_majority_ho > 0 else 0.0, 4
            ),
            "inter_agent_agreement": round(
                n_majority_ho / n_held_out if n_held_out > 0 else 0.0, 4
            ),
        }

        # Step 7: compute and log metrics
        interaction_m, stable_counts = compute_interaction_metrics(
            consensus, ground_truth, stable_counts
        )

        curr_snap_rich = snapshot_centroids_rich(agents)
        drift_records.extend(compute_centroid_drift_rich(
            prev_snap_rich, curr_snap_rich, initial_snap, round_num, "exp_003b"
        ))
        prev_snap_rich = curr_snap_rich

        row: dict = {
            "round":       round_num,
            **interaction_m,
            **{f"per_agent_accuracy_{aid}": acc
               for aid, acc in held_out["per_agent_accuracy"].items()},
            "held_out_consensus_accuracy": held_out["held_out_consensus_accuracy"],
            "inter_agent_agreement":       held_out["inter_agent_agreement"],
        }
        metrics_rows.append(row)

        # Only print every 10 rounds to keep output manageable (results are flat)
        if round_num == 1 or round_num % 10 == 0 or round_num == MAX_ROUNDS:
            print(
                f"  R{round_num:3d} | "
                f"major={interaction_m['majority_agreement']:.3f} "
                f"unan={interaction_m['unanimous_agreement']:.3f} | "
                f"held={held_out['held_out_consensus_accuracy']:.3f} "
                f"agree={held_out['inter_agent_agreement']:.3f} | "
                f"H={interaction_m['mean_entropy']:.3f} "
                f"cov={interaction_m['label_coverage']}"
            )

    # Write all artifacts
    print(f"\n[Artifacts]")
    write_metrics_csv(metrics_rows, RESULTS_DIR / "round_by_round_metrics.csv")
    write_ledger_events_csv(ledger_events, RESULTS_DIR / "ledger_events.csv")
    write_drift_rich_csv(drift_records,    RESULTS_DIR / "centroid_drift.csv")
    plot_convergence_curve(metrics_rows, "003b no feedback", RESULTS_DIR / "convergence_curve.png")
    plot_entropy_curve(metrics_rows,     "003b no feedback", RESULTS_DIR / "entropy_curve.png")
    plot_per_agent_accuracy(metrics_rows, "003b no feedback", RESULTS_DIR / "per_agent_accuracy.png")
    write_report(config, metrics_rows, RESULTS_DIR)

    first = metrics_rows[0]
    print(f"\n[Summary] All {MAX_ROUNDS} rounds completed (baseline — flat results expected)")
    print(f"  Baseline majority agreement:  {first['majority_agreement']:.4f}")
    print(f"  Baseline unanimous agreement: {first['unanimous_agreement']:.4f}")
    print(f"  Baseline held-out accuracy:   {first['held_out_consensus_accuracy']:.4f}")
    print(f"\n[Done] All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
