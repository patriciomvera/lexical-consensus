"""
run_003a.py — EXP_003A: Multi-agent consensus WITH feedback.

Three agents trained on disjoint seed sets (5 images each per category)
attempt to reach a shared Carroll vocabulary through repeated labeling
rounds with consensus feedback.

Feedback rule (conservative — REJECTED means "no update", not penalty):
    ACCEPTED   — label matches majority (>= 2/3 agents): centroid update
    REJECTED   — label does not match majority:          no update
    UNRESOLVED — no majority (all three labels differ):  no update

The interaction pool [15:30] is labeled each round; the held-out pool
[30:50] is evaluated silently (no feedback, no lexicon update) to track
generalization across rounds.

Usage:
    python -m experiments.exp_003_multi_agent_consensus.run_003a
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

from _shared import (  # noqa: E402
    CATEGORIES, AGENT_SEED_SLICES, N_AGENTS, RANDOM_SEED,
    CARROLL_LABELS, CAT_TO_CARROLL, INTERACTION_SLICE, HELD_OUT_SLICE,
    CONFIDENCE_THRESHOLD, MAX_ROUNDS, STOP_UNANIMOUS_ROUNDS, SUCCESS_CRITERIA,
    load_all_images, encode_all_images, build_pools,
    label_pool, compute_consensus, compute_interaction_metrics, evaluate_held_out,
    snapshot_centroids, compute_centroid_drift,
    write_metrics_csv, write_drift_csv,
    plot_convergence_curve, plot_entropy_curve, plot_per_agent_accuracy,
)
from src.agents.learner_agent import LearnerAgent  # noqa: E402
from src.agents.perception import PerceptionLayer   # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_003a_consensus_feedback"


# ─── Seeding ──────────────────────────────────────────────────────────────────

def seed_agents(
    agents: list[LearnerAgent],
    agent_seed_hashes: dict[str, dict[str, str]],
    embeddings: dict[str, np.ndarray],
) -> None:
    """
    Assign tutor seed images to each agent via direct lexicon injection.
    Uses pre-computed embeddings — avoids re-encoding with DINOv2.
    """
    for agent in agents:
        seed_map = agent_seed_hashes[agent.agent_id]
        for img_hash, cat_name in seed_map.items():
            label = CAT_TO_CARROLL[cat_name]
            agent.lexicon.assign(embeddings[img_hash], label, round_number=0)
        print(f"  {agent.agent_id}: labels={agent.lexicon.get_labels()}, "
              f"seeds={len(seed_map)} images")


# ─── Feedback (003a only) ─────────────────────────────────────────────────────

def apply_feedback(
    agents: list[LearnerAgent],
    consensus: dict[str, dict],
    embeddings: dict[str, np.ndarray],
    round_num: int,
) -> None:
    """
    Propagate consensus results to agent lexicons.

    ACCEPTED  (agent's label == majority_label): add embedding to centroid
    REJECTED  (agent's label != majority_label): no update
    UNRESOLVED (no majority):                    no update

    Conservative design for first run: rejected labels are NOT penalized
    (no lexicon.penalize call), only correct consensus updates centroids.
    """
    for agent in agents:
        for img_hash, cons in consensus.items():
            agent_label = cons["agent_labels"].get(agent.agent_id)
            if agent_label is None or agent_label == "UNCERTAIN":
                continue

            majority = cons["majority_label"]
            if majority is not None and agent_label == majority:
                agent.lexicon.assign(embeddings[img_hash], agent_label, round_num)


# ─── Report ───────────────────────────────────────────────────────────────────

def write_report(
    config: dict,
    metrics_rows: list[dict],
    results_dir: Path,
) -> None:
    if not metrics_rows:
        return

    last      = metrics_rows[-1]
    n_rounds  = last["round"]
    agent_ids = [f"agent_{i:02d}" for i in range(N_AGENTS)]

    passed_agreement = last["inter_agent_agreement"]       >= SUCCESS_CRITERIA["inter_agent_agreement"]
    passed_accuracy  = last["held_out_consensus_accuracy"] >= SUCCESS_CRITERIA["consensus_accuracy"]
    passed_coverage  = last["label_coverage"]              == SUCCESS_CRITERIA["label_coverage"]
    verdict          = "PASS" if (passed_agreement and passed_accuracy and passed_coverage) else "PARTIAL"

    agent_acc_rows = "\n    ".join(
        f"| {aid} | {last.get(f'per_agent_accuracy_{aid}', 0.0):.4f} |"
        for aid in agent_ids
    )

    # Check if collective outperforms individual (protocol criterion)
    max_individual = max(
        last.get(f"per_agent_accuracy_{aid}", 0.0) for aid in agent_ids
    )
    consensus_beats_individual = last["held_out_consensus_accuracy"] >= max_individual

    report = textwrap.dedent(f"""\
    # Experiment 003a — Multi-Agent Consensus with Feedback

    **Date:** {config['date']}
    **Encoder:** {config['encoder']}
    **Dataset:** {config['dataset']}
    **Label assignment:** frog → slithy | horse → mimsy | ship → vorpal
    **Feedback rule:** ACCEPTED → centroid update | REJECTED/UNRESOLVED → no update

    **Seed split:**
    - agent_00: indices [0:5]   per category (5 seeds × 3 labels = 15 total)
    - agent_01: indices [5:10]  per category
    - agent_02: indices [10:15] per category

    **Interaction pool:** [15:30] per category (45 images)
    **Held-out pool:**    [30:50] per category (60 images)
    **Total rounds run:** {n_rounds}

    ---

    ## Verdict: {verdict}

    ---

    ## Final-Round Metrics

    | Metric | Value | Target | Result |
    |---|---|---|---|
    | Held-out inter-agent agreement | {last['inter_agent_agreement']:.4f} | >= 0.90 | {'PASS' if passed_agreement else 'FAIL'} |
    | Held-out consensus accuracy | {last['held_out_consensus_accuracy']:.4f} | >= 0.90 | {'PASS' if passed_accuracy else 'FAIL'} |
    | Label coverage | {last['label_coverage']} | == 3 | {'PASS' if passed_coverage else 'FAIL'} |
    | Consensus beats individual | {consensus_beats_individual} | True | {'PASS' if consensus_beats_individual else 'FAIL'} |

    **Interaction pool (final round):**
    - Majority agreement:  {last['majority_agreement']:.4f}
    - Unanimous agreement: {last['unanimous_agreement']:.4f}
    - Consensus accuracy:  {last['consensus_accuracy']:.4f}
    - Mean entropy:        {last['mean_entropy']:.4f} bits
    - Stable label count:  {last['stable_label_count']}

    ---

    ## Per-Agent Accuracy — Held-out (final round)

    | Agent | Accuracy |
    |---|---|
    {agent_acc_rows}
    | Consensus | {last['held_out_consensus_accuracy']:.4f} |

    ---

    ## Label Balance — Final Round (interaction pool)

    | Label | Category | Fraction as majority |
    |---|---|---|
    | slithy | frog  | {last['label_balance_slithy']:.4f} |
    | mimsy  | horse | {last['label_balance_mimsy']:.4f}  |
    | vorpal | ship  | {last['label_balance_vorpal']:.4f} |

    Expected: ~0.333 each (15 images per category in interaction pool).

    ---

    ## Key Question

    Did consensus feedback accelerate agreement and improve accuracy
    beyond what shared DINOv2 perception alone produces?

    Compare with: results/exp_003b_no_feedback_baseline/report.md
    Full comparison: results/exp_003_comparison/summary_report.md
    """)

    out = results_dir / "report.md"
    out.write_text(report, encoding="utf-8")
    print(f"  Saved: {out.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment":            "exp_003a_consensus_feedback",
        "date":                  datetime.now().strftime("%Y-%m-%d"),
        "encoder":               "facebook/dinov2-small",
        "dataset":               "CIFAR-10",
        "label_assignment":      CAT_TO_CARROLL,
        "agent_seed_slices":     {k: list(v) for k, v in AGENT_SEED_SLICES.items()},
        "interaction_slice":     list(INTERACTION_SLICE),
        "held_out_slice":        list(HELD_OUT_SLICE),
        "n_agents":              N_AGENTS,
        "max_rounds":            MAX_ROUNDS,
        "stop_unanimous_rounds": STOP_UNANIMOUS_ROUNDS,
        "feedback":              "accepted_centroid_update_only",
        "confidence_threshold":  CONFIDENCE_THRESHOLD,
        "random_seed":           RANDOM_SEED,
        "success_criteria":      SUCCESS_CRITERIA,
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

    print("\n[Seeding] Assigning tutor seed images...")
    seed_agents(agents, agent_seed_hashes, embeddings)

    # Round loop
    print(f"\n[Rounds] Max={MAX_ROUNDS} | "
          f"Stop: {STOP_UNANIMOUS_ROUNDS} consecutive unanimous + "
          f"held_acc≥0.90 + coverage==3")
    print("-" * 75)

    metrics_rows:  list[dict] = []
    drift_records: list[dict] = []
    stable_counts: dict       = {h: (None, 0) for h in interaction_hashes}
    prev_centroids            = snapshot_centroids(agents)
    consecutive_unanimous     = 0

    for round_num in range(1, MAX_ROUNDS + 1):
        for agent in agents:
            agent.advance_round()

        # Step 1-2: label interaction pool (read-only)
        agent_labels = label_pool(agents, interaction_hashes, embeddings)

        # Step 3: compute consensus
        consensus = compute_consensus(agent_labels)

        # Steps 4-5: feedback → lexicon update (ACCEPTED only)
        apply_feedback(agents, consensus, embeddings, round_num)

        # Step 6: evaluate held-out (no feedback, no update)
        held_out = evaluate_held_out(agents, held_out_hashes, embeddings, ground_truth)

        # Step 7: compute and log round metrics
        interaction_m, stable_counts = compute_interaction_metrics(
            consensus, ground_truth, stable_counts
        )

        curr_centroids = snapshot_centroids(agents)
        drift_records.extend(compute_centroid_drift(prev_centroids, curr_centroids, round_num))
        prev_centroids = curr_centroids

        row: dict = {
            "round":       round_num,
            **interaction_m,
            **{f"per_agent_accuracy_{aid}": acc
               for aid, acc in held_out["per_agent_accuracy"].items()},
            "held_out_consensus_accuracy": held_out["held_out_consensus_accuracy"],
            "inter_agent_agreement":       held_out["inter_agent_agreement"],
        }
        metrics_rows.append(row)

        print(
            f"  R{round_num:3d} | "
            f"major={interaction_m['majority_agreement']:.3f} "
            f"unan={interaction_m['unanimous_agreement']:.3f} | "
            f"held={held_out['held_out_consensus_accuracy']:.3f} "
            f"agree={held_out['inter_agent_agreement']:.3f} | "
            f"H={interaction_m['mean_entropy']:.3f} "
            f"cov={interaction_m['label_coverage']}"
        )

        # Stopping criterion
        if interaction_m["unanimous_agreement"] >= 1.0:
            consecutive_unanimous += 1
        else:
            consecutive_unanimous = 0

        if (consecutive_unanimous >= STOP_UNANIMOUS_ROUNDS
                and held_out["held_out_consensus_accuracy"] >= SUCCESS_CRITERIA["consensus_accuracy"]
                and interaction_m["label_coverage"] == SUCCESS_CRITERIA["label_coverage"]):
            print(f"\n[Stop] Criterion met at round {round_num} "
                  f"({consecutive_unanimous} consecutive unanimous rounds).")
            break

    # Write all artifacts
    print(f"\n[Artifacts]")
    write_metrics_csv(metrics_rows, RESULTS_DIR / "round_by_round_metrics.csv")
    write_drift_csv(drift_records,  RESULTS_DIR / "centroid_drift.csv")
    plot_convergence_curve(metrics_rows, "003a with feedback", RESULTS_DIR / "convergence_curve.png")
    plot_entropy_curve(metrics_rows,     "003a with feedback", RESULTS_DIR / "entropy_curve.png")
    plot_per_agent_accuracy(metrics_rows, "003a with feedback", RESULTS_DIR / "per_agent_accuracy.png")
    write_report(config, metrics_rows, RESULTS_DIR)

    last = metrics_rows[-1]
    print(f"\n[Summary] Rounds run: {last['round']}")
    print(f"  Majority agreement:  {last['majority_agreement']:.4f}")
    print(f"  Unanimous agreement: {last['unanimous_agreement']:.4f}")
    print(f"  Held-out accuracy:   {last['held_out_consensus_accuracy']:.4f}")
    print(f"  Label coverage:      {last['label_coverage']}")
    print(f"\n[Done] All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
