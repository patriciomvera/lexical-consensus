"""
metrics_and_plots.py — exp_005b: compute inter-agent distances in raw and
projected spaces, write CSV / JSON artifacts, and generate the four plots
specified by the experiment plan.

Run AFTER run.py so the centroid .npz files and round logs exist:

    python -m experiments.exp_005b_language_conditioned_geometry.run
    python -m experiments.exp_005b_language_conditioned_geometry.metrics_and_plots

Inputs (under results/exp_005b_language_conditioned_geometry/):
  round_log_by_condition.csv
  adapter_drift.csv
  projected_centroid_vectors.npz
  raw_centroid_vectors.npz

Outputs:
  inter_agent_distances_raw.csv
  inter_agent_distances_projected.csv
  held_out_accuracy_by_round.csv
  alignment_summary.json
  projected_alignment_curve.png
  raw_vs_projected_alignment.png
  adapter_drift.png
  accuracy_by_condition.png

Note on the metric: "projected centroid" here means the full operational
centroid in projected space (seeds + accepted interaction images, projected
through the agent's current W). The spec calls out projected and full
operational as separate quantities; in our implementation they are the same
because the centroid is always built from the agent's full accepted set,
which always includes the tutor seeds. The accepted-image set varies between
conditions (different W gives different consensus), so the raw centroid is
not identical across conditions either — it's only frozen in the sense that
the DINOv2 encoder never changes.
"""

from __future__ import annotations

import csv
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

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.exp_005b_language_conditioned_geometry._shared import (  # noqa: E402
    AGENT_IDS, CARROLL_LABELS, CONDITIONS, CONDITION_LABELS,
    SUCCESS_CRITERIA,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_005b_language_conditioned_geometry"

AGENT_PAIRS = list(combinations(AGENT_IDS, 2))

COND_COLOR = {
    "frozen":    "tab:gray",
    "consensus": "tab:blue",
    "random":    "tab:orange",
}


# ─── IO ───────────────────────────────────────────────────────────────────────

def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
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


# ─── Distance computation ─────────────────────────────────────────────────────

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.dot(u, v)))


def compute_inter_agent_distances(
    vectors: dict[str, np.ndarray],
    space_name: str,
) -> list[dict]:
    """
    For each (condition, label, round), compute pairwise cosine distances
    between all 3 agent pairs and their mean. Emits one row per
    (condition, label, round). Skips any (condition, label, round)
    where fewer than 2 agents have a centroid.

    The npz keys are formatted `{condition}_{agent_id}_{label}_{round}` —
    see run.py for the source of truth on the key layout.
    """
    # Parse keys into a nested structure (condition, label, round) -> {agent: vec}
    by_key: dict[tuple, dict[str, np.ndarray]] = defaultdict(dict)
    for k, vec in vectors.items():
        parts = k.split("_")
        # parts: [condition, "agent", "00", label, round]  (5)
        # Reconstruct robustly: condition is always first token, agent_id is
        # always "agent_NN", label is one of CARROLL_LABELS, round is last.
        condition = parts[0]
        agent_id  = f"{parts[1]}_{parts[2]}"
        label     = parts[3]
        round_num = int(parts[4])
        by_key[(condition, label, round_num)][agent_id] = vec

    rows: list[dict] = []
    for (condition, label, round_num), agent_vecs in sorted(by_key.items()):
        pairs: dict[str, float] = {}
        for a, b in AGENT_PAIRS:
            va, vb = agent_vecs.get(a), agent_vecs.get(b)
            if va is None or vb is None:
                pairs[f"d_{a[-2:]}_{b[-2:]}"] = float("nan")
            else:
                pairs[f"d_{a[-2:]}_{b[-2:]}"] = round(cosine_distance(va, vb), 8)
        valid = [v for v in pairs.values() if not math.isnan(v)]
        mean_d = round(sum(valid) / len(valid), 8) if valid else float("nan")
        rows.append({
            "space":     space_name,
            "condition": condition,
            "label":     label,
            "round":     round_num,
            **pairs,
            "mean_pairwise_distance": mean_d,
        })
    return rows


# ─── Held-out accuracy reshape ────────────────────────────────────────────────

def extract_held_out_accuracy(round_rows: list[dict]) -> list[dict]:
    """Pull held_out_consensus_accuracy per (condition, round) from round_log."""
    out = []
    for r in round_rows:
        out.append({
            "condition":                  r["condition"],
            "round":                      int(r["round"]),
            "held_out_consensus_accuracy": float(r["held_out_consensus_accuracy"]),
            "majority_agreement":          float(r["majority_agreement"]),
            "unanimous_agreement":         float(r["unanimous_agreement"]),
            "mean_entropy":                float(r["mean_entropy"]),
            "label_coverage":              int(r["label_coverage"]),
        })
    return out


# ─── alignment_summary.json ───────────────────────────────────────────────────

def _per_label_rows(rows: list[dict], condition: str, label: str) -> list[dict]:
    return sorted(
        [r for r in rows if r["condition"] == condition and r["label"] == label],
        key=lambda r: r["round"],
    )


def _first_last_mean_distance(
    rows: list[dict],
    condition: str,
    label: str,
) -> tuple[float, float]:
    pl = _per_label_rows(rows, condition, label)
    if not pl:
        return float("nan"), float("nan")
    return float(pl[0]["mean_pairwise_distance"]), float(pl[-1]["mean_pairwise_distance"])


def _final_held_out_accuracy(round_rows: list[dict], condition: str) -> float:
    rows = sorted(
        [r for r in round_rows if r["condition"] == condition],
        key=lambda r: int(r["round"]),
    )
    if not rows:
        return float("nan")
    return float(rows[-1]["held_out_consensus_accuracy"])


def _final_max_label_fraction(round_rows: list[dict], condition: str) -> float:
    """
    Highest single-label fraction at the final round. Used by the success
    criterion `max_label_fraction <= 0.50` (no label collapse).
    """
    rows = sorted(
        [r for r in round_rows if r["condition"] == condition],
        key=lambda r: int(r["round"]),
    )
    if not rows:
        return float("nan")
    last = rows[-1]
    return max(
        float(last["label_balance_slithy"]),
        float(last["label_balance_mimsy"]),
        float(last["label_balance_vorpal"]),
    )


def _final_label_coverage(round_rows: list[dict], condition: str) -> int:
    rows = sorted(
        [r for r in round_rows if r["condition"] == condition],
        key=lambda r: int(r["round"]),
    )
    if not rows:
        return 0
    return int(rows[-1]["label_coverage"])


def _mean_drift(drift_rows: list[dict], condition: str) -> float:
    """Mean frobenius_drift_from_initial across all (agent, round) rows for this condition."""
    vals = [
        float(r["frobenius_drift_from_initial"])
        for r in drift_rows
        if r["condition"] == condition
    ]
    return round(sum(vals) / len(vals), 6) if vals else 0.0


def build_alignment_summary(
    proj_dist_rows: list[dict],
    raw_dist_rows:  list[dict],
    round_rows:     list[dict],
    drift_rows:     list[dict],
) -> dict:
    """
    Build alignment_summary.json. Per spec, contains per-condition per-label:
      alignment_gain_projected
      alignment_gain_raw
      consensus_specific_gain (B - C in projected space)
      distortion_penalty (mean frobenius drift)
      held_out_accuracy_final
      h1_supported (B only — meets all 5 success criteria)

    For A and C the consensus_specific_gain field is reported as the same
    arithmetic (this_condition - random_baseline_C in projected space),
    which is 0 for C itself and is informative for A vs C.

    Methodological note on the success criteria. The spec phrases its first
    two criteria as "projected distance decreases more than Condition A /
    Condition C". A naive reading interprets "decreases more" as the largest
    `alignment_gain = d_round1 - d_final`. That measure has a known artifact
    in our setup: Adam's first step on B / C perturbs W enough to inflate
    d_round1, so any subsequent recovery shows up as a large "gain" even
    when the final state is no better aligned than A. We therefore evaluate
    the H1 criteria on the FINAL distance directly: B beats A iff its final
    mean projected centroid distance is lower than A's. The alignment_gain
    field is still reported for full transparency.
    """
    proj_by_cl: dict[tuple, tuple[float, float]] = {}
    raw_by_cl:  dict[tuple, tuple[float, float]] = {}
    for cond in CONDITIONS:
        for label in CARROLL_LABELS:
            proj_by_cl[(cond, label)] = _first_last_mean_distance(proj_dist_rows, cond, label)
            raw_by_cl[(cond, label)]  = _first_last_mean_distance(raw_dist_rows,  cond, label)

    def alignment_gain(d_first: float, d_last: float) -> float:
        """Positive when distances decreased (alignment improved)."""
        if math.isnan(d_first) or math.isnan(d_last):
            return float("nan")
        return round(d_first - d_last, 8)

    summary: dict = {
        "experiment":           "exp_005b_language_conditioned_geometry",
        "spec_question":        ("Does mutable per-agent geometry let consensus "
                                 "feedback act as an attractor over inter-agent "
                                 "representational distance?"),
        "n_agents":             len(AGENT_IDS),
        "labels":               list(CARROLL_LABELS),
        "conditions":           list(CONDITIONS),
        "condition_labels":     dict(CONDITION_LABELS),
        "success_criteria":     dict(SUCCESS_CRITERIA),
        "per_condition":        {},
    }

    for cond in CONDITIONS:
        per_label: dict[str, dict] = {}
        held_final = _final_held_out_accuracy(round_rows, cond)
        coverage   = _final_label_coverage(round_rows, cond)
        max_frac   = _final_max_label_fraction(round_rows, cond)
        drift_mean = _mean_drift(drift_rows, cond)

        for label in CARROLL_LABELS:
            df_proj, dl_proj = proj_by_cl[(cond, label)]
            df_raw,  dl_raw  = raw_by_cl[(cond, label)]
            gain_proj = alignment_gain(df_proj, dl_proj)
            gain_raw  = alignment_gain(df_raw,  dl_raw)

            # consensus_specific_gain = this condition's projected gain -
            #                            random condition's projected gain
            #                            (informative only for the consensus row)
            df_c, dl_c = proj_by_cl[("random", label)]
            random_gain = alignment_gain(df_c, dl_c)
            csg = (round(gain_proj - random_gain, 8)
                   if not math.isnan(gain_proj) and not math.isnan(random_gain)
                   else float("nan"))

            per_label[label] = {
                "d_round1_projected":     round(df_proj, 8) if not math.isnan(df_proj) else float("nan"),
                "d_final_projected":      round(dl_proj, 8) if not math.isnan(dl_proj) else float("nan"),
                "d_round1_raw":           round(df_raw,  8) if not math.isnan(df_raw)  else float("nan"),
                "d_final_raw":            round(dl_raw,  8) if not math.isnan(dl_raw)  else float("nan"),
                "alignment_gain_projected": gain_proj,
                "alignment_gain_raw":       gain_raw,
                "consensus_specific_gain":  csg,
                "distortion_penalty":       drift_mean,
                "held_out_accuracy_final":  round(held_final, 4),
            }

        # h1_supported is the per-condition rollup; the meaningful test is on B.
        # All five success criteria must hold:
        #   (1) projected final distance LOWER than frozen baseline
        #   (2) projected final distance LOWER than random baseline
        #   (3) held_out_accuracy_final >= 0.90
        #   (4) label_coverage == 3
        #   (5) max_label_fraction <= 0.50
        # (1) and (2) are evaluated on the FINAL distance rather than the
        # alignment_gain to avoid the round-1 perturbation artifact described
        # in this function's docstring.
        def mean_final_distance(condition_key: str) -> float:
            finals = [proj_by_cl[(condition_key, lbl)][1] for lbl in CARROLL_LABELS]
            finals = [f for f in finals if not math.isnan(f)]
            return sum(finals) / len(finals) if finals else float("nan")

        mean_final_this   = mean_final_distance(cond)
        mean_final_frozen = mean_final_distance("frozen")
        mean_final_random = mean_final_distance("random")

        # Also report the spec-literal alignment_gain rollup for transparency.
        def mean_first_minus_last(pairs):
            v = [d0 - d1 for d0, d1 in pairs if not (math.isnan(d0) or math.isnan(d1))]
            return sum(v) / len(v) if v else float("nan")
        this_gains  = [proj_by_cl[(cond, lbl)] for lbl in CARROLL_LABELS]
        mean_gain_this = mean_first_minus_last(this_gains)

        criteria_eval = {
            "beats_frozen_in_projected_final":  bool(mean_final_this < mean_final_frozen),
            "beats_random_in_projected_final":  bool(mean_final_this < mean_final_random),
            "held_out_accuracy_passes":         bool(held_final >= SUCCESS_CRITERIA["held_out_accuracy"]),
            "label_coverage_passes":            bool(coverage == SUCCESS_CRITERIA["label_coverage"]),
            "no_label_collapse":                bool(max_frac <= SUCCESS_CRITERIA["max_label_fraction"]),
        }
        h1_supported = all(criteria_eval.values()) if cond == "consensus" else False

        summary["per_condition"][cond] = {
            "label":                       CONDITION_LABELS[cond],
            "per_label":                   per_label,
            "mean_projected_gain":         round(mean_gain_this,   8),
            "mean_final_distance_projected": round(mean_final_this, 8),
            "mean_final_minus_frozen":     round(mean_final_this - mean_final_frozen, 8),
            "mean_final_minus_random":     round(mean_final_this - mean_final_random, 8),
            "mean_drift":                  drift_mean,
            "held_out_final":              round(held_final, 4),
            "label_coverage_final":        coverage,
            "max_label_fraction":          round(max_frac, 4),
            "criteria_evaluated":          criteria_eval,
            "h1_supported":                h1_supported,
        }

    return summary


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_projected_alignment_curve(
    proj_rows: list[dict],
    out_path: Path,
) -> None:
    """One subplot per Carroll label, three curves (A / B / C) of mean
    pairwise projected centroid distance vs round."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax, label in zip(axes, CARROLL_LABELS):
        for cond in CONDITIONS:
            rows = sorted(
                [r for r in proj_rows if r["label"] == label and r["condition"] == cond],
                key=lambda r: int(r["round"]),
            )
            xs = [int(r["round"]) for r in rows]
            ys = [float(r["mean_pairwise_distance"]) for r in rows]
            ax.plot(xs, ys, lw=2, marker="o", ms=4,
                    color=COND_COLOR[cond], label=CONDITION_LABELS[cond])
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean pairwise cosine distance (projected)")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Inter-agent centroid alignment in projected space\n"
        "Lower = more aligned. B below A and C would support weak Sapir-Whorf.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_raw_vs_projected_alignment(
    raw_rows:  list[dict],
    proj_rows: list[dict],
    out_path: Path,
) -> None:
    """2x3 grid. Top row = raw DINOv2 space, bottom = projected space.
    Columns = labels. Three lines per panel."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey="row")
    for col, label in enumerate(CARROLL_LABELS):
        for row_idx, (rows_src, space_name) in enumerate([
            (raw_rows,  "raw DINOv2"),
            (proj_rows, "projected"),
        ]):
            ax = axes[row_idx, col]
            for cond in CONDITIONS:
                rows = sorted(
                    [r for r in rows_src if r["label"] == label and r["condition"] == cond],
                    key=lambda r: int(r["round"]),
                )
                xs = [int(r["round"]) for r in rows]
                ys = [float(r["mean_pairwise_distance"]) for r in rows]
                ax.plot(xs, ys, lw=2, marker="o", ms=3,
                        color=COND_COLOR[cond], label=CONDITION_LABELS[cond])
            ax.set_xlabel("Round")
            ax.set_ylabel(f"d (mean pairwise) — {space_name}")
            ax.set_title(f"{label} — {space_name}")
            ax.grid(True, alpha=0.2)
            if row_idx == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle(
        "Raw vs projected alignment\n"
        "Raw rows should be ~flat across conditions (DINOv2 unchanged). "
        "Projected rows are where W's effect shows.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_adapter_drift(drift_rows: list[dict], out_path: Path) -> None:
    """One subplot per condition; 3 lines per subplot (one per agent).
    A should be flat at 0. B and C should grow."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    agent_colors = {"agent_00": "tab:red", "agent_01": "tab:green", "agent_02": "tab:purple"}
    for ax, cond in zip(axes, CONDITIONS):
        for aid in AGENT_IDS:
            rows = sorted(
                [r for r in drift_rows if r["condition"] == cond and r["agent_id"] == aid],
                key=lambda r: int(r["round"]),
            )
            xs = [int(r["round"]) for r in rows]
            ys = [float(r["frobenius_drift_from_initial"]) for r in rows]
            ax.plot(xs, ys, lw=2, marker="o", ms=4, color=agent_colors[aid], label=aid)
        ax.set_xlabel("Round")
        ax.set_ylabel(r"$\|W_t - W_0\|_F$")
        ax.set_title(CONDITION_LABELS[cond])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        "Adapter drift from initial weights\n"
        "Frozen (A) should be flat at 0. Consensus (B) and random (C) drift; "
        "compare magnitudes.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_accuracy_by_condition(round_rows: list[dict], out_path: Path) -> None:
    """Held-out consensus accuracy per round, three lines (one per condition)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond in CONDITIONS:
        rows = sorted(
            [r for r in round_rows if r["condition"] == cond],
            key=lambda r: int(r["round"]),
        )
        xs = [int(r["round"]) for r in rows]
        ys = [float(r["held_out_consensus_accuracy"]) for r in rows]
        ax.plot(xs, ys, lw=2, marker="o", ms=4,
                color=COND_COLOR[cond], label=CONDITION_LABELS[cond])
    ax.axhline(SUCCESS_CRITERIA["held_out_accuracy"], color="black",
               lw=1, linestyle=":", label=f"target {SUCCESS_CRITERIA['held_out_accuracy']}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Held-out consensus accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Held-out consensus accuracy by condition\n"
                 "Geometric reshaping must not degrade task performance.")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] {RESULTS_DIR} missing — run run.py first.")
        sys.exit(1)

    # Load all inputs
    print("[Load] Inputs from run.py ...")
    round_rows = read_csv(RESULTS_DIR / "round_log_by_condition.csv")
    drift_rows = read_csv(RESULTS_DIR / "adapter_drift.csv")
    proj_npz   = np.load(RESULTS_DIR / "projected_centroid_vectors.npz")
    raw_npz    = np.load(RESULTS_DIR / "raw_centroid_vectors.npz")
    proj_vecs  = dict(proj_npz)
    raw_vecs   = dict(raw_npz)
    print(f"  round_rows={len(round_rows)}, drift_rows={len(drift_rows)}, "
          f"projected_vecs={len(proj_vecs)}, raw_vecs={len(raw_vecs)}")

    # Compute distances per space
    print("\n[Compute] Pairwise centroid distances ...")
    proj_dist_rows = compute_inter_agent_distances(proj_vecs, "projected")
    raw_dist_rows  = compute_inter_agent_distances(raw_vecs,  "raw")
    print(f"  projected: {len(proj_dist_rows)} (condition, label, round) rows")
    print(f"  raw:       {len(raw_dist_rows)}  (condition, label, round) rows")

    # Held-out accuracy per round (subset of round_log)
    held_rows = extract_held_out_accuracy(round_rows)

    # Write CSVs
    print("\n[CSV]")
    write_csv(proj_dist_rows, RESULTS_DIR / "inter_agent_distances_projected.csv")
    write_csv(raw_dist_rows,  RESULTS_DIR / "inter_agent_distances_raw.csv")
    write_csv(held_rows,      RESULTS_DIR / "held_out_accuracy_by_round.csv")

    # alignment_summary.json
    print("\n[JSON]")
    summary = build_alignment_summary(proj_dist_rows, raw_dist_rows, round_rows, drift_rows)
    write_json(summary, RESULTS_DIR / "alignment_summary.json")

    # Plots
    print("\n[Plots]")
    plot_projected_alignment_curve(proj_dist_rows,           RESULTS_DIR / "projected_alignment_curve.png")
    plot_raw_vs_projected_alignment(raw_dist_rows, proj_dist_rows,
                                    RESULTS_DIR / "raw_vs_projected_alignment.png")
    plot_adapter_drift(drift_rows,                            RESULTS_DIR / "adapter_drift.png")
    plot_accuracy_by_condition(round_rows,                    RESULTS_DIR / "accuracy_by_condition.png")

    print(f"\n[Done] Metrics + plots in: {RESULTS_DIR}")


if __name__ == "__main__":
    run()
