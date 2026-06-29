"""
analyze_exp007.py — aggregate the ledger into CIs, tables, and figures
(plan §8, §12, §13).

Reads results/exp_007/ledger.jsonl (+ descriptors.csv) and produces:

    aggregate.csv     — one row per (sub, concept_type, learner, ways, shots):
                        mean + 95% bootstrap CI for c1 / c2 / margin /
                        nmi_pred_category, n_episodes, chance.
    paired_tests.csv  — per (sub, concept_type, ways, shots): EVERY learner vs
                        the centroid baseline, on C1 AND C2 — i.e. the
                        prototype-failure gap. Directional `supported`
                        (preserved from 007a). margin is NOT compared across
                        learners (directive 3): score scales differ, so a margin
                        comparison is meaningless — it stays a within-learner
                        diagnostic in aggregate.csv only.
    accuracy_vs_shots.png       — 007a/native readout (H2: curves coincide).
    figure1_accuracy_vs_ns.png  — CENTRAL 007b figure: per-concept C1 vs NS(c),
                        one curve per learner. The dissociation at a glance.
    figure2_gap_boxplots.png    — prototype-failure gap by concept type (007b).

All CIs / paired tests come from src.eval.stats. No learning here — only reads
the ledger — so it is cheap to iterate on figures.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from _shared import RESULTS_DIR  # noqa: E402
from src.eval import stats        # noqa: E402

LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"
DESCRIPTORS_PATH = RESULTS_DIR / "descriptors.csv"
AGG_PATH = RESULTS_DIR / "aggregate.csv"
PAIRED_PATH = RESULTS_DIR / "paired_tests.csv"

SCALAR_METRICS = ["c1", "c2", "margin", "nmi_pred_category"]
CELL_KEYS = ("sub", "concept_type", "ways", "shots")
REFERENCE_LEARNER = "centroid"     # the prototype baseline the gap is measured against
FIG_SHOTS = 5                      # shots level for the per-concept figures


def _load_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        raise FileNotFoundError(f"{LEDGER_PATH} not found — run run_exp007.py first.")
    return [json.loads(l) for l in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_descriptors() -> dict[tuple, dict]:
    """(sub, label) -> descriptor row (ns_silhouette, concept_type, ...)."""
    if not DESCRIPTORS_PATH.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(DESCRIPTORS_PATH, encoding="utf-8")):
        out[(r["sub"], r["label"])] = r
    return out


def _groupby(rows, keys):
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


# ─── aggregate.csv ────────────────────────────────────────────────────────────

def aggregate_ledger(rows: list[dict]) -> None:
    groups = _groupby(rows, [*CELL_KEYS, "learner"])
    out_rows = []
    for (sub, ctype, ways, shots, learner), grp in sorted(groups.items(), key=lambda x: str(x[0])):
        rec = {"sub": sub, "concept_type": ctype, "learner": learner,
               "ways": ways, "shots": shots, "n_episodes": len(grp),
               "chance": grp[0]["chance"]}
        for m in SCALAR_METRICS:
            vals = np.array([g[m] for g in grp], dtype=float)
            ci = stats.bootstrap_ci(vals, seed=0)
            rec[f"{m}_mean"] = round(ci.mean, 4)
            rec[f"{m}_ci_low"] = round(ci.ci_low, 4)
            rec[f"{m}_ci_high"] = round(ci.ci_high, 4)
        out_rows.append(rec)
    with open(AGG_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"  saved {AGG_PATH.name} ({len(out_rows)} rows)")


# ─── paired_tests.csv — prototype-failure gap (every learner vs centroid) ─────

def _aligned(grp_a, grp_b, metric):
    a = {g["episode_id"]: g[metric] for g in grp_a}
    b = {g["episode_id"]: g[metric] for g in grp_b}
    common = sorted(set(a) & set(b))
    return (np.array([a[e] for e in common], dtype=float),
            np.array([b[e] for e in common], dtype=float))


def run_paired_tests(rows: list[dict]) -> None:
    """Per cell, every learner vs the centroid baseline (a=learner, b=centroid),
    on C1 and C2. A positive, supported gap = that learner beats the prototype
    (H1/H3 on disjunctive; expected NOT supported on native)."""
    cells = _groupby(rows, list(CELL_KEYS))
    out_rows = []
    for (sub, ctype, ways, shots), grp in sorted(cells.items(), key=lambda x: str(x[0])):
        by_learner = _groupby(grp, ["learner"])
        if (REFERENCE_LEARNER,) not in by_learner:
            continue
        ref = by_learner[(REFERENCE_LEARNER,)]
        for (learner,), lgrp in sorted(by_learner.items()):
            if learner == REFERENCE_LEARNER:
                continue
            for metric in ("c1", "c2"):
                a, b = _aligned(lgrp, ref, metric)
                if len(a) < 2:
                    continue
                res = stats.paired_test(a, b, seed=0)
                out_rows.append({
                    "sub": sub, "concept_type": ctype, "ways": ways, "shots": shots,
                    "comparison": f"{learner}_vs_{REFERENCE_LEARNER}",
                    "metric": metric, "n": res.n,
                    "mean_diff": round(res.mean_diff, 4),
                    "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
                    "wilcoxon_p": round(res.wilcoxon_p, 5),
                    "effect_size": round(res.effect_size, 4),
                    "excludes_zero": res.excludes_zero,
                    "supported": res.supported,
                })
    with open(PAIRED_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"  saved {PAIRED_PATH.name} ({len(out_rows)} comparisons)")


# ─── figures ──────────────────────────────────────────────────────────────────

def figure_accuracy_vs_shots(rows: list[dict]) -> None:
    """C1 vs shots, one curve per learner, a facet per ways (007a/native)."""
    rows = [r for r in rows if r["sub"] == "007a"]
    if not rows:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ways_vals = sorted({r["ways"] for r in rows})
    learners = sorted({r["learner"] for r in rows})
    fig, axes = plt.subplots(1, len(ways_vals), figsize=(6 * len(ways_vals), 4.5), squeeze=False)
    for ax, ways in zip(axes[0], ways_vals):
        sub = [r for r in rows if r["ways"] == ways]
        chance = sub[0]["chance"]
        for learner in learners:
            cell = _groupby([r for r in sub if r["learner"] == learner], ["shots"])
            keys = sorted(cell)
            shots, means, los, his = [], [], [], []
            for key in keys:
                vals = np.array([g["c1"] for g in cell[key]], dtype=float)
                ci = stats.bootstrap_ci(vals, seed=0)
                shots.append(key[0]); means.append(ci.mean); los.append(ci.ci_low); his.append(ci.ci_high)
            ax.plot(shots, means, marker="o", label=learner)
            ax.fill_between(shots, los, his, alpha=0.15)
        ax.axhline(chance, ls="--", color="gray", lw=1, label="chance")
        ax.set_title(f"{ways}-way (NATIVE)"); ax.set_xlabel("shots K")
        ax.set_ylabel("C1 naming accuracy (macro)"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("exp_007a — naming accuracy vs shots (H2: curves coincide on native)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "accuracy_vs_shots.png", dpi=130)
    print("  saved accuracy_vs_shots.png")


def figure_accuracy_vs_ns(rows: list[dict], desc: dict) -> None:
    """CENTRAL FIGURE 1: per-concept C1 vs NS(c), one curve per learner.

    Per-concept accuracy is pooled over all 007b episodes at K=FIG_SHOTS that
    included the concept; x is the concept's NS_silhouette. Native concepts
    (high NS) and disjunctive concepts (low NS) on one axis trace the collapse:
    the centroid curve drops at low NS while episodic / multi-anchor hold."""
    rows = [r for r in rows if r["sub"] == "007b" and r["shots"] == FIG_SHOTS]
    if not rows:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (learner -> {label: [accuracies]}) from per_concept_c1.
    acc: dict[str, dict[str, list]] = {}
    for r in rows:
        d = acc.setdefault(r["learner"], {})
        for label, a in r.get("per_concept_c1", {}).items():
            d.setdefault(label, []).append(a)

    def ns_of(label):
        row = desc.get(("007b", label))
        return float(row["ns_silhouette"]) if row else None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for learner in sorted(acc):
        pts = []
        for label, accs in acc[learner].items():
            ns = ns_of(label)
            if ns is not None:
                pts.append((ns, float(np.mean(accs))))
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=4, label=learner, alpha=0.85)
    # mark the native/disjunctive NS bands
    dis_ns = [float(v["ns_silhouette"]) for (s, _), v in desc.items()
              if s == "007b" and v["concept_type"] == "disjunctive"]
    nat_ns = [float(v["ns_silhouette"]) for (s, _), v in desc.items()
              if s == "007b" and v["concept_type"] == "native"]
    if dis_ns:
        ax.axvspan(min(dis_ns) - 0.01, max(dis_ns) + 0.01, color="orange", alpha=0.06)
    if nat_ns:
        ax.axvspan(min(nat_ns) - 0.01, max(nat_ns) + 0.01, color="green", alpha=0.06)
    ax.set_xlabel("NS(c) — native separability (silhouette, cosine)")
    ax.set_ylabel(f"per-concept C1 accuracy (K={FIG_SHOTS})")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("exp_007b · Figure 1 — accuracy vs native separability\n"
                 "(left band: disjunctive concepts; right band: native)")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "figure1_accuracy_vs_ns.png", dpi=130)
    print("  saved figure1_accuracy_vs_ns.png")


def figure_gap_boxplots(rows: list[dict]) -> None:
    """FIGURE 2: prototype-failure gap (exemplar_knn_k3 − centroid) per episode,
    boxplots by concept type (007b)."""
    rows = [r for r in rows if r["sub"] == "007b" and r["shots"] == FIG_SHOTS]
    if not rows:
        return
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = "exemplar_knn_k3"
    data, labels = [], []
    for ctype in ("native", "disjunctive"):
        cell = [r for r in rows if r["concept_type"] == ctype]
        by_l = _groupby(cell, ["learner", "ways"])
        gaps = []
        for ways in sorted({r["ways"] for r in cell}):
            t = by_l.get((target, ways)); c = by_l.get((REFERENCE_LEARNER, ways))
            if not t or not c:
                continue
            a, b = _aligned(t, c, "c1")
            gaps.extend((a - b).tolist())
        if gaps:
            data.append(gaps); labels.append(f"{ctype}\n(n={len(gaps)})")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.set_ylabel(f"prototype-failure gap: C1({target}) − C1(centroid)")
    ax.set_title(f"exp_007b · Figure 2 — gap by concept type (K={FIG_SHOTS})")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(RESULTS_DIR / "figure2_gap_boxplots.png", dpi=130)
    print("  saved figure2_gap_boxplots.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_007 analysis / figures")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_ledger()
    desc = _load_descriptors()
    if not args.figures_only:
        aggregate_ledger(rows)
        run_paired_tests(rows)
    figure_accuracy_vs_shots(rows)
    figure_accuracy_vs_ns(rows, desc)
    figure_gap_boxplots(rows)


if __name__ == "__main__":
    main()
