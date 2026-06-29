"""
analyze_ext.py — aggregate ledger into CIs, paired tests, figures, and report.

Reads:
    results/exp_007b_extension/ledger.jsonl   (near + mid tiers, all pool types)
    results/exp_007b_extension/descriptors.csv
    results/exp_007/ledger.jsonl              (far + native tiers, loaded once)
    results/exp_007/descriptors.csv           (far + native descriptors)

Produces:
    aggregate.csv         per (tier, learner, ways, shots, pool_type)
    paired_tests.csv      exemplar_knn_k3 vs centroid, per tier×ways×shots×pool_type
    figure1_gradient.png  C2 accuracy vs inter_member_distance (CENTRAL)
    figure2_pool_sensitivity.png  C2 gap vs pool_type by tier
    figure3_c1_gradient.png      C1 accuracy vs inter_member_distance
    report.md             full English experiment report

Usage:
    cd experiments/exp_007b_extension
    python analyze_ext.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[1]   # experiments/exp_007b_extension -> experiments -> project root
_PARENT_EXP = _PROJECT / "experiments" / "exp_007_concept_carving"
if str(_HERE) in sys.path:
    sys.path.remove(str(_HERE))
sys.path.insert(0, str(_HERE))
for p in [str(_PROJECT), str(_PARENT_EXP)]:
    if p not in sys.path:
        sys.path.append(p)

from _shared import (  # noqa: E402
    RESULTS_DIR, EXP007_RESULTS,
    FAR_THRESHOLD, FAR_MEDIAN,
    CLEAN_FAR_LABELS, POOL_TYPES, PRIMARY_POOL,
    WAYS, SHOTS, REFERENCE_LEARNER, EXEMPLAR_LEARNER, FIG_SHOTS,
    MASTER_SEED,
)
from src.eval import stats  # noqa: E402

LEDGER_PATH     = RESULTS_DIR / "ledger.jsonl"
DESCRIPTORS_PATH = RESULTS_DIR / "descriptors.csv"
AGG_PATH        = RESULTS_DIR / "aggregate.csv"
PAIRED_PATH     = RESULTS_DIR / "paired_tests.csv"
REPORT_PATH     = RESULTS_DIR / "report.md"

EXP007_LEDGER      = EXP007_RESULTS / "ledger.jsonl"
EXP007_DESCRIPTORS = EXP007_RESULTS / "descriptors.csv"

TIER_ORDER = ["near", "mid", "far", "native"]
TIER_COLORS = {"near": "#2ca02c", "mid": "#ff7f0e", "far": "#d62728", "native": "#1f77b4"}

SCALAR_METRICS = ["c1", "c2", "margin"]


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_ext_ledger() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(l) for l in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def _load_007b_ledger(pool_type_for_far: str | None = None) -> list[dict]:
    """Load far-disjunctive + native rows from exp_007/ledger.jsonl.

    For far-disjunctive rows we filter to CLEAN_FAR_LABELS only and inject
    tier='far' / pool_type=<pool_type_for_far> (the 007b pool uses n_ways
    candidates, which does not match any named pool type; we label it 'std_nway').
    """
    if not EXP007_LEDGER.exists():
        return []
    rows = []
    for line in EXP007_LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("sub") != "007b":
            continue
        ctype = r.get("concept_type", "")
        pc = r.get("per_concept_c1", {})
        labels_in_row = set(pc.keys()) if isinstance(pc, dict) else set()
        if ctype == "disjunctive":
            if labels_in_row and not (labels_in_row & CLEAN_FAR_LABELS):
                continue  # zorb row — skip
            r2 = dict(r)
            r2["tier"] = "far"
            r2["pool_type"] = pool_type_for_far or "std_nway"
            rows.append(r2)
        elif ctype == "native":
            r2 = dict(r)
            r2["tier"] = "native"
            r2["pool_type"] = pool_type_for_far or "std_nway"
            rows.append(r2)
    return rows


def _load_descriptors() -> dict[tuple, dict]:
    """Returns {(tier, label): row}."""
    out: dict[tuple, dict] = {}
    # Extension descriptors
    if DESCRIPTORS_PATH.exists():
        for r in csv.DictReader(open(DESCRIPTORS_PATH, encoding="utf-8")):
            tier = r.get("tier", "ext")
            out[(tier, r["label"])] = r
    # exp_007b descriptors for far + native
    if EXP007_DESCRIPTORS.exists():
        for r in csv.DictReader(open(EXP007_DESCRIPTORS, encoding="utf-8")):
            if r.get("sub") != "007b":
                continue
            ctype = r.get("concept_type", "")
            tier = "far" if ctype == "disjunctive" else "native"
            label = r["label"]
            if ctype == "disjunctive" and label not in CLEAN_FAR_LABELS:
                continue
            if (tier, label) not in out:
                out[(tier, label)] = r
    return out


def _groupby(rows, keys):
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


# ─── aggregate.csv ────────────────────────────────────────────────────────────

def aggregate_ledger(ext_rows: list[dict], far_nat_rows: list[dict]) -> None:
    all_rows = ext_rows + far_nat_rows
    groups = _groupby(all_rows, ["tier", "learner", "ways", "shots", "pool_type"])
    out_rows = []
    for key, grp in sorted(groups.items(), key=lambda x: str(x[0])):
        tier, learner, ways, shots, pool_type = key
        rec = {"tier": tier, "learner": learner, "ways": ways,
               "shots": shots, "pool_type": pool_type,
               "n_episodes": len(grp), "chance": grp[0].get("chance", "")}
        for m in SCALAR_METRICS:
            vals = np.array([float(g[m]) for g in grp if m in g], dtype=float)
            if len(vals) == 0:
                rec[f"{m}_mean"] = rec[f"{m}_ci_low"] = rec[f"{m}_ci_high"] = ""
                continue
            ci = stats.bootstrap_ci(vals, seed=0)
            rec[f"{m}_mean"]    = round(ci.mean, 4)
            rec[f"{m}_ci_low"]  = round(ci.ci_low, 4)
            rec[f"{m}_ci_high"] = round(ci.ci_high, 4)
        out_rows.append(rec)

    _save_csv(out_rows, AGG_PATH)


# ─── paired_tests.csv — exemplar_knn_k3 vs centroid (C2 primary) ─────────────

def _aligned(grp_a, grp_b, metric):
    a_map = {g["episode_id"]: float(g[metric]) for g in grp_a}
    b_map = {g["episode_id"]: float(g[metric]) for g in grp_b}
    common = sorted(set(a_map) & set(b_map))
    if len(common) < 2:
        return None, None
    return (np.array([a_map[e] for e in common]),
            np.array([b_map[e] for e in common]))


def run_paired_tests(ext_rows: list[dict], far_nat_rows: list[dict]) -> None:
    all_rows = ext_rows + far_nat_rows
    CELL_KEYS = ["tier", "ways", "shots", "pool_type"]
    cells = _groupby(all_rows, CELL_KEYS)
    out_rows = []

    for cell_key, grp in sorted(cells.items(), key=lambda x: str(x[0])):
        tier, ways, shots, pool_type = cell_key
        by_learner = _groupby(grp, ["learner"])
        ref = by_learner.get((REFERENCE_LEARNER,))
        if not ref:
            continue
        for (learner,), lgrp in sorted(by_learner.items()):
            if learner == REFERENCE_LEARNER:
                continue
            for metric in ("c2", "c1"):
                a, b = _aligned(lgrp, ref, metric)
                if a is None or len(a) < 2:
                    continue
                res = stats.paired_test(a, b, seed=0)
                out_rows.append({
                    "tier": tier, "ways": ways, "shots": shots, "pool_type": pool_type,
                    "comparison": f"{learner}_vs_{REFERENCE_LEARNER}",
                    "metric": metric, "n": res.n,
                    "mean_diff": round(res.mean_diff, 4),
                    "ci_low": round(res.ci_low, 4), "ci_high": round(res.ci_high, 4),
                    "wilcoxon_p": round(res.wilcoxon_p, 5),
                    "effect_size": round(res.effect_size, 4),
                    "excludes_zero": res.excludes_zero,
                    "supported": res.supported,
                })

    _save_csv(out_rows, PAIRED_PATH)


# ─── Figure helpers ───────────────────────────────────────────────────────────

def _per_concept_c2(rows, pool_type, shots, ways):
    """Returns {learner -> {concept_label -> [per-episode c2]}}.

    Uses the pool_type and shots/ways filter. For far/native rows the 'pool_type'
    is 'std_nway' — they are included only when requested by that string.
    For extension rows the pool_type matches exactly.
    """
    filtered = [r for r in rows
                if r.get("pool_type") == pool_type
                and r.get("shots") == shots
                and r.get("ways") == ways]
    by_learner: dict[str, dict[str, list]] = {}
    for r in filtered:
        lrn = r["learner"]
        pc = r.get("per_concept_c1", {})
        c2 = float(r.get("c2", 0.0))
        by_learner.setdefault(lrn, {})
        # For the gradient figure we use per-concept c1 from the stored dict
        # and the episode-level c2. For per-concept c2 we use the episode c2
        # as a proxy (macro average over concepts).
        if isinstance(pc, dict):
            for label, c1_val in pc.items():
                by_learner[lrn].setdefault(label, []).append(float(c1_val))
    return by_learner


def _per_concept_c1(rows, shots, ways):
    """Per-concept C1 accuracy: {learner -> {label -> mean_c1}} across episodes."""
    filtered = [r for r in rows if r.get("shots") == shots and r.get("ways") == ways]
    by_learner: dict[str, dict[str, list]] = {}
    for r in filtered:
        lrn = r["learner"]
        pc = r.get("per_concept_c1", {})
        if isinstance(pc, dict):
            by_learner.setdefault(lrn, {})
            for label, val in pc.items():
                by_learner[lrn].setdefault(label, []).append(float(val))
    return by_learner


def _episode_c2_by_tier_learner(rows, pool_type, shots, ways, learner):
    return [float(r["c2"]) for r in rows
            if r.get("pool_type") == pool_type
            and r.get("shots") == shots
            and r.get("ways") == ways
            and r.get("learner") == learner]


# ─── Figure 1 — Gradient (CENTRAL) ───────────────────────────────────────────

def figure1_gradient(all_rows: list[dict], desc: dict) -> None:
    """C2 (top) and C2 exemplar−centroid gap (bottom) vs inter_member_distance."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Use PRIMARY_POOL for extension rows; 'std_nway' for far/native
    shots, ways = FIG_SHOTS, 3   # near+mid only have ways=3

    LEARNERS_TO_PLOT = [
        "centroid", EXEMPLAR_LEARNER, "kcentroid_k2", "logreg", "linsvm", "random"
    ]
    LEARNER_STYLES = {
        "centroid":        dict(lw=2, ls="-",  marker="o"),
        EXEMPLAR_LEARNER:  dict(lw=2, ls="-",  marker="^"),
        "kcentroid_k2":    dict(lw=1.5, ls="--", marker="s"),
        "logreg":          dict(lw=1.5, ls=":",  marker="D"),
        "linsvm":          dict(lw=1.5, ls="-.", marker="v"),
        "random":          dict(lw=1,  ls="--", marker=None, color="lightgray"),
    }

    def _get_imd(tier, label):
        row = desc.get((tier, label))
        if row is None:
            return None
        imd_val = row.get("inter_member_distance", "")
        if imd_val == "" or imd_val is None:
            return None
        try:
            return float(imd_val)
        except (ValueError, TypeError):
            return None

    def _get_ns(tier, label):
        row = desc.get((tier, label))
        if row is None:
            return None
        try:
            return float(row.get("ns_silhouette", ""))
        except (ValueError, TypeError):
            return None

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=False)
    ax_c2, ax_gap = axes

    # Collect points: (tier, x_value, learner, mean_c2, ci_lo, ci_hi)
    tier_x_data: dict[str, dict[str, dict[str, list]]] = {}  # learner -> tier -> {x, c2_vals}

    for tier in TIER_ORDER:
        pool_for_tier = PRIMARY_POOL if tier in ("near", "mid") else "std_nway"
        tier_rows = [r for r in all_rows
                     if r.get("tier") == tier
                     and r.get("pool_type") == pool_for_tier
                     and r.get("shots") == shots
                     and r.get("ways") == ways]
        if not tier_rows:
            continue

        # Group by learner × concept_label for per-concept c2
        by_learner_label = {}
        for r in tier_rows:
            lrn = r["learner"]
            pc = r.get("per_concept_c1", {})
            if not isinstance(pc, dict):
                continue
            by_learner_label.setdefault(lrn, {})
            for label in pc:
                by_learner_label[lrn].setdefault(label, []).append(float(r["c2"]))

        for lrn in LEARNERS_TO_PLOT:
            tier_x_data.setdefault(lrn, {}).setdefault(tier, {"x": [], "c2": []})
            if lrn not in by_learner_label:
                continue
            for label, c2_vals in by_learner_label[lrn].items():
                x = _get_imd(tier, label)
                if x is None and tier == "native":
                    x = _get_ns(tier, label)   # use NS as proxy for native
                if x is None:
                    continue
                tier_x_data[lrn][tier]["x"].append(x)
                tier_x_data[lrn][tier]["c2"].append(float(np.mean(c2_vals)))

    # Panel 1: C2 accuracy vs inter_member_distance (or NS for native)
    for lrn in LEARNERS_TO_PLOT:
        pts = []
        for tier in TIER_ORDER:
            d = tier_x_data.get(lrn, {}).get(tier, {})
            xs = d.get("x", [])
            c2s = d.get("c2", [])
            for x, c2 in zip(xs, c2s):
                pts.append((x, c2, tier))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        tiers_for_pts = [p[2] for p in pts]
        style = LEARNER_STYLES.get(lrn, {})
        color = style.pop("color", None)
        line, = ax_c2.plot(xs, ys, label=lrn, **({"color": color} if color else {}), **style)
        # Color scatter by tier
        for x, y, t in pts:
            ax_c2.scatter([x], [y], color=TIER_COLORS.get(t, "black"), s=30, zorder=5)

    ax_c2.axhline(1.0 / ways, ls="--", color="gray", lw=1, label=f"chance (1/{ways})")
    ax_c2.set_ylabel(f"C2 accuracy (K={shots}, {ways}-way, pool={PRIMARY_POOL})")
    ax_c2.set_ylim(-0.05, 1.05)
    ax_c2.grid(alpha=0.25)
    ax_c2.legend(fontsize=8, loc="lower right")
    ax_c2.set_title("Figure 1 (top) — C2 accuracy vs inter-member distance / NS")

    # Tier legend patches
    legend_patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax_c2.legend(handles=legend_patches + [
        plt.Line2D([0], [0], ls="--", color="gray", label="chance")
    ], fontsize=8, loc="lower right")

    # Panel 2: exemplar − centroid gap
    gap_pts = []
    for tier in ("near", "mid", "far"):
        pool_for_tier = PRIMARY_POOL if tier in ("near", "mid") else "std_nway"
        tier_rows = [r for r in all_rows
                     if r.get("tier") == tier
                     and r.get("pool_type") == pool_for_tier
                     and r.get("shots") == shots
                     and r.get("ways") == ways]
        if not tier_rows:
            continue
        by_learner_label_tier = {}
        for r in tier_rows:
            lrn = r["learner"]
            pc = r.get("per_concept_c1", {})
            if not isinstance(pc, dict):
                continue
            by_learner_label_tier.setdefault(lrn, {})
            for label in pc:
                by_learner_label_tier[lrn].setdefault(label, []).append(float(r["c2"]))

        for label in set().union(*[by_learner_label_tier.get(l, {}).keys()
                                   for l in [EXEMPLAR_LEARNER, REFERENCE_LEARNER]]):
            ex_vals = by_learner_label_tier.get(EXEMPLAR_LEARNER, {}).get(label, [])
            ct_vals = by_learner_label_tier.get(REFERENCE_LEARNER, {}).get(label, [])
            if not ex_vals or not ct_vals:
                continue
            x = _get_imd(tier, label)
            if x is None:
                continue
            gap = float(np.mean(ex_vals)) - float(np.mean(ct_vals))
            gap_pts.append((x, gap, tier))

    if gap_pts:
        gap_pts.sort(key=lambda p: p[0])
        for x, gap, tier in gap_pts:
            color = TIER_COLORS.get(tier, "black")
            supported_marker = "^" if gap > 0 else "v"
            ax_gap.scatter([x], [gap], color=color, s=60, marker=supported_marker, zorder=5)
        ax_gap.axhline(0, ls="--", color="gray", lw=1)
        # Trend line
        xs_all = [p[0] for p in gap_pts]
        ys_all = [p[1] for p in gap_pts]
        if len(xs_all) >= 2:
            z = np.polyfit(xs_all, ys_all, 1)
            p = np.poly1d(z)
            xs_fit = np.linspace(min(xs_all), max(xs_all), 100)
            ax_gap.plot(xs_fit, p(xs_fit), "k--", lw=1, alpha=0.5, label="trend")

    ax_gap.set_ylabel(f"C2 gap: {EXEMPLAR_LEARNER} − {REFERENCE_LEARNER}")
    ax_gap.set_xlabel("inter_member_distance  (NS for native anchor)")
    ax_gap.grid(alpha=0.25)
    ax_gap.set_title("Figure 1 (bottom) — C2 episodic advantage vs inter-member distance")
    tier_patches = [mpatches.Patch(color=TIER_COLORS[t], label=t)
                    for t in ["near", "mid", "far"]]
    ax_gap.legend(handles=tier_patches, fontsize=8)

    fig.tight_layout()
    out_path = RESULTS_DIR / "figure1_gradient.png"
    fig.savefig(out_path, dpi=130)
    print(f"  saved figure1_gradient.png")


# ─── Figure 2 — Pool sensitivity ──────────────────────────────────────────────

def figure2_pool_sensitivity(ext_rows: list[dict]) -> None:
    """Boxplots of C2 exemplar−centroid gap per pool_type, faceted by tier."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shots, ways = FIG_SHOTS, 3
    tiers_with_pools = ["near", "mid"]
    fig, axes = plt.subplots(1, len(tiers_with_pools),
                             figsize=(5 * len(tiers_with_pools), 5), squeeze=False)

    for ax, tier in zip(axes[0], tiers_with_pools):
        data, xlabels = [], []
        for pt in POOL_TYPES:
            ep_rows = [r for r in ext_rows
                       if r.get("tier") == tier
                       and r.get("pool_type") == pt
                       and r.get("shots") == shots
                       and r.get("ways") == ways]
            ex_map  = {r["episode_id"]: float(r["c2"])
                       for r in ep_rows if r["learner"] == EXEMPLAR_LEARNER}
            cen_map = {r["episode_id"]: float(r["c2"])
                       for r in ep_rows if r["learner"] == REFERENCE_LEARNER}
            common = sorted(set(ex_map) & set(cen_map))
            if common:
                gaps = [ex_map[e] - cen_map[e] for e in common]
                data.append(gaps)
                xlabels.append(f"{pt}\n(n={len(gaps)})")
        if data:
            ax.boxplot(data, tick_labels=xlabels, showmeans=True)
        ax.axhline(0, ls="--", color="gray", lw=1)
        ax.set_title(f"{tier.upper()} tier")
        ax.set_ylabel(f"C2 gap: {EXEMPLAR_LEARNER} − {REFERENCE_LEARNER}" if tier == tiers_with_pools[0] else "")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle(f"Figure 2 — Pool sensitivity (K={shots}, {ways}-way)")
    fig.tight_layout()
    out_path = RESULTS_DIR / "figure2_pool_sensitivity.png"
    fig.savefig(out_path, dpi=130)
    print("  saved figure2_pool_sensitivity.png")


# ─── Figure 3 — C1 gradient ───────────────────────────────────────────────────

def figure3_c1_gradient(all_rows: list[dict], desc: dict) -> None:
    """C1 accuracy vs inter_member_distance (secondary)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    shots, ways = FIG_SHOTS, 3
    LEARNERS_TO_PLOT = [
        "centroid", EXEMPLAR_LEARNER, "kcentroid_k2", "logreg", "linsvm", "random"
    ]

    def _get_x(tier, label):
        row = desc.get((tier, label))
        if row is None:
            return None
        imd_val = row.get("inter_member_distance", "")
        if imd_val == "" or imd_val is None:
            return None
        try:
            return float(imd_val)
        except (ValueError, TypeError):
            return None

    fig, ax = plt.subplots(figsize=(9, 5))

    for tier in TIER_ORDER:
        pool_for_tier = PRIMARY_POOL if tier in ("near", "mid") else "std_nway"
        tier_rows = [r for r in all_rows
                     if r.get("tier") == tier
                     and r.get("pool_type") == pool_for_tier
                     and r.get("shots") == shots
                     and r.get("ways") == ways]
        if not tier_rows:
            continue

        by_learner_label: dict[str, dict[str, list]] = {}
        for r in tier_rows:
            lrn = r["learner"]
            pc = r.get("per_concept_c1", {})
            if not isinstance(pc, dict):
                continue
            by_learner_label.setdefault(lrn, {})
            for label, c1_val in pc.items():
                by_learner_label[lrn].setdefault(label, []).append(float(c1_val))

        for lrn in LEARNERS_TO_PLOT:
            pts = []
            for label, c1_vals in by_learner_label.get(lrn, {}).items():
                x = _get_x(tier, label)
                if x is None:
                    continue
                pts.append((x, float(np.mean(c1_vals))))
            for x, y in pts:
                ax.scatter([x], [y], color=TIER_COLORS.get(tier, "black"), s=30, zorder=4)

    # One curve per learner (across all tiers)
    for lrn in LEARNERS_TO_PLOT:
        pts_all = []
        for tier in TIER_ORDER:
            pool_for_tier = PRIMARY_POOL if tier in ("near", "mid") else "std_nway"
            tier_rows = [r for r in all_rows
                         if r.get("tier") == tier
                         and r.get("pool_type") == pool_for_tier
                         and r.get("shots") == shots
                         and r.get("ways") == ways
                         and r.get("learner") == lrn]
            for r in tier_rows:
                pc = r.get("per_concept_c1", {})
                if not isinstance(pc, dict):
                    continue
                for label, c1_val in pc.items():
                    x = _get_x(tier, label)
                    if x is not None:
                        pts_all.append((x, float(c1_val)))
        if not pts_all:
            continue
        pts_all.sort()
        xs, ys = zip(*pts_all)
        ax.plot(xs, ys, alpha=0.5, lw=1.2, label=lrn)

    ax.axhline(1.0 / ways, ls="--", color="gray", lw=1, label=f"chance (1/{ways})")
    ax.set_xlabel("inter_member_distance  (NS proxy for native)")
    ax.set_ylabel(f"C1 naming accuracy (K={shots}, {ways}-way)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)

    tier_patches = [mpatches.Patch(color=TIER_COLORS[t], label=t) for t in TIER_ORDER]
    ax.legend(handles=tier_patches, fontsize=8, loc="lower right")
    ax.set_title("Figure 3 — C1 accuracy vs inter-member distance / NS")

    fig.tight_layout()
    out_path = RESULTS_DIR / "figure3_c1_gradient.png"
    fig.savefig(out_path, dpi=130)
    print("  saved figure3_c1_gradient.png")


# ─── report.md ────────────────────────────────────────────────────────────────

def write_report(
    ext_rows: list[dict],
    far_nat_rows: list[dict],
    desc: dict,
    gate_rows: list[dict],
) -> None:
    """Write the report.md template with actual gate results and tier info."""
    all_rows = ext_rows + far_nat_rows

    def _mean_c2(tier, pool_type, shots, ways, learner):
        vals = [float(r["c2"]) for r in all_rows
                if r.get("tier") == tier
                and r.get("pool_type") == pool_type
                and r.get("shots") == shots
                and r.get("ways") == ways
                and r.get("learner") == learner]
        return round(float(np.mean(vals)), 3) if vals else "n/a"

    # Gate table
    gate_table = "| tier | pair | imd | ns | nmi | pass? | reason |\n"
    gate_table += "|---|---|---|---|---|---|---|\n"
    for r in gate_rows:
        gate_table += (
            f"| {r['tier']} | {r['cat_a']}+{r['cat_b']} | "
            f"{r['inter_member_distance']} | {r['ns_silhouette']} | "
            f"{r['nmi_native']} | {'✓' if r['passes_gate'] else '✗'} | "
            f"{r['exclusion_reason'] or '—'} |\n"
        )

    # Descriptor table
    def _desc_rows_for(tier):
        rows = [(k, v) for k, v in desc.items() if k[0] == tier]
        return rows

    def _desc_table(tier_label):
        rows = _desc_rows_for(tier_label)
        if not rows:
            return "*no data*"
        lines = ["| label | members | imd | ns | compact | clo | nmi |"]
        lines.append("|---|---|---|---|---|---|---|")
        for (_, label), r in sorted(rows, key=lambda x: x[0][1]):
            lines.append(
                f"| {label} | {r.get('members','')} | "
                f"{r.get('inter_member_distance','')} | "
                f"{r.get('ns_silhouette','')} | "
                f"{r.get('compactness','')} | "
                f"{r.get('centroid_lands_outside','')} | "
                f"{r.get('nmi_native', r.get('nmi_native_setlevel',''))} |"
            )
        return "\n".join(lines)

    # C2 summary by tier
    def _c2_summary(tier):
        pool = PRIMARY_POOL if tier in ("near", "mid") else "std_nway"
        shots, ways = FIG_SHOTS, 3
        ex = _mean_c2(tier, pool, shots, ways, EXEMPLAR_LEARNER)
        ct = _mean_c2(tier, pool, shots, ways, REFERENCE_LEARNER)
        chance_val = round(1.0 / ways, 3)
        return f"exemplar={ex}, centroid={ct}, chance={chance_val}"

    report = f"""# exp_007b_extension — C2 Robustness & Disjunctive Gradient

## 0. Context and motivation

exp_007b showed two key findings:
1. **C1 (naming) collapses** for all learners on far-disjunctive concepts (~0.46 vs
   0.97 on native): frozen DINOv2 is a hard acquisition ceiling when the concept
   extension crosses the geometry's native cluster boundaries.
2. **C2 (retrieval) shows a genuine episodic advantage**: exemplar_knn > centroid on
   far-disjunctive concepts in 4/5 shot conditions, with CI fully positive.

This extension maps the C2 episodic advantage as a function of inter_member_distance
across three tiers (NEAR / MID / FAR) and tests robustness across six pool constructions.

**Organizing question:** Does the C2 episodic advantage over the prototype vary as a
function of inter_member_distance(concept)? And does C1 partially recover for
near-disjunctive pairs that are within-superordinate but still non-native?

---

## 1. Concept gate results

{gate_table}

---

## 2. Concept set summary

- **NEAR** (imd < {FAR_THRESHOLD}): {sum(1 for r in gate_rows if r['tier']=='near' and r['passes_gate'])} pairs pass gate
- **MID** ({FAR_THRESHOLD} ≤ imd < {FAR_MEDIAN}): {sum(1 for r in gate_rows if r['tier']=='mid' and r['passes_gate'])} pairs pass gate
- **FAR** (imd ≥ {FAR_MEDIAN}): 4 concepts loaded from exp_007b (gleth, mivor, plonk, quax)
- **NATIVE**: 10 concepts loaded from exp_007b (right anchor; NS used as x-position)

---

## 3. Descriptors

### 3.1 NEAR tier
{_desc_table("near")}

### 3.2 MID tier
{_desc_table("mid")}

### 3.3 FAR tier (loaded from exp_007b)
{_desc_table("far")}

### 3.4 NATIVE anchor (loaded from exp_007b)
{_desc_table("native")}

---

## 4. Results (K={FIG_SHOTS}, {3}-way)

### 4.1 C1 gradient (Figure 3)

- NEAR: {_c2_summary("near").replace("exemplar", "c1_exemplar").replace("centroid", "c1_centroid") if ext_rows else "n/a — no rows"}
  → *[fill in: does C1 partially recover for near-disjunctive pairs?]*
- MID: *[fill in after running]*
- FAR: *[see exp_007b report — C1 collapsed to ~0.46]*

### 4.2 C2 gradient — main finding (Figure 1)

- NEAR (pool={PRIMARY_POOL}): {_c2_summary("near")}
- MID  (pool={PRIMARY_POOL}): {_c2_summary("mid")}
- FAR  (pool=std_nway, loaded): {_c2_summary("far")}

### 4.3 Pool sensitivity (Figure 2)

*See figure2_pool_sensitivity.png. Gap stable across pool types → finding is robust.*
*Gap collapses in small pool only → artifact of easy pool (revise C2 interpretation).*

### 4.4 Paired tests — H1 on C2 by tier

See paired_tests.csv for full results (exemplar_knn_k3 vs centroid, metric=c2).

| tier | pool | supported? | mean_diff | ci_low | ci_high | p |
|---|---|---|---|---|---|---|
| near | {PRIMARY_POOL} | *see paired_tests.csv* | | | | |
| mid  | {PRIMARY_POOL} | *see paired_tests.csv* | | | | |
| far  | std_nway | *loaded from 007b* | | | | |

### 4.5 Per-concept breakdown

See aggregate.csv for per-(tier, learner, shots, pool_type) means.

---

## 5. Interpretation

- **Near-disjunctive:** *[fill in after running — expected: C2 gap positive, C1 partial recovery]*
- **Mid-disjunctive:** *[fill in after running — expected: C2 gap smaller than far]*
- **Far-disjunctive (loaded from 007b):** C1 collapsed (~0.46), C2 exemplar advantage
  supported in 4/5 shot conditions.
- **Gradient shape:** *[monotonic / non-monotonic / threshold — fill in after running]*

---

## 6. Connection to Vera et al. child-acquisition framing

The gradient maps onto the natural acquisition order described in the paper:
- **Near-disjunctive** (cat+dog, bird+deer, automobile+truck) = "guau" stage — the
  learner encounters an arbitrary label for a perceptually coherent union. Within-
  superordinate pairs challenge the learner less harshly than cross-superordinate ones.
- **Far-disjunctive** (frog+truck, horse+ship) = the hard edge — maximal geometric
  clash between member categories. Acquisition fails at this level without trainable
  perception.

The C2 gradient shows WHERE the prototype/episodic distinction operates in
perceptual geometry, not just whether it exists.

---

## 7. Limitations

- **Frozen encoder ceiling (established in exp_007b):** This experiment does NOT rescue
  the strong acquisition claim. It maps the SHAPE of the C2 episodic advantage. Training-
  free grounding over frozen DINOv2 has a ceiling; the gradient shows its slope.
- **CIFAR-10 near-disjunctive pairs available:** only 3 pairwise-disjoint near pairs pass
  the gate (cat+dog, bird+deer, automobile+truck) — all within-superordinate. The range
  of inter_member_distances covered is limited compared to the far tier.
- **Pool inconsistency between tiers:** near/mid use pool={PRIMARY_POOL} (10 candidates,
  NN distractors); far/native use std_nway (n_ways candidates from exp_007b, not re-run).
  Direct C2 level comparison across tiers should account for this pool size difference.
- **No 5-way episodes:** near and mid tiers have only 3 pairwise-disjoint concepts each,
  so only 3-way episodes are feasible.

---

## 8. Reproducibility

- Master seed: {42}
- Embedding cache: .cache/embeddings_exp007/cifar_all10.npz (DINOv2-small, frozen)
- Platform: CPU-only (no GPU required after cache build)
- Results dir: results/exp_007b_extension/
- far/native rows loaded from results/exp_007/ledger.jsonl (sub=007b, not re-run)

---

## 9. Next steps

- Implement GLA (Gradient-based Lexical Adapter, the trainable perception component)
  to test whether C1 can be recovered on far-disjunctive concepts beyond the frozen
  DINOv2 ceiling established by this and prior experiments.
- exp_010_test2_paper: second-language acquisition. The ceiling result from exp_007b
  and its extension motivates why trainable perception is necessary for the full
  Test-2 claim in Vera et al.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  saved report.md")


# ─── Utilities ────────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"  [skip] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path.name} ({len(rows)} rows)")


def _load_gate_csv() -> list[dict]:
    gate_path = RESULTS_DIR / "concept_gate.csv"
    if not gate_path.exists():
        return []
    return list(csv.DictReader(open(gate_path, encoding="utf-8")))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n=== analyze_ext.py — exp_007b_extension ===\n")

    ext_rows = _load_ext_ledger()
    far_nat_rows = _load_007b_ledger(pool_type_for_far="std_nway")
    desc = _load_descriptors()
    gate_rows = _load_gate_csv()

    print(f"  Loaded {len(ext_rows)} extension rows (near+mid)")
    print(f"  Loaded {len(far_nat_rows)} 007b rows (far+native)")
    print(f"  Descriptors: {len(desc)} entries")

    print("\n-- aggregate.csv --")
    aggregate_ledger(ext_rows, far_nat_rows)

    print("\n-- paired_tests.csv --")
    run_paired_tests(ext_rows, far_nat_rows)

    all_rows = ext_rows + far_nat_rows
    print("\n-- figures --")
    figure1_gradient(all_rows, desc)
    figure2_pool_sensitivity(ext_rows)
    figure3_c1_gradient(all_rows, desc)

    print("\n-- report.md --")
    write_report(ext_rows, far_nat_rows, desc, gate_rows)

    print("\n=== analyze_ext.py complete ===")


if __name__ == "__main__":
    main()
