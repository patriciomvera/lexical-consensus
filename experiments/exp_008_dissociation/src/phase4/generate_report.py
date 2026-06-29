"""
generate_report.py — Phase 4, Step 11. Final step.

Reads all output files from phases 1-4 and produces:
    <run_dir>/report.md           — full human-readable audit trail
    <run_dir>/exp_008_decision.md — which rule triggered + evidence (also written
                                    by dissociation_analysis.py; this script
                                    overwrites with the same content for consistency)

EVERY number in report.md is read from raw CSVs/JSONs — nothing is hardcoded.
The raw-data cross-check section selects 3 pairs deterministically:
positions 0, 33, 66 in sampled_pairs.csv (pre-registered reproducibility rule).

Usage:
    python src/phase4/generate_report.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import get_run_dir, load_config


PRIMARY_LEARNER = "centroid"
LEARNER_NAMES   = ["centroid", "exemplar_knn_k3", "logreg", "random"]
CHECK_POSITIONS = [0, 33, 66]


def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _file_inventory(run_dir: Path) -> list[tuple[str, int, int | None]]:
    """List all files in run_dir with size and (if CSV) row count."""
    rows = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size // 1024
            n_rows = None
            if p.suffix == ".csv":
                try:
                    n_rows = sum(1 for _ in open(p, encoding="utf-8")) - 1  # minus header
                except Exception:
                    pass
            rows.append((str(p.relative_to(run_dir)), size_kb, n_rows))
    return rows


def generate(run_dir: Path, config: dict) -> str:
    lines: list[str] = []

    def h1(s):   lines.append(f"\n# {s}\n")
    def h2(s):   lines.append(f"\n## {s}\n")
    def h3(s):   lines.append(f"\n### {s}\n")
    def row(s):  lines.append(s + "\n")
    def blank(): lines.append("\n")

    # ── Run metadata ──────────────────────────────────────────────────────────
    h1("exp_008 Dissociation Experiment — Run Report")
    h2("Run metadata")

    ts = run_dir.name.replace("exp_008_", "")
    emb_path = run_dir / "phase1" / "raw" / "centroids.npy"
    pairs_path = run_dir / "phase2" / "sampled_pairs.csv"
    ep_path = run_dir / "phase3" / "exp_008_episode_results.csv"

    n_embedded = "N/A"
    emb_cache = run_dir.parents[2] / ".cache" / "embeddings_exp008" / "cifar100.npz"
    if emb_cache.exists():
        d = np.load(emb_cache, allow_pickle=False)
        n_embedded = str(len(d["emb"]))

    row(f"- Timestamp: {ts}")
    row(f"- Master seed: {config['phase3']['master_seed']}")
    row(f"- Encoder: {config['experiment']['encoder']}")
    row(f"- Dataset: CIFAR-100 (torchvision)")
    row(f"- Total images embedded: {n_embedded}")
    row(f"- Embedding dimension: {config['experiment']['embedding_dim']}")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    h2("Phase 1 — Distance Matrices")

    corr_path = run_dir / "phase1" / "phase1_correlation_report.json"
    if corr_path.exists():
        with open(corr_path) as f:
            corr = json.load(f)
        row(f"- Pearson r(d_perc, d_sem): **{_fmt(corr['pearson_r'])}** "
            f"(p={corr['pearson_p']:.2e})")
        row(f"- Spearman ρ(d_perc, d_sem): **{_fmt(corr['spearman_rho'])}** "
            f"(p={corr['spearman_p']:.2e})")
        if corr.get("collinearity_flag"):
            row(f"- ⚠ **COLLINEARITY FLAG**: {corr['interpretation']}")
        else:
            row(f"- Interpretation: {corr['interpretation']}")
        blank()
        row("![Phase 1 scatter](phase1/phase1_scatter.png)")
        blank()

        h3("Top 5 most perceptually distant pairs")
        row("| class_u | class_v | d_perc | d_sem |")
        row("|---------|---------|--------|-------|")
        for p in corr.get("top5_most_perceptually_distant", []):
            row(f"| {p['class_u']} | {p['class_v']} | {p['d_perc']} | {p['d_sem']} |")

        h3("Top 5 most semantically distant pairs")
        row("| class_u | class_v | d_perc | d_sem |")
        row("|---------|---------|--------|-------|")
        for p in corr.get("top5_most_semantically_distant", []):
            row(f"| {p['class_u']} | {p['class_v']} | {p['d_perc']} | {p['d_sem']} |")

        h3("Top 5 most dissociated pairs (largest |rank_perc − rank_sem|)")
        row("| class_u | class_v | d_perc | d_sem | rank_gap |")
        row("|---------|---------|--------|-------|----------|")
        for p in corr.get("top5_most_dissociated", []):
            row(f"| {p['class_u']} | {p['class_v']} | {p['d_perc']} | {p['d_sem']} | {p.get('rank_gap','N/A')} |")
    else:
        row("*(phase1_correlation_report.json not found)*")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    h2("Phase 2 — Pair Selection")

    counts_path = run_dir / "phase2" / "quadrant_counts.json"
    if counts_path.exists():
        with open(counts_path) as f:
            counts = json.load(f)
        row(f"- Pairs per quadrant: "
            f"Q1={counts.get('Q1_concordant_near','?')}, "
            f"Q2={counts.get('Q2_concordant_far','?')}, "
            f"Q3={counts.get('Q3_dissociation_A','?')}, "
            f"Q4={counts.get('Q4_dissociation_B','?')}")

    if pairs_path.exists():
        pairs_df = pd.read_csv(pairs_path)
        n_sampled_per_q = pairs_df["quadrant"].value_counts().to_dict()
        row(f"- Sampled: "
            f"Q1={n_sampled_per_q.get('Q1',0)}, "
            f"Q2={n_sampled_per_q.get('Q2',0)}, "
            f"Q3={n_sampled_per_q.get('Q3',0)}, "
            f"Q4={n_sampled_per_q.get('Q4',0)}")
        blank()
        row("**Sampled pairs (first 10 rows):**")
        row("| sample_id | class_u | class_v | quadrant | d_perc | d_sem |")
        row("|-----------|---------|---------|----------|--------|-------|")
        for _, r in pairs_df.head(10).iterrows():
            row(f"| {r['sample_id']} | {r['class_u']} | {r['class_v']} "
                f"| {r['quadrant']} | {r['d_perc']:.4f} | {r['d_sem']:.4f} |")
        if len(pairs_df) > 10:
            row(f"*(... {len(pairs_df)-10} more rows in sampled_pairs.csv)*")

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    h2("Phase 3 — C1 Acquisition Results")

    agg_path = run_dir / "phase3" / "exp_008_c1_results.csv"
    if ep_path.exists():
        ep_df = pd.read_csv(ep_path)
        row(f"- Total episodes run: {len(ep_df)}")

    if agg_path.exists():
        agg_df = pd.read_csv(agg_path)
        centroid = agg_df[agg_df["learner"] == PRIMARY_LEARNER]

        row(f"\n**Per-quadrant mean C1 (centroid learner):**\n")
        row("| Quadrant | Description | Mean C1 | 95% CI |")
        row("|----------|-------------|---------|--------|")
        for q, desc in [("Q1","both near"), ("Q2","both far"),
                        ("Q3","perc-far, sem-near"), ("Q4","perc-near, sem-far")]:
            sub = centroid[centroid["quadrant"] == q]
            if len(sub) == 0:
                row(f"| {q} | {desc} | N/A | N/A |")
            else:
                mean = sub["c1_accuracy"].mean()
                lo   = sub["c1_ci_low"].mean()
                hi   = sub["c1_ci_high"].mean()
                row(f"| {q} | {desc} | {mean:.4f} | [{lo:.4f}, {hi:.4f}] |")

        blank()
        row("![Quadrant boxplot](phase4/fig_quadrant_boxplot.png)")
        blank()

        row("\n**Per-learner summary (mean C1 by quadrant):**\n")
        row("| Learner | Q1 | Q2 | Q3 | Q4 |")
        row("|---------|----|----|----|----|")
        for lname in LEARNER_NAMES:
            sub = agg_df[agg_df["learner"] == lname]
            vals = []
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                sq = sub[sub["quadrant"] == q]["c1_accuracy"]
                vals.append(f"{sq.mean():.3f}" if len(sq) > 0 else "N/A")
            row(f"| {lname} | {' | '.join(vals)} |")

        # Raw data cross-check (pre-registered positions)
        h3("Raw data cross-check (positions 0, 33, 66 in sampled_pairs.csv)")
        if pairs_path.exists():
            pairs_df = pd.read_csv(pairs_path)
            ep_df_check = pd.read_csv(ep_path) if ep_path.exists() else pd.DataFrame()

            for pos in CHECK_POSITIONS:
                if pos >= len(pairs_df):
                    continue
                pair = pairs_df.iloc[pos]
                row(f"\n**Position {pos}:** {pair['class_u']} + {pair['class_v']} "
                    f"| {pair['quadrant']} | d_perc={pair['d_perc']:.4f} | d_sem={pair['d_sem']:.4f}")
                if len(ep_df_check) > 0:
                    ep_sub = ep_df_check[
                        (ep_df_check["sample_id"] == pair["sample_id"]) &
                        (ep_df_check["learner"] == PRIMARY_LEARNER)
                    ]
                    if len(ep_sub) > 0:
                        c1_vals = ep_sub["c1"].values
                        row(f"- Episode C1 values (centroid): "
                            f"{', '.join(f'{v:.3f}' for v in c1_vals)}")
                        agg_sub = agg_df[
                            (agg_df["sample_id"] == pair["sample_id"]) &
                            (agg_df["learner"] == PRIMARY_LEARNER)
                        ]
                        if len(agg_sub) > 0:
                            r2 = agg_sub.iloc[0]
                            row(f"- Aggregated C1: {r2['c1_accuracy']:.4f} "
                                f"[{r2['c1_ci_low']:.4f}, {r2['c1_ci_high']:.4f}]")

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    h2("Phase 4 — Regression Analysis")

    reg_path = run_dir / "phase4" / "regression_table.csv"
    if reg_path.exists():
        reg_df = pd.read_csv(reg_path)
        cr = reg_df[reg_df["learner"] == PRIMARY_LEARNER]
        if len(cr) > 0:
            r = cr.iloc[0]
            row(f"\n**Model P (perceptual only, centroid learner):**")
            row(f"- β_perc = {_fmt(r['P_beta_perc'])} "
                f"[{_fmt(r['P_beta_perc_ci_low'])}, {_fmt(r['P_beta_perc_ci_high'])}], "
                f"p = {r['P_beta_perc_p']:.4f}, R² = {_fmt(r['P_r2'])}")
            blank()
            row(f"**Model S (semantic only, centroid learner):**")
            row(f"- β_sem = {_fmt(r['S_beta_sem'])} "
                f"[{_fmt(r['S_beta_sem_ci_low'])}, {_fmt(r['S_beta_sem_ci_high'])}], "
                f"p = {r['S_beta_sem_p']:.4f}, R² = {_fmt(r['S_r2'])}")
            blank()
            row(f"**Model PS (both predictors, centroid learner):**")
            row(f"- β_perc = {_fmt(r['PS_beta_perc'])} "
                f"[{_fmt(r['PS_beta_perc_ci_low'])}, {_fmt(r['PS_beta_perc_ci_high'])}], "
                f"p = {r['PS_beta_perc_p']:.4f}")
            row(f"- β_sem = {_fmt(r['PS_beta_sem'])} "
                f"[{_fmt(r['PS_beta_sem_ci_low'])}, {_fmt(r['PS_beta_sem_ci_high'])}], "
                f"p = {r['PS_beta_sem_p']:.4f}")
            row(f"- R² = {_fmt(r['PS_r2'])}, VIF_perc = {_fmt(r['PS_vif_perc'])}, "
                f"VIF_sem = {_fmt(r['PS_vif_sem'])}")
            row(f"- Partial R²_perc = {_fmt(r['PS_partial_r2_perc'])}, "
                f"Partial R²_sem = {_fmt(r['PS_partial_r2_sem'])}")
            blank()
            row(f"**Likelihood ratio test (Model P vs Model PS):** "
                f"χ² = {_fmt(r['lrt_P_vs_PS_chi2'])}, p = {r['lrt_P_vs_PS_p']:.4f}")
            row(f"**Likelihood ratio test (Model S vs Model PS):** "
                f"χ² = {_fmt(r['lrt_S_vs_PS_chi2'])}, p = {r['lrt_S_vs_PS_p']:.4f}")
        blank()
        row("![Partial regression](phase4/fig_partial_regression.png)")

    # ── Dissociation Subset ───────────────────────────────────────────────────
    h2("Dissociation Subset Analysis (Q3 + Q4 only)")

    diss_path = run_dir / "phase4" / "dissociation_analysis.json"
    if diss_path.exists():
        with open(diss_path) as f:
            diss = json.load(f)

        dc = diss["dissociation_subset"].get(PRIMARY_LEARNER, {})
        qm = diss["quadrant_medians"].get(PRIMARY_LEARNER, {})
        row(f"- N pairs in dissociation subset (Q3+Q4): {dc.get('n_pairs', 'N/A')}")
        row(f"- Spearman(C1, d_perc) in Q3+Q4: "
            f"ρ = {_fmt(dc.get('rho_perc'))}, p = {_fmt(dc.get('rho_perc_p'))}")
        row(f"- Spearman(C1, d_sem) in Q3+Q4: "
            f"ρ = {_fmt(dc.get('rho_sem'))}, p = {_fmt(dc.get('rho_sem_p'))}")
        row(f"- Median C1: Q3 = {_fmt(qm.get('Q3'))}, Q4 = {_fmt(qm.get('Q4'))}")
        blank()
        row("![d_perc vs C1](phase4/fig_scatter_perc_vs_c1.png)")
        row("![d_sem  vs C1](phase4/fig_scatter_sem_vs_c1.png)")

    # ── Decision ──────────────────────────────────────────────────────────────
    h2("Decision")

    dec_path = run_dir / "exp_008_decision.md"
    if diss_path.exists():
        with open(diss_path) as f:
            diss2 = json.load(f)
        decision = diss2.get("decision", {})
        rule = decision.get("rule", "?")
        row(f"**Rule triggered: {rule}**")
        blank()
        row("**Evidence:**")
        for e in decision.get("evidence", []):
            row(f"- {e}")
        blank()
        paper_action = {
            "A": "Add §3.X 'The perceptual-coherence gradient is not a measurement tautology' "
                 "with quadrant boxplot as main-text figure.",
            "B": "Add paragraph in Limitations: collinearity between d_perc and d_sem "
                 "prevented clean dissociation. Schedule cross-encoder replication.",
            "C": "Revise Results claim from 'constrained by perceptual geometry' to "
                 "'constrained by concept coherence'. Discuss semantic contribution explicitly.",
        }
        row(f"**Recommended paper action:** {paper_action.get(rule, 'See exp_008_decision.md')}")

    # ── Raw data inventory ────────────────────────────────────────────────────
    h2("Raw Data Inventory")
    row("| File | Size (KB) | Rows |")
    row("|------|-----------|------|")
    for fname, size_kb, n_rows in _file_inventory(run_dir):
        nr = str(n_rows) if n_rows is not None else "—"
        row(f"| {fname} | {size_kb} | {nr} |")

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 4 — generate final report"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    config  = load_config()

    print("[report] generating report.md ...")
    content = generate(run_dir, config)

    report_path = run_dir / "report.md"
    report_path.write_text(content, encoding="utf-8")
    print(f"[report] saved -> {report_path}")
    print(f"\nExp_008 complete. See {run_dir}/report.md for the full audit trail.")
    print(f"Decision: {run_dir}/exp_008_decision.md")


if __name__ == "__main__":
    main()
