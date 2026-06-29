"""
dissociation_analysis.py — Phase 4, Step 9.

Secondary analysis restricted to dissociation quadrants Q3 and Q4.

For each learner:
  - Spearman(C1, d_perc) in Q3+Q4
  - Spearman(C1, d_sem)  in Q3+Q4

Tertiary analysis: same-superclass pairs vs different-superclass pairs,
controlling for d_perc (OLS with superclass_same dummy).

Also evaluates the pre-registered Rule A/B/C decision criteria for the
primary (centroid) learner and writes exp_008_decision.md.

Outputs:
    <run_dir>/phase4/dissociation_analysis.json
    <run_dir>/exp_008_decision.md

Usage:
    python src/phase4/dissociation_analysis.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import get_run_dir, load_config


LEARNER_PRIMARY = "centroid"
LEARNER_NAMES   = ["centroid", "exemplar_knn_k3", "logreg", "random"]


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


def _apply_rule(cfg: dict, reg: dict, diss: dict) -> tuple[str, list[str]]:
    """Evaluate pre-registered Rule A / B / C on centroid learner.

    Returns: (rule_letter, list_of_evidence_strings)
    """
    evidence = []

    # Extract centroid row from regression table
    beta_perc_p  = reg.get("PS_beta_perc_p", 1.0)
    beta_perc    = reg.get("PS_beta_perc", 0.0)
    beta_sem_p   = reg.get("PS_beta_sem_p", 0.0)
    beta_sem     = reg.get("PS_beta_sem", 0.0)
    partial_r2_p = reg.get("PS_partial_r2_perc", 0.0)
    partial_r2_s = reg.get("PS_partial_r2_sem", 0.0)

    # Dissociation stats for centroid
    rho_perc = diss.get("rho_perc", float("nan"))
    rho_sem  = diss.get("rho_sem",  float("nan"))
    n_diss   = diss.get("n_pairs",  0)

    # Median Q3 vs Q4
    q3_median = diss.get("q3_median_c1", float("nan"))
    q4_median = diss.get("q4_median_c1", float("nan"))

    a_pval_thr  = cfg["phase4"]["rule_a_beta_perc_pvalue_threshold"]
    a_sem_thr   = cfg["phase4"]["rule_a_beta_sem_pvalue_threshold"]
    a_r2_ratio  = cfg["phase4"]["rule_a_partial_r2_ratio"]
    a_rho_min   = cfg["phase4"]["rule_a_dissociation_rho_min"]
    c_rho_gap   = cfg["phase4"]["rule_c_rho_gap"]

    # Rule C checks (checked first — if true, no need to check A)
    rule_c_flags = []
    if beta_perc_p >= a_pval_thr and beta_sem_p < a_pval_thr:
        rule_c_flags.append(
            f"β_perc NOT significant (p={beta_perc_p:.3f}) but β_sem IS (p={beta_sem_p:.3f})"
        )
    if (not np.isnan(rho_perc) and not np.isnan(rho_sem) and
            abs(rho_sem) > abs(rho_perc) + c_rho_gap):
        rule_c_flags.append(
            f"|ρ_sem|={abs(rho_sem):.3f} > |ρ_perc|+{c_rho_gap}={abs(rho_perc)+c_rho_gap:.3f}"
        )
    if (not np.isnan(q3_median) and not np.isnan(q4_median) and q3_median > q4_median):
        rule_c_flags.append(
            f"Q3 median C1={q3_median:.3f} > Q4 median C1={q4_median:.3f} (opposite of Rule A prediction)"
        )

    if rule_c_flags:
        evidence.extend(rule_c_flags)
        return "C", evidence

    # Rule A checks (all must hold)
    rule_a_flags: list[bool] = []

    a1 = (beta_perc_p < a_pval_thr and beta_perc < 0)
    rule_a_flags.append(a1)
    if a1:
        evidence.append(f"A1: β_perc={beta_perc:.4f} significant (p={beta_perc_p:.3f}) and negative ✓")
    else:
        evidence.append(f"A1 FAIL: β_perc={beta_perc:.4f} p={beta_perc_p:.3f} "
                        f"(need p<{a_pval_thr} and β<0)")

    a2_sem_ns   = beta_sem_p >= a_sem_thr
    a2_r2_ratio = (partial_r2_s > 0 and
                   partial_r2_p / max(partial_r2_s, 1e-9) >= a_r2_ratio)
    a2 = a2_sem_ns or a2_r2_ratio
    rule_a_flags.append(a2)
    if a2:
        evidence.append(
            f"A2: β_sem not dominant (p={beta_sem_p:.3f}, "
            f"partial_R²_perc/partial_R²_sem={partial_r2_p:.4f}/{partial_r2_s:.4f}) ✓"
        )
    else:
        evidence.append(
            f"A2 FAIL: β_sem significant (p={beta_sem_p:.3f} < {a_sem_thr}) "
            f"AND partial_R² ratio={partial_r2_p/max(partial_r2_s,1e-9):.2f} < {a_r2_ratio}"
        )

    a3 = (not np.isnan(rho_perc) and abs(rho_perc) > a_rho_min and
          rho_perc < 0 and abs(rho_perc) > abs(rho_sem))
    rule_a_flags.append(a3)
    if a3:
        evidence.append(
            f"A3: Spearman(C1,d_perc) in Q3+Q4 = {rho_perc:.3f} (negative, |ρ|>{a_rho_min}, "
            f"|ρ_perc|>|ρ_sem|={abs(rho_sem):.3f}) ✓"
        )
    else:
        evidence.append(
            f"A3 FAIL: ρ_perc={rho_perc:.3f}, ρ_sem={rho_sem:.3f}, "
            f"n_diss={n_diss} (need n≥20, ρ<0, |ρ|>{a_rho_min}, |ρ_perc|>|ρ_sem|)"
        )

    a4 = (not np.isnan(q4_median) and not np.isnan(q3_median) and q4_median > q3_median)
    rule_a_flags.append(a4)
    if a4:
        evidence.append(f"A4: Q4 median C1={q4_median:.3f} > Q3 median C1={q3_median:.3f} ✓")
    else:
        evidence.append(
            f"A4 FAIL: Q4 median C1={q4_median:.3f} NOT > Q3 median C1={q3_median:.3f}"
        )

    if all(rule_a_flags):
        return "A", evidence

    # Rule B: neither A nor C
    evidence.append(
        f"Rule B: not all Rule A criteria met, no Rule C criteria met — "
        f"collinearity or insufficient dissociation power."
    )
    return "B", evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 4 — dissociation subset analysis"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    config  = load_config()

    agg_path = run_dir / "phase3" / "exp_008_c1_results.csv"
    reg_path = run_dir / "phase4" / "regression_table.csv"

    if not agg_path.exists():
        raise FileNotFoundError(f"Missing: {agg_path}")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing: {reg_path}. Run regression_analysis.py first.")

    from _shared import CLASS_TO_SUPERCLASS

    df     = pd.read_csv(agg_path)
    reg_df = pd.read_csv(reg_path)

    # Add same_superclass flag if not already present
    if "same_superclass" not in df.columns:
        df["same_superclass"] = df.apply(
            lambda r: CLASS_TO_SUPERCLASS.get(r["class_u"]) == CLASS_TO_SUPERCLASS.get(r["class_v"]),
            axis=1,
        )

    results: dict = {
        "dissociation_subset": {},
        "quadrant_medians": {},
        "superclass_analysis": {},
    }

    # Per-learner dissociation subset analysis
    for lname in LEARNER_NAMES:
        sub = df[df["learner"] == lname]
        diss = sub[sub["quadrant"].isin(["Q3", "Q4"])].copy()
        n_diss = len(diss)

        rho_perc, p_perc = _spearman(diss["d_perc"].values, diss["c1_accuracy"].values)
        rho_sem,  p_sem  = _spearman(diss["d_sem"].values,  diss["c1_accuracy"].values)

        # Median C1 per quadrant
        q_medians = {
            q: float(sub[sub["quadrant"] == q]["c1_accuracy"].median())
            for q in ["Q1", "Q2", "Q3", "Q4"]
        }

        results["dissociation_subset"][lname] = {
            "n_pairs": n_diss,
            "rho_perc": round(rho_perc, 4) if not np.isnan(rho_perc) else None,
            "rho_perc_p": round(p_perc, 4) if not np.isnan(p_perc) else None,
            "rho_sem":  round(rho_sem,  4) if not np.isnan(rho_sem)  else None,
            "rho_sem_p":  round(p_sem,  4) if not np.isnan(p_sem)    else None,
        }
        results["quadrant_medians"][lname] = {
            q: round(v, 4) for q, v in q_medians.items()
        }

        print(f"  [{lname}] Q3+Q4 n={n_diss}: "
              f"ρ_perc={rho_perc:.3f}(p={p_perc:.3f}), "
              f"ρ_sem={rho_sem:.3f}(p={p_sem:.3f})")

    # Tertiary: same-superclass vs different-superclass, controlling for d_perc
    import statsmodels.api as sm
    centroid_df = df[df["learner"] == LEARNER_PRIMARY].copy()
    if len(centroid_df) > 0:
        y     = centroid_df["c1_accuracy"].values.astype(np.float64)
        dp    = centroid_df["d_perc"].values.astype(np.float64)
        sc    = centroid_df["same_superclass"].values.astype(np.float64) \
                    if "same_superclass" in centroid_df.columns \
                    else np.zeros(len(y))
        X = sm.add_constant(np.column_stack([dp, sc]))
        m = sm.OLS(y, X).fit()
        _ci_sc = m.conf_int()   # numpy ndarray in statsmodels>=0.14
        results["superclass_analysis"] = {
            "beta_d_perc":        round(float(m.params[1]), 5),
            "beta_same_sc":       round(float(m.params[2]), 5),
            "beta_same_sc_p":     float(m.pvalues[2]),
            "beta_same_sc_ci_low":  round(float(_ci_sc[2, 0]), 5),
            "beta_same_sc_ci_high": round(float(_ci_sc[2, 1]), 5),
            "r2": round(float(m.rsquared), 4),
            "interpretation": (
                "same-superclass pairs have significantly higher C1 after controlling for d_perc"
                if m.pvalues[2] < 0.05 else
                "no significant superclass effect beyond d_perc"
            ),
        }

    # Apply decision rule using centroid learner
    reg_centroid = reg_df[reg_df["learner"] == LEARNER_PRIMARY]
    reg_row = reg_centroid.iloc[0].to_dict() if len(reg_centroid) > 0 else {}

    diss_centroid = results["dissociation_subset"].get(LEARNER_PRIMARY, {})
    q_med = results["quadrant_medians"].get(LEARNER_PRIMARY, {})
    diss_for_rule = {
        "n_pairs":      diss_centroid.get("n_pairs", 0),
        "rho_perc":     diss_centroid.get("rho_perc") or float("nan"),
        "rho_sem":      diss_centroid.get("rho_sem") or float("nan"),
        "q3_median_c1": q_med.get("Q3", float("nan")),
        "q4_median_c1": q_med.get("Q4", float("nan")),
    }

    rule, evidence = _apply_rule(config, reg_row, diss_for_rule)
    results["decision"] = {"rule": rule, "evidence": evidence}

    # Save JSON
    out_path = run_dir / "phase4" / "dissociation_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[dissociation] saved -> {out_path}")

    # Write decision doc
    decision_lines = [
        f"# exp_008 Decision\n",
        f"**Rule triggered: {rule}**\n",
        f"\n## Evidence (centroid learner)\n",
    ]
    for e in evidence:
        decision_lines.append(f"- {e}\n")

    paper_action = {
        "A": "Add §3.X 'The perceptual-coherence gradient is not a measurement tautology' "
             "with quadrant boxplot as main-text figure.",
        "B": "Add paragraph in Limitations: collinearity between d_perc and d_sem "
             "prevented clean dissociation. Schedule cross-encoder replication.",
        "C": "Revise Results claim from 'constrained by perceptual geometry' to "
             "'constrained by concept coherence'. Discuss semantic contribution explicitly.",
    }
    decision_lines.append(f"\n## Recommended paper action\n{paper_action[rule]}\n")

    dec_path = run_dir / "exp_008_decision.md"
    dec_path.write_text("".join(decision_lines), encoding="utf-8")
    print(f"[dissociation] decision doc -> {dec_path}")
    print(f"\n*** PRE-REGISTERED DECISION: Rule {rule} ***")
    print(f"\nNext step:\n  python src/phase4/generate_figures.py")


if __name__ == "__main__":
    main()
