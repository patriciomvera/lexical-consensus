"""
regression_analysis.py — Phase 4, Step 8.

Fits three OLS regression models for each learner:
    Model P:  C1 ~ β_perc * d_perc + ε
    Model S:  C1 ~ β_sem  * d_sem  + ε
    Model PS: C1 ~ β_perc * d_perc + β_sem * d_sem + ε

All models include an intercept (added automatically by statsmodels).
Reports coefficients, 95% CIs, p-values, R², partial R² (for Model PS),
VIF, and likelihood-ratio tests (P vs PS, S vs PS).

The primary learner for pre-registered Rule A/B/C decisions is centroid,
matching exp_007's interpretive focus. The other three learners are
robustness checks.

Outputs:
    <run_dir>/phase4/regression_table.csv   — all models × learners
    <run_dir>/phase4/raw/statsmodels_summary_<learner>.txt

Usage:
    python src/phase4/regression_analysis.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import get_run_dir


LEARNER_PRIMARY = "centroid"
LEARNER_NAMES   = ["centroid", "exemplar_knn_k3", "logreg", "random"]


def _partial_r2(model_restricted, model_full) -> float:
    """Partial R² = (RSS_restricted - RSS_full) / RSS_restricted."""
    rss_r = model_restricted.ssr
    rss_f = model_full.ssr
    if rss_r <= 0:
        return float("nan")
    return float((rss_r - rss_f) / rss_r)


def _lrt(model_restricted, model_full) -> tuple[float, float]:
    """Likelihood ratio test: returns (chi2_stat, p_value)."""
    from scipy.stats import chi2
    stat = 2.0 * (model_full.llf - model_restricted.llf)
    df   = model_full.df_model - model_restricted.df_model
    if df <= 0:
        return float("nan"), float("nan")
    pval = float(chi2.sf(stat, df=df))
    return float(stat), pval


def fit_models(
    y: np.ndarray,
    d_perc: np.ndarray,
    d_sem: np.ndarray,
    learner: str,
    raw_dir: Path,
) -> dict:
    """Fit Models P, S, PS. Return a summary dict with all reported quantities."""
    X_const = sm.add_constant(np.ones(len(y)))          # intercept only
    X_perc  = sm.add_constant(d_perc)
    X_sem   = sm.add_constant(d_sem)
    X_both  = sm.add_constant(np.column_stack([d_perc, d_sem]))
    X_both_noconst = np.column_stack([d_perc, d_sem])  # for VIF

    model_P  = sm.OLS(y, X_perc).fit()
    model_S  = sm.OLS(y, X_sem).fit()
    model_PS = sm.OLS(y, X_both).fit()

    # Dump statsmodels summaries for audit
    for m, name in [(model_P, "P"), (model_S, "S"), (model_PS, "PS")]:
        txt_path = raw_dir / f"statsmodels_summary_{learner}_model{name}.txt"
        txt_path.write_text(m.summary().as_text(), encoding="utf-8")

    # VIF for Model PS (only meaningful with both predictors)
    vif_perc = variance_inflation_factor(X_both_noconst, 0)
    vif_sem  = variance_inflation_factor(X_both_noconst, 1)

    # Partial R²: effect of adding d_perc to Model S (how much does perc help beyond sem?)
    model_S_base = sm.OLS(y, X_sem).fit()
    partial_r2_perc = _partial_r2(model_S_base, model_PS)
    # Effect of adding d_sem to Model P
    model_P_base = sm.OLS(y, X_perc).fit()
    partial_r2_sem = _partial_r2(model_P_base, model_PS)

    # LRT: does adding sem help beyond perc alone?
    chi2_P_vs_PS, p_P_vs_PS = _lrt(model_P, model_PS)
    # LRT: does adding perc help beyond sem alone?
    chi2_S_vs_PS, p_S_vs_PS = _lrt(model_S, model_PS)

    def _coef(m, coef_idx, n_params=None):
        """Extract coef, CI low, CI high, p from a fitted model (1-based index).
        conf_int() returns numpy ndarray in statsmodels>=0.14; use direct indexing.
        """
        ci = m.conf_int()
        return {
            "coef":   round(float(m.params[coef_idx]), 5),
            "ci_low": round(float(ci[coef_idx, 0]), 5),
            "ci_high":round(float(ci[coef_idx, 1]), 5),
            "pvalue": float(m.pvalues[coef_idx]),
        }

    ci_P  = model_P.conf_int()
    ci_S  = model_S.conf_int()
    ci_PS = model_PS.conf_int()

    return {
        "learner": learner,
        "n": len(y),
        # Model P
        "P_beta_perc":   round(float(model_P.params[1]), 5),
        "P_beta_perc_ci_low":   round(float(ci_P[1, 0]), 5),
        "P_beta_perc_ci_high":  round(float(ci_P[1, 1]), 5),
        "P_beta_perc_p":        float(model_P.pvalues[1]),
        "P_r2":                 round(float(model_P.rsquared), 4),
        # Model S
        "S_beta_sem":    round(float(model_S.params[1]), 5),
        "S_beta_sem_ci_low":    round(float(ci_S[1, 0]), 5),
        "S_beta_sem_ci_high":   round(float(ci_S[1, 1]), 5),
        "S_beta_sem_p":         float(model_S.pvalues[1]),
        "S_r2":                 round(float(model_S.rsquared), 4),
        # Model PS
        "PS_beta_perc":  round(float(model_PS.params[1]), 5),
        "PS_beta_perc_ci_low":  round(float(ci_PS[1, 0]), 5),
        "PS_beta_perc_ci_high": round(float(ci_PS[1, 1]), 5),
        "PS_beta_perc_p":       float(model_PS.pvalues[1]),
        "PS_beta_sem":   round(float(model_PS.params[2]), 5),
        "PS_beta_sem_ci_low":   round(float(ci_PS[2, 0]), 5),
        "PS_beta_sem_ci_high":  round(float(ci_PS[2, 1]), 5),
        "PS_beta_sem_p":        float(model_PS.pvalues[2]),
        "PS_r2":                round(float(model_PS.rsquared), 4),
        "PS_vif_perc":          round(float(vif_perc), 3),
        "PS_vif_sem":           round(float(vif_sem), 3),
        "PS_partial_r2_perc":   round(float(partial_r2_perc), 4),
        "PS_partial_r2_sem":    round(float(partial_r2_sem), 4),
        # Likelihood ratio tests
        "lrt_P_vs_PS_chi2":     round(float(chi2_P_vs_PS), 4),
        "lrt_P_vs_PS_p":        float(p_P_vs_PS),
        "lrt_S_vs_PS_chi2":     round(float(chi2_S_vs_PS), 4),
        "lrt_S_vs_PS_p":        float(p_S_vs_PS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 4 — regression analysis"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    raw_dir = run_dir / "phase4" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    agg_path = run_dir / "phase3" / "exp_008_c1_results.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing: {agg_path}. Run run_episodes.py first.")

    df = pd.read_csv(agg_path)
    print(f"[regression] loaded {len(df)} rows ({df['sample_id'].nunique()} pairs)")

    results = []
    for lname in LEARNER_NAMES:
        sub = df[df["learner"] == lname].copy()
        if len(sub) == 0:
            print(f"  [skip] no data for learner '{lname}'")
            continue
        y      = sub["c1_accuracy"].values.astype(np.float64)
        d_perc = sub["d_perc"].values.astype(np.float64)
        d_sem  = sub["d_sem"].values.astype(np.float64)
        print(f"  [{lname}] n={len(sub)}, mean C1={y.mean():.3f}")
        row = fit_models(y, d_perc, d_sem, lname, raw_dir)
        results.append(row)
        print(f"    PS: β_perc={row['PS_beta_perc']:.4f}(p={row['PS_beta_perc_p']:.3f}), "
              f"β_sem={row['PS_beta_sem']:.4f}(p={row['PS_beta_sem_p']:.3f}), "
              f"R²={row['PS_r2']:.3f}, VIF_perc={row['PS_vif_perc']:.2f}")

    out_df = pd.DataFrame(results)
    out_path = run_dir / "phase4" / "regression_table.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n[regression] saved -> {out_path}")
    print(f"\nNext step:\n  python src/phase4/dissociation_analysis.py")


if __name__ == "__main__":
    main()
