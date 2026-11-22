"""STAGE 2 (plan §5.2, B1): per encoder x item set x variant, Delta = mean(beta | "3-1") - mean(beta | "1-3") over the 12 stage-1 slopes.
Exact permutation over all C(12,6) = 924 assignments of the order labels (two-sided p = share of |T*| >= |T_obs|).
CIs by test inversion (shift the "3-1" slopes by delta0, grid step 0.001): 95% CI = {delta0: p2(delta0) > .05}, 90% CI = {p2 > .10}.
One-sided permutation p-values: p_opposite = P(T* <= T_obs) at delta0 = 0 (H1: Delta < 0);
TOST at each delta in DELTAS: p_TOST = max(P(T* >= T_obs | shift -delta), P(T* <= T_obs | shift +delta)).
Secondary: Welch t-test and Welch 95% CI. If any of the 12 slopes is missing -> status "inconcluso_tecnico" (no imputation).
Input: stage-1 tables from 05b_select_structure.py (primary = common rung; per_encoder_rung and S9_mixed descriptive) or from 05 at a fixed rung.
Usage: python 06_stage2_permutation.py <out.csv> <stage1.csv> [stage1.csv ...]"""
import sys, itertools, numpy as np, pandas as pd
from scipy import stats
DELTAS = (0.05, 0.10, 0.20); EPS = 1e-12; STEP = 0.001
COMBOS = np.array([[i in c for i in range(12)] for c in itertools.combinations(range(12), 6)])  # 924 x 12 (True = "3-1")
assert COMBOS.shape == (924, 12)
def tstat(y, A): return y[A].mean() - y[~A].mean()
def perm_parts(y, s):
    """T*(y - delta0*s) = T*(y) - delta0 * T*(s) for every assignment."""
    Ty = COMBOS @ y / 6 - (~COMBOS) @ y / 6; Ts = COMBOS @ s / 6 - (~COMBOS) @ s / 6; return Ty, Ts
def analyse(y, is31):
    y = np.asarray(y, float); s = is31.astype(float); Ty, Ts = perm_parts(y, s); D = tstat(y, is31)
    def p2(d0): t = D - d0; return np.mean(np.abs(Ty - d0 * Ts) >= abs(t) - EPS)
    def pge(d0): t = D - d0; return np.mean((Ty - d0 * Ts) >= t - EPS)
    def ple(d0): t = D - d0; return np.mean((Ty - d0 * Ts) <= t + EPS)
    span = y.max() - y.min() + 1.0
    grid = np.round(np.arange(D - 3 * span, D + 3 * span + STEP, STEP), 6)
    pv = np.array([p2(g) for g in grid]); out = dict(delta_hat=D, p_two_sided=p2(0.0), p_opposite=ple(0.0), p_positive=pge(0.0))
    for lvl, a in (("95", .05), ("90", .10)):
        acc = grid[pv > a]; out[f"ci{lvl}_lo"], out[f"ci{lvl}_hi"] = (acc.min(), acc.max()) if len(acc) else (np.nan, np.nan)
        out[f"ci{lvl}_contiguous"] = bool(len(acc) and np.allclose(np.diff(acc), STEP))
        out[f"ci{lvl}_at_grid_edge"] = bool(len(acc) and (acc.min() <= grid[0] or acc.max() >= grid[-1]))
    for dl in DELTAS:
        out[f"p_tost_{dl:.2f}"] = max(pge(-dl), ple(dl))
    a, b = y[is31], y[~is31]; w = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / 6, b.var(ddof=1) / 6; df = (va + vb) ** 2 / (va ** 2 / 5 + vb ** 2 / 5); h = stats.t.ppf(.975, df) * np.sqrt(va + vb)
    out.update(welch_p=w.pvalue, welch_df=df, welch_ci95_lo=D - h, welch_ci95_hi=D + h, mean_31=a.mean(), mean_13=b.mean()); return out
if __name__ == "__main__":
    s1 = pd.concat([pd.read_csv(f, keep_default_na=True) for f in sys.argv[2:]]); s1["suffix"] = s1["suffix"].fillna("")
    if "converged" in s1.columns:   # rows straight from 05 at a fixed rung (sensitivities): a non-converged fit is a missing slope
        conv = s1.converged.fillna(True).astype(str).str.upper().isin(["TRUE", "1"])
        if "re_used" not in s1.columns: s1["re_used"] = np.nan
        used = s1["rung"].fillna(s1["re_used"]) if "rung" in s1.columns else s1["re_used"]
        s1.loc[~conv, "beta"] = np.nan; s1["re_used"] = np.where(conv, used, "failed")
    assert not s1.duplicated(["encoder", "items", "variant", "suffix", "exp"]).any(), "one stage-1 row per experiment expected (ladder files go through 05b)"
    rows = []
    for k, g in s1.groupby(["encoder", "items", "variant", "suffix"], dropna=False):
        g = g.sort_values("exp"); base = dict(zip(["encoder", "items", "variant", "suffix"], k), n_exp=len(g))
        for t in ("L1", "L2", "L3", "L4", "failed"):      # REVISION3 I1: fit type (ladder rung) x order, per encoder
            for o in ("1-3", "3-1"): base[f"n_{t}_{o}"] = int(((g.re_used == t) & (g.order == o)).sum())
        if len(g) != 12 or g.beta.isna().any() or set(g.order) != {"1-3", "3-1"} or (g.order == "3-1").sum() != 6:
            rows.append(dict(base, status="inconcluso_tecnico")); continue
        rows.append(dict(base, status="ok", **analyse(g.beta.to_numpy(), (g.order == "3-1").to_numpy())))
    out = pd.DataFrame(rows); out.to_csv(sys.argv[1], index=False); print("wrote", sys.argv[1], len(rows), "rows")
    ft = [c for c in out.columns if c.startswith(("n_L1_", "n_L2_", "n_L3_", "n_L4_", "n_failed_"))]
    out[["encoder", "items", "variant", "suffix"] + ft].to_csv(sys.argv[1].replace(".csv", "_fit_type_by_order.csv"), index=False)
