import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
SEED = 42; NB = 10_000
def rank(x): return stats.rankdata(x)
def spearman_boot(x, y, seed=SEED, nb=NB):
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    rho = stats.spearmanr(x, y).statistic; rng = np.random.default_rng(seed); bs = np.empty(nb)
    for b in range(nb):
        i = rng.integers(0, n, n); bs[b] = stats.spearmanr(x[i], y[i]).statistic
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return rho, lo, hi
def mantel(pairs_df, col_x, col_y, a="a", b="b", seed=SEED, nperm=NB):
    """Class-label permutation of the y matrix; Spearman over pairs with both values; two-sided p."""
    d = pairs_df[[a, b, col_x, col_y]]
    classes = sorted(set(d[a]) | set(d[b])); ix = {c: i for i, c in enumerate(classes)}; K = len(classes)
    MX = np.full((K, K), np.nan); MY = np.full((K, K), np.nan)
    for r in d.itertuples(index=False):
        i, j = ix[r[0]], ix[r[1]]; MX[i, j] = MX[j, i] = r[2]; MY[i, j] = MY[j, i] = r[3]
    iu = np.triu_indices(K, 1); x = MX[iu]; y = MY[iu]; ok = ~np.isnan(x) & ~np.isnan(y)
    obs = stats.spearmanr(x[ok], y[ok]).statistic; rng = np.random.default_rng(seed); cnt = 0
    for _ in range(nperm):
        p = rng.permutation(K); yp = MY[p][:, p][iu]; ok2 = ~np.isnan(x) & ~np.isnan(yp)
        r = stats.spearmanr(x[ok2], yp[ok2]).statistic
        cnt += abs(r) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (nperm + 1), int(ok.sum())
z = lambda v: (v - v.mean()) / v.std(ddof=1)
def ols_hc3(df, y, xs):
    X = sm.add_constant(pd.DataFrame({x: z(df[x]) for x in xs})); m = sm.OLS(df[y].values, X).fit(cov_type="HC3")
    m_cl = sm.OLS(df[y].values, X).fit(); out = {"n": len(df), "R2": m_cl.rsquared}
    dfres = m_cl.df_resid
    for x in xs:
        others = [o for o in xs if o != x]
        if others:
            Xo = sm.add_constant(pd.DataFrame({o: z(df[o]) for o in others}))
            ry = sm.OLS(df[y].values, Xo).fit().resid; rx = sm.OLS(z(df[x]).values, Xo).fit().resid
        else: ry, rx = df[y].values - df[y].mean(), z(df[x]).values
        pr = np.corrcoef(rx, ry)[0, 1]; zf = np.arctanh(pr); se = 1 / np.sqrt(len(df) - len(others) - 3)
        t_cl = m_cl.tvalues[x]
        out[x] = dict(beta=m.params[x], se_hc3=m.bse[x], ci_lo=m.conf_int().loc[x, 0], ci_hi=m.conf_int().loc[x, 1], p=m.pvalues[x],
                      r_partial=pr, r_ci_lo=np.tanh(zf - 1.96 * se), r_ci_hi=np.tanh(zf + 1.96 * se), R2_partial=t_cl**2 / (t_cl**2 + dfres))
    return out
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o): run = max(run, (m - k) * p[i]); adj[i] = min(1, run)
    return adj
def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)[::-1]; adj = np.empty(m); run = 1
    for k, i in enumerate(o): run = min(run, p[i] * m / (m - k)); adj[i] = run
    return adj
