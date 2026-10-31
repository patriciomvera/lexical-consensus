"""Plan §4: between-source dispersion. Forest plot (beta_hum and partial r per source, with paper beta_perc on the same subset),
random-effects I2/tau (DerSimonian-Laird; descriptive, sources share pairs), combined models (composite z; long-format
interaction LR test; intersection), Bland-Altman with Delta = z(d_hum) - z(d_perc) and the top-10 |Delta| pairs per source."""
import sys, json, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import ols_hc3, z
B = pathlib.Path(__file__).resolve().parents[1]; (B / "figs").mkdir(exist_ok=True)
R = json.load(open(B / "out/h1_h2_results.json")); s = pd.read_csv(B / "data/sampled_100.csv")
SRC = {"THINGS": "d_things", "HSJ": "d_hsj"}; out = {}
def dl(b, se):
    w = 1 / se**2; bf = (w * b).sum() / w.sum(); Q = (w * (b - bf)**2).sum(); k = len(b)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum())); I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    wr = 1 / (se**2 + t2); br = (wr * b).sum() / wr.sum(); ser = np.sqrt(1 / wr.sum())
    return dict(k=k, Q=float(Q), I2=float(I2), tau=float(np.sqrt(t2)), beta_RE=float(br), ci_lo=float(br - 1.96 * ser), ci_hi=float(br + 1.96 * ser))
b = np.array([R["H2"][k][c]["beta"] for k, c in SRC.items()]); se = np.array([R["H2"][k][c]["se_hc3"] for k, c in SRC.items()])
out["meta_beta_hum"] = dl(b, se)
r = np.array([R["H2"][k][c]["r_partial"] for k, c in SRC.items()]); n = np.array([R["H2"][k]["n"] for k in SRC])
out["meta_rpartial_fisherz"] = dl(np.arctanh(r), 1 / np.sqrt(n - 4)); out["range_beta_hum"] = [float(b.min()), float(b.max())]
# forest plot
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); lab, yy = [], 0
for k, c in SRC.items():
    h = R["H2"][k][c]; p = R["H2b"][k]["paper_perc_same_subset"]["d_perc"]
    for nm, e in ((f"{k}: β_hum (n={R['H2'][k]['n']})", h), (f"{k}: β_perc paper, mismo subconjunto", p)):
        ax[0].errorbar(e["beta"], yy, xerr=[[e["beta"] - e["ci_lo"]], [e["ci_hi"] - e["beta"]]], fmt="o" if "hum" in nm else "s", color="k" if "hum" in nm else "grey")
        ax[1].errorbar(e["r_partial"], yy, xerr=[[e["r_partial"] - e["r_ci_lo"]], [e["r_ci_hi"] - e["r_partial"]]], fmt="o" if "hum" in nm else "s", color="k" if "hum" in nm else "grey")
        lab.append(nm); yy += 1
m = out["meta_beta_hum"]; ax[0].errorbar(m["beta_RE"], yy, xerr=[[m["beta_RE"] - m["ci_lo"]], [m["ci_hi"] - m["beta_RE"]]], fmt="D", color="C0"); lab.append(f"RE β_hum (I²={m['I2']:.2f}, τ={m['tau']:.3f})")
for a, t in zip(ax, ("β (z, HC3 IC95%)", "r parcial (IC95%)")): a.axvline(0, ls=":", c="k"); a.set_yticks(range(len(lab))); a.set_xlabel(t); a.invert_yaxis()
ax[1].axvline(-0.2, ls="--", c="r", lw=.8); ax[0].set_yticklabels(lab, fontsize=8); ax[1].set_yticklabels([]); plt.tight_layout(); plt.savefig(B / "figs/forest_h2.png", dpi=150); plt.close()
# combined (a) composite z over the union; (c) intersection
s2 = s.copy()
for k, c in SRC.items(): s2["z_" + k] = (s2[c] - s2[c].mean()) / s2[c].std()
s2["d_comp"] = s2[["z_THINGS", "z_HSJ"]].mean(axis=1, skipna=True)
u = s2.dropna(subset=["d_comp"]); out["combined_a_composite_union"] = ols_hc3(u, "acc_pair", ["d_comp", "d_sem"])
it = s2.dropna(subset=["z_THINGS", "z_HSJ"]); out["combined_c_intersection_composite"] = ols_hc3(it, "acc_pair", ["d_comp", "d_sem"])
for k, c in SRC.items(): out[f"combined_c_intersection_{k}"] = ols_hc3(it, "acc_pair", [c, "d_sem"])
# (b) long-format, random intercept per pair, LR test of z(d_hum) x source (ML fits)
L = pd.concat([pd.DataFrame(dict(pair=g.sample_id, acc=g.acc_pair, zh=z(g[c]), zs=z(g.d_sem), source=k)) for k, c in SRC.items() for g in [s.dropna(subset=[c])]])
# acc is identical for a pair across sources, so a pair random intercept absorbs all residual variance (MixedLM degenerate,
# LR<0: see DESVIACIONES). Fallback: OLS with pair-clustered SEs; LR test of the interaction from the nested OLS fits.
g0 = smf.ols("acc ~ zh + source + zs", L).fit(); g1 = smf.ols("acc ~ zh * source + zs", L).fit()
gc = smf.ols("acc ~ zh * source + zs", L).fit(cov_type="cluster", cov_kwds={"groups": L["pair"]}); it_ = "zh:source[T.THINGS]"
lr = 2 * (g1.llf - g0.llf); out["combined_b_long_LR"] = dict(n_rows=len(L), n_pairs=int(L.pair.nunique()), model="OLS, pair-clustered SE (MixedLM degenerate)",
    LR=float(lr), df=1, p_LR=float(stats.chi2.sf(lr, 1)), beta_zh_HSJ=float(g1.params["zh"]), beta_int_THINGS_minus_HSJ=float(g1.params[it_]), p_int_cluster_wald=float(gc.pvalues[it_]))
# Bland-Altman over all covered pairs
P = pd.read_csv(B / "data/pairs_c100.csv"); P10 = pd.read_csv(B / "data/pairs_c10.csv"); fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); ba = {}
for i, (k, df, c) in enumerate((("THINGS", P, "d_things"), ("HSJ", P, "d_hsj"), ("CIFAR10H", P10, "d_c10h"))):
    d = df.dropna(subset=[c]).copy(); d["zh"] = z(d[c]); d["zp"] = z(d.d_perc); d["delta"] = d.zh - d.zp; d["mean"] = (d.zh + d.zp) / 2
    sd = d.delta.std(); ba[k] = dict(n=len(d), sd_delta=float(sd), mean_delta=float(d.delta.mean()),
        top10=d.reindex(d.delta.abs().sort_values(ascending=False).index).head(10)[["a", "b", c, "d_perc", "delta"]].round(3).to_dict("records"))
    ax[i].scatter(d["mean"], d.delta, s=4 if len(d) > 100 else 15, alpha=.4); ax[i].axhline(0, c="k", lw=.5)
    for v in (-1.96 * sd, 1.96 * sd): ax[i].axhline(v, ls="--", c="r", lw=.8)
    ax[i].set_title(f"{k} (n={len(d)}, DE Δ={sd:.2f})"); ax[i].set_xlabel("media de z"); ax[0].set_ylabel("Δ = z(d_hum) − z(d_perc)")
plt.tight_layout(); plt.savefig(B / "figs/bland_altman.png", dpi=150); plt.close(); out["bland_altman"] = ba
json.dump(out, open(B / "out/dispersion_results.json", "w"), indent=1, default=float); print(json.dumps({k: v for k, v in out.items() if k != "bland_altman"}, indent=1, default=float))
for k, v in ba.items(): print(k, v["n"], round(v["sd_delta"], 3), [(t["a"], t["b"], t["delta"]) for t in v["top10"]])
