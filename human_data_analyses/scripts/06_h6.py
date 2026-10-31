"""H6 analysis (exploratory, not human data). Uses CVCL embeddings: c10 from cvcl_cifar_all10.npz (same 2,000 images as the
exp_007 DINOv2 cache), c100 from data/cvcl_dperc_c100.csv + data/cvcl_acc_c100.csv (scripts/h6_c100.py) when available.
(a) OLS HC3 acc_pair_CVCL ~ z(d_perc_CVCL)+z(d_sem); (b) Spearman+Mantel CVCL vs DINOv2; (c) CVCL vs each human distance;
(d) tiers descriptive. BH over (a: beta_perc, beta_sem),(b),(c). Partial results saved whatever is available."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, pathlib, itertools, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import spearman_boot, mantel, ols_hc3, bh
B = pathlib.Path(__file__).resolve().parents[1]; out = {"tests": {}}
z = np.load(f"{_P.CACHE}/cvcl_cifar_all10.npz"); E, cat = z["emb"], z["categories"]; E = E / np.linalg.norm(E, axis=1, keepdims=True)
C10 = sorted(set(cat)); cen = {c: E[cat == c].mean(0) for c in C10}; cen = {c: v / np.linalg.norm(v) for c, v in cen.items()}
p10 = pd.read_csv(B / "data/pairs_c10.csv"); p10["d_perc_cvcl"] = [1 - float(cen[a] @ cen[b]) for a, b in zip(p10.a, p10.b)]
def corr(df, x, y, tag):
    d = df.dropna(subset=[x, y]); r, lo, hi = spearman_boot(d[x], d[y]); mr, mp, n = mantel(df, x, y)
    out["tests"][tag] = dict(n=int(n), rho=float(r), ci_lo=float(lo), ci_hi=float(hi), mantel_p=float(mp))
corr(p10, "d_perc_cvcl", "d_perc", "b_c10_cvcl_vs_dino")
for h in ("d_things", "d_hsj", "d_c10h"):
    corr(p10, "d_perc_cvcl", h, f"c_c10_cvcl_vs_{h}"); r = spearman_boot(p10.dropna(subset=[h]).d_perc, p10.dropna(subset=[h])[h]); out[f"ref_c10_dino_vs_{h}"] = float(r[0])
t = json.load(open(B / "out/h6_tiers_cvcl.json")); td = json.load(open(B / "out/h6_tiers_dino.json"))
out["d_tiers"] = dict(c1_by_tier_cvcl=t["c1_by_tier"], c1_by_tier_dino=td["c1_by_tier"])
tc = pd.read_csv(B / "data/tiers_c10.csv"); tc["acc_cvcl"] = [t[f"{a}|{b}"] for a, b in zip(tc.tier, tc.word)]
tc["d_perc_cvcl"] = [1 - float(cen[a] @ cen[b]) for a, b in zip(tc.a, tc.b)]; r = spearman_boot(tc.acc_cvcl, tc.d_perc_cvcl)
out["d_tiers"]["spearman_acc_cvcl_vs_dperc_cvcl"] = dict(n=len(tc), rho=float(r[0]), ci_lo=float(r[1]), ci_hi=float(r[2]))
p10.to_csv(B / "data/h6_pairs_c10.csv", index=False)
fa, fd = B / "data/cvcl_acc_c100.csv", B / "data/cvcl_dperc_c100.csv"
if fd.exists():
    P = pd.read_csv(B / "data/pairs_c100.csv").merge(pd.read_csv(fd), on=["a", "b"], how="left"); assert P.d_perc_cvcl.notna().all()
    corr(P, "d_perc_cvcl", "d_perc", "b_c100_cvcl_vs_dino")
    for h in ("d_things", "d_hsj"): corr(P, "d_perc_cvcl", h, f"c_c100_cvcl_vs_{h}")
if fa.exists():
    s = pd.read_csv(B / "data/sampled_100.csv").merge(pd.read_csv(fa)[["a", "b", "acc_pair_cvcl", "d_perc_cvcl"]], on=["a", "b"]); assert len(s) == 100
    m = ols_hc3(s, "acc_pair_cvcl", ["d_perc_cvcl", "d_sem"]); out["a_ols"] = m
    out["base_acc_c100_cvcl"] = float(s.acc_pair_cvcl.mean()); out["base_acc_c100_dino_paper"] = float(s.acc_pair.mean())
    out["tests"]["a_beta_perc"] = dict(p=m["d_perc_cvcl"]["p"]); out["tests"]["a_beta_sem"] = dict(p=m["d_sem"]["p"])
k = list(out["tests"]); ps = [out["tests"][x].get("mantel_p", out["tests"][x].get("p")) for x in k]
for x, q in zip(k, bh(ps)): out["tests"][x]["p_bh"] = float(q)
out["complete"] = fa.exists() and fd.exists()
json.dump(out, open(B / "out/h6_results.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))
