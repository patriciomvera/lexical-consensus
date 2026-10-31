"""H1 (convergence), H2 (dissociation with human axis), H2b, robustness. Raw p-values; Holm applied in 05_holm_report.py."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import *
B = pathlib.Path(_P.HDA); D, O = B / "data", B / "out"
c100 = pd.read_csv(D / "pairs_c100.csv"); c10 = pd.read_csv(D / "pairs_c10.csv")
res = {"H1": {}, "H1_desc_c10": {}, "H2": {}, "H2b": {}, "H2_rob": {}}
for src, df, col in [("THINGS", c100, "d_things"), ("HSJ", c100, "d_hsj"), ("CIFAR10H", c10, "d_c10h")]:
    d = df.dropna(subset=[col]); rho, lo, hi = spearman_boot(d.d_perc, d[col]); mr, mp, n = mantel(df, "d_perc", col)
    res["H1"][src] = dict(n_pairs=len(d), rho=rho, ci_lo=lo, ci_hi=hi, mantel_r=mr, mantel_p=mp); print("H1", src, res["H1"][src])
for src, col in [("THINGS", "d_things"), ("HSJ", "d_hsj")]:
    d = c10.dropna(subset=[col]); rho, lo, hi = spearman_boot(d.d_perc, d[col]); mr, mp, n = mantel(c10, "d_perc", col)
    res["H1_desc_c10"][src] = dict(n_pairs=len(d), rho=rho, ci_lo=lo, ci_hi=hi, mantel_r=mr, mantel_p=mp)
for (s1, c1), (s2, c2) in [(("THINGS", "d_things"), ("HSJ", "d_hsj"))]:
    d = c100.dropna(subset=[c1, c2]); rho, lo, hi = spearman_boot(d[c1], d[c2]); mr, mp, n = mantel(c100, c1, c2)
    res["agree_c100_THINGS_HSJ"] = dict(n_pairs=len(d), n_sampled=int(d.acc_pair.notna().sum()), rho=rho, ci_lo=lo, ci_hi=hi, mantel_r=mr, mantel_p=mp)
for c1, c2 in [("d_things", "d_hsj"), ("d_things", "d_c10h"), ("d_hsj", "d_c10h")]:
    d = c10.dropna(subset=[c1, c2]); rho, lo, hi = spearman_boot(d[c1], d[c2]); mr, mp, n = mantel(c10, c1, c2)
    res[f"agree_c10_{c1}_{c2}"] = dict(n_pairs=len(d), rho=rho, ci_lo=lo, ci_hi=hi, mantel_r=mr, mantel_p=mp)
s = c100.dropna(subset=["acc_pair"]).copy()
s["d_hsj_ge1"] = np.where(s.hsj_n >= 1, s.d_hsj_raw, np.nan); s["d_hsj_ge30"] = np.where(s.hsj_n >= 30, s.d_hsj_raw, np.nan)
s.to_csv(D / "sampled_100.csv", index=False)
rng = np.random.default_rng(SEED)
for src, col in [("THINGS", "d_things"), ("HSJ", "d_hsj")]:
    d = s.dropna(subset=[col]).reset_index(drop=True)
    res["H2"][src] = ols_hc3(d, "acc_pair", [col, "d_sem"]); res["H2"][src]["quadrants"] = d.quadrant.value_counts().to_dict()
    m = ols_hc3(d, "acc_pair", ["d_perc", col, "d_sem"]); diffs = []
    for _ in range(NB):
        i = rng.integers(0, len(d), len(d)); dd = d.iloc[i].reset_index(drop=True)
        try: mm = ols_hc3(dd, "acc_pair", ["d_perc", col, "d_sem"]); diffs.append(mm["d_perc"]["beta"] - mm[col]["beta"])
        except Exception: pass
    m["diff_beta_perc_minus_hum"] = dict(est=m["d_perc"]["beta"] - m[col]["beta"], ci_lo=float(np.percentile(diffs, 2.5)), ci_hi=float(np.percentile(diffs, 97.5)), n_boot=len(diffs))
    m["paper_perc_same_subset"] = ols_hc3(d, "acc_pair", ["d_perc", "d_sem"])
    res["H2b"][src] = m; print("H2", src, {k: v for k, v in res["H2"][src][col].items()})
rob = {}
rob["THINGS_origmap_dsem"] = ols_hc3(s.dropna(subset=["d_things"]), "acc_pair", ["d_things", "d_sem_orig"])
rob["THINGS_scout"] = ols_hc3(s.dropna(subset=["d_things_scout"]), "acc_pair", ["d_things_scout", "d_sem"])
rob["THINGS_strict"] = ols_hc3(s.dropna(subset=["d_things_strict"]), "acc_pair", ["d_things_strict", "d_sem"])
rob["HSJ_origmap_dsem"] = ols_hc3(s.dropna(subset=["d_hsj"]), "acc_pair", ["d_hsj", "d_sem_orig"])
rob["HSJ_origmap_both"] = ols_hc3(s.dropna(subset=["d_hsj_origmap"]), "acc_pair", ["d_hsj_origmap", "d_sem_orig"])
rob["HSJ_ge1"] = ols_hc3(s.dropna(subset=["d_hsj_ge1"]), "acc_pair", ["d_hsj_ge1", "d_sem"])
rob["HSJ_ge30"] = ols_hc3(s.dropna(subset=["d_hsj_ge30"]), "acc_pair", ["d_hsj_ge30", "d_sem"])
for src, col in [("THINGS", "d_things"), ("HSJ", "d_hsj")]:
    d = s[s.quadrant.isin(["Q3", "Q4"])].dropna(subset=[col])
    rob[f"{src}_Q3Q4_desc"] = ols_hc3(d, "acc_pair", [col, "d_sem"]) if len(d) >= 6 else {"n": len(d)}
res["H2_rob"] = rob
json.dump(res, open(O / "h1_h2_results.json", "w"), indent=1, default=float); print("ok")
