"""Partial correlation of naming accuracy with semantic distance given perceptual distance,
within the discordant quadrants Q3+Q4 (selection makes d_perc and d_sem anti-correlated there).
Light analysis on committed per-pair results only (exp_008 phase3 c1 results), plus the corrected-mapping
quadrants recomputed exactly as in r01. Partial Spearman = Pearson correlation of residuals of ranks.
"""
import json, sys, numpy as np, pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn
R = sys.argv[1] if len(sys.argv) > 1 else "results/exp_008_dissociation"
OUT = sys.argv[2] if len(sys.argv) > 2 else "revision_analyses/out"
c1 = pd.read_csv(f"{R}/phase3/exp_008_c1_results.csv")
mapping = json.load(open(f"{R}/phase1/wordnet_mappings_used.json"))
CORR = {"beaver": "beaver.n.07", "flatfish": "flatfish.n.02", "leopard": "leopard.n.02",
        "lobster": "lobster.n.02", "maple_tree": "maple.n.02", "oak_tree": "oak.n.02",
        "otter": "otter.n.02", "palm_tree": "palm.n.03", "raccoon": "raccoon.n.02",
        "ray": "ray.n.07", "shrew": "shrew.n.02", "skunk": "skunk.n.04", "table": "table.n.02",
        "television": "television_receiver.n.01", "tiger": "tiger.n.02", "trout": "trout.n.02",
        "turtle": "turtle.n.02", "whale": "whale.n.02", "castle": "castle.n.02", "cloud": "cloud.n.02",
        "pickup_truck": "pickup.n.01"}
Sy = {k: wn.synset(v) for k, v in {**mapping, **CORR}.items()}
allp = pd.read_csv(f"{R}/phase2/raw/all_pairs_quadrants.csv")
allp["d_sem_corr"] = [1 - Sy[u].wup_similarity(Sy[v]) for u, v in zip(allp.class_u, allp.class_v)]
mp, ms = allp.d_perc.median(), allp.d_sem_corr.median()
qd = lambda p, s: "Q1" if (p < mp and s < ms) else "Q2" if (p >= mp and s >= ms) else "Q3" if p >= mp else "Q4"
allp["quad_corr"] = [qd(p, s) for p, s in zip(allp.d_perc, allp.d_sem_corr)]
def partial(x, y, z, rank):
    if rank: x, y, z = (stats.rankdata(v) for v in (x, y, z))
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]; ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = np.corrcoef(rx, ry)[0, 1]; n = len(x); df = n - 3
    t = r * np.sqrt(df / (1 - r ** 2)); p = 2 * stats.t.sf(abs(t), df)
    return float(r), float(p), n
out = {}
for learner in sorted(c1.learner.unique()):
    d = c1[c1.learner == learner].merge(allp[["class_u", "class_v", "d_sem_corr", "quad_corr"]], on=["class_u", "class_v"], how="left")
    res = {}
    for name, sub, sem in [("original_Q34", d[d.quadrant.isin(["Q3", "Q4"])], "d_sem"),
                           ("corrected_Q34", d[d.quad_corr.isin(["Q3", "Q4"])], "d_sem_corr"),
                           ("original_all100", d, "d_sem")]:
        x, y, z = sub.c1_accuracy.values, sub[sem].values, sub.d_perc.values
        res[name] = {"n": len(sub),
                     "spearman_perc_sem": float(stats.spearmanr(z, y)[0]),
                     "zero_order_rho_sem": float(stats.spearmanr(y, x)[0]),
                     "partial_spearman_sem_given_perc": partial(x, y, z, True)[:2],
                     "partial_pearson_sem_given_perc": partial(x, y, z, False)[:2],
                     "partial_spearman_perc_given_sem": partial(x, z, y, True)[:2]}
    out[learner] = res
json.dump(out, open(f"{OUT}/r07_partial_corr_q34.json", "w"), indent=1)
for l, res in out.items():
    for k, v in res.items():
        print(l, k, "n=%d rho(perc,sem)=%.3f rho_sem=%.3f partial_sem|perc(S)=%.3f p=%.3f (P)=%.3f p=%.3f partial_perc|sem(S)=%.3f p=%.2g" % (
            v["n"], v["spearman_perc_sem"], v["zero_order_rho_sem"], *v["partial_spearman_sem_given_perc"],
            *v["partial_pearson_sem_given_perc"], *v["partial_spearman_perc_given_sem"]))
