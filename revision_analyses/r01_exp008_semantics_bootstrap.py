"""exp_008 dissociation: semantic-measure robustness and bootstrap.
Uses ONLY committed artifacts (no re-encoding):
  results/exp_008_dissociation/phase1/{semantic_distance_matrix.csv,wordnet_mappings_used.json}
  results/exp_008_dissociation/phase3/exp_008_c1_results.csv  (per-pair C1 accuracy, 100 pairs x 4 learners)
Steps
  (a) verify that 1-WuP with the frozen mapping reproduces the committed d_sem matrix
  (b) reproduce the committed OLS P+S regression (beta_perc=-0.2119, beta_sem p=0.660)
  (c) pairs bootstrap (10,000 resamples, seed 20261026) of beta_perc, beta_sem, Q3+Q4 Spearman rhos, Q4-Q3 median diff
  (d) alternative semantic measures: path, LCH (normalised), Lin/JCN (Brown IC), WuP with sense-corrected mapping
      -> rerun P+S regression on the same 100 pairs and Q3+Q4 Spearman under each
"""
import json, sys, numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import spearmanr
from nltk.corpus import wordnet as wn, wordnet_ic
R = sys.argv[1] if len(sys.argv) > 1 else "results/exp_008_dissociation"
OUT = sys.argv[2] if len(sys.argv) > 2 else "revision_analyses/out"
mapping = json.load(open(f"{R}/phase1/wordnet_mappings_used.json"))
semM = pd.read_csv(f"{R}/phase1/semantic_distance_matrix.csv", index_col=0)
c1 = pd.read_csv(f"{R}/phase3/exp_008_c1_results.csv")
# sense corrections: frozen mapping sense -> intended CIFAR-100 sense (definitions checked in WordNet 3.0)
CORR = {"beaver": "beaver.n.07", "flatfish": "flatfish.n.02", "leopard": "leopard.n.02",
        "lobster": "lobster.n.02", "maple_tree": "maple.n.02", "oak_tree": "oak.n.02",
        "otter": "otter.n.02", "palm_tree": "palm.n.03", "raccoon": "raccoon.n.02",
        "ray": "ray.n.07", "shrew": "shrew.n.02", "skunk": "skunk.n.04", "table": "table.n.02",
        "television": "television_receiver.n.01", "tiger": "tiger.n.02", "trout": "trout.n.02",
        "turtle": "turtle.n.02", "whale": "whale.n.02", "castle": "castle.n.02", "cloud": "cloud.n.02",
        "pickup_truck": "pickup.n.01"}
log = {}
# (a)
S = {k: wn.synset(v) for k, v in mapping.items()}
maxdiff = 0.0
for u in semM.index:
    for v in semM.columns:
        if u < v:
            w = S[u].wup_similarity(S[v]); maxdiff = max(maxdiff, abs((1 - w) - semM.loc[u, v]))
log["a_max_abs_diff_recomputed_vs_committed_dsem"] = maxdiff
log["frozen_mapping_definitions_of_corrected_classes"] = {k: [mapping[k], S[k].definition(), CORR[k], wn.synset(CORR[k]).definition()] for k in CORR}
brown = wordnet_ic.ic("ic-brown.dat")
def measures(mp):
    Sy = {k: wn.synset(v) for k, v in mp.items()}
    def f(u, v):
        a, b = Sy[u], Sy[v]
        lch_max = -np.log(1.0 / (2 * 20))  # max depth 19 for nouns in WN3.0 -> lch upper bound
        return {"wup": 1 - a.wup_similarity(b), "path": 1 - a.path_similarity(b),
                "lch": 1 - a.lch_similarity(b) / a.lch_similarity(a),
                "lin": 1 - a.lin_similarity(b, brown),
                "jcn_dist": 1.0 / max(a.jcn_similarity(b, brown), 1e-12)}
    return f
def fit(df, sem):
    X = sm.add_constant(np.column_stack([df.d_perc.values, sem]))
    m = sm.OLS(df.c1_accuracy.values, X).fit()
    return m.params, m.pvalues, m.rsquared
res = {}
for learner in ["centroid", "exemplar_knn_k3", "logreg", "random"]:
    df = c1[c1.learner == learner].sort_values("sample_id").reset_index(drop=True)
    p, pv, r2 = fit(df, df.d_sem.values)
    out = {"n": len(df), "beta_perc": p[1], "p_perc": pv[1], "beta_sem": p[2], "p_sem": pv[2], "r2": r2}
    diss = df[df.quadrant.isin(["Q3", "Q4"])]
    out["rho_perc_Q34"] = spearmanr(diss.d_perc, diss.c1_accuracy)[0]
    out["rho_sem_Q34"] = spearmanr(diss.d_sem, diss.c1_accuracy)[0]
    # (c) bootstrap
    rng = np.random.default_rng(20261026); B = 10000
    bs = {k: [] for k in ["beta_perc", "beta_sem", "rho_perc_Q34", "rho_sem_Q34", "q4_minus_q3_median", "q1_minus_q2_median"]}
    n = len(df)
    q = {k: df[df.quadrant == k].reset_index(drop=True) for k in ["Q1", "Q2", "Q3", "Q4"]}
    for b in range(B):
        idx = rng.integers(0, n, n); d = df.iloc[idx]
        pp, _, _ = fit(d, d.d_sem.values); bs["beta_perc"].append(pp[1]); bs["beta_sem"].append(pp[2])
        # stratified (within-quadrant) resampling for quadrant statistics, preserving 25/quadrant design
        qs = {k: v.iloc[rng.integers(0, len(v), len(v))] for k, v in q.items()}
        dd = pd.concat([qs["Q3"], qs["Q4"]])
        bs["rho_perc_Q34"].append(spearmanr(dd.d_perc, dd.c1_accuracy)[0])
        bs["rho_sem_Q34"].append(spearmanr(dd.d_sem, dd.c1_accuracy)[0])
        bs["q4_minus_q3_median"].append(qs["Q4"].c1_accuracy.median() - qs["Q3"].c1_accuracy.median())
        bs["q1_minus_q2_median"].append(qs["Q1"].c1_accuracy.median() - qs["Q2"].c1_accuracy.median())
    out["bootstrap"] = {k: {"mean": float(np.nanmean(v)), "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                            "frac_lt0": float(np.mean(np.array(v) < 0))} for k, v in bs.items()}
    # (d) alternative semantic measures
    alt = {}
    for mname, mp in [("frozen", mapping), ("corrected", {**mapping, **CORR})]:
        f = measures(mp)
        vals = pd.DataFrame([f(u, v) for u, v in zip(df.class_u, df.class_v)])
        for col in vals.columns:
            if mname == "frozen" and col == "wup":
                assert np.allclose(vals[col].values, df.d_sem.values, atol=1e-5)
            pp, pv2, r2b = fit(df, vals[col].values)
            dd = df.assign(s=vals[col].values)[df.quadrant.isin(["Q3", "Q4"])]
            alt[f"{mname}_{col}"] = {"beta_perc": pp[1], "p_perc": pv2[1], "beta_sem": pp[2], "p_sem": pv2[2], "r2": r2b,
                                     "rho_sem_Q34_origquadrants": spearmanr(dd.s, dd.c1_accuracy)[0],
                                     "spearman_with_frozen_wup": spearmanr(vals[col], df.d_sem)[0]}
        if mname == "corrected":
            df_corr = df.assign(d_sem_corr=vals["wup"].values)
    # quadrant reassignment under corrected WuP: medians over all 4,950 pairs
    out["alt_semantics"] = alt
    res[learner] = out
# corrected full matrix medians and reassignment of the 100 sampled pairs
mp = {**mapping, **CORR}; Sy = {k: wn.synset(v) for k, v in mp.items()}
classes = list(semM.index)
allp = pd.read_csv(f"{R}/phase2/raw/all_pairs_quadrants.csv")
allp["d_sem_corr"] = [1 - Sy[u].wup_similarity(Sy[v]) for u, v in zip(allp.class_u, allp.class_v)]
med_perc = allp.d_perc.median(); med_sem_c = allp.d_sem_corr.median()
def qd(p, s):
    return "Q1" if (p < med_perc and s < med_sem_c) else "Q2" if (p >= med_perc and s >= med_sem_c) else "Q3" if p >= med_perc else "Q4"
allp["quad_corr"] = [qd(p, s) for p, s in zip(allp.d_perc, allp.d_sem_corr)]
log["n_pairs_changed_quadrant_all4950"] = int((allp.quad_corr != allp.quadrant).sum())
log["spearman_dsem_frozen_vs_corrected_all4950"] = float(spearmanr(allp.d_sem, allp.d_sem_corr)[0])
log["median_dsem_frozen"] = float(allp.d_sem.median()); log["median_dsem_corr"] = float(med_sem_c)
df = c1[c1.learner == "centroid"].sort_values("sample_id").reset_index(drop=True)
m = df.merge(allp[["class_u", "class_v", "d_sem_corr", "quad_corr"]], on=["class_u", "class_v"], how="left")
log["sampled100_quadrant_crosstab_orig_vs_corr"] = pd.crosstab(m.quadrant, m.quad_corr).to_dict()
log["sampled100_n_pairs_touching_corrected_class"] = int(sum((u in CORR) or (v in CORR) for u, v in zip(m.class_u, m.class_v)))
d34 = m[m.quad_corr.isin(["Q3", "Q4"])]
log["centroid_corrected_quadrants"] = {"n_Q34": len(d34), "rho_perc": spearmanr(d34.d_perc, d34.c1_accuracy)[0],
    "rho_sem": spearmanr(d34.d_sem_corr, d34.c1_accuracy)[0],
    "medians": m.groupby("quad_corr").c1_accuracy.median().to_dict(), "counts": m.quad_corr.value_counts().to_dict()}
log["results"] = res
def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(conv(log), open(f"{OUT}/r01_exp008_semantics_bootstrap.json", "w"), indent=1)
pd.DataFrame([{"class": k, "frozen_synset": mapping[k], "frozen_definition": wn.synset(mapping[k]).definition(),
               "corrected_synset": v, "corrected_definition": wn.synset(v).definition()} for k, v in CORR.items()]).to_csv(f"{OUT}/r01_wordnet_sense_corrections.csv", index=False)
print(json.dumps(conv({k: log[k] for k in log if k not in ("results", "frozen_mapping_definitions_of_corrected_classes")}), indent=1))
for l, o in conv(res).items():
    print(l, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ("bootstrap", "alt_semantics")})
    print("  boot", o["bootstrap"])
    for k, v in o["alt_semantics"].items(): print("  ", k, {a: round(b, 4) for a, b in v.items()})
