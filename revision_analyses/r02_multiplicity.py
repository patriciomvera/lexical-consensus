"""Holm and Benjamini-Hochberg corrections over the paired-Wilcoxon families.
Exact (unrounded) p values are recomputed from the committed per-episode ledger
results/exp_007b_extension_clean/ledger_c2_homogeneous.jsonl with the same call the repo uses
(scipy.stats.wilcoxon(a, b), src/eval/stats.py), and checked against the rounded values in
paired_tests_c2_homogeneous.csv.
Families:
  F1 = the 4 hard-pool exemplar-vs-centroid retrieval tests in Table 2
  F2 = all 96 paired tests in paired_tests_c2_homogeneous.csv (4 tiers x 7 pools x 3 contrasts on C2 = 84, + 4 tiers x 3 contrasts on C1 = 12)
  F3 = dissociation-study inferential tests reported in the text for the centroid learner (from committed phase4 outputs)
"""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
B = "results/exp_007b_extension_clean"
rows = [json.loads(l) for l in open(f"{B}/ledger_c2_homogeneous.jsonl")]
led = pd.DataFrame(rows)
pt = pd.read_csv(f"{B}/paired_tests_c2_homogeneous.csv")
ps = []
for i, r in pt.iterrows():
    alt = r.comparison.split(" vs ")[0]
    pool = "std_nway" if r.metric == "c1" else r.pool_type
    sub = led[(led.tier == r.tier) & (led.pool_type == pool)]
    a = sub[sub.learner == alt].sort_values("episode_id")[r.metric].values
    b = sub[sub.learner == "centroid"].sort_values("episode_id")[r.metric].values
    ps.append(float(wilcoxon(a, b).pvalue))
pt["p_exact"] = ps
pt["abs_diff_vs_rounded"] = (pt.p_exact.round(5) - pt.wilcoxon_p).abs()
assert pt.abs_diff_vs_rounded.max() < 1.1e-5, pt.abs_diff_vs_rounded.max()
for fam, mask in [("F2_all", np.ones(len(pt), bool)),
                  ("F1_table2", ((pt.pool_type == "hard") & (pt.comparison == "exemplar_knn_k3 vs centroid") & (pt.metric == "c2")).values)]:
    p = pt.loc[mask, "p_exact"].values
    pt.loc[mask, f"{fam}_holm"] = multipletests(p, method="holm")[1]
    pt.loc[mask, f"{fam}_bh"] = multipletests(p, method="fdr_bh")[1]
pt.to_csv("revision_analyses/out/r02_paired_tests_adjusted.csv", index=False)
summ = {}
for alpha in [0.05]:
    raw_sig = pt.p_exact < alpha
    summ["F2_n_tests"] = int(len(pt)); summ["F2_raw_sig"] = int(raw_sig.sum())
    summ["F2_holm_sig"] = int((pt.F2_all_holm < alpha).sum()); summ["F2_bh_sig"] = int((pt.F2_all_bh < alpha).sum())
    lost_h = pt[raw_sig & (pt.F2_all_holm >= alpha)]; lost_b = pt[raw_sig & (pt.F2_all_bh >= alpha)]
    summ["F2_lost_under_holm"] = lost_h[["tier", "pool_type", "comparison", "metric", "mean_diff", "p_exact", "F2_all_holm"]].to_dict("records")
    summ["F2_lost_under_bh"] = lost_b[["tier", "pool_type", "comparison", "metric", "mean_diff", "p_exact", "F2_all_bh"]].to_dict("records")
    c2 = pt[pt.metric == "c2"]
    summ["F2_c2_positive_and_holm_sig"] = int(((c2.mean_diff > 0) & (c2.F2_all_holm < alpha)).sum()); summ["F2_c2_n"] = int(len(c2))
summ["F1_table2"] = pt[pt.F1_table2_holm.notna()][["tier", "mean_diff", "p_exact", "F1_table2_holm", "F1_table2_bh", "F2_all_holm", "F2_all_bh"]].to_dict("records")
# F3 dissociation (centroid): p values from committed phase4 outputs (regression_table.csv, dissociation_analysis.json)
D = "results/exp_008_dissociation/phase4"
rt = pd.read_csv(f"{D}/regression_table.csv").set_index("learner").loc["centroid"]
da = json.load(open(f"{D}/dissociation_analysis.json"))
f3 = {"beta_perc (P+S)": rt.PS_beta_perc_p, "beta_sem (P+S)": rt.PS_beta_sem_p, "LRT add sem": rt.lrt_P_vs_PS_p,
      "LRT add perc": rt.lrt_S_vs_PS_p, "rho_perc Q3+Q4": da["dissociation_subset"]["centroid"]["rho_perc_p"],
      "rho_sem Q3+Q4": da["dissociation_subset"]["centroid"]["rho_sem_p"], "beta_same_superclass": da["superclass_analysis"]["beta_same_sc_p"]}
p = np.array(list(f3.values()), float)
summ["F3_dissociation"] = [{"test": k, "p": float(v), "holm": float(h), "bh": float(b)} for k, v, h, b in
                           zip(f3, p, multipletests(p, method="holm")[1], multipletests(p, method="fdr_bh")[1])]
json.dump(summ, open("revision_analyses/out/r02_multiplicity.json", "w"), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
