"""Master-seed robustness of the naming gradient (C1) and the hard-pool retrieval dissociation (Table 2).
Inputs: revision_analyses/out/r04_seeds/<mode>_seed_<s>/ledger_c2_homogeneous.jsonl produced by r04_run_clean_seed.py
(run_clean.py of exp_007b_extension_clean, unmodified apart from MASTER_SEED / output paths).
  episodes mode: MASTER_SEED changes episode sampling (support/query/concept draws) and learner seeds; image pool = committed 200/class.
  full mode:     additionally re-draws the 200 images/class from CIFAR-10 train with the new seed and re-encodes them.
Seed 42 (episodes mode) reproduces the committed ledger exactly for all non-random learners.
Real learners only; the random baseline is not reproducible bit-for-bit and is reported only as a chance check."""
import json, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
TIERS = ["native", "near", "mid", "far"]
REAL = ["centroid", "exemplar_knn_k3", "kcentroid_k2", "logreg", "linsvm"]
def summarize(path):
    L = pd.DataFrame([json.loads(l) for l in open(path)])
    out = {"n_rows": len(L)}
    c1 = L[L.pool_type == "std_nway"].groupby(["tier", "learner"]).c1.mean().unstack()
    c2h = L[L.pool_type == "hard"].groupby(["tier", "learner"]).c2.mean().unstack()
    out["c1"] = {t: {l: float(c1.loc[t, l]) for l in REAL + ["random"]} for t in TIERS}
    out["c2_hard"] = {t: {l: float(c2h.loc[t, l]) for l in REAL + ["random"]} for t in TIERS}
    out["c1_monotone"] = {l: bool(all(c1.loc[TIERS[i], l] > c1.loc[TIERS[i+1], l] for i in range(3))) for l in REAL}
    gaps = {}
    H = L[L.pool_type == "hard"]
    for t in TIERS:
        s = H[H.tier == t]
        a = s[s.learner == "exemplar_knn_k3"].sort_values("episode_id").c2.values
        b = s[s.learner == "centroid"].sort_values("episode_id").c2.values
        gaps[t] = {"gap": float((a - b).mean()), "p": float(wilcoxon(a, b).pvalue), "n": int(len(a))}
    out["table2_gap"] = gaps
    out["near_gap_minus_native_gap"] = gaps["near"]["gap"] - gaps["native"]["gap"]
    return out
runs = {}
for f in sorted(glob.glob("revision_analyses/out/r04_seeds/*_seed_*/ledger_c2_homogeneous.jsonl")):
    name = os.path.basename(os.path.dirname(f))
    n = sum(1 for _ in open(f))
    if n < 5040: print("SKIP incomplete", name, n); continue
    runs[name] = summarize(f)
sets = {"episodes (seeds 42,0,1,2,3; fixed image pool)": [k for k in runs if k.startswith("episodes")],
        "full (seed 42 + seeds 0-3 with re-drawn, re-encoded image pools)": ["episodes_seed_42"] + [k for k in runs if k.startswith("full")]}
report = {"runs": runs, "aggregate": {}}
for sname, keys in sets.items():
    keys = [k for k in keys if k in runs]
    if len(keys) < 2: continue
    agg = {"seeds": keys}
    for metric in ["c1", "c2_hard"]:
        agg[metric] = {t: {l: (float(np.mean([runs[k][metric][t][l] for k in keys])), float(np.std([runs[k][metric][t][l] for k in keys], ddof=1)))
                           for l in REAL} for t in TIERS}
    agg["c1_monotone_all_seeds_all_learners"] = all(all(runs[k]["c1_monotone"].values()) for k in keys)
    agg["gap"] = {t: (float(np.mean([runs[k]["table2_gap"][t]["gap"] for k in keys])), float(np.std([runs[k]["table2_gap"][t]["gap"] for k in keys], ddof=1)),
                      int(sum(runs[k]["table2_gap"][t]["gap"] > 0 for k in keys)), int(sum(runs[k]["table2_gap"][t]["p"] < 0.05 for k in keys)),
                      float(max(runs[k]["table2_gap"][t]["p"] for k in keys))) for t in TIERS}
    agg["near_gap_minus_native_gap"] = (float(np.mean([runs[k]["near_gap_minus_native_gap"] for k in keys])), float(np.std([runs[k]["near_gap_minus_native_gap"] for k in keys], ddof=1)),
                                        int(sum(runs[k]["near_gap_minus_native_gap"] > 0 for k in keys)))
    agg["near_gap_is_max_tier"] = int(sum(max(TIERS, key=lambda t: runs[k]["table2_gap"][t]["gap"]) == "near" for k in keys))
    report["aggregate"][sname] = agg
    print(f"\n=== {sname}: {keys}")
    print("C1 mean±SD (centroid / exemplar / logreg):")
    for t in TIERS:
        print(f"  {t:6s}", "  ".join(f"{l}={agg['c1'][t][l][0]:.3f}±{agg['c1'][t][l][1]:.3f}" for l in ["centroid", "exemplar_knn_k3", "logreg"]))
    print("  monotone native>near>mid>far for every real learner and seed:", agg["c1_monotone_all_seeds_all_learners"])
    print("Hard C2 mean±SD (centroid / exemplar / logreg) and gap (mean±SD, #pos, #p<.05, max p):")
    for t in TIERS:
        g = agg["gap"][t]
        print(f"  {t:6s}", "  ".join(f"{l}={agg['c2_hard'][t][l][0]:.3f}±{agg['c2_hard'][t][l][1]:.3f}" for l in ["centroid", "exemplar_knn_k3", "logreg"]),
              f" gap={g[0]:+.3f}±{g[1]:.3f} pos={g[2]}/{len(keys)} sig={g[3]}/{len(keys)} maxp={g[4]:.4f}")
    print("  near gap - native gap:", agg["near_gap_minus_native_gap"], " near is largest-gap tier in", agg["near_gap_is_max_tier"], "of", len(keys))
for k, r in runs.items():
    print(k, {t: (round(r["table2_gap"][t]["gap"], 3), round(r["table2_gap"][t]["p"], 4)) for t in TIERS})
json.dump(report, open("revision_analyses/out/r06_seed_summary.json", "w"), indent=1)
