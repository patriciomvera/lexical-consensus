"""Tier boundaries and per-tier distance distributions for exp_007b_extension(_clean).
Recomputes the 10x10 CIFAR-10 inter-category distance d(a,b) = 1 - cos(mean_a, mean_b) over the rebuilt
exp_007 cache (200 DINOv2-small embeddings/class), checks it against the values hard-coded in
experiments/exp_007b_extension/_shared.py and results/exp_007b_extension/concept_gate.csv, and tabulates all 45 pairs by tier."""
import json, numpy as np, pandas as pd, itertools
d = np.load(".cache/embeddings_exp007/cifar_all10.npz")
E, C = d["emb"], d["categories"]
cats = list(dict.fromkeys(C))
mu = {c: E[C == c].mean(0) for c in cats}
def dist(a, b):
    x, y = mu[a], mu[b]; return 1 - float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
NEAR_T, FAR_MED = 0.4767, 0.6964
rows = []
for a, b in itertools.combinations(sorted(cats), 2):
    v = dist(a, b); tier = "near" if v < NEAR_T else ("mid" if v < FAR_MED else "far")
    rows.append(dict(cat_a=a, cat_b=b, d=round(v, 4), band=tier))
df = pd.DataFrame(rows).sort_values("d")
gate = pd.read_csv("results/exp_007b_extension/concept_gate.csv")
chk = gate.merge(df, on=["cat_a", "cat_b"], how="left")
print("max |d_recomputed - concept_gate| =", float((chk.d - chk.inter_member_distance).abs().max()))
used = {("cat","dog"):"near:tromble",("bird","deer"):"near:varble",("automobile","truck"):"near:glintz",
        ("deer","dog"):"mid:brumble",("airplane","ship"):"mid:strenk",("bird","horse"):"mid:worblix",
        ("airplane","cat"):"far:gleth",("frog","truck"):"far:mivor",("automobile","deer"):"far:plonk",("horse","ship"):"far:quax",
        ("bird","dog"):"exp_007b:zorb (excluded)"}
df["used_as"] = [used.get((a, b), "") for a, b in zip(df.cat_a, df.cat_b)]
far_used = {k: dist(*k) for k, v in used.items() if v.startswith("far")}
print("far concepts:", {used[k]: round(v, 4) for k, v in far_used.items()})
summ = df.groupby("band").d.describe()[["count", "min", "50%", "max"]]
print(summ); print(df.to_string(index=False))
df.to_csv("revision_analyses/out/r05_tier_pairs.csv", index=False)
# distance of each native category to its nearest other category (context for 'native')
nn = {c: round(min(dist(c, o) for o in cats if o != c), 4) for c in cats}
json.dump(dict(near_threshold=NEAR_T, far_median=FAR_MED, band_summary=summ.round(4).reset_index().to_dict("records"),
               far_concepts={used[k]: round(v, 4) for k, v in far_used.items()}, native_nearest_other=nn,
               check_vs_concept_gate=float((chk.d - chk.inter_member_distance).abs().max())),
          open("revision_analyses/out/r05_tier_distances.json", "w"), indent=1)
