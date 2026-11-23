"""Encoder equivalence table, step 2 (no participant data; only out/emb_*.npz from 01_embed.py).
RULES FIXED BEFORE RUNNING THIS SCRIPT (written 2026-11-18 10:16 America/Santiago, before any RSA/CKA among the new encoders was seen;
the only prior glimpse was a smoke-test full RSA R0 vs random ResNet-50 seed 0 = .158, declared in the design §8.3):
- Images: 45 stimuli, domain = filename prefix c1/c2/c3 (15 per domain).
- Distances: cosine distance 1-cos on L2-normalised embeddings.
- RSA_full: Spearman over the 990 upper-triangle pairs. RSA_within: Spearman over the 315 within-domain pairs (3 x 105), pooled.
- CKA: linear CKA (Kornblith et al. 2019) on column-centred L2-normalised embeddings, 45 rows.
- Bootstrap: B = 1000, numpy default_rng(42); each replicate resamples 15 images with replacement WITHIN each domain; the same
  resample is used for all encoders (paired). RSA uses only pairs of distinct original images; CKA uses resampled rows.
  95% percentile intervals.
- Grouping (primary): encoders in PRIMARY set; dissimilarity 1 - RSA_within (point estimates); average linkage (UPGMA);
  cut at height 0.5 (groups whose members have mean within-domain RSA >= .50). Sensitivities: same rule on 1-RSA_full and 1-CKA.
  Stability: same rule in every bootstrap replicate -> co-clustering proportion per encoder pair.
- The .90 rule of the first design draft is kept only as a reference: flag encoders whose RSA_within with R0 has upper CI < .90.
- Supplementary encoders (E4s MAE-CLS, F1 seeds 1-2) are reported in the matrices but do not enter the primary grouping.
- AMENDMENT 2026-11-18 10:17 America/Santiago, BEFORE the first run of this script: Patricio decided that ALL encoders enter,
  so O1 (DINOv2-base) and O2 (OpenCLIP LAION) move from supplementary to PRIMARY. No other rule changed.
"""
import json, pathlib, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
B = pathlib.Path(__file__).resolve().parents[1]; OUT = B / "out"
PRIMARY = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
SUPP = ["E4s_mae_cls", "F1_rn50rand1", "F1_rn50rand2"]
E, names = {}, None
for ID in PRIMARY + SUPP:
    f = OUT / f"emb_{ID}.npz"
    if not f.exists(): print("MISSING", ID); continue
    z = np.load(f); E[ID] = z["emb"].astype(np.float64)
    assert names is None or list(z["names"]) == names; names = list(z["names"])
IDS = [i for i in PRIMARY + SUPP if i in E]; PRIM = [i for i in PRIMARY if i in E]
dom = np.array([n[:2] for n in names]); idx_by_dom = [np.where(dom == d)[0] for d in ("c1", "c2", "c3")]
def dist(X): return 1 - X @ X.T
def pairs(sel):
    i, j = np.triu_indices(len(sel), 1); a, b = sel[i], sel[j]; keep = a != b
    return a[keep], b[keep], (dom[a] == dom[b])[keep]
def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
D = {k: dist(v) for k, v in E.items()}
def stats(sel):
    a, b, w = pairs(sel); out = {}
    vec = {k: D[k][a, b] for k in IDS}
    rk_f = {k: rankdata(v) for k, v in vec.items()}; rk_w = {k: rankdata(v[w]) for k, v in vec.items()}
    for x in IDS:
        for y in IDS:
            out[("rsa_full", x, y)] = np.corrcoef(rk_f[x], rk_f[y])[0, 1]
            out[("rsa_within", x, y)] = np.corrcoef(rk_w[x], rk_w[y])[0, 1]
            out[("cka", x, y)] = cka(E[x][sel], E[y][sel])
    return out
full = stats(np.arange(45))
rng = np.random.default_rng(42); NB = 1000; boots = []
for bi in range(NB):
    sel = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_dom]); boots.append(stats(sel))
rows = []
for (m, x, y), v in full.items():
    bs = np.array([b[(m, x, y)] for b in boots])
    rows.append(dict(measure=m, enc_a=x, enc_b=y, est=v, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
T = pd.DataFrame(rows); T.to_csv(OUT / "equivalence_long.csv", index=False)
for m in ("rsa_full", "rsa_within", "cka"):
    T[T.measure == m].pivot(index="enc_a", columns="enc_b", values="est").loc[IDS, IDS].round(3).to_csv(OUT / f"matrix_{m}.csv")
# distance-scale descriptives
desc = {}
for k in IDS:
    a, b, w = pairs(np.arange(45)); v = D[k][a, b]
    desc[k] = dict(mean=v.mean(), sd=v.std(ddof=1), min=v.min(), max=v.max(), mean_within=v[w].mean(), mean_between=v[~w].mean(),
                   between_over_within=v[~w].mean() / v[w].mean())
pd.DataFrame(desc).T.round(4).to_csv(OUT / "distance_scale.csv")
# grouping
def groups(est, measure, ids):
    M = np.array([[1 - est[(measure, x, y)] for y in ids] for x in ids]); M = (M + M.T) / 2; np.fill_diagonal(M, 0)
    Z = linkage(squareform(np.clip(M, 0, None), checks=False), "average"); return Z, fcluster(Z, t=0.5, criterion="distance")
res = {"rules": __doc__, "B": NB, "seed": 42, "n_images": 45, "primary": PRIM, "supplementary": [i for i in SUPP if i in E], "grouping": {}}
for m in ("rsa_within", "rsa_full", "cka"):
    Z, lab = groups(full, m, PRIM)
    co = np.zeros((len(PRIM), len(PRIM)))
    for b in boots:
        _, lb = groups(b, m, PRIM); co += lb[:, None] == lb[None, :]
    co /= NB
    res["grouping"][m] = dict(labels=dict(zip(PRIM, map(int, lab))), linkage=Z.round(4).tolist(),
                              coclustering=pd.DataFrame(co, index=PRIM, columns=PRIM).round(3).to_dict())
    pd.DataFrame(co, index=PRIM, columns=PRIM).round(3).to_csv(OUT / f"coclustering_{m}.csv")
res["ref90_vs_R0_rsa_within"] = {r.enc_b: dict(est=round(r.est, 3), hi=round(r.hi, 3), distinct=bool(r.hi < .90))
                                 for r in T[(T.measure == "rsa_within") & (T.enc_a == "R0_dinov2s")].itertuples() if r.enc_b != "R0_dinov2s"}
res["distance_scale"] = desc
(OUT / "equivalence_results.json").write_text(json.dumps(res, indent=1, default=float))
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 4, figsize=(26, 6.5), gridspec_kw=dict(width_ratios=[1, 1, 1, .8]))
for a, m, t in zip(ax[:3], ("rsa_full", "rsa_within", "cka"), ("RSA (Spearman), all 990 pairs", "RSA (Spearman), 315 within-domain pairs", "Linear CKA")):
    P = T[T.measure == m].pivot(index="enc_a", columns="enc_b", values="est").loc[IDS, IDS].values
    im = a.imshow(P, vmin=-.2 if m != "cka" else 0, vmax=1, cmap="viridis"); a.set_title(t)
    a.set_xticks(range(len(IDS))); a.set_xticklabels(IDS, rotation=90, fontsize=8); a.set_yticks(range(len(IDS))); a.set_yticklabels(IDS, fontsize=8)
    for i in range(len(IDS)):
        for j in range(len(IDS)): a.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if P[i, j] < .6 else "k")
    a.axhline(len(PRIM) - .5, color="r", lw=1); a.axvline(len(PRIM) - .5, color="r", lw=1); plt.colorbar(im, ax=a, fraction=.046)
Z = np.array(res["grouping"]["rsa_within"]["linkage"]); dendrogram(Z, labels=PRIM, ax=ax[3], orientation="right", color_threshold=0.5)
ax[3].axvline(0.5, ls="--", color="k"); ax[3].set_title("Primary grouping: UPGMA on 1 - RSA_within, cut 0.5"); ax[3].set_xlabel("1 - RSA_within")
fig.suptitle("Encoder equivalence on the 45 Lewis & Frank (2018) stimuli (no participant data). Red lines separate primary | supplementary encoders.")
fig.tight_layout(); fig.savefig(B / "figs/encoder_equivalence_heatmap.png", dpi=200); fig.savefig(B / "figs/encoder_equivalence_heatmap.pdf")
print(json.dumps(res["grouping"]["rsa_within"]["labels"])); print(json.dumps(res["ref90_vs_R0_rsa_within"]))
