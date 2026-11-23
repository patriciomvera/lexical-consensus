"""Encoder equivalence (v2 of arXiv 2611.14116). TO BE RUN ONLY AFTER THE INTERNAL FREEZE (Ariadna review B3).
Supersedes 02_equivalence_SUPERSEDED_never_run.py (never executed).

PART A - DESCRIPTIVE CONTEXT ONLY (does not decide anything):
  45 stimuli, out/emb_<ID>.npz. Cosine distance on L2 vectors. RSA_full = Spearman over 990 pairs; RSA_within = Spearman over
  315 within-domain pairs; linear CKA on centred L2 embeddings. Bootstrap B = 1000, numpy default_rng(42), resampling 15 images
  with replacement within each domain (paired across encoders); RSA only on pairs of distinct original images. 95% percentile CI.
  Distance scale per encoder: mean, SD, min, max, mean within/between domain, between_over_within.
PART B - THE SINGLE GEOMETRY RULE (measured on what enters the model):
  Input: data/h4_long_<ID>.csv from h4_prep_encoders.py (same rows/order as v1 h4_long.csv). The column `selected` is NEVER read.
  Rows: 8 target-domain items (in_target == 1), 57,600 rows. zd(e) = z-score of dist over those rows (as in the 8-item model).
  Partial on level: residuals of zd(e) on level dummies (OLS). r(e,f) = Pearson correlation of residuals across the 57,600 rows.
  Grouping: encoders in GROUP_SET; dissimilarity 1 - r; average linkage (UPGMA); cut at heights 0.3, 0.5 AND 0.7 (all three, always;
  decision of Patricio 18-11 10:44). Calibration is checked separately at each cut; a cut failing calibration => "sin grupos" there.
  Grouping profile: "robusto" if the partition AND the calibration status coincide at the three cuts, else "depende del corte".
  Reference only: r(e, R0) >= .90 flagged "practically equivalent to R0" (does not decide).
  Calibration (pre-specified): the criterion "discriminates" only if O1 is in R0's group AND neither F1 nor F2 shares a group with
  any trained encoder. If calibration fails, NO group-level interpretation is made (encoder-level regions only).
  Also reported (descriptive): same r on the 24-item rows (zd over 24 items, residualised on level x in_target).
"""
import json, sys, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
B = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else B / "data"
# argv[2] = output dir, argv[3] = embeddings dir: ONLY for the synthetic tests; the weekend run uses the defaults (README_weekend.md)
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else B / "out"; EMB = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else B / "out"
FIG = B / "figs" if len(sys.argv) <= 2 else OUT
CUTS = (0.3, 0.5, 0.7)  # Patricio 18-11 10:44: the cut is a RANGE; all three are always computed and reported, none privileged
ALL = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels",
       "E4s_mae_cls", "F1_rn50rand1", "F1_rn50rand2"]
GROUP_SET = ALL[:10]; TRAINED = ALL[:8]; FLOORS = ["F1_rn50rand0", "F2_pixels"]
def partA():
    E = {}; names = None
    for ID in ALL:
        z = np.load(EMB / f"emb_{ID}.npz"); E[ID] = z["emb"].astype(np.float64)
        assert names is None or list(z["names"]) == names; names = list(z["names"])
    dom = np.array([n[:2] for n in names]); byd = [np.where(dom == d)[0] for d in ("c1", "c2", "c3")]
    D = {k: 1 - v @ v.T for k, v in E.items()}
    def cka(X, Y):
        X = X - X.mean(0); Y = Y - Y.mean(0); return np.linalg.norm(Y.T @ X) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
    def stats(sel):
        i, j = np.triu_indices(len(sel), 1); a, b = sel[i], sel[j]; k = a != b; a, b = a[k], b[k]; w = dom[a] == dom[b]
        rf = {e: rankdata(D[e][a, b]) for e in ALL}; rw = {e: rankdata(D[e][a, b][w]) for e in ALL}; o = {}
        for x in ALL:
            for y in ALL:
                o[("rsa_full", x, y)] = np.corrcoef(rf[x], rf[y])[0, 1]; o[("rsa_within", x, y)] = np.corrcoef(rw[x], rw[y])[0, 1]
                o[("cka", x, y)] = cka(E[x][sel], E[y][sel])
        return o
    est = stats(np.arange(45)); rng = np.random.default_rng(42)
    boots = [stats(np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in byd])) for _ in range(1000)]
    T = pd.DataFrame([dict(measure=m, enc_a=x, enc_b=y, est=v, lo=np.percentile([b[(m, x, y)] for b in boots], 2.5),
                           hi=np.percentile([b[(m, x, y)] for b in boots], 97.5)) for (m, x, y), v in est.items()])
    T.to_csv(OUT / "equivalence_descriptive_long.csv", index=False)
    i, j = np.triu_indices(45, 1); w = dom[i] == dom[j]
    sc = {e: dict(mean=D[e][i, j].mean(), sd=D[e][i, j].std(ddof=1), min=D[e][i, j].min(), max=D[e][i, j].max(), mean_within=D[e][i, j][w].mean(),
                  mean_between=D[e][i, j][~w].mean(), between_over_within=D[e][i, j][~w].mean() / D[e][i, j][w].mean()) for e in ALL}
    pd.DataFrame(sc).T.to_csv(OUT / "distance_scale.csv"); return T
def canon(lab):
    """partition as a sorted tuple of sorted member tuples (label-free, comparable across cuts)"""
    g = {}
    for k, v in lab.items(): g.setdefault(v, []).append(k)
    return sorted(tuple(sorted(m)) for m in g.values())
def group_all(ids, R, cuts=CUTS):
    """UPGMA on 1 - r, cut at every height in CUTS; calibration per cut (plan §6.1). No argument selects a cut."""
    M = 1 - R; np.fill_diagonal(M, 0); Zl = linkage(squareform((M + M.T) / 2, checks=False), "average"); per = {}
    for c in cuts:
        lab = dict(zip(ids, map(int, fcluster(Zl, c, "distance")))); g = lab.get
        calib = dict(O1_with_R0=g("O1_dinov2b") == g("R0_dinov2s"),
                     floors_apart=all(g(f) != g(t) for f in FLOORS for t in TRAINED if f in lab and t in lab))
        calib["discriminates"] = bool(calib["O1_with_R0"] and calib["floors_apart"])
        n_tr = len({lab[t] for t in TRAINED if t in lab})
        per[f"{c:.1f}"] = dict(cut=c, labels=lab, partition=canon(lab), calibration=calib, n_trained_groups=n_tr,
                               reading=("grupos" if calib["discriminates"] else "sin grupos"))
    parts = [json.dumps(v["partition"]) for v in per.values()]; cal = [v["calibration"]["discriminates"] for v in per.values()]
    prof = "robusto" if len(set(parts)) == 1 and len(set(cal)) == 1 else "depende del corte"
    r0 = ids.index("R0_dinov2s")
    return dict(rule="UPGMA on 1 - r (8 items, partial level), cuts 0.3/0.5/0.7, calibration per cut", cuts=per, profile=prof,
                linkage=Zl.round(4).tolist(), ref90_vs_R0={k: bool(R[r0, i] >= .90) for i, k in enumerate(ids)})
def partB():
    KEY = ["exp", "subid", "trial", "image", "condition", "level", "in_target"]
    Z8, Z24, key0 = {}, {}, None
    for ID in GROUP_SET + ["F1_rn50rand1", "F1_rn50rand2", "E4s_mae_cls"]:
        f = DATA / f"h4_long_{ID}.csv"
        if not f.exists(): print("MISSING", f); continue
        d = pd.read_csv(f, usecols=KEY + ["dist"])  # `selected` is never read
        if key0 is None: key0 = d[KEY]
        else: assert d[KEY].equals(key0), f"row mismatch {ID}"
        for Zs, mask, lev in ((Z8, d.in_target == 1, d.level), (Z24, np.ones(len(d), bool), d.level + "_" + d.in_target.astype(str))):
            x = d.dist[mask].to_numpy(); zd = (x - x.mean()) / x.std(ddof=1)
            X = pd.get_dummies(lev[mask]).to_numpy(float); res = zd - X @ np.linalg.lstsq(X, zd, rcond=None)[0]; Zs[ID] = res
    out = {}
    for name, Zs in (("items8_partial_level", Z8), ("items24_partial_level_x_domain", Z24)):
        ids = list(Zs); R = np.corrcoef(np.stack([Zs[k] for k in ids])); pd.DataFrame(R, index=ids, columns=ids).to_csv(OUT / f"zd_corr_{name}.csv")
        out[name] = dict(ids=ids, r=R.round(4).tolist())
    ids = [k for k in GROUP_SET if k in Z8]; R = np.corrcoef(np.stack([Z8[k] for k in ids]))
    out["grouping"] = group_all(ids, R)
    (OUT / "equivalence_grouping.json").write_text(json.dumps(out, indent=1, default=bool)); return out
if __name__ == "__main__":
    T = partA(); out = partB()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    for a, m in zip(ax, ("rsa_full", "rsa_within", "cka")):
        P = T[T.measure == m].pivot(index="enc_a", columns="enc_b", values="est").loc[ALL, ALL].values
        im = a.imshow(P, vmin=0, vmax=1); a.set_title(m + " (descriptive)"); a.set_xticks(range(len(ALL))); a.set_xticklabels(ALL, rotation=90, fontsize=8)
        a.set_yticks(range(len(ALL))); a.set_yticklabels(ALL, fontsize=8); plt.colorbar(im, ax=a, fraction=.046)
        for i in range(len(ALL)):
            for j in range(len(ALL)): a.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w" if P[i, j] < .6 else "k")
    fig.tight_layout(); fig.savefig(FIG / "encoder_equivalence_descriptive.png", dpi=200)
    print(json.dumps(out["grouping"], default=bool))
