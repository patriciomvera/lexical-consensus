"""H5 (exploratory, BH within H5): class coherence (DINOv2 within-class dispersion = mean cosine distance to the class
centroid) vs Glasgow AoA (Scott et al. 2019, CC BY 4.0), partial Spearman controlling log frequency (wordfreq Zipf, 'en').
Classes sharing a word (the 5 *_tree -> 'tree') are collapsed (images pooled). CIFAR-100: 100 train images/class (seed 42);
CIFAR-10: 200/class exp_007 cifar_all10 cache. Wordbank AoA: NOT run (data now only via Redivis with login; see DESVIACIONES)."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, csv, json, glob, pathlib, numpy as np, pandas as pd
from scipy import stats
from wordfreq import zipf_frequency
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import bh
B = pathlib.Path(__file__).resolve().parents[1]; S = B.parent / "human_data_scouting"; REPO = B.parent / "revision_runs/repo"
rows = list(csv.reader(open(S / "norms/glasgow_esm2.csv", encoding="latin-1")))[2:]
G = {r[0].strip().lower(): float(r[20]) for r in rows if r[20].strip()}
SENSE = {"pepper": "pepper (vegetable)", "seal": "seal (animal)", "tank": "tank (army)", "mouse": "mouse (animal)", "plate": "plate",
         "ray": "ray (fish)", "can": "can (tin)", "palm": "palm (tree)"}  # decision #2: sense shown by CIFAR images, if a sense entry exists
ALT = {"airplane": ["airplane", "aeroplane", "plane (aircraft)"], "automobile": ["car", "automobile"], "ship": ["ship"], "aquarium_fish": ["fish"],
       "cattle": ["cattle", "cow"], "sweet_pepper": ["pepper (vegetable)", "pepper"], "maple_tree": ["tree"], "oak_tree": ["tree"],
       "palm_tree": ["tree"], "pine_tree": ["tree"], "willow_tree": ["tree"], "pickup_truck": ["truck"], "lawn_mower": ["lawnmower", "lawn mower"],
       "television": ["television", "tv"], "telephone": ["telephone", "phone"]}
def word(c):
    for f in ALT.get(c, [SENSE.get(c, c.replace("_", " ")), c.replace("_", " "), c.split("_")[-1]]):
        if f in G: return f
    return None
def disp(E, lab, classes):
    E = E / np.linalg.norm(E, axis=1, keepdims=True); out = {}
    for w in set(classes.values()):
        X = E[np.isin(lab, [k for k, v in classes.items() if v == w])]; m = X.mean(0); m /= np.linalg.norm(m); out[w] = float(np.mean(1 - X @ m))
    return out
def pspear(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z)); A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r = stats.pearsonr(ex, ey).statistic; n = len(x); t = r * np.sqrt((n - 3) / (1 - r * r)); p = 2 * stats.t.sf(abs(t), n - 3)
    se = 1 / np.sqrt(n - 4); lo, hi = np.tanh(np.arctanh(r) + np.array([-1.96, 1.96]) * se)
    return dict(n=n, rho_partial=float(r), ci_lo=float(lo), ci_hi=float(hi), p=float(p), rho_raw=float(stats.spearmanr(x, y).statistic))
res, tabs = {}, {}
z = np.load(f"{_P.CACHE}/dino_c100_train_sub100.npz"); names = list(z["class_names"])
cw = {i: word(c) for i, c in enumerate(names)}; cw = {i: w for i, w in cw.items() if w}
d100 = disp(z["emb"], z["labels"], cw)
z10 = np.load(REPO / ".cache/embeddings_exp007/cifar_all10.npz"); c10 = list(dict.fromkeys(z10["categories"]))
cw10 = {c: word(c) for c in c10}; cw10 = {k: v for k, v in cw10.items() if v}; d10 = disp(z10["emb"], z10["categories"], cw10)
for tag, d, cmap in (("C100_glasgow", d100, {names[i]: w for i, w in cw.items()}), ("C10_glasgow", d10, cw10)):
    t = pd.DataFrame([dict(word=w, classes="|".join(sorted(k for k, v in cmap.items() if v == w)), dispersion=d[w], aoa_glasgow=G[w],
                           zipf=zipf_frequency(w.split(" (")[0], "en")) for w in d]).sort_values("word")
    t.to_csv(B / f"data/h5_{tag}.csv", index=False); res[tag] = pspear(t.dispersion.values, t.aoa_glasgow.values, t.zipf.values)
ps = [res[k]["p"] for k in res]; for_bh = bh(ps)
for k, q in zip(res, for_bh): res[k]["p_BH"] = float(q)
res["wordbank"] = "not run: Wordbank item-level data now served only via Redivis (login/API token required); no account created (plan §5)"
res["overextensions"] = "not run: license unresolved (plan H5)"
json.dump(res, open(B / "out/h5_results.json", "w"), indent=1); print(json.dumps(res, indent=1))
