"""H3 (descriptive): (i) tier accuracy vs each human distance (Spearman, bootstrap CI over tiers, seed 42);
(ii) CIFAR-10H per-image entropy vs learner margin = d(own centroid) - d(nearest rival centroid), cosine distance,
centroids from the exp_007 cifar_all10 DINOv2 cache (200/class), over the 10,000 CIFAR-10 test images."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import spearman_boot
B = pathlib.Path(__file__).resolve().parents[1]; REPO = pathlib.Path(_P.MAIN_REPO)
out = {}
t = pd.read_csv(B / "data/tiers_c10.csv")
for c in ["d_perc", "d_sem", "d_things", "d_hsj", "d_c10h"]:
    s = t.dropna(subset=[c]); r = spearman_boot(s.acc_tier.values, s[c].values)
    out[f"tier_{c}"] = dict(n=len(s), **dict(zip(["rho","ci_lo","ci_hi"], map(float, r))))
json.dump(out, open(B / "out/h3_tiers_partial.json", "w"), indent=1); print(json.dumps(out, indent=1))
if not pathlib.Path(f"{_P.CACHE}/dino_c10_test.npz").exists(): sys.exit("tier part saved; image part waits for dino_c10_test.npz")
C10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
z = np.load(REPO / ".cache/embeddings_exp007/cifar_all10.npz"); E = z["embeddings"] if "embeddings" in z else z["emb"]; cats = z["categories"]
E = E / np.linalg.norm(E, axis=1, keepdims=True)
cen = np.stack([E[cats == c].mean(0) for c in C10]); cen /= np.linalg.norm(cen, axis=1, keepdims=True)
te = np.load(f"{_P.CACHE}/dino_c10_test.npz"); X = te["emb"] / np.linalg.norm(te["emb"], axis=1, keepdims=True); y = te["labels"]
D = 1 - X @ cen.T; own = D[np.arange(len(y)), y]; Dr = D.copy(); Dr[np.arange(len(y)), y] = np.inf
margin = own - Dr.min(1)
cnt = np.load(_P.SCOUT / "cifar10h-counts.npy").astype(float)
P = cnt / cnt.sum(1, keepdims=True); H = -(np.where(P > 0, P * np.log2(np.where(P > 0, P, 1)), 0)).sum(1)
assert (P.argmax(1) == y).mean() > 0.9  # label order sanity
r = spearman_boot(H, margin)
out["c10h_entropy_vs_margin"] = dict(n=len(y), centroid_acc=float((D.argmin(1) == y).mean()), human_majority_acc=float((P.argmax(1) == y).mean()),
                                     **dict(zip(["rho","ci_lo","ci_hi"], map(float, r))))
pd.DataFrame(dict(label=y, entropy=H, margin=margin)).to_csv(B / "data/h3_c10test_margin.csv", index=False)
json.dump(out, open(B / "out/h3_results.json", "w"), indent=1); print(json.dumps(out, indent=1))
