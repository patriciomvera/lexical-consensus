"""Build pair tables: learner metrics (from original repo results, unchanged) + human distances per source.
Outputs data/pairs_c100.csv (4950 pairs), data/pairs_c10.csv (45 pairs), data/tiers_c10.csv (10 tier concepts)."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, gzip, csv, pickle, pathlib, itertools, collections, numpy as np, pandas as pd
ROOT = pathlib.Path(_P.ROOT); SC = ROOT / "human_data_scouting"; D = ROOT / "human_data_analyses/data"
REPO = ROOT / "revision_runs/repo"; R8 = REPO / "experiments/exp_008_dissociation/results/exp_008_20260608_173255"
sys.path.insert(0, str(SC / "scripts")); import cifar_wn as W
from nltk.corpus import wordnet as wn
key = lambda a, b: tuple(sorted((a, b)))
# ---------- learner metrics (CIFAR-100)
P = pd.read_csv(R8 / "phase1/perceptual_distance_matrix.csv", index_col=0); C100 = list(P.index)
pairs = pd.DataFrame([dict(a=a, b=b, d_perc=P.loc[a, b]) for a, b in itertools.combinations(sorted(C100), 2)])
wup = lambda m, a, b: 1 - wn.synset(m[a]).wup_similarity(wn.synset(m[b]))
orig = {c: s for c, s in W.m.items() if not c.startswith("_")}
pairs["d_sem"] = [wup(W.C100, a, b) for a, b in zip(pairs.a, pairs.b)]
pairs["d_sem_orig"] = [wup(orig, a, b) for a, b in zip(pairs.a, pairs.b)]
c1 = pd.read_csv(R8 / "phase3/exp_008_c1_results.csv"); c1 = c1[c1.learner == "centroid"]
c1["k"] = [key(a, b) for a, b in zip(c1.class_u, c1.class_v)]
pairs["k"] = [key(a, b) for a, b in zip(pairs.a, pairs.b)]
pairs = pairs.merge(c1[["k", "sample_id", "quadrant", "c1_accuracy", "d_sem"]].rename(columns={"d_sem": "d_sem_paper", "c1_accuracy": "acc_pair"}), on="k", how="left")
s = pairs.dropna(subset=["acc_pair"]); assert len(s) == 100
print("check d_sem_orig vs paper d_sem (sampled): max|diff| =", float((s.d_sem_orig - s.d_sem_paper).abs().max()))
# ---------- THINGS SPoSE
X = np.loadtxt(SC / "spose_embedding_66d_sorted.txt"); S = X @ X.T; E = np.exp(S)
def p_ij(i, j):
    k = np.array([t for t in range(len(X)) if t != i and t != j])
    return float(np.mean(E[i, j] / (E[i, j] + E[i, k] + E[j, k])))
cache = {}
def sim(I, J):
    v = []
    for i in I:
        for j in J:
            if i == j: return None
            kk = key(i, j); cache.setdefault(kk, p_ij(*kk)); v.append(cache[kk])
    return float(np.mean(v))
TM = json.load(open(D / "map_things_v1.json"))["maps"]
for name, col in [("PRIMARY", "d_things"), ("SENS_SCOUT", "d_things_scout"), ("SENS_STRICT", "d_things_strict")]:
    m = TM[name]; vals = []
    for a, b in zip(pairs.a, pairs.b):
        sv = sim(m[a], m[b]) if a in m and b in m else None
        vals.append(np.nan if sv is None else 1 - sv)
    pairs[col] = vals; print(col, "pairs covered", int(pairs[col].notna().sum()))
# ---------- HSJ
H = json.load(open(D / "map_hsj_v1.json"))
T = pd.read_csv(D / "hsj_trials.csv.gz")
def hsj_dist(cmap):
    w2c = {w: c for c, ws in cmap.items() for w in ws}
    n = collections.Counter(); k = collections.Counter()
    for q, refs, r0, r1 in zip(T.q, T.refs, T.r0, T.r1):
        a = w2c.get(q)
        if a is None: continue
        top = {r0, r1}
        for r in refs.split("|"):
            b = w2c.get(r)
            if b is None or b == a: continue
            n[key(a, b)] += 1; k[key(a, b)] += (r in top)
    return n, k
for tag, cm, sfx in [("corr", H["C100_corrected"], ""), ("orig", H["C100_original_mapping"], "_origmap")]:
    n, k = hsj_dist(cm)
    pairs["hsj_n" + sfx] = [n.get(x, 0) for x in pairs.k]
    pairs["d_hsj_raw" + sfx] = [1 - k[x] / n[x] if n.get(x, 0) > 0 else np.nan for x in pairs.k]
    print("HSJ", tag, "pairs n>=1", int((pairs["hsj_n" + sfx] >= 1).sum()), ">=10", int((pairs["hsj_n" + sfx] >= 10).sum()), ">=30", int((pairs["hsj_n" + sfx] >= 30).sum()))
pairs["d_hsj"] = np.where(pairs.hsj_n >= 10, pairs.d_hsj_raw, np.nan)
pairs["d_hsj_origmap"] = np.where(pairs.hsj_n_origmap >= 10, pairs.d_hsj_raw_origmap, np.nan)
pairs.drop(columns=["k"]).to_csv(D / "pairs_c100.csv", index=False)
print("sampled covered: THINGS", int(s.shape[0] and pairs.dropna(subset=["acc_pair"]).d_things.notna().sum()), "HSJ>=10", int(pairs.dropna(subset=["acc_pair"]).d_hsj.notna().sum()))
# ---------- CIFAR-10
z = np.load(REPO / ".cache/embeddings_exp007/cifar_all10.npz"); cats = z["categories"]; C10 = list(dict.fromkeys(cats))
cen = {c: z["emb"][cats == c].mean(0) for c in C10}; cen = {c: v / np.linalg.norm(v) for c, v in cen.items()}
c10 = pd.DataFrame([dict(a=a, b=b, d_perc=1 - float(cen[a] @ cen[b])) for a, b in itertools.combinations(sorted(C10), 2)])
r05 = pd.read_csv(ROOT / "revision_runs/out/r05_tier_pairs.csv"); r05["k"] = [key(a, b) for a, b in zip(r05.cat_a, r05.cat_b)]
c10["k"] = [key(a, b) for a, b in zip(c10.a, c10.b)]; c10 = c10.merge(r05[["k", "d", "used_as"]], on="k", how="left")
print("check C10 d_perc vs r05: max|diff| =", float((c10.d_perc - c10.d).abs().max()))
c10["d_sem"] = [wup(W.C10, a, b) for a, b in zip(c10.a, c10.b)]
m = TM["C10"]; c10["d_things"] = [1 - sim(m[a], m[b]) for a, b in zip(c10.a, c10.b)]
n, k = hsj_dist(H["C10"]); c10["hsj_n"] = [n.get(x, 0) for x in c10.k]
c10["d_hsj_raw"] = [1 - k[x] / n[x] if n.get(x, 0) else np.nan for x in c10.k]; c10["d_hsj"] = np.where(c10.hsj_n >= 10, c10.d_hsj_raw, np.nan)
# CIFAR-10H
counts = np.load(SC / "cifar10h-counts.npy"); tb = pickle.load(open(REPO / ".cache/cifar10/cifar-10-batches-py/test_batch", "rb"), encoding="latin1")
lab = np.array(tb["labels"]); names = pickle.load(open(REPO / ".cache/cifar10/cifar-10-batches-py/batches.meta", "rb"))["label_names"]
assert counts.shape == (10000, 10)
Cm = np.array([counts[lab == i].sum(0) / counts[lab == i].sum() for i in range(10)])
ix = {nm: i for i, nm in enumerate(names)}
c10["d_c10h"] = [1 - (Cm[ix[a], ix[b]] + Cm[ix[b], ix[a]]) / 2 for a, b in zip(c10.a, c10.b)]
c10.drop(columns=["k", "d"]).to_csv(D / "pairs_c10.csv", index=False)
print("C10 HSJ>=10:", int(c10.d_hsj.notna().sum()), "/45")
# ---------- tiers: acc_tier = mean per-concept C1 of the centroid learner, std_nway pool, 30 episodes (exp_007b_extension_clean ledger)
acc = collections.defaultdict(list)
for line in open(D / "exp007b_clean_ledger.jsonl"):
    r = json.loads(line)
    if r["learner"] == "centroid" and r["pool_type"] == "std_nway" and r["tier"] in ("near", "mid", "far"):
        for w, v in r["per_concept_c1"].items(): acc[(r["tier"], w)].append(v)
rows = []
for _, r in c10[c10.used_as.notna() & ~c10.used_as.fillna("").str.contains("excluded")].iterrows():
    tier, w = r.used_as.split(":"); v = acc[(tier, w)]
    rows.append(dict(tier=tier, word=w, a=r.a, b=r.b, n_episodes=len(v), acc_tier=float(np.mean(v)), d_perc=r.d_perc, d_sem=r.d_sem, d_things=r.d_things, d_hsj=r.d_hsj, hsj_n=r.hsj_n, d_c10h=r.d_c10h))
pd.DataFrame(rows).to_csv(D / "tiers_c10.csv", index=False); print(pd.DataFrame(rows).round(3).to_string())
