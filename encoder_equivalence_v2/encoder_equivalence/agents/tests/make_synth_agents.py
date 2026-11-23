"""SYNTHETIC world for the agents-addendum tests. NO real data is read: no CIFAR image, no real embedding, no participant response,
no human-similarity value. Only NAMES are reused (CIFAR class names and the v1 concept definitions, which are code constants).
Planted structure (seed-controlled):
  * a latent position z_c in R^8 for each of the 100 'CIFAR-100' and 10 'CIFAR-10' classes; synthetic human distances (THINGS-, HSJ-,
    CIFAR-10H-like) = latent distance + noise;
  * GOOD encoders: class means = random projection of z_c (+ small noise) -> d_perc tracks the human distances;
  * NULL encoders (the floors F1_rn50rand0, F2_pixels): class means random, unrelated to z_c -> H1 rho ~ 0 with a narrow CI;
  * DEAD encoder (--dead, default E4_mae): no class structure at all -> agents' positive control must fail;
  * people: per-trial tables with selected ~ Bernoulli(logit(-1 + b * zd)), b < 0 for good encoders and 0 for the floors.
The v1 "results" (ledger of T, perceptual matrix, regression_table.csv) are produced by the UNCHANGED v1 code on the synthetic R0.
Usage: python make_synth_agents.py /tmp/synth_agents_XXX [--h4-full] [--dead E4_mae]"""
import sys, os, json, zlib, itertools, pathlib, argparse, numpy as np, pandas as pd
seed_of = lambda *k: zlib.crc32(repr(k).encode())
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--h4-full", action="store_true"); ap.add_argument("--dead", default="E4_mae")
ap.add_argument("--dim", type=int, default=384); ap.add_argument("--no-v1-ledger", action="store_true"); a = ap.parse_args()
T = pathlib.Path(a.root); assert str(T).startswith("/tmp/synth_agents"); os.environ["AGENTS_SYNTH_DIR"] = str(T)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts")); import lib_agents as L
for d in ("v1", "hda", "human/data", "human/out", "out/emb", "out/sims", "out/results", "logs"): (T / d).mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261122); DIM = a.dim
re8 = L.run_episodes(); C100 = sorted(re8.CIFAR100_CLASSES)
s7 = L._load("s7_synth", L.E7 / "_shared.py"); C10 = list(s7.DATASET_CATEGORIES["cifar_all10"])   # v1 category order (frog, horse, ...)
Z100 = rng.normal(size=(100, 8)); Z10 = rng.normal(size=(10, 8)); Zs = rng.normal(size=(100, 6))   # latent; Zs = 'semantic' latent
GOOD = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "E4s_mae_cls"]
NULL = ["F1_rn50rand0", "F2_pixels", "F1_rn50rand1", "F1_rn50rand2"]
def means(ID, Z, k):
    r = np.random.default_rng(seed_of(ID, k))
    if ID == a.dead: return np.zeros((len(Z), DIM))
    if ID in NULL: return r.normal(size=(len(Z), DIM)) * 1.2
    W = r.normal(size=(Z.shape[1], DIM)) / np.sqrt(Z.shape[1]); return Z @ W * 1.6 + r.normal(size=(len(Z), DIM)) * 0.25
def images(M, lab, ID, k):
    r = np.random.default_rng(seed_of(ID, k, "img")); X = M[lab] + r.normal(size=(len(lab), DIM)); X /= np.linalg.norm(X, axis=1, keepdims=True); return X.astype(np.float32)
lab100 = np.repeat(np.arange(100), 500); lab10 = np.repeat(np.arange(10), 200); labte = np.tile(np.arange(10), 1000)
hashes = [f"{c}_{j:03d}" for c in C10 for j in range(200)]; cats = [c for c in C10 for _ in range(200)]
for ID in GOOD + NULL:
    M100 = means(ID, Z100, "c100"); M10 = means(ID, Z10, "c10")
    np.savez(L.emb_path(ID, "c100train"), emb=images(M100, lab100, ID, "c100"), labels=lab100.astype(np.int32), class_names=np.array(C100))
    np.savez(L.emb_path(ID, "c10pool"), emb=images(M10, lab10, ID, "c10"), hashes=np.array(hashes), categories=np.array(cats))
    ALPHA = sorted(C10); np.savez(L.emb_path(ID, "c10test"), emb=images(M10, np.array([C10.index(ALPHA[y]) for y in labte]), ID, "te"), labels=labte.astype(np.int64))
    for s in L.TSEM_SEEDS:
        L.emb_path(ID, f"c10pool_seed{s}").parent.mkdir(parents=True, exist_ok=True); np.savez(L.emb_path(ID, f"c10pool_seed{s}"), emb=images(M10, lab10, ID, f"c10s{s}"), hashes=np.array(hashes), categories=np.array(cats))
# ---- v1 synthetic "results" with the unchanged v1 code on the synthetic R0
z = np.load(L.emb_path("R0_dinov2s", "c100train")); E = z["emb"]
Cen = np.stack([E[lab100 == i].mean(0) for i in range(100)]); Cen /= np.linalg.norm(Cen, axis=1, keepdims=True); Dp = 1 - Cen @ Cen.T
Dp_gpu = Dp + np.triu(rng.normal(scale=2e-6, size=Dp.shape), 1); Dp_gpu = np.triu(Dp_gpu, 1) + np.triu(Dp_gpu, 1).T   # "GPU" noise << 1e-4
pd.DataFrame(Dp_gpu, index=C100, columns=C100).to_csv(L.V1_PERC)
Ds = np.linalg.norm(Zs[:, None] - Zs[None], axis=2); Ds = Ds / Ds.max(); np.fill_diagonal(Ds, 0); pd.DataFrame(Ds, index=C100, columns=C100).to_csv(L.V1_SEM)
iu = list(itertools.combinations(range(100), 2)); mp = np.median([Dp_gpu[i, j] for i, j in iu]); ms = np.median([Ds[i, j] for i, j in iu])
cq = L._load("cq_synth", L.E8 / "src/phase2/classify_quadrants.py", [L.E8])
allp = pd.DataFrame([dict(pair_id=k, class_u=C100[i], class_v=C100[j], d_perc=round(Dp_gpu[i, j], 6), d_sem=round(Ds[i, j], 6), quadrant=cq.classify_pair(Dp_gpu[i, j], Ds[i, j], mp, ms)) for k, (i, j) in enumerate(iu)])
sp = pd.concat([g.sample(25, random_state=42) for _, g in allp.groupby("quadrant")]).reset_index(drop=True); sp.insert(0, "sample_id", range(100))
sp["superclass_u"] = "-"; sp["superclass_v"] = "-"; sp["same_superclass"] = False; sp.to_csv(L.V1_SAMPLED, index=False)
cfg = re8.load_config(); pool, h2c = re8.build_episode_pool(E, lab100, C100, pool_per_class=200, seed=42); agg = []
for _, p in sp.iterrows(): agg += re8._run_pair(p.to_dict(), pool, h2c, cfg, cfg["phase3"]["contrast_base_seed"])[1]
A = pd.DataFrame(agg); reg = L.regression_v1(); (T / "v1/raw").mkdir(exist_ok=True); rows = []
for ln in ["centroid", "exemplar_knn_k3", "logreg", "random"]:
    s = A[A.learner == ln]; rows.append(reg.fit_models(s.c1_accuracy.to_numpy(float), s.d_perc.to_numpy(float), s.d_sem.to_numpy(float), ln, T / "v1/raw"))
pd.DataFrame(rows).to_csv(L.V1_REGTAB, index=False)
# ---- human-similarity-like synthetic distances
def hum(Z, noise, r): D = np.linalg.norm(Z[:, None] - Z[None], axis=2); D = D / D.max(); return D + r.normal(scale=noise, size=D.shape)
Ht, Hh, H10 = hum(Z100, 0.08, rng), hum(Z100, 0.15, rng), hum(Z10, 0.05, rng)
P = pd.DataFrame([dict(a=x, b=y) for x, y in itertools.combinations(C100, 2)]); ix = {c: i for i, c in enumerate(C100)}
P["d_perc"] = [Dp_gpu[ix[x], ix[y]] for x, y in zip(P.a, P.b)]; P["d_sem"] = [Ds[ix[x], ix[y]] for x, y in zip(P.a, P.b)]
key = {tuple(sorted((u, v))) for u, v in zip(sp.class_u, sp.class_v)}; P["is_s"] = [(x, y) in key for x, y in zip(P.a, P.b)]
sel_t = set(P[P.is_s].sample(61, random_state=1).index) | set(P[~P.is_s].sample(1100, random_state=2).index)
sel_h = set(P[P.is_s].sample(32, random_state=3).index) | set(P[~P.is_s].sample(600, random_state=4).index)
P["d_things"] = [Ht[ix[x], ix[y]] if k in sel_t else np.nan for k, x, y in zip(P.index, P.a, P.b)]
P["d_hsj"] = [Hh[ix[x], ix[y]] if k in sel_h else np.nan for k, x, y in zip(P.index, P.a, P.b)]
P["hsj_n"] = np.where(P.d_hsj.notna(), 20, 0); P["d_hsj_raw"] = P.d_hsj
A0 = A[A.learner == "centroid"]; acc = {tuple(sorted((u, v))): c for u, v, c in zip(A0.class_u, A0.class_v, A0.c1_accuracy)}
P["acc_pair"] = [acc.get((x, y), np.nan) for x, y in zip(P.a, P.b)]
P.drop(columns="is_s").to_csv(L.PAIRS_C100, index=False); P[P.acc_pair.notna()].drop(columns="is_s").to_csv(L.SAMPLED_100, index=False)
P10 = pd.DataFrame([dict(a=x, b=y) for x, y in itertools.combinations(sorted(C10), 2)]); j = {c: C10.index(c) for c in C10}
P10["d_perc"] = 0.5; P10["d_sem"] = rng.uniform(0.2, 0.9, len(P10)); P10["d_things"] = [H10[j[x], j[y]] for x, y in zip(P10.a, P10.b)]
P10["d_hsj"] = [H10[j[x], j[y]] + rng.normal(scale=.05) for x, y in zip(P10.a, P10.b)]; P10["d_c10h"] = [H10[j[x], j[y]] + rng.normal(scale=.03) for x, y in zip(P10.a, P10.b)]
P10.to_csv(L.PAIRS_C10, index=False)
rc = L.run_clean(); tr = []
for t, defs in (("near", rc.NEAR_DEF), ("mid", rc.MID_DEF), ("far", rc.FAR_DEF)):
    for w, (x, y) in defs:
        r = P10[((P10.a == min(x, y)) & (P10.b == max(x, y)))].iloc[0]; tr.append(dict(tier=t, word=w, a=x, b=y, d_sem=r.d_sem, d_things=r.d_things, d_hsj=r.d_hsj, d_c10h=r.d_c10h))
pd.DataFrame(tr).to_csv(L.TIERS_C10, index=False)
C10ALPHA = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]; yte = np.load(L.emb_path("R0_dinov2s", "c10test"))["labels"]
cnt = np.zeros((10000, 10), int)
for i, y in enumerate(yte): cnt[i] = rng.multinomial(50, np.where(np.arange(10) == y, 0.85, 0.15 / 9))
np.save(L.C10H_COUNTS, cnt)
# ---- H5 inputs (synthetic Glasgow-like norms; real class NAMES only)
words = sorted({w for w in ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] + [c.replace("_", " ") for c in C100] + ["tree", "fish", "pepper (vegetable)", "cattle", "lawnmower", "television", "telephone"]})
with open(L.GLASGOW, "w", encoding="latin-1") as f:
    f.write("h1\nh2\n"); [f.write(",".join([w] + ["0"] * 19 + [f"{rng.uniform(2, 6):.2f}"]) + "\n") for w in words]
idx = np.concatenate([np.where(lab100 == i)[0][:100] for i in range(100)]); np.savez(L.SUB100, idx=idx, labels=lab100[idx], class_names=np.array(C100), emb=np.zeros((len(idx), 2), np.float32))
# ---- v1 CIFAR-10 cache = synthetic R0 pool (EXISTING), v1 T ledger with the UNCHANGED v1 code (exp_007 cache branch)
import shutil; shutil.copy(L.emb_path("R0_dinov2s", "c10pool"), T / "v1/cifar_all10.npz") if L.emb_path("R0_dinov2s", "c10pool") != T / "v1/cifar_all10.npz" else None
if not a.no_v1_ledger:
    rcv = L._load("rc_v1_synth", L.E7B / "run_clean.py"); rcv._shared007.EMB_CACHE_DIR = T / "v1"; rcv.RESULTS_DIR = T / "v1/ledger"; rcv.RESULTS_DIR.mkdir(exist_ok=True)
    rcv.load_or_build_embeddings = rcv._shared007.load_or_build_embeddings; rcv.main(); shutil.copy(T / "v1/ledger/ledger_c2_homogeneous.jsonl", L.V1_LEDGER)
# ---- people (synthetic per-trial tables), human-part outputs
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
nP = 600 if a.h4_full else 60; subids = np.concatenate([np.arange(1, 521), rng.integers(1, 521, 80)]) if a.h4_full else np.arange(1, nP + 1)
conds = ["one", "three_subordinate", "three_basic", "three_superordinate"]; cats3 = ["vegetables", "vehicles", "animals"]
base = []
for pi in range(nP):
    for t in range(1, 13):
        cat = cats3[(t - 1) % 3]; cn = {"animals": 3, "vehicles": 2}.get(cat, 1); cond = conds[(pi + t) % 4]
        for it in TEST: base.append((pi % 12 + 1, int(subids[pi]), t, cat, cond, it, int(int(it[1]) == cn), it[3:6]))
base = pd.DataFrame(base, columns=["exp", "subid", "trial", "category", "condition", "image", "in_target", "level"])
for ID in GOOD + NULL:
    r = np.random.default_rng(seed_of(ID, "h4")); lev = base.level.map({"sub": 0.2, "bas": 0.45, "sup": 0.7}).to_numpy()
    dist = np.clip(lev + (1 - base.in_target) * 0.4 + r.normal(scale=0.12, size=len(base)), 0, 2); zd = (dist - dist.mean()) / dist.std()
    b = 0.0 if ID in NULL else -2.0; pr = 1 / (1 + np.exp(-(-1 + b * zd + r.normal(scale=.5, size=base.subid.max() + 1)[base.subid])))
    d = base.copy(); d["selected"] = (r.random(len(d)) < pr).astype(int); d["dist"] = dist; d["exp_recoded"] = d.exp; d["preregistered"] = True
    d.to_csv(T / f"human/data/h4_long_{ID}.csv", index=False)
pd.DataFrame([dict(encoder=e, model="poscontrol_slope", items=8, term="zd", estimate=(-2.0 if e not in NULL else 0.01), se=.05, z=0, p=.001, warn="", singular=False, nobs=1,
                   gate_pass=e not in NULL, zd_lo=(-2.1 if e not in NULL else -0.09), zd_hi=(-1.9 if e not in NULL else 0.11)) for e in L.P10]).to_csv(L.H_POSCONTROL, index=False)
pd.DataFrame([dict(encoder=e, control_positivo="SINTETICO", **{f"region_d{d}": "SINTETICO" for d in ("0.05", "0.10", "0.20")}, region_24="SINTETICO", perfil="SINTETICO") for e in L.P10]).to_csv(L.H_REGIONS, index=False)
import importlib.util; spec = importlib.util.spec_from_file_location("eq02s", L.EE / "scripts/02_equivalence.py"); eq = importlib.util.module_from_spec(spec); argv = sys.argv; sys.argv = [argv[0]]; spec.loader.exec_module(eq); sys.argv = argv
ids = eq.GROUP_SET; Rm = np.full((10, 10), 0.2); g1 = {"R0_dinov2s", "O1_dinov2b", "E2_vitsup", "E3_rn50sup"}; g2 = {"R1_cvcl", "E1_siglip", "O2_openclip", "E4_mae"}
for i, x in enumerate(ids):
    for k, y in enumerate(ids): Rm[i, k] = 1.0 if i == k else (0.9 if ({x, y} <= g1 or {x, y} <= g2) else 0.2)
json.dump(dict(grouping=eq.group_all(ids, Rm)), open(L.H_GROUPING, "w"), indent=1, default=bool)
print("synthetic world ready:", T)
