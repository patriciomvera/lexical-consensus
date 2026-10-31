"""H6(a,b,c) inputs: CVCL d_perc over all 4,950 CIFAR-100 pairs (repo phase1 rule: L2 centroids of ALL 500 train images,
1 - cos) and acc_pair_CVCL on the same 100 sampled pairs via the repo's exp_008 phase3 _run_pair UNCHANGED
(pool 200/class seed 42, contrast seed 9000+pair_id, 30 episodes 3-way 5-shot, master_seed 42), learner = centroid only."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, pathlib, importlib.util, itertools, numpy as np, pandas as pd
REPO = pathlib.Path(_P.MAIN_REPO); B = pathlib.Path(__file__).resolve().parents[1]
E8 = REPO / "experiments/exp_008_dissociation"; sys.path.insert(0, str(E8)); sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("re8", E8 / "src/phase3/run_episodes.py"); re8 = importlib.util.module_from_spec(spec); spec.loader.exec_module(re8)
z = np.load(f"{_P.CACHE}/cvcl_cifar100_train.npz"); emb, lab, names = z["emb"], z["labels"].astype(int), [str(n) for n in z["class_names"]]
assert names == list(re8.CIFAR100_CLASSES) or sorted(names) == sorted(re8.CIFAR100_CLASSES)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
C = np.stack([emb[lab == i].mean(0) for i in range(100)]); C /= np.linalg.norm(C, axis=1, keepdims=True); D = 1 - C @ C.T
pd.DataFrame([dict(a=names[i], b=names[j], d_perc_cvcl=float(D[i, j])) for i, j in itertools.combinations(range(100), 2)]).to_csv(B / "data/cvcl_dperc_c100.csv", index=False)
cfg = re8.load_config(); re8.LEARNER_NAMES = ["centroid"]
pool, h2c = re8.build_episode_pool(emb, lab, names, pool_per_class=cfg["dataset"]["episode_pool_per_class"], seed=42)
sp = pd.read_csv(sorted((E8 / "results").glob("*/phase2/sampled_pairs.csv"))[0]); rows = []
for _, p in sp.iterrows():
    _, agg = re8._run_pair(p.to_dict(), pool, h2c, cfg, cfg["phase3"]["contrast_base_seed"])
    a = agg[0]; rows.append(dict(sample_id=a["sample_id"], a=p.class_u, b=p.class_v, contrast1=a["contrast1"], contrast2=a["contrast2"],
                                 acc_pair_cvcl=a["c1_accuracy"], d_perc_cvcl=float(D[names.index(p.class_u), names.index(p.class_v)])))
    print(a["sample_id"], a["c1_accuracy"], flush=True)
pd.DataFrame(rows).to_csv(B / "data/cvcl_acc_c100.csv", index=False); print("saved")
