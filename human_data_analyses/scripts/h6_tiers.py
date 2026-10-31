"""H6(d): CIFAR-10 tiers with the CVCL encoder. Runs the repo's exp_007b_extension_clean/run_clean.py main() UNCHANGED
except: embeddings loader -> given npz; LEARNER_NAMES -> ['centroid'] (li=0, so its seed is unchanged);
POOL_CONFIG -> std_nway only (C1 is pool-independent); RESULTS_DIR -> out/. Arg: dino|cvcl (dino = reproduction check)."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, pathlib, importlib.util, numpy as np
REPO = pathlib.Path(_P.MAIN_REPO); B = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location("rc", REPO / "experiments/exp_007b_extension_clean/run_clean.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
which = sys.argv[1]
path = {"dino": REPO / ".cache/embeddings_exp007/cifar_all10.npz", "cvcl": pathlib.Path(f"{_P.CACHE}/cvcl_cifar_all10.npz")}[which]
from src.eval.concepts import ImageRecord
def loader(_):
    z = np.load(path); hs = [str(h) for h in z["hashes"]]; cs = [str(c) for c in z["categories"]]
    return {h: z["emb"][i] for i, h in enumerate(hs)}, [ImageRecord(img_hash=h, object_category=c) for h, c in zip(hs, cs)]
rc.load_or_build_embeddings = loader; rc.LEARNER_NAMES = ["centroid"]; rc.POOL_CONFIG = {"std_nway": rc.POOL_CONFIG["std_nway"]}
rc.POOL_TYPES = ["std_nway"]; rc.RESULTS_DIR = B / f"out/h6_tiers_{which}"; rc.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
rc.main()
rows = [json.loads(l) for l in open(rc.RESULTS_DIR / "ledger_c2_homogeneous.jsonl")]
acc = {}
for r in rows:
    for k, v in r["per_concept_c1"].items(): acc.setdefault((r["tier"], k), []).append(v)
out = {f"{t}|{k}": float(np.mean(v)) for (t, k), v in acc.items()}
out["c1_by_tier"] = {t: float(np.mean([r["c1"] for r in rows if r["tier"] == t])) for t in ["native", "near", "mid", "far"]}
json.dump(out, open(B / f"out/h6_tiers_{which}.json", "w"), indent=1); print(json.dumps(out, indent=1))
