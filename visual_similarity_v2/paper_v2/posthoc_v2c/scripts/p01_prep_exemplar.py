"""Post hoc (v2c, after the DeepReview / paperreview.ai reviews; outside every frozen plan).
Builds per-trial tables with, for every trial x test item and representation:
  dist  = 1 - cos(item, normalized centroid of the teaching images)  (identical to the frozen h4_long_<ID>.csv; asserted)
  dmin  = min_i (1 - cos(item, example_i))        (nearest-exemplar distance)
  dmean = mean_i (1 - cos(item, example_i))       (mean exemplar distance)
  disp  = mean pairwise cosine distance among the teaching images (0 with one example)
Same row order, same 24 items, same embeddings (frozen emb_<ID>.npz) as h4_prep_encoders.py at a945ed2.
Per-trial derivatives of Lewis & Frank (2018) are written locally only and are NOT redistributed.
Usage: python p01_prep_exemplar.py <repo> <all_raw_A.csv> <experiment_key.csv> <out_dir>"""
import sys, re, pathlib, hashlib, json, numpy as np, pandas as pd
repo, raw_p, key_p, out = map(pathlib.Path, sys.argv[1:5])
EE = repo / "visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence"
IDS = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
SPLIT = r"[,;\- ]+"
assert hashlib.md5(raw_p.read_bytes()).hexdigest() == "e42712b8a15187983620a1d9f6edb5e9"   # OSF md5 listed in download_data.sh
raw = pd.read_csv(raw_p); key = pd.read_csv(key_p).iloc[:12]; key["exp"] = key.exp.astype(int)
def load_emb(path):
    z = np.load(path); return {str(n): v / np.linalg.norm(v) for n, v in zip(z["names"], z["emb"].astype(np.float32))}
report = {}
for ID in IDS:
    emb = load_emb(EE / f"out/emb_{ID}.npz"); rows = []
    for _, r in raw.iterrows():
        for t in range(1, 13):
            tp = re.split(SPLIT, str(r[f"trainPics_T{t}"]).strip())
            sel = set(re.split(SPLIT, str(r[f"selected_T{t}"]).strip())) if pd.notna(r[f"selected_T{t}"]) else set()
            E = np.stack([emb[x] for x in tp]); m = E.mean(0); m /= np.linalg.norm(m)
            disp = 0.0 if len(tp) == 1 else float(np.mean([1 - E[i] @ E[j] for i in range(len(tp)) for j in range(i + 1, len(tp))]))
            cat = r[f"category_T{t}"]; cnum = {"animals": 3, "vehicles": 2}.get(cat, 1)
            for it in TEST:
                de = 1 - E @ emb[it]
                rows.append(dict(exp=int(r.exp), subid=int(r.subids), trial=t, category=cat, condition=r[f"condition_T{t}"], image=it,
                                 in_target=int(int(it[1]) == cnum), level=it[3:6], selected=int(it in sel), dist=1 - float(emb[it] @ m),
                                 dmin=float(de.min()), dmean=float(de.mean()), disp=disp, nsup=len(tp)))
    d = pd.DataFrame(rows).merge(key[["exp", "exp_recoded", "preregistered"]], on="exp")
    fr = pd.read_csv(EE / f"data/h4_long_{ID}.csv")
    assert d[["exp", "subid", "trial", "image", "selected"]].equals(fr[["exp", "subid", "trial", "image", "selected"]]), ID
    err = float(np.abs(d.dist.to_numpy() - fr.dist.to_numpy()).max()); assert err <= 1e-5, (ID, err)
    report[ID] = dict(rows=len(d), max_abs_diff_dist_vs_frozen=err)
    d.to_csv(out / f"ex_long_{ID}.csv", index=False); print(ID, d.shape, "max|dist - frozen| =", f"{err:.2e}", flush=True)
(out / "p01_check.json").write_text(json.dumps(report, indent=1))
