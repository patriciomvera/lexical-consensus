"""SYNTHETIC data only (no participant data is read). Writes, under <dir>:
  raw/all_raw_A_SYNTH.csv  (same columns that h4_prep.py uses: exp, subids, AcceptTime, trainPics_T*, selected_T*, category_T*, condition_T*)
  raw/experiment_key_SYNTH.csv (exp, exp_recoded, preregistered, order; 14 rows, only the first 12 are used, as in v1)
  emb/emb_<ID>.npz for the 13 sources (45 fake stimuli, L2, float32)
  raw/h4_first_session_SYNTH.csv (subid, first_exp)
Ground truth: selection is generated from R0's distance; true zd slope -1.0 in "1-3" experiments and -0.6 in "3-1" (Delta > 0),
plus a per-person random slope (SD SLOPE_SD, REVISION3 I1) and a per-person random intercept (SD 0.8)."""
import sys, re, pathlib, numpy as np, pandas as pd
D = pathlib.Path(sys.argv[1]); N_PER_EXP = int(sys.argv[2]) if len(sys.argv) > 2 else 15; SLOPE_SD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
rng = np.random.default_rng(2026)
(D / "raw").mkdir(parents=True, exist_ok=True); (D / "emb").mkdir(parents=True, exist_ok=True)
NAMES = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 5), ("bas", 4), ("sup", 6)) for i in range(1, k + 1)]
ORDER = {e: ("1-3" if e in (1, 2, 5, 7, 10, 12) else "3-1") for e in range(1, 13)}
dim = 64; dom = {c: rng.normal(0, 1, dim) for c in (1, 2, 3)}; lev = {(c, l): rng.normal(0, .7, dim) for c in (1, 2, 3) for l in ("sub", "bas", "sup")}
base = np.stack([dom[int(n[1])] + lev[(int(n[1]), n[3:6])] + rng.normal(0, .5, dim) for n in NAMES]); other = rng.normal(0, 1, (45, dim))
def save(ID, X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True); np.savez(D / "emb" / f"emb_{ID}.npz", emb=X.astype(np.float32), names=np.array(NAMES))
save("R0_dinov2s", base); save("O1_dinov2b", base + rng.normal(0, .15, base.shape))
for ID, s in (("R1_cvcl", .6), ("E1_siglip", .6), ("O2_openclip", .6)): save(ID, base + rng.normal(0, s, base.shape))
alt = np.stack([dom[int(n[1])] + rng.normal(0, 1.2, dim) for n in NAMES])
for ID in ("E2_vitsup", "E3_rn50sup", "E4_mae", "E4s_mae_cls"): save(ID, alt + rng.normal(0, .5, base.shape))
for ID in ("F1_rn50rand0", "F1_rn50rand1", "F1_rn50rand2", "F2_pixels"): save(ID, rng.normal(0, 1, (45, dim)) + 3)
E = {n: v for n, v in zip(NAMES, np.load(D / "emb" / "emb_R0_dinov2s.npz")["emb"])}
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
CAT = {1: "vegetables", 2: "vehicles", 3: "animals"}; CONDS = ["one", "three_subordinate", "3bas", "three_superordinate"]
TRAIN = {"one": ["sub3"], "three_subordinate": ["sub3", "sub4", "sub5"], "3bas": ["sub3", "bas3", "bas4"], "three_superordinate": ["sub3", "bas3", "sup5"]}
SEPS = [",", ";", "-", " ", ", "]
rows = []; sid = 1000; prev = []
for e in range(1, 13):
    for p in range(N_PER_EXP):
        if prev and rng.random() < .1: s = int(rng.choice(prev))
        else: sid += 1; s = sid; prev.append(s)
        r = dict(exp=e, subids=s, AcceptTime=f"2016-0{1 + e % 9}-1{p % 10}T10:{p:02d}:00Z"); u = rng.normal(0, .8); us = rng.normal(0, SLOPE_SD)
        trials = [(c, k) for c in CONDS for k in (1, 2, 3)]; rng.shuffle(trials)
        for t, (cond, c) in enumerate(trials, 1):
            tp = [f"c{c}_{x}" for x in TRAIN[cond]]; m = np.mean([E[x] for x in tp], 0); m /= np.linalg.norm(m)
            beta = -1.0 if ORDER[e] == "1-3" else -0.6; sel = []
            for it in TEST:
                d = 1 - float(E[it] @ m); tgt = int(it[1]) == c
                logit = (1.0 + (beta + us) * (d - .5) / .15 + u) if tgt else -4 + u
                if rng.random() < 1 / (1 + np.exp(-logit)): sel.append(it)
            r[f"trainPics_T{t}"] = SEPS[t % 5].join(tp); r[f"selected_T{t}"] = SEPS[(t + 1) % 5].join(sel) if sel else np.nan
            r[f"category_T{t}"] = CAT[c]; r[f"condition_T{t}"] = cond
        rows.append(r)
raw = pd.DataFrame(rows); raw.to_csv(D / "raw" / "all_raw_A_SYNTH.csv", index=False)
pd.DataFrame(dict(exp=[str(e) for e in range(1, 13)] + ["x", "y"], exp_recoded=list(range(101, 115)), preregistered=[True] * 14,
                  order=[ORDER[e] for e in range(1, 13)] + ["", ""])).to_csv(D / "raw" / "experiment_key_SYNTH.csv", index=False)
raw.assign(t=pd.to_datetime(raw.AcceptTime, utc=True)).sort_values(["subids", "t", "exp"]).groupby("subids", as_index=False).first() \
   .rename(columns={"subids": "subid", "exp": "first_exp"})[["subid", "first_exp"]].to_csv(D / "raw" / "h4_first_session_SYNTH.csv", index=False)
print("synthetic:", raw.shape, "persons", raw.subids.nunique(), "->", D)
