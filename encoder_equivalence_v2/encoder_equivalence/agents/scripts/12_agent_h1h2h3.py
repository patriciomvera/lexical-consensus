"""AGENTS addendum, step 3: H1, H2 (+H2b), H3 and the optional H5 per encoder (proposal §1-§2), with the v1 code of
human_data_analyses (stats_lib.spearman_boot / mantel / ols_hc3; same columns, sources and seeds as 02_h1_h2.py, 03_h3.py,
05_h5.py and 06_h6.py). Only d_perc and the learner accuracy change: they are recomputed in the encoder's own space.
  H1  rho Spearman(d_perc_e, d_human) + bootstrap CI (10,000, seed 42) + Mantel (10,000, seed 42): THINGS and HSJ on the 4,950
      CIFAR-100 pairs (centroids of all 500 train images), CIFAR-10H on the 45 CIFAR-10 pairs (centroids of the 2,000-image pool).
      Also: one-sided bootstrap p for 'rho >= SESOI' (region "No apoyo", ACLARACIONES b5) from the SAME bootstrap replicates.
  H2  OLS HC3 acc_pair_e ~ z(d_human) + z(d_sem) on the v1 pairs covered (THINGS n = 61, HSJ n = 32); acc = centroid C1 of D (11).
      H2b adds z(d_perc_e) (descriptive). Descriptive too: H2 on the D-todos pairs covered by each source (does not decide).
  H3  (descriptive) tiers: Spearman(acc_tier_e, d) for d_perc_e, d_sem, d_things, d_hsj, d_c10h over the 10 tier concepts;
      per image (if c10test embeddings exist): CIFAR-10H entropy vs learner margin over the 10,000 test images.
  H5  (optional, --h5) class coherence vs Glasgow AoA, partial Spearman on Zipf; functions taken verbatim from 05_h5.py.
Usage: python 12_agent_h1h2h3.py --encoder ID [--h5]      Output: agents/out/results/H_<ID>.json"""
import sys, json, argparse, pathlib, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
import importlib; sims = importlib.import_module("11_agent_sims")
from scipy import stats
SL = L.stats_lib()
def spearman_boot_dist(x, y, seed=SL.SEED, nb=SL.NB):
    """Same loop, RNG and draws as stats_lib.spearman_boot (tested identical); also returns the replicates."""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x); rho = stats.spearmanr(x, y).statistic; rng = np.random.default_rng(seed); bs = np.empty(nb)
    for b in range(nb):
        i = rng.integers(0, n, n); bs[b] = stats.spearmanr(x[i], y[i]).statistic
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); return rho, lo, hi, bs
def p_rho_below(bs, sesoi):
    """one-sided bootstrap p for H0: rho >= SESOI (small p = rho below SESOI)."""
    b = bs[np.isfinite(bs)]; return float((np.sum(b >= sesoi) + 1) / (len(b) + 1))
def h1_one(df, col, xcol="d_perc_e"):
    d = df.dropna(subset=[col]); rho, lo, hi, bs = spearman_boot_dist(d[xcol], d[col]); mr, mp, n = SL.mantel(df, xcol, col)
    r = dict(n_pairs=len(d), rho=float(rho), ci_lo=float(lo), ci_hi=float(hi), mantel_r=float(mr), mantel_p=float(mp),
             p_below_sesoi={f"{s:.2f}": p_rho_below(bs, s) for s in L.SESOIS})
    r["status"] = "ok" if len(d) >= 3 and all(np.isfinite([rho, lo, hi, mp])) else "tecnico: fuente sin pares o estadístico no finito"
    return r
def h2_one(d, col, acc="acc_e", extra=()):
    m = SL.ols_hc3(d, acc, [*extra, col, "d_sem"]); t = m[col]; s = m["d_sem"]
    r = dict(n=len(d), beta=float(t["beta"]), p=float(t["p"]), ci_lo=float(t["ci_lo"]), ci_hi=float(t["ci_hi"]), r_partial=float(t["r_partial"]),
             r_ci_lo=float(t["r_ci_lo"]), r_ci_hi=float(t["r_ci_hi"]), r2p=float(t["R2_partial"]), p_sem=float(s["p"]), r2p_sem=float(s["R2_partial"]),
             n_others=len(extra) + 1, full=m)
    r["status"] = "ok" if all(np.isfinite([r["beta"], r["p"], r["r_partial"], r["r_ci_lo"], r["r_ci_hi"]])) else "tecnico: estadístico no finito"
    return r
def tech(e): return dict(status=f"tecnico: {e!r}")
def run(ID, h5=False):
    out = {"encoder": ID, "H1": {}, "H1_desc_c10": {}, "H2": {}, "H2b": {}, "H2_Dall_desc": {}, "H3": {}}
    # ---- H1
    try:
        D, names = sims.dperc_c100(ID); e100 = pd.read_csv(L.SIM_DIR / f"dperc_c100_{ID}.csv")
        P100 = pd.read_csv(L.PAIRS_C100).merge(e100, on=["a", "b"], how="left"); assert P100.d_perc_e.notna().all() and len(P100) == 4950
        for src in ("THINGS", "HSJ"):
            try: out["H1"][src] = h1_one(P100, L.HCOL[src])
            except Exception as e: out["H1"][src] = tech(e)
    except Exception as e: P100 = None; out["H1"]["THINGS"] = out["H1"]["HSJ"] = tech(e)
    try:
        cen = sims.c10_centroids(ID); P10 = pd.read_csv(L.PAIRS_C10); P10["d_perc_e"] = [1 - float(cen[a] @ cen[b]) for a, b in zip(P10.a, P10.b)]
        out["H1"]["CIFAR10H"] = h1_one(P10, "d_c10h")
        for src in ("THINGS", "HSJ"): out["H1_desc_c10"][src] = h1_one(P10, L.HCOL[src])
    except Exception as e: cen = None; out["H1"]["CIFAR10H"] = tech(e)
    # ---- H2 / H2b (pairs of D; v1 d_sem column of pairs_c100 = corrected WordNet map, as 02_h1_h2.py)
    try:
        A = pd.read_csv(L.SIM_DIR / f"D_{ID}_c1.csv"); A = A[A.learner == "centroid"][["class_u", "class_v", "c1_accuracy"]]
        A["a"] = [min(u, v) for u, v in zip(A.class_u, A.class_v)]; A["b"] = [max(u, v) for u, v in zip(A.class_u, A.class_v)]
        s = pd.read_csv(L.SAMPLED_100).merge(A[["a", "b", "c1_accuracy"]].rename(columns={"c1_accuracy": "acc_e"}), on=["a", "b"], how="left")
        assert s.acc_e.notna().all() and len(s) == 100
        if P100 is not None: s = s.merge(P100[["a", "b", "d_perc_e"]], on=["a", "b"], how="left")
        for src in L.H2_SOURCES:
            col = L.HCOL[src]; d = s.dropna(subset=[col]).reset_index(drop=True)
            try: out["H2"][src] = h2_one(d, col)
            except Exception as e: out["H2"][src] = tech(e)
            try: out["H2b"][src] = h2_one(d, col, extra=("d_perc_e",)) if "d_perc_e" in d else tech("sin d_perc_e")
            except Exception as e: out["H2b"][src] = tech(e)
    except Exception as e:
        for src in L.H2_SOURCES: out["H2"][src] = tech(e)
    try:
        Aall = pd.read_csv(L.SIM_DIR / f"Dall_{ID}_c1.csv"); Aall["a"] = [min(u, v) for u, v in zip(Aall.class_u, Aall.class_v)]; Aall["b"] = [max(u, v) for u, v in zip(Aall.class_u, Aall.class_v)]
        Q = P100.merge(Aall[["a", "b", "c1_accuracy"]].rename(columns={"c1_accuracy": "acc_e"}), on=["a", "b"])
        for src in L.H2_SOURCES:
            col = L.HCOL[src]; out["H2_Dall_desc"][src] = {k: v for k, v in h2_one(Q.dropna(subset=[col]).reset_index(drop=True), col).items() if k != "full"} | {"note": "descriptive, does not decide"}
    except Exception as e: out["H2_Dall_desc"] = tech(e)
    # ---- H3 (descriptive)
    try:
        rows = [json.loads(l) for l in open(L.SIM_DIR / f"T_{ID}_episodes_seed42/ledger_c2_homogeneous.jsonl")]
        acc = {}
        for r in rows:
            if r["learner"] == "centroid" and r["pool_type"] == "std_nway" and r["tier"] in ("near", "mid", "far"):
                for w, v in r["per_concept_c1"].items(): acc.setdefault((r["tier"], w), []).append(v)
        t = pd.read_csv(L.TIERS_C10); t["acc_e"] = [float(np.mean(acc[(a, b)])) for a, b in zip(t.tier, t.word)]
        t["d_perc_e"] = [1 - float(cen[a] @ cen[b]) for a, b in zip(t.a, t.b)]
        for c in ["d_perc_e", "d_sem", "d_things", "d_hsj", "d_c10h"]:
            x = t.dropna(subset=[c]); r = SL.spearman_boot(x.acc_e.values, x[c].values); out["H3"][f"tier_{c}"] = dict(n=len(x), rho=float(r[0]), ci_lo=float(r[1]), ci_hi=float(r[2]))
    except Exception as e: out["H3"]["tiers"] = tech(e)
    te = L.emb_path(ID, "c10test")
    if te.exists() and cen is not None:
        try:
            C10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]; C = np.stack([cen[c] for c in C10])
            z = np.load(te); X = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True); y = z["labels"]
            Dm = 1 - X @ C.T; own = Dm[np.arange(len(y)), y]; Dr = Dm.copy(); Dr[np.arange(len(y)), y] = np.inf; margin = own - Dr.min(1)
            cnt = np.load(L.C10H_COUNTS).astype(float); Pp = cnt / cnt.sum(1, keepdims=True); H = -(np.where(Pp > 0, Pp * np.log2(np.where(Pp > 0, Pp, 1)), 0)).sum(1)
            assert (Pp.argmax(1) == y).mean() > 0.9; r = SL.spearman_boot(H, margin)
            out["H3"]["c10h_entropy_vs_margin"] = dict(n=len(y), centroid_acc=float((Dm.argmin(1) == y).mean()), rho=float(r[0]), ci_lo=float(r[1]), ci_hi=float(r[2]))
        except Exception as e: out["H3"]["c10h_entropy_vs_margin"] = tech(e)
    else: out["H3"]["c10h_entropy_vs_margin"] = dict(status="no corrido: faltan embeddings c10test (opcional en la opción mínima)")
    if h5: out["H5"] = run_h5(ID)
    L.write_json(L.RES_DIR / f"H_{ID}.json", out); return out
def run_h5(ID):
    ns, _ = L.v1_functions(L.HDA / "scripts/05_h5.py", {"SENSE", "ALT", "word", "disp", "pspear"})
    try:
        import csv
        from wordfreq import zipf_frequency
        rows = list(csv.reader(open(L.GLASGOW, encoding="latin-1")))[2:]; ns["G"] = {r[0].strip().lower(): float(r[20]) for r in rows if r[20].strip()}
        sub = np.load(L.SUB100); z = np.load(L.emb_path(ID, "c100train")); idx = sub["idx"]; names = [str(n) for n in z["class_names"]]
        assert names == [str(n) for n in sub["class_names"]] and (z["labels"][idx] == sub["labels"]).all()
        cw = {i: ns["word"](c) for i, c in enumerate(names)}; cw = {i: w for i, w in cw.items() if w}; d100 = ns["disp"](z["emb"][idx], z["labels"][idx], cw)
        z10 = np.load(L.emb_path(ID, "c10pool")); c10 = list(dict.fromkeys(z10["categories"])); cw10 = {c: ns["word"](c) for c in c10}; cw10 = {k: v for k, v in cw10.items() if v}
        d10 = ns["disp"](z10["emb"], z10["categories"], cw10); res = {}
        for tag, d in (("C100_glasgow", d100), ("C10_glasgow", d10)):
            w = sorted(d); res[tag] = ns["pspear"](np.array([d[x] for x in w]), np.array([ns["G"][x] for x in w]), np.array([zipf_frequency(x.split(" (")[0], "en") for x in w]))
        for k, q in zip(res, SL.bh([res[k]["p"] for k in res])): res[k]["p_BH"] = float(q)
        res["status"] = "ok"; return res
    except Exception as e: return tech(e)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder", required=True); ap.add_argument("--h5", action="store_true"); a = ap.parse_args()
    L.ensure_dirs(); r = run(a.encoder, a.h5); print(a.encoder, "H1/H2/H3 written; statuses:", {k: {s: v.get("status") for s, v in r[k].items() if isinstance(v, dict)} for k in ("H1", "H2")})
