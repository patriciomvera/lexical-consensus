"""AGENTS addendum, step 2: learner simulations per encoder with the v1 code UNCHANGED (proposal §2), generalising
human_data_analyses/scripts/h6_tiers.py (T) and h6_c100.py (D) to --encoder <ID> and to all v1 learners. v1 files are not edited:
the modules are loaded with importlib and only the embedding loader / RESULTS_DIR / MASTER_SEED / LEARNER_NAMES are replaced,
exactly as h6_tiers.py, h6_c100.py and revision_runs/scripts/r04_run_clean_seed.py do.
Analyses (--analysis):
  dispersion  §3.6, BEFORE any accuracy: per encoder on the 100 v1 pairs: SD and range of d_perc, Spearman(d_perc, d_sem), pairs per
              recalculated quadrant -> results/agent_pairs_dispersion.json  (use --encoder all)
  T           exp_007b run_clean.main(): 6 learners, 7 pools, 3-way 5-shot, 30 episodes, --seed (42 = primary) and --pool
              (c10pool = fixed pool, T-sem 'episodes'; c10pool_seed<s> = T-sem 'full'). With R0, seed 42 and the v1 cache it is the
              pipeline control 1 (§3.5): the ledger is compared with the v1 ledger -> results/control_T_exact.json
  D           exp_008 _run_pair on the SAME 100 v1 pairs, 4 v1 learners, pool 200/class seed 42, contrast 9000 + sample_id (v1 code);
              Model PS with the v1 code (raw d, classical OLS) and with stats_lib.ols_hc3 (z, HC3, partial r + Fisher CI); rules A/B/C
              (v1 _apply_rule) with quadrants recalculated: d_perc median of this encoder over the 4,950 pairs, as v1 (REVISION5 b3);
              the 100-pair-median quadrants are kept as a descriptive column (quadrant_e100). With R0 it also evaluates
              pipeline control 2 (§3.5, Ariadna's three-level tolerance) -> results/control_c100_recode.json
  Dall        D-todos (I-A2, mandatory sensitivity): the 4,950 CIFAR-100 pairs, centroid only, 30 episodes, contrast 9000 + pair_id
              (pair_id = index of the pair in the v1 triu order of perceptual_distance_matrix.csv), Model PS.
Outputs in agents/out/sims/ (per-pair tables, ledgers) and agents/out/results/<analysis>_<ID>[...].json."""
import os, sys, json, time, argparse, pathlib, itertools, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
from scipy import stats
V1_LEARNERS_T = ["centroid", "exemplar_knn_k3", "kcentroid_k2", "logreg", "linsvm", "random"]
TIERS = ["native", "near", "mid", "far"]
CONTROL_T_RULE = os.environ.get("AGENTS_CONTROL_T_RULE", "propuesto_b1")   # ACLARACIONES (b2; key name kept): 'estricto' = every field of every row identical
CONTROL2 = dict(pass_maxabs=1e-4, pass_spearman=0.99999, dev_maxabs=2e-3, dev_spearman=0.9999, dev_beta=0.005)   # REVISION4 (5), verbatim
# ---------------------------------------------------------------- embeddings / distances
def load_c100(ID):
    z = np.load(L.emb_path(ID, "c100train")); emb, lab, names = z["emb"], z["labels"].astype(int), [str(n) for n in z["class_names"]]
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True); return emb, lab, names
def centroid_dist_c100(emb, lab, names):
    """h6_c100.py / exp_008 phase1 rule: L2 vectors, centroid of ALL train images of the class, renormalised, d = 1 - cos."""
    C = np.stack([emb[lab == i].mean(0) for i in range(len(names))]); C /= np.linalg.norm(C, axis=1, keepdims=True); return 1 - C @ C.T
def dperc_c100(ID):
    """cached 100x100 matrix (names order) + long table with a < b (alphabetical, as pairs_c100.csv)."""
    f = L.SIM_DIR / f"dperc_c100_{ID}.npz"
    if f.exists(): z = np.load(f); return z["D"], [str(n) for n in z["names"]]
    emb, lab, names = load_c100(ID); D = centroid_dist_c100(emb, lab, names); L.ensure_dirs(); np.savez(f, D=D, names=np.array(names))
    ix = {n: i for i, n in enumerate(names)}
    pd.DataFrame([dict(a=a, b=b, d_perc_e=float(D[ix[a], ix[b]])) for a, b in itertools.combinations(sorted(names), 2)]).to_csv(L.SIM_DIR / f"dperc_c100_{ID}.csv", index=False)
    return D, names
def c10_centroids(ID, s="c10pool"):
    z = np.load(L.emb_path(ID, s)); E = z["emb"]; cats = np.array([str(c) for c in z["categories"]])
    E = E / np.linalg.norm(E, axis=1, keepdims=True); cen = {c: E[cats == c].mean(0) for c in dict.fromkeys(cats)}
    return {c: v / np.linalg.norm(v) for c, v in cen.items()}
def v1_sem():
    S = pd.read_csv(L.V1_SEM, index_col=0); return S
def v1_sem_median():
    S = v1_sem().values.astype(np.float64); return float(np.median(S[np.triu_indices(S.shape[0], 1)]))
def quadrants(d_perc, d_sem, med_perc, med_sem):
    cq = L._load("cq8_agents", L.E8 / "src/phase2/classify_quadrants.py", [L.E8])   # v1 classify_pair, unchanged
    return [cq.classify_pair(a, b, med_perc, med_sem) for a, b in zip(d_perc, d_sem)]
def sampled_with_e(ID):
    D, names = dperc_c100(ID); ix = {n: i for i, n in enumerate(names)}; sp = pd.read_csv(L.V1_SAMPLED)
    sp["d_perc_e"] = [float(D[ix[u], ix[v]]) for u, v in zip(sp.class_u, sp.class_v)]
    med_all = float(np.median(D[np.triu_indices(len(names), 1)]))   # REVISION5 b3: median over the 4,950 pairs, as v1 phase2
    sp["quadrant_e"] = quadrants(sp.d_perc_e, sp.d_sem, med_all, v1_sem_median())
    sp["quadrant_e100"] = quadrants(sp.d_perc_e, sp.d_sem, float(np.median(sp.d_perc_e)), v1_sem_median()); return sp   # descriptive only
# ---------------------------------------------------------------- §3.6 dispersion (before any accuracy)
def dispersion(ids):
    out = {"_rule": "100 v1 pairs (phase2/sampled_pairs.csv); d_sem = exp_008 d_sem; quadrants (quadrant_counts): d_perc median of the encoder over the 4,950 pairs and d_sem median over the 4,950 (v1 rule, REVISION5 b3); quadrant_counts_100: d_perc median over the 100 pairs (descriptive). Descriptive; changes no rule."}
    for ID in ids:
        try:
            sp = sampled_with_e(ID); x = sp.d_perc_e.to_numpy()
            out[ID] = dict(n=len(sp), sd=float(x.std(ddof=1)), min=float(x.min()), max=float(x.max()), range=float(x.max() - x.min()),
                           spearman_dperc_dsem=float(stats.spearmanr(x, sp.d_sem).statistic), quadrant_counts=sp.quadrant_e.value_counts().sort_index().to_dict(),
                           quadrant_counts_100=sp.quadrant_e100.value_counts().sort_index().to_dict())
        except FileNotFoundError as e: out[ID] = dict(status=f"tecnico: falta {e.filename}")
    L.write_json(L.RES_DIR / "agent_pairs_dispersion.json", out); print(json.dumps(out, indent=1)); return out
# ---------------------------------------------------------------- T (exp_007b) and pipeline control 1
def run_T(ID, seed=42, pool="c10pool"):
    rc = L.run_clean(); path = L.emb_path(ID, pool)
    from src.eval.concepts import ImageRecord
    def loader(_):   # = h6_tiers.loader = the cache branch of exp_007 _shared.load_or_build_embeddings
        z = np.load(path); hs = [str(h) for h in z["hashes"]]; cs = [str(c) for c in z["categories"]]
        matrix = z["emb"]   # read ONCE, rows are views: identical to the exp_007 cache branch (h6_tiers re-read the npz per row: ~6 GB RSS)
        return {h: matrix[i] for i, h in enumerate(hs)}, [ImageRecord(img_hash=h, object_category=c) for h, c in zip(hs, cs)]
    mode = "episodes" if pool == "c10pool" else "full"
    od = L.SIM_DIR / f"T_{ID}_{mode}_seed{seed}"; od.mkdir(parents=True, exist_ok=True)
    rc.load_or_build_embeddings = loader; rc.MASTER_SEED = seed; rc.RESULTS_DIR = od
    assert rc.LEARNER_NAMES == V1_LEARNERS_T and len(rc.POOL_TYPES) == 7 and (rc.WAYS, rc.SHOTS, rc.N_EPISODES) == (3, 5, 30)
    t0 = time.time(); rc.main(); secs = time.time() - t0
    rows = [json.loads(l) for l in open(od / "ledger_c2_homogeneous.jsonl")]
    res = summarize_T(rows, ID, pool); res.update(encoder=ID, seed=seed, pool=pool, mode=mode, seconds=round(secs, 1), ledger=str(od / "ledger_c2_homogeneous.jsonl"))
    tag = f"T_{ID}" if (seed == 42 and pool == "c10pool") else f"Tsem_{ID}_{mode}_seed{seed}"
    L.write_json(L.RES_DIR / f"{tag}.json", res)
    if ID == L.R0 and seed == 42 and pool == "c10pool" and L.emb_path(ID, pool) == L.V1_C10_CACHE: control_T(rows)
    return res
def summarize_T(rows, ID, pool):
    from src.eval.stats import bootstrap_ci   # v1 analyze_clean.py: bootstrap_ci(n_resamples=2000, seed=42) over episodes
    sl = L.stats_lib(); out = {"c1_by_tier": {}, "c1_ci": {}}
    std = [r for r in rows if r["pool_type"] == "std_nway"]   # C1 is pool-independent (same value in the 7 pool rows)
    for ln in V1_LEARNERS_T:
        out["c1_by_tier"][ln] = {}; out["c1_ci"][ln] = {}
        for t in TIERS:
            v = np.array([r["c1"] for r in std if r["learner"] == ln and r["tier"] == t]); ci = bootstrap_ci(v, n_resamples=2000, seed=42)
            out["c1_by_tier"][ln][t] = ci.mean; out["c1_ci"][ln][t] = [ci.ci_low, ci.ci_high]
        out.setdefault("tier_order", {})[ln] = " > ".join(sorted(TIERS, key=lambda t: -out["c1_by_tier"][ln][t]))
    nat = out["c1_ci"]["centroid"]["native"]
    out["positive_control"] = dict(rule="prototype (centroid), seed 42, native words: 95% bootstrap CI lower bound > 1/3 (lax gate, declared)",
                                   ci_lo=nat[0], ci_hi=nat[1], pass_=bool(nat[0] > 1 / 3))
    acc = {}
    for r in std:
        if r["learner"] == "centroid":
            for w, v in r["per_concept_c1"].items(): acc.setdefault((r["tier"], w), []).append(v)
    out["per_concept_c1_centroid"] = {f"{t}|{w}": float(np.mean(v)) for (t, w), v in acc.items()}
    try:   # descriptive: Spearman(C1 of the 10 disjunctive concepts, d between their two classes in THIS encoder's pool space)
        rc = L.run_clean(); cen = c10_centroids(ID, pool); xs, ys = [], []
        for t, defs in (("near", rc.NEAR_DEF), ("mid", rc.MID_DEF), ("far", rc.FAR_DEF)):
            for w, (a, b) in defs: xs.append(out["per_concept_c1_centroid"][f"{t}|{w}"]); ys.append(1 - float(cen[a] @ cen[b]))
        rho, lo, hi = sl.spearman_boot(np.array(xs), np.array(ys))
        out["rho_c1_vs_dconcept"] = dict(n=len(xs), rho=float(rho), ci_lo=float(lo), ci_hi=float(hi), note="descriptive, n = 10, no p")
    except Exception as e: out["rho_c1_vs_dconcept"] = dict(status=f"tecnico: {e!r}")
    out["status"] = "ok" if all(np.isfinite(v) for d in out["c1_by_tier"].values() for v in d.values()) else "tecnico: C1 no finita"
    return out
def compare_ledgers(new, old):
    """Row-by-row comparison with the v1 ledger (same order). Returns both readings (ACLARACIONES b2).
    REVISION5 b2: the random learner is reported as a distribution (mean C1 per tier, new vs v1 vs chance 1/3), never required."""
    strict = len(new) == len(old) and all(a == b for a, b in zip(new, old)); keys_ok = len(new) == len(old); bad, mx = {}, 0.0
    for a, b in zip(new, old):
        if any(a[k] != b[k] for k in ("sub", "tier", "concept_type", "learner", "ways", "shots", "episode_id", "pool_type", "chance")): keys_ok = False; break
        if a["learner"] == "random": continue
        for k in ("c1", "c2", "per_concept_c1"):
            if a[k] != b[k]: bad[k] = bad.get(k, 0) + 1
        mx = max(mx, abs(a["margin"] - b["margin"]))
    b1 = bool(keys_ok and not bad and mx <= 1e-6 + 1e-12)
    rnd = [(a["c1"], b["c1"]) for a, b in zip(new, old) if a["learner"] == "random" and a["pool_type"] == "std_nway"]
    rdist = {}
    for a, b in zip(new, old):
        if a["learner"] == "random" and a["pool_type"] == "std_nway": rdist.setdefault(a["tier"], ([], []))[0].append(a["c1"]); rdist[a["tier"]][1].append(b["c1"])
    rdist = {t: dict(mean_c1_new=float(np.mean(x)), mean_c1_v1=float(np.mean(y)), sd_c1_new=float(np.std(x)), sd_c1_v1=float(np.std(y)), n=len(x), chance=1 / 3) for t, (x, y) in rdist.items()}
    return dict(n_new=len(new), n_v1=len(old), estricto=bool(strict), propuesto_b1=b1, keys_identical=bool(keys_ok), nonrandom_mismatches=bad, random_c1_by_tier=rdist,
                nonrandom_margin_max_abs_diff=mx, random_mean_c1_new=float(np.mean([x for x, _ in rnd])) if rnd else None,
                random_mean_c1_v1=float(np.mean([y for _, y in rnd])) if rnd else None)
def control_T(rows):
    old = [json.loads(l) for l in open(L.V1_LEDGER)]; c = compare_ledgers(rows, old); c["rule_used"] = CONTROL_T_RULE; c["ok"] = c[CONTROL_T_RULE]
    c["rule_text"] = {"estricto": "every field of every one of the 5,040 rows identical",
                      "propuesto_b1": "same 5,040 rows/keys; for the 5 non-random learners c1, c2 and per_concept_c1 identical and |margin diff| <= 1e-6 (6-decimal rounding); the random learner (Gumbel noise hashed from the embedding BYTES) is reported, not required"}[CONTROL_T_RULE]
    L.write_json(L.RES_DIR / "control_T_exact.json", c); print("CONTROL_T", json.dumps(c)); return c
# ---------------------------------------------------------------- D (exp_008) and pipeline control 2
def model_ps(df, y="c1_accuracy", xp="d_perc", xs="d_sem", tag="x"):
    """Model PS twice: v1 code (raw d, classical OLS; regression_analysis.fit_models) and stats_lib.ols_hc3 (z-scores, HC3, partial r)."""
    reg = L.regression_v1(); sl = L.stats_lib(); raw = L.SIM_DIR / f"raw_{tag}"; raw.mkdir(parents=True, exist_ok=True)
    v1 = reg.fit_models(df[y].to_numpy(np.float64), df[xp].to_numpy(np.float64), df[xs].to_numpy(np.float64), "centroid", raw)
    d = df.rename(columns={xp: "d_perc_x", xs: "d_sem_x"}); h = sl.ols_hc3(d, y, ["d_perc_x", "d_sem_x"])
    t = dict(n=len(df), beta_v1=v1["PS_beta_perc"], p_v1=v1["PS_beta_perc_p"], beta_sem_v1=v1["PS_beta_sem"], p_sem_v1=v1["PS_beta_sem_p"],
             r2p_perc_v1=v1["PS_partial_r2_perc"], r2p_sem_v1=v1["PS_partial_r2_sem"], vif=v1["PS_vif_perc"],
             beta_z=h["d_perc_x"]["beta"], p_hc3=h["d_perc_x"]["p"], ci_lo_z=h["d_perc_x"]["ci_lo"], ci_hi_z=h["d_perc_x"]["ci_hi"],
             p_sem_hc3=h["d_sem_x"]["p"], r_partial=h["d_perc_x"]["r_partial"], r_ci_lo=h["d_perc_x"]["r_ci_lo"], r_ci_hi=h["d_perc_x"]["r_ci_hi"],
             r2p_perc=h["d_perc_x"]["R2_partial"], r2p_sem=h["d_sem_x"]["R2_partial"], n_others=1)
    t["status"] = "ok" if all(np.isfinite(v) for k, v in t.items() if isinstance(v, float)) else "tecnico: estadístico no finito"
    return t, v1
def run_D(ID):
    re8 = L.run_episodes(); cfg = re8.load_config(); assert re8.LEARNER_NAMES == ["centroid", "exemplar_knn_k3", "logreg", "random"]
    emb, lab, names = load_c100(ID); assert sorted(names) == sorted(re8.CIFAR100_CLASSES)
    sp = sampled_with_e(ID); pool, h2c = re8.build_episode_pool(emb, lab, names, pool_per_class=cfg["dataset"]["episode_pool_per_class"], seed=42)
    t0 = time.time(); eps, agg = [], []
    for _, p in sp.iterrows():
        pd_ = p.to_dict(); pd_["d_perc"] = p.d_perc_e; pd_["quadrant"] = p.quadrant_e   # bookkeeping columns only; _run_pair is unchanged
        e, a = re8._run_pair(pd_, pool, h2c, cfg, cfg["phase3"]["contrast_base_seed"]); eps += e; agg += a
    secs = time.time() - t0; A = pd.DataFrame(agg); A["quadrant_e100"] = A.sample_id.map(dict(zip(sp.sample_id, sp.quadrant_e100))); A.to_csv(L.SIM_DIR / f"D_{ID}_c1.csv", index=False); pd.DataFrame(eps).to_csv(L.SIM_DIR / f"D_{ID}_episodes.csv", index=False)
    res = analyze_D(A, ID); res["seconds"] = round(secs, 1); L.write_json(L.RES_DIR / f"D_{ID}.json", res)
    if ID == L.R0: control_c100(res)
    return res
def analyze_D(A, ID):
    dis = L.dissociation_v1(); cfg = L.run_episodes().load_config(); out = {"encoder": ID, "learners": {}}
    try:
        for ln in ["centroid", "exemplar_knn_k3", "logreg", "random"]:
            sub = A[A.learner == ln].copy(); t, v1 = model_ps(sub, tag=f"D_{ID}_{ln}"); diss = sub[sub.quadrant.isin(["Q3", "Q4"])]
            rp = stats.spearmanr(diss.d_perc, diss.c1_accuracy).statistic if len(diss) >= 3 else np.nan
            rs = stats.spearmanr(diss.d_sem, diss.c1_accuracy).statistic if len(diss) >= 3 else np.nan
            qm = {q: float(sub[sub.quadrant == q].c1_accuracy.median()) for q in ["Q1", "Q2", "Q3", "Q4"]}
            # exactly as dissociation_analysis.main(): rho rounded to 4 decimals, `or nan`
            dr = {"n_pairs": len(diss), "rho_perc": (round(float(rp), 4) if np.isfinite(rp) else None) or float("nan"),
                  "rho_sem": (round(float(rs), 4) if np.isfinite(rs) else None) or float("nan"), "q3_median_c1": qm["Q3"], "q4_median_c1": qm["Q4"]}
            rule, ev = dis._apply_rule(cfg, v1, dr)
            out["learners"][ln] = dict(model_ps=t, q3q4=dr, quadrant_medians=qm, quadrant_counts=sub.quadrant.value_counts().sort_index().to_dict(), rule=rule, evidence=ev)
        out["test"] = out["learners"]["centroid"]["model_ps"]; out["rule_centroid"] = out["learners"]["centroid"]["rule"]; out["status"] = out["test"]["status"]
        if "quadrant_e100" in A:   # REVISION5 b3: rule letter with the 100-pair d_perc median, DESCRIPTIVE only
            sub = A[A.learner == "centroid"]; d1 = sub[sub.quadrant_e100.isin(["Q3", "Q4"])]; q = {k: float(sub[sub.quadrant_e100 == k].c1_accuracy.median()) for k in ["Q3", "Q4"]}
            rp = stats.spearmanr(d1.d_perc, d1.c1_accuracy).statistic if len(d1) >= 3 else np.nan; rs = stats.spearmanr(d1.d_sem, d1.c1_accuracy).statistic if len(d1) >= 3 else np.nan
            dr = {"n_pairs": len(d1), "rho_perc": (round(float(rp), 4) if np.isfinite(rp) else None) or float("nan"), "rho_sem": (round(float(rs), 4) if np.isfinite(rs) else None) or float("nan"), "q3_median_c1": q["Q3"], "q4_median_c1": q["Q4"]}
            _, v1c = model_ps(sub, tag=f"D_{ID}_centroid_q100"); out["rule_centroid_q100_descriptiva"] = dis._apply_rule(cfg, v1c, dr)[0]
            out["quadrant_counts_100_descriptiva"] = sub.quadrant_e100.value_counts().sort_index().to_dict()
    except Exception as e: out["status"] = f"tecnico: {e!r}"
    return out
def control_c100(resD):
    """REVISION4 (5), adopted verbatim: CPU re-encoding of CIFAR-100 with R0 vs the v1 GPU matrix."""
    D, names = dperc_c100(L.R0); P = pd.read_csv(L.V1_PERC, index_col=0); P = P.loc[names, names].to_numpy(np.float64)
    iu = np.triu_indices(len(names), 1); a, b = D[iu], P[iu]; mx = float(np.abs(a - b).max()); sp = float(stats.spearmanr(a, b).statistic)
    beta = resD.get("test", {}).get("beta_v1", np.nan); b1 = L.v1_beta_perc(); db = abs(beta - b1) if np.isfinite(beta) else np.inf
    c = CONTROL2
    if mx <= c["pass_maxabs"] and sp >= c["pass_spearman"]: v = "PASA"
    elif mx <= c["dev_maxabs"] and sp >= c["dev_spearman"] and db <= c["dev_beta"]: v = "PASA con desviación registrada"
    else: v = "FALLA"
    import torch, transformers
    r = dict(verdict=v, ok=v != "FALLA", max_abs_diff=mx, spearman=sp, n_pairs=len(a), beta_perc_R0_cpu=beta, beta_perc_v1=b1, abs_beta_diff=db, thresholds=c,
             versions=dict(torch=torch.__version__, transformers=transformers.__version__), note="beta on the v1 scale (raw d, classical OLS, regression_table.csv PS_beta_perc)")
    L.write_json(L.RES_DIR / "control_c100_recode.json", r); print("CONTROL_C100", json.dumps(r)); return r
def run_Dall(ID, max_pairs=None):
    re8 = L.run_episodes(); cfg = re8.load_config(); re8.LEARNER_NAMES = ["centroid"]   # as h6_c100.py
    emb, lab, names = load_c100(ID); D, dn = dperc_c100(ID); ix = {n: i for i, n in enumerate(dn)}
    P = pd.read_csv(L.V1_PERC, index_col=0); classes = list(P.index); S = v1_sem()
    pool, h2c = re8.build_episode_pool(emb, lab, names, pool_per_class=cfg["dataset"]["episode_pool_per_class"], seed=42)
    allp = [(k, classes[i], classes[j]) for k, (i, j) in enumerate(itertools.combinations(range(len(classes)), 2))]
    if max_pairs: assert L.SYNTH, "--max-pairs is for synthetic tests only"; allp = allp[:max_pairs]
    part = L.SIM_DIR / f"Dall_{ID}_c1.partial.csv"; done = set(pd.read_csv(part).sample_id) if part.exists() else set(); buf = []; t0 = time.time()
    for k, u, v in allp:
        if k in done: continue
        p = dict(sample_id=k, class_u=u, class_v=v, quadrant="-", d_perc=float(D[ix[u], ix[v]]), d_sem=float(S.loc[u, v]))
        _, a = re8._run_pair(p, pool, h2c, cfg, cfg["phase3"]["contrast_base_seed"]); buf += a
        if len(buf) >= 250: pd.DataFrame(buf).to_csv(part, mode="a", header=not part.exists(), index=False); buf = []
    if buf: pd.DataFrame(buf).to_csv(part, mode="a", header=not part.exists(), index=False)
    secs = time.time() - t0; A = pd.read_csv(part).drop_duplicates("sample_id").sort_values("sample_id"); assert len(A) == len(allp)
    A.to_csv(L.SIM_DIR / f"Dall_{ID}_c1.csv", index=False); part.unlink()
    try: t, _ = model_ps(A, tag=f"Dall_{ID}"); res = dict(encoder=ID, test=t, status=t["status"])
    except Exception as e: res = dict(encoder=ID, status=f"tecnico: {e!r}")
    res.update(n_pairs=len(A), seconds=round(secs, 1)); L.write_json(L.RES_DIR / f"Dall_{ID}.json", res); return res
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", required=True, choices=["dispersion", "T", "D", "Dall"]); ap.add_argument("--encoder", required=True)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--pool", default="c10pool"); ap.add_argument("--max-pairs", type=int); a = ap.parse_args()
    L.ensure_dirs()
    if a.analysis == "dispersion": dispersion(L.P10 + L.EXTRA if a.encoder == "all" else a.encoder.split(","))
    elif a.analysis == "T": r = run_T(a.encoder, a.seed, a.pool); print(json.dumps({k: r[k] for k in ("c1_by_tier", "positive_control", "seconds")}, default=float))
    elif a.analysis == "D": r = run_D(a.encoder); print(a.encoder, "D", r.get("status"), r.get("seconds"))
    else: r = run_Dall(a.encoder, a.max_pairs); print(a.encoder, "Dall", r.get("status"), r.get("seconds"))
