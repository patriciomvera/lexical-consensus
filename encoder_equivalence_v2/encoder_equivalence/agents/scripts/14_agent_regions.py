"""AGENTS addendum, step 5: regions per encoder (no adjustment), families C and D (Holm only for GA1), floors (C-pisos, D-pisos),
GA0-GA3 with precedence GA0 > GA0b > GA1 > GA2 > GA2b > GA3, geometry concordance with CIFAR (§3.3), 'sensible al muestreo de
pares' label (§3.6), the agents x humans descriptive table and the per-encoder replica of the v1 Holm of 6 (proposal §3).
Implements the proposal literally; every point where the proposal does not fix a detail is a NAMED CONSTANT below and is listed
in agents/ACLARACIONES_scripts.md (section b). SESOIs and cuts are constants and are always all computed and reported together.
Reads agents/out/results/*.json + h4_by_encoder.csv + controls, the human-part outputs (poscontrol.csv, regions_table.csv,
equivalence_grouping.json) and writes agents/out/results/regions/. Nothing is interpreted here."""
import sys, json, argparse, pathlib, importlib.util, numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
from scipy import stats
TEC = "inconcluso (técnico)"; NI = "no interpretable (control positivo de agentes)"; AP = "Apoyo"; NA_ = "No apoyo"; INC = "Inconcluso"
SENS_PAIRS = "sensible al muestreo de pares"
AP_LOW = "Apoyo con IC bajo SESOI"   # REVISION5 b4: descriptive label; the region stays "Apoyo" (literal precedence)
def ap_low(t, sesoi):
    k = t.get("kind")
    if k == "A": return t["ci_hi"] < sesoi
    if k in ("P", "D"): return t["r_ci_lo"] > -sesoi
    return False
def region_hc3(t, sesoi, gate_ok, gated):   # REVISION5 b6: D-type region with the HC3 p, to report whether it changes the region
    global D_P
    old = D_P; D_P = "hc3"
    try: return region(t, sesoi, gate_ok, gated)
    finally: D_P = old
# ---- decisions the proposal does not fix (ACLARACIONES b): defaults, changeable only before the freeze
D_P = "v1"                 # b6: p of beta_perc in D / D-todos: "v1" = classical OLS of exp_008 regression_analysis (v1 Rule A); "hc3" = stats_lib
A_AGG = "todas"            # b8: an encoder is "Apoyo" in A (H1) only if Apoyo in the 3 sources; GA1 fires on any source in No apoyo (Holm p < .05)
GROUPS_OVER = NEW_TRAINED = L.NEW_TRAINED   # b10: ">= 2 grupos" counted among the 6 new trained encoders
CONCORD = "particion_identica"   # b9: L&F and CIFAR groups "coinciden" = identical partition of the 10 primaries at that cut
GATED = set(L.TRAINED + ["E4s_mae_cls"])    # §3.2 0b: positive control is a gate only for trained encoders (floors: reported)
ALL = L.P10 + L.EXTRA
def rd(name):
    p = L.RES_DIR / (name if name.endswith(".json") else name + ".json")   # T_<ID> -> T_<ID>.json
    return json.loads(p.read_text()) if p.exists() else None
def ok(x): return isinstance(x, dict) and x.get("status") == "ok"
def phi_sf(z): return float(stats.norm.sf(z))
# ---------------------------------------------------------------- units of test
def units(e):
    """dict (analysis, source) -> stats dict with 'kind' and 'status'."""
    H = rd(f"H_{e}") or {}; D = rd(f"D_{e}") or {}; DA = rd(f"Dall_{e}") or {}; u = {}
    for s in L.H1_SOURCES: u[("A", s)] = dict((H.get("H1") or {}).get(s) or {"status": "tecnico: falta H1"}, kind="A")
    for s in L.H2_SOURCES: u[("P", s)] = dict((H.get("H2") or {}).get(s) or {"status": "tecnico: falta H2"}, kind="P")
    u[("D", "-")] = dict(D.get("test") or {"status": D.get("status", "tecnico: falta D")}, kind="D")
    if D and not ok(D): u[("D", "-")]["status"] = D["status"]
    u[("Dall", "-")] = dict(DA.get("test") or {"status": DA.get("status", "tecnico: falta D-todos")}, kind="D")
    h4 = H4.get(e); u[("H4", "-")] = dict(h4 if h4 else {"status": "tecnico: falta H4"}, kind="H4")
    return u
def dstats(t):
    """beta / p used for D-type tests (D_P constant)."""
    if D_P == "v1": return t["beta_v1"], t["p_v1"], t["p_sem_v1"], t["r2p_perc_v1"], t["r2p_sem_v1"]
    return t["beta_z"], t["p_hc3"], t["p_sem_hc3"], t["r2p_perc"], t["r2p_sem"]
def region(t, sesoi, gate_ok, gated):
    if not ok(t): return TEC
    if gated and not gate_ok: return NI
    k = t["kind"]
    if k == "A":
        if t["rho"] > 0 and t["mantel_p"] < .05: return AP
        if t["ci_hi"] < sesoi: return NA_
        return INC
    if k in ("P", "D"):
        b, p, ps, r2, r2s = (t["beta"], t["p"], t["p_sem"], t["r2p"], t["r2p_sem"]) if k == "P" else dstats(t)
        if b < 0 and p < .05 and (ps >= .10 or r2s < r2 / 3): return AP
        if t["r_ci_lo"] > -sesoi or (b > 0 and p < .05): return NA_
        return INC
    if k == "H4":
        if t["logOR"] < 0 and t["p"] < .05: return AP
        if t["logOR"] > 0 and t["p"] < .05: return NA_   # b7: H4 has no SESOI; "No apoyo" = opposite sign with p < .05
        return INC
    raise ValueError(k)
def p_region(t, sesoi):
    """b5: p of the 'No apoyo' hypothesis (used only by Holm for GA1)."""
    if not ok(t): return np.nan
    k = t["kind"]
    if k == "A": return float(t["p_below_sesoi"][f"{sesoi:.2f}"])
    if k in ("P", "D"):
        b, p = (t["beta"], t["p"]) if k == "P" else dstats(t)[:2]
        se = 1 / np.sqrt(t["n"] - t.get("n_others", 1) - 3); peq = phi_sf((np.arctanh(t["r_partial"]) - np.arctanh(-sesoi)) / se)
        return float(min(peq, p if b > 0 else 1.0))
    if k == "H4": return float(t["p"]) if t["logOR"] > 0 else 1.0
def holm(p):
    sl = L.stats_lib(); p = np.asarray(p, float); return sl.holm(p) if len(p) else p
# ---------------------------------------------------------------- core
def build(controls_ok, control_detail, grouping, cifar):
    pc = {e: bool(((rd(f"T_{e}") or {}).get("positive_control") or {}).get("pass_", False)) for e in ALL}
    pc_known = {e: ok(rd(f"T_{e}") or {}) for e in ALL}
    U = {e: units(e) for e in ALL}
    for e in ALL:   # positive control not evaluable (T missing/technical) -> technical for a gated encoder (ACLARACIONES a)
        if e in GATED and not pc_known[e]:
            for k in U[e]:
                if ok(U[e][k]): U[e][k] = dict(U[e][k], status="tecnico: control positivo no evaluable (T)")
    R = {(e, k, s): region(U[e][k], s, pc[e], e in GATED) for e in ALL for k in U[e] for s in L.SESOIS}
    P = {(e, k, s): p_region(U[e][k], s) for e in ALL for k in U[e] for s in L.SESOIS}
    C_KEYS = [("A", s) for s in L.H1_SOURCES] + [("P", s) for s in L.H2_SOURCES] + [("H4", "-")]
    H4K = ("H4", "-")
    def members(cands, keys, gate=True, h4_test_only=False):
        # b12: an encoder leaves the family if technical (or NI) in ANY test of that family. REVISION5b (b12 fix): in family C a missing /
        # technical H4 (14 h cap cut or failed fit) removes ONLY that encoder's H4 test (C then has 30 tests); its H1/H2 stay and can give GA1.
        inc, exc = [], {}
        for e in cands:
            if gate and not pc[e]: exc[e] = NI if pc_known[e] else TEC
            elif any(R[(e, k, L.SESOIS[0])] == TEC for k in keys if not (h4_test_only and k == H4K)): exc[e] = TEC
            else: inc.append(e)
        return inc, exc
    fam = {"C": members(L.NEW_TRAINED, C_KEYS, h4_test_only=True) + (C_KEYS,), "D": members(L.NEW_TRAINED, [("D", "-")]) + ([("D", "-")],),
           "C-pisos": members(L.FLOORS, C_KEYS, gate=False) + (C_KEYS,), "D-pisos": members(L.FLOORS, [("D", "-")], gate=False) + ([("D", "-")],)}
    HP = {}
    for f, (inc, exc, keys) in fam.items():
        for s in L.SESOIS:
            tests = [(e, k) for e in inc for k in keys if not (f == "C" and k == H4K and R[(e, k, L.SESOIS[0])] == TEC)]; adj = holm([P[(e, k, s)] for e, k in tests])
            for (e, k), a in zip(tests, adj): HP[(e, k, s)] = (f, float(a), len(tests))
    # ---- GA per analysis (D, A, P) and descriptively per source
    def agg(e, keys, s):
        r = [R[(e, k, s)] for k in keys]
        if TEC in r: return TEC
        if NI in r: return NI
        if all(x == AP for x in r): return AP
        if any(x == NA_ for x in r): return NA_
        return INC
    def ga(keys, s, cut, famname):
        if not controls_ok: return "GA0", "sin lectura: falla (o falta) el control de tubería de agentes (§3.5): " + control_detail
        r0 = agg(L.R0, keys, s)
        if r0 == TEC: return "GA0", "sin lectura: R0 en 'inconcluso (técnico)'"
        if r0 != AP: return "GA0b", f"descriptivo, encoder por encoder: R0 en '{r0}'"
        g = grouping["cuts"][cut]; disc = g["calibration"]["discriminates"]; conc = cifar["concordance"].get(cut, False)
        fails = [e for e in L.NEW_TRAINED if any(R[(e, k, s)] == NA_ and (e, k, s) in HP and HP[(e, k, s)][1] < .05 for k in keys)]
        if fails:
            where = {e: g["labels"].get(e) for e in fails} if (disc and conc) else "sin grupos"
            return "GA1", f"depende del encoder: {fails}; grupos: {where}"
        allr = {e: agg(e, keys, s) for e in L.NEW_TRAINED}
        if all(pc[e] for e in L.NEW_TRAINED) and all(v == AP for v in allr.values()):
            ng = len({g["labels"].get(e) for e in GROUPS_OVER})
            if disc and conc and ng >= 2: return "GA2", f"no es específico de DINOv2: {ng} grupos (L&F, concuerdan con CIFAR)"
            why = "calibración L&F falla" if not disc else ("grupos L&F y CIFAR discrepan" if not conc else "un solo grupo")
            return "GA2b", f"se mantiene en todos, sin evidencia de diferencia geométrica ({why})"
        bad = {e: v for e, v in allr.items() if v != AP}
        return "GA3", f"inconcluso o mixto: {bad}"
    AN = {"D": [("D", "-")], "A": [("A", s) for s in L.H1_SOURCES], "P": [("P", s) for s in L.H2_SOURCES]}
    G = {(a, s, c): ga(keys, s, c, a) for a, keys in AN.items() for s in L.SESOIS for c in L.CUTS}
    GS = {(f"{a}_{k[1]}", s, c): ga([k], s, c, a) for a, keys in AN.items() if a != "D" for k in keys for s in L.SESOIS for c in L.CUTS}
    for f, v in list(fam.items()):   # actual tests per family (C: minus the technical H4 tests) + which H4 tests were dropped
        drop = [e for e in v[0] if f == "C" and R[(e, H4K, L.SESOIS[0])] == TEC]
        fam[f] = (v[0], v[1], v[2], len(v[0]) * len(v[2]) - len(drop), drop)
    return pc, pc_known, U, R, P, HP, fam, G, GS
def profile(labels):
    vals = [labels[s] for s in L.SESOIS]
    if len(set(vals)) == 1: return f"robusto: {vals[0]}"
    return "dependiente del margen: " + "; ".join(f"{v} (SESOI = {', '.join(f'{s:.2f}' for s in L.SESOIS if labels[s] == v)})" for v in dict.fromkeys(vals))
def grid_profile(G):   # same declared rule as 08_regions.grid_profile (plan §7.4), with SESOI in place of delta
    cells = [G[(s, c)] for s in L.SESOIS for c in L.CUTS]
    if len(set(cells)) == 1: return f"robusta: {cells[0]}"
    by_s = all(len({G[(s, c)] for c in L.CUTS}) == 1 for s in L.SESOIS); by_c = all(len({G[(s, c)] for s in L.SESOIS}) == 1 for c in L.CUTS)
    if by_s: return "depende solo del margen: " + "; ".join(f"SESOI = {s:.2f}: {G[(s, L.CUTS[0])]}" for s in L.SESOIS)
    if by_c: return "depende solo del corte: " + "; ".join(f"corte {c}: {G[(L.SESOIS[0], c)]}" for c in L.CUTS)
    return "depende del margen y del corte: " + "; ".join(f"(SESOI = {s:.2f}, corte {c}): {G[(s, c)]}" for s in L.SESOIS for c in L.CUTS)
def cifar_grouping(grouping):
    """§3.3: Spearman between encoders of the 4,950 CIFAR-100 distances, grouped with the SAME rule as plan §6.1 (02_equivalence.group_all)."""
    spec = importlib.util.spec_from_file_location("eq02_agents", L.EE / "scripts/02_equivalence.py"); eq = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [argv[0]]; spec.loader.exec_module(eq); sys.argv = argv
    ids = eq.GROUP_SET; vec = {}
    for e in ids:
        f = L.SIM_DIR / f"dperc_c100_{e}.npz"
        if not f.exists(): return dict(status=f"tecnico: falta {f.name}", concordance={c: False for c in L.CUTS})
        z = np.load(f); names = [str(n) for n in z["names"]]; o = np.argsort(names); Dm = z["D"][np.ix_(o, o)]; vec[e] = Dm[np.triu_indices(len(o), 1)]
    Rm = np.array([[stats.spearmanr(vec[a], vec[b]).statistic for b in ids] for a in ids]); g = eq.group_all(ids, Rm)
    conc = {}
    for c in L.CUTS:
        same = json.loads(json.dumps(g["cuts"][c]["partition"])) == json.loads(json.dumps(grouping["cuts"][c]["partition"])) if CONCORD == "particion_identica" else None
        conc[c] = bool(same)
    from sklearn.metrics import adjusted_rand_score   # REVISION5 b9: ARI L&F vs CIFAR per cut, descriptive only
    ari = {c: float(adjusted_rand_score([grouping["cuts"][c]["labels"][e] for e in ids], [g["cuts"][c]["labels"][e] for e in ids])) for c in L.CUTS}
    return dict(status="ok", ids=ids, rho=Rm.round(4).tolist(), grouping=g, concordance=conc, ari_descriptivo=ari, label="grupos medidos en estímulos de Lewis & Frank; ρ entre encoders en CIFAR-100 (4.950 pares) solo para concordancia")
def md_table(df):
    h = "| " + " | ".join(map(str, df.columns)) + " |"; sep = "|" + "---|" * len(df.columns)
    return "\n".join([h, sep] + ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |" for r in df.itertuples(index=False)])
H4 = {}
def main(out_dir):
    global H4
    o = pathlib.Path(out_dir); o.mkdir(parents=True, exist_ok=True)
    h4f = L.RES_DIR / "h4_by_encoder.csv"
    if h4f.exists():
        t = pd.read_csv(h4f); H4 = {r["encoder"]: dict(r, status=r["status"]) for r in t.to_dict("records")}
    cT, cC = rd("control_T_exact.json"), rd("control_c100_recode.json")
    controls_ok = bool(cT and cT.get("ok")) and bool(cC and cC.get("ok"))
    detail = f"T exacto: {(cT or {}).get('ok', 'falta')}; CIFAR-100 recodificado: {(cC or {}).get('verdict', 'falta')}"
    grouping = json.loads(L.H_GROUPING.read_text())["grouping"]; cifar = cifar_grouping(grouping)
    pc, pck, U, R, P, HP, fam, G, GS = build(controls_ok, detail, grouping, cifar)
    rows = []
    for e in ALL:
        for k, t in U[e].items():
            r = dict(encoder=e, analisis=k[0], fuente=k[1], control_positivo_agentes=("no es puerta" if e not in GATED else ("pasa" if pc[e] else ("NO pasa" if pck[e] else "no evaluable"))), status=t.get("status"))
            for s in L.SESOIS:
                r[f"region_{s:.2f}"] = R[(e, k, s)]; r[f"p_region_{s:.2f}"] = P[(e, k, s)]
                r[f"rotulo_{s:.2f}"] = AP_LOW if (R[(e, k, s)] == AP and ap_low(t, s)) else ""
                if t.get("kind") == "D": r[f"region_hc3_{s:.2f}"] = region_hc3(t, s, pc[e], e in GATED)
                h = HP.get((e, k, s)); r[f"familia_{s:.2f}"] = h[0] if h else ""; r[f"p_holm_{s:.2f}"] = h[1] if h else np.nan
            r["perfil"] = profile({s: R[(e, k, s)] for s in L.SESOIS})
            if t.get("kind") == "D": r["hc3_cambia_region"] = any(r[f"region_hc3_{s:.2f}"] != r[f"region_{s:.2f}"] for s in L.SESOIS)
            rows.append(r)
    tab = pd.DataFrame(rows); tab.to_csv(o / "regions_agents.csv", index=False)
    grid = pd.DataFrame([dict(analisis=a, sesoi=s, corte=c, conclusion=v[0], detalle=v[1]) for (a, s, c), v in G.items()]); grid.to_csv(o / "ga_grid.csv", index=False)
    pd.DataFrame([dict(analisis=a, sesoi=s, corte=c, conclusion=v[0], detalle=v[1]) for (a, s, c), v in GS.items()]).to_csv(o / "ga_grid_por_fuente_descriptivo.csv", index=False)
    prof = {a: grid_profile({(s, c): G[(a, s, c)][0] for s in L.SESOIS for c in L.CUTS}) for a in ("D", "A", "P")}
    sens = []
    for e in ALL:
        r1 = {s: R[(e, ("D", "-"), s)] for s in L.SESOIS}; r2 = {s: R[(e, ("Dall", "-"), s)] for s in L.SESOIS}
        sens.append(dict(encoder=e, **{f"D100_{s:.2f}": r1[s] for s in L.SESOIS}, **{f"Dtodos_{s:.2f}": r2[s] for s in L.SESOIS},
                         rotulo=(SENS_PAIRS if any(r1[s] != r2[s] for s in L.SESOIS) else "estable"), rule_centroid_D100=(rd(f"D_{e}") or {}).get("rule_centroid")))
    sens = pd.DataFrame(sens); sens.to_csv(o / "sampling_sensitivity.csv", index=False)
    fams = {f: dict(incluidos=v[0], excluidos=v[1], tests_per_encoder=[f"{a}:{b}" for a, b in v[2]], n_tests=v[3], h4_tecnico_fuera_solo_prueba=v[4]) for f, v in fam.items()}
    # ---- replica of the v1 Holm of 6 per encoder (descriptive)
    sl = L.stats_lib(); rep = []
    for e in ALL:
        u = U[e]; ps = [u[("A", s)].get("mantel_p") for s in L.H1_SOURCES] + [u[("P", s)].get("p") for s in L.H2_SOURCES] + [u[("H4", "-")].get("p")]
        names = [f"H1_{s}" for s in L.H1_SOURCES] + [f"H2_{s}" for s in L.H2_SOURCES] + ["H4"]
        if all(p is not None and np.isfinite(p) for p in ps):
            adj = sl.holm(np.array(ps, float)); rep += [dict(encoder=e, test=n, p=p, p_holm6=float(a)) for n, p, a in zip(names, ps, adj)]
        else: rep += [dict(encoder=e, test=n, p=p, p_holm6=np.nan, status="incompleto") for n, p in zip(names, ps)]
    pd.DataFrame(rep).to_csv(o / "replica_holm6_v1.csv", index=False)
    # ---- agents x humans (descriptive)
    pcH = pd.read_csv(L.H_POSCONTROL); pcH = pcH[pcH.term == "zd"].set_index("encoder"); rh = pd.read_csv(L.H_REGIONS).set_index("encoder"); tab2 = []
    for e in L.P10 + L.EXTRA:
        T = rd(f"T_{e}") or {}; D = rd(f"D_{e}") or {}; H = rd(f"H_{e}") or {}; c1 = ((T.get("c1_by_tier") or {}).get("centroid") or {})
        r = dict(encoder=e, humano_beta_zd_control_positivo=pcH.estimate.get(e, np.nan), humano_ic_lo=pcH.zd_lo.get(e, np.nan), humano_ic_hi=pcH.zd_hi.get(e, np.nan),
                 humano_control_positivo=pcH.gate_pass.get(e, np.nan), H4_logOR=(H4.get(e) or {}).get("logOR", np.nan), H4_lo=(H4.get(e) or {}).get("lo", np.nan), H4_hi=(H4.get(e) or {}).get("hi", np.nan), H4_RE=(H4.get(e) or {}).get("RE", ""))
        for c in ("region_d0.05", "region_d0.10", "region_d0.20", "region_24", "perfil"): r[f"humano_{c}"] = rh[c].get(e, "") if c in rh else ""
        for t in ("native", "near", "mid", "far"): r[f"C1_{t}"] = c1.get(t, np.nan)
        dt = D.get("test") or {}; r["D_beta_perc_v1escala"] = dt.get("beta_v1", np.nan); r["D_beta_perc_z"] = dt.get("beta_z", np.nan); r["D_regla_v1"] = D.get("rule_centroid", "")
        for s in L.H1_SOURCES: r[f"H1_rho_{s}"] = ((H.get("H1") or {}).get(s) or {}).get("rho", np.nan)
        for s in L.H2_SOURCES: r[f"H2_beta_humana_{s}"] = ((H.get("H2") or {}).get(s) or {}).get("beta", np.nan)
        tab2.append(r)
    tab2 = pd.DataFrame(tab2); tab2.to_csv(o / "tabla_agentes_personas.csv", index=False)
    t10 = tab2[tab2.encoder.isin(L.P10)]; sp = {"_nota": "Spearman entre encoders (los 10 primarios), solo descriptiva: sin p ni IC inferencial"}
    for c in [f"H1_rho_{s}" for s in L.H1_SOURCES] + [f"H2_beta_humana_{s}" for s in L.H2_SOURCES]:
        d = t10[["humano_beta_zd_control_positivo", c]].dropna(); sp[f"beta_zd_humano_vs_{c}"] = dict(n=len(d), rho=float(stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1]).statistic) if len(d) >= 3 else None)
    disp = rd("agent_pairs_dispersion.json")
    summary = dict(controls=dict(ok=controls_ok, control_T=cT, control_c100=cC), families=fams, ga_profile=prof, cifar_geometry=cifar,
                   spearman_descriptiva=sp, constants=dict(D_P=D_P, A_AGG=A_AGG, CONCORD=CONCORD, GROUPS_OVER=GROUPS_OVER, SESOIS=L.SESOIS, CUTS=L.CUTS))
    L.write_json(o / "agents_summary.json", summary)
    md = ["# Agentes: regiones y conclusiones GA (generado por 14_agent_regions.py; sin interpretación)", "",
          *([f"> **GA0 en todos los análisis: falla (o falta) el control de tubería de agentes (§3.5): {detail}. Nada de lo que sigue se lee.**", ""] if not controls_ok else []),
          "- §3.7 (circularidad): frente a las personas solo se leen H1 y H2; un GA2 en D significa \"suficiencia del mecanismo\" (el mecanismo basta con otros encoders), no correspondencia con las personas.",
          f"- Control de tubería de agentes (§3.5): {'PASA' if controls_ok else 'FALLA o falta'} ({detail}).",
          f"- Márgenes de agentes (SESOI, escala de correlación; NO comparables con los δ humanos): {', '.join(f'{s:.2f}' for s in L.SESOIS)}.",
          f"- Geometría: grupos medidos en estímulos de Lewis & Frank; concordancia con CIFAR por corte: {cifar.get('concordance')}.", "",
          "## Familias Holm (solo para GA1)", "", *[f"- **{f}:** incluidos {v['incluidos']}; excluidos {v['excluidos']}; {v['n_tests']} pruebas por SESOI" + (f"; H4 técnico (sale solo la prueba H4, b12/REVISION5b): {v['h4_tecnico_fuera_solo_prueba']}" if v['h4_tecnico_fuera_solo_prueba'] else "") + "." for f, v in fams.items()], "",
          "## Regiones por encoder (sin ajuste)", "", md_table(tab[["encoder", "analisis", "fuente", "control_positivo_agentes", *[f"region_{s:.2f}" for s in L.SESOIS], "perfil"]]), "",
          "## GA por análisis (SESOI × corte)", "", md_table(grid), "", *[f"- **Perfil {a}:** {p}" for a, p in prof.items()], "",
          "## D con 100 pares vs D-todos (rótulo descriptivo)", "", "Nota: los p de D-todos son anticonservadores (los 4.950 pares comparten clases), así que el rótulo puede dispararse por eso; es solo descriptivo y no cambia GA.", "", md_table(sens), "",
          "## H4: estructura de efectos aleatorios usada por encoder (caída a `noslope` por encoder, como 04c)", "", md_table(pd.DataFrame([dict(encoder=e, RE=(H4.get(e) or {}).get("RE", ""), status=(H4.get(e) or {}).get("status", "falta")) for e in ALL])), "",
          f"- Rótulo descriptivo «{AP_LOW}» (b4) y región con HC3 en D (b6): columnas `rotulo_*`, `region_hc3_*` y `hc3_cambia_region` de regions_agents.csv. ARI L&F vs CIFAR por corte (b9, descriptivo): {cifar.get('ari_descriptivo')}.", "",
          "## Tabla agentes × personas (descriptiva)", "", md_table(tab2), "", "Spearman entre encoders (descriptiva, sin p): " + json.dumps(sp, ensure_ascii=False), "",
          "## Dispersión de d_perc en los 100 pares (§3.6)", "", "```", json.dumps(disp, indent=1, ensure_ascii=False) if disp else "falta", "```"]
    (o / "agents_regions_report.md").write_text("\n".join(md)); print(f"wrote {o}")
    return summary
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=str(L.RES_DIR / "regions")); a = ap.parse_args(); main(a.out_dir)
