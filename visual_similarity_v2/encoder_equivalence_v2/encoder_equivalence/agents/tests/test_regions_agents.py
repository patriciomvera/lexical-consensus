"""Unit tests of 14_agent_regions.py on HAND-WRITTEN synthetic result files (no data of any kind): region rules and their
precedence (0 técnico > 0b no interpretable > Apoyo > No apoyo > Inconcluso), Holm only for GA1, GA precedence
GA0 > GA0b > GA1 > GA2 > GA2b > GA3, CIFAR concordance ('sin grupos'), the 'sensible al muestreo de pares' label and the tables."""
import os, sys, json, shutil, pathlib, importlib, numpy as np, pandas as pd
T = pathlib.Path("/tmp/synth_agents_regions"); shutil.rmtree(T, ignore_errors=True); os.environ["AGENTS_SYNTH_DIR"] = str(T)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts")); import lib_agents as L
R14 = importlib.import_module("14_agent_regions")
ALL = L.P10 + L.EXTRA; IDS10 = L.P10
eq = R14.importlib.util.spec_from_file_location("eq02t", L.EE / "scripts/02_equivalence.py"); EQ = R14.importlib.util.module_from_spec(eq); a0 = sys.argv; sys.argv = [a0[0]]; eq.loader.exec_module(EQ); sys.argv = a0
def Rmat(groups):
    R = np.full((10, 10), 0.1)
    for i, x in enumerate(IDS10):
        for k, y in enumerate(IDS10): R[i, k] = 1 if i == k else (0.95 if any({x, y} <= g for g in groups) else 0.1)
    return R
TWO = [{"R0_dinov2s", "O1_dinov2b", "E2_vitsup", "E3_rn50sup"}, {"R1_cvcl", "E1_siglip", "O2_openclip", "E4_mae"}]
ONE = [set(L.TRAINED)]
def h1(rho=0.6, lo=0.5, hi=0.7, mp=1e-4, pb=None): return dict(n_pairs=100, rho=rho, ci_lo=lo, ci_hi=hi, mantel_p=mp, status="ok", p_below_sesoi=pb or {f"{s:.2f}": 0.9 for s in L.SESOIS})
def h2(beta=-0.05, p=0.001, r=-0.5, lo=-0.65, hi=-0.3, n=61, psem=0.5): return dict(n=n, beta=beta, p=p, r_partial=r, r_ci_lo=lo, r_ci_hi=hi, r2p=0.25, p_sem=psem, r2p_sem=0.01, n_others=1, status="ok")
def dt(beta=-0.2, p=1e-6, r=-0.5, lo=-0.6, hi=-0.35, n=100): return dict(n=n, beta_v1=beta, p_v1=p, beta_sem_v1=0.0, p_sem_v1=0.6, r2p_perc_v1=0.25, r2p_sem_v1=0.002, vif=1.0,
    beta_z=beta, p_hc3=p, ci_lo_z=-1, ci_hi_z=0, p_sem_hc3=0.6, r_partial=r, r_ci_lo=lo, r_ci_hi=hi, r2p_perc=0.25, r2p_sem=0.002, n_others=1, status="ok")
def write_base(groups_lf=TWO, groups_cifar=TWO, ctrl=True):
    shutil.rmtree(T, ignore_errors=True); [(T / d).mkdir(parents=True) for d in ("out/results", "out/sims", "human")]
    for e in ALL:
        floor = e in L.FLOORS or e.startswith("F1")
        L.write_json(L.RES_DIR / f"T_{e}.json", dict(status="ok", c1_by_tier={"centroid": dict(native=.9, near=.8, mid=.7, far=.6)}, positive_control=dict(pass_=True, ci_lo=.8, ci_hi=.95)))
        L.write_json(L.RES_DIR / f"D_{e}.json", dict(status="ok", test=dt(beta=0.0, p=0.9, r=0.0, lo=-0.05, hi=0.05, n=100) if floor else dt(), rule_centroid="A"))
        L.write_json(L.RES_DIR / f"Dall_{e}.json", dict(status="ok", test=dt(beta=0.0, p=0.9, r=0.0, lo=-0.03, hi=0.03, n=4950) if floor else dt(n=4950, lo=-0.55, hi=-0.45)))
        H = dict(H1={s: (h1(0.01, -0.03, 0.05, 0.6, {f"{x:.2f}": 1e-4 for x in L.SESOIS}) if floor else h1()) for s in L.H1_SOURCES},
                 H2={s: (h2(0.0, 0.9, 0.0, -0.08, 0.08) if floor else h2()) for s in L.H2_SOURCES})
        L.write_json(L.RES_DIR / f"H_{e}.json", H)
    pd.DataFrame([dict(encoder=e, RE="slope", logOR=-3.0, se=.03, lo=-3.06, hi=-2.94, p=1e-10, status="ok") for e in ALL]).to_csv(L.RES_DIR / "h4_by_encoder.csv", index=False)
    L.write_json(L.RES_DIR / "control_T_exact.json", dict(ok=ctrl)); L.write_json(L.RES_DIR / "control_c100_recode.json", dict(ok=True, verdict="PASA"))
    json.dump(dict(grouping=EQ.group_all(IDS10, Rmat(groups_lf))), open(L.H_GROUPING, "w"), default=bool)
    pd.DataFrame([dict(encoder=e, term="zd", estimate=-2.0 - i * .1, zd_lo=-2.2, zd_hi=-1.8, gate_pass=True) for i, e in enumerate(IDS10)]).to_csv(L.H_POSCONTROL, index=False)
    pd.DataFrame([dict(encoder=e, **{f"region_d{d}": "x" for d in ("0.05", "0.10", "0.20")}, region_24="x", perfil="x") for e in IDS10]).to_csv(L.H_REGIONS, index=False)
    # CIFAR distances whose between-encoder Spearman reproduces groups_cifar: encoders in a group share a base vector
    rng = np.random.default_rng(0); names = sorted(f"c{i:02d}" for i in range(100)); bases = [rng.random((100, 100)) for _ in groups_cifar]
    for e in IDS10:
        gi = [i for i, g in enumerate(groups_cifar) if e in g]; B = bases[gi[0]] if gi else rng.random((100, 100)); M = B + rng.normal(scale=.02 if gi else 0, size=B.shape); M = (M + M.T) / 2
        np.savez(L.SIM_DIR / f"dperc_c100_{e}.npz", D=M, names=np.array(names))
def edit(name, path, value):
    f = L.RES_DIR / name; d = json.loads(f.read_text()); cur = d
    for k in path[:-1]: cur = cur[k]
    cur[path[-1]] = value; f.write_text(json.dumps(d))
def run():
    importlib.reload(R14); s = R14.main(str(L.RES_DIR / "regions")); g = pd.read_csv(L.RES_DIR / "regions/ga_grid.csv")
    return s, g, pd.read_csv(L.RES_DIR / "regions/regions_agents.csv")
def ga(g, a, s=0.20, c="0.5"): return g[(g.analisis == a) & (np.isclose(g.sesoi, s)) & (g.corte.astype(str) == c)].conclusion.iloc[0]
def reg(t, e, a, src="-", s=0.20): return t[(t.encoder == e) & (t.analisis == a) & (t.fuente.astype(str) == src)][f"region_{s:.2f}"].iloc[0]
# 1. all trained in Apoyo, 2 concordant groups -> GA2 everywhere; floors No apoyo; profile robust
write_base(); s, g, t = run()
assert {ga(g, a) for a in "DAP"} == {"GA2"}, g; assert all(v.startswith("robusta: GA2") for v in s["ga_profile"].values()), s["ga_profile"]
assert reg(t, "F2_pixels", "A", "THINGS") == R14.NA_ and reg(t, "F2_pixels", "P", "HSJ") == R14.NA_ and reg(t, "F2_pixels", "D") == R14.NA_
assert s["families"]["C"]["n_tests"] == 36 and s["families"]["D"]["n_tests"] == 6 and s["families"]["C-pisos"]["n_tests"] == 12 and s["families"]["D-pisos"]["n_tests"] == 2
print("[14] all Apoyo + 2 concordant groups -> GA2 (robust); floors No apoyo; family sizes C=36, D=6, C-pisos=12, D-pisos=2")
# 2. CIFAR groups discordant -> GA2b; one L&F group -> GA2b; calibration fails -> GA2b
write_base(groups_cifar=ONE); s, g, _ = run(); assert ga(g, "A") == "GA2b" and "discrepan" in g[g.analisis == "A"].detalle.iloc[0]; print("[14] CIFAR discordant -> GA2b ('sin grupos')")
write_base(groups_lf=[set(L.TRAINED) | {"F1_rn50rand0"}], groups_cifar=[set(L.TRAINED) | {"F1_rn50rand0"}]); s, g, _ = run(); assert ga(g, "D") == "GA2b"; print("[14] calibration fails / one group -> GA2b")
# 3. one new trained in No apoyo with small p -> GA1 (A only); with Holm p >= .05 -> not GA1
write_base(); edit("H_E3_rn50sup.json", ["H1", "HSJ"], h1(0.02, -0.02, 0.06, 0.4, {f"{x:.2f}": 1e-5 for x in L.SESOIS})); s, g, t = run()
assert ga(g, "A") == "GA1" and ga(g, "D") == "GA2" and ga(g, "P") == "GA2" and reg(t, "E3_rn50sup", "A", "HSJ") == R14.NA_; print("[14] one trained No apoyo (Holm p < .05) -> GA1 in A only")
write_base(); edit("H_E3_rn50sup.json", ["H1", "HSJ"], h1(0.02, -0.02, 0.06, 0.4, {f"{x:.2f}": 0.01 for x in L.SESOIS})); s, g, t = run()
assert reg(t, "E3_rn50sup", "A", "HSJ") == R14.NA_ and ga(g, "A") == "GA3", g[g.analisis == "A"]; print("[14] No apoyo with raw p .01 but Holm(36) p >= .05 -> no GA1 (GA3)")
# 4. R0 not Apoyo in P -> GA0b in P; R0 technical in D -> GA0 in D; pipeline control fails -> GA0 everywhere
write_base(); edit("H_R0_dinov2s.json", ["H2", "HSJ"], h2(-0.02, 0.3, -0.2, -0.5, 0.15)); s, g, _ = run(); assert ga(g, "P") == "GA0b" and ga(g, "A") == "GA2"; print("[14] R0 Inconcluso in P -> GA0b (descriptive)")
write_base(); edit("D_R0_dinov2s.json", ["status"], "tecnico: singular"); s, g, t = run(); assert ga(g, "D") == "GA0" and reg(t, "R0_dinov2s", "D") == R14.TEC; print("[14] R0 technical in D -> GA0 'sin lectura'")
write_base(ctrl=False); s, g, _ = run(); assert set(g.conclusion) == {"GA0"}; print("[14] pipeline control fails -> GA0 in every cell")
# 5. GA0 > GA0b precedence: R0 technical AND not Apoyo -> GA0
write_base(); edit("H_R0_dinov2s.json", ["H2", "THINGS", "status"], "tecnico: sin pares"); edit("H_R0_dinov2s.json", ["H2", "HSJ"], h2(-0.02, 0.3, -0.2, -0.5, 0.15)); s, g, _ = run(); assert ga(g, "P") == "GA0"; print("[14] precedence GA0 > GA0b")
# 6. a trained encoder fails the agents' positive control -> 'no interpretable', out of Holm, GA3 (not GA2); floors are not gated
write_base(); edit("T_E4_mae.json", ["positive_control", "pass_"], False); edit("T_F2_pixels.json", ["positive_control", "pass_"], False); s, g, t = run()
assert reg(t, "E4_mae", "D") == R14.NI and "E4_mae" in s["families"]["C"]["excluidos"] and ga(g, "D") == "GA3" and reg(t, "F2_pixels", "D") == R14.NA_; print("[14] trained fails positive control -> NI, out of Holm, GA3; floor not gated")
# 7. technical new trained -> out of family, blocks GA2 -> GA3
write_base(); edit("H_O2_openclip.json", ["H1", "CIFAR10H", "status"], "tecnico: NaN"); s, g, t = run(); assert "O2_openclip" in s["families"]["C"]["excluidos"] and ga(g, "A") == "GA3" and ga(g, "P") == "GA2"; print("[14] technical trained -> excluded from family C; GA3 in A")
# 8. region rules: P/D No apoyo through the partial-r CI and through an opposite effect; Apoyo needs the beta_sem part; H4 regions
write_base(); edit("H_E1_siglip.json", ["H2", "THINGS"], h2(0.01, 0.6, 0.05, -0.15, 0.25)); edit("D_E2_vitsup.json", ["test", "beta_v1"], 0.1); edit("D_E2_vitsup.json", ["test", "p_v1"], 0.01)
edit("H_E1_siglip.json", ["H2", "HSJ"], dict(h2(), p_sem=0.01) | {"r2p_sem": 0.2}); s, g, t = run()
assert reg(t, "E1_siglip", "P", "THINGS", 0.20) == R14.NA_ and reg(t, "E1_siglip", "P", "THINGS", 0.10) == R14.INC and reg(t, "E2_vitsup", "D") == R14.NA_ and reg(t, "E1_siglip", "P", "HSJ") == R14.INC
print("[14] P No apoyo via CI (SESOI-dependent), D No apoyo via opposite effect, Apoyo requires the beta_sem condition")
tab = pd.read_csv(L.RES_DIR / "regions/regions_agents.csv"); pr = tab[(tab.encoder == "E1_siglip") & (tab.analisis == "P") & (tab.fuente == "THINGS")].perfil.iloc[0]; assert pr.startswith("dependiente del margen"), pr
write_base(); pd.DataFrame([dict(encoder=e, RE="slope", logOR=(0.5 if e == "E1_siglip" else (-0.01 if e == "E2_vitsup" else -3.0)), se=.03, lo=0, hi=0, p=(1e-5 if e != "E2_vitsup" else 0.7), status="ok") for e in ALL]).to_csv(L.RES_DIR / "h4_by_encoder.csv", index=False)
s, g, t = run(); assert reg(t, "E1_siglip", "H4") == R14.NA_ and reg(t, "E2_vitsup", "H4") == R14.INC and reg(t, "O1_dinov2b", "H4") == R14.AP; print("[14] H4: Apoyo / No apoyo (opposite, p<.05) / Inconcluso")
# 9. H1: Apoyo has precedence over the CI rule (significant tiny rho = Apoyo, ACLARACIONES b3)
write_base(); edit("H_O1_dinov2b.json", ["H1", "THINGS"], h1(0.05, 0.02, 0.08, 0.001)); s, g, t = run(); assert reg(t, "O1_dinov2b", "A", "THINGS") == R14.AP; print("[14] H1: rho = .05 with Mantel p < .05 -> Apoyo (first rule wins)")
# 10. sampling sensitivity label; agents x humans table; replica Holm of 6
write_base(); edit("Dall_O1_dinov2b.json", ["test", "beta_v1"], -0.001); edit("Dall_O1_dinov2b.json", ["test", "p_v1"], 0.5); edit("Dall_O1_dinov2b.json", ["test", "r_ci_lo"], -0.02); s, g, t = run()
ss = pd.read_csv(L.RES_DIR / "regions/sampling_sensitivity.csv").set_index("encoder"); assert ss.rotulo["O1_dinov2b"] == R14.SENS_PAIRS and ss.rotulo["E1_siglip"] == "estable"
assert ga(g, "D") == "GA2", "the label must not change GA"; tb = pd.read_csv(L.RES_DIR / "regions/tabla_agentes_personas.csv"); assert len(tb) == 13 and "H1_rho_THINGS" in tb
rp = pd.read_csv(L.RES_DIR / "regions/replica_holm6_v1.csv"); assert (rp.groupby("encoder").size() == 6).all(); assert s["spearman_descriptiva"]["beta_zd_humano_vs_H1_rho_THINGS"]["n"] == 10
print("[14] 'sensible al muestreo de pares' label (descriptive, GA unchanged); agents x humans table; replica Holm of 6")
# 11. b12 fix (REVISION5b): a missing H4 (14 h cap cut) or a technical H4 (failed fit) removes ONLY that H4 test from family C;
#     the encoder and its H1/H2 stay in C and can give GA1; other technical exclusions unchanged (scenario 7); all 6 H4 missing -> C = 30
write_base(); pd.DataFrame([dict(encoder=e, RE="slope", logOR=-3.0, se=.03, lo=-3.06, hi=-2.94, p=1e-10, status=("tecnico: no converge" if e == "E2_vitsup" else "ok")) for e in ALL if e != "E1_siglip"]).to_csv(L.RES_DIR / "h4_by_encoder.csv", index=False)
edit("H_E1_siglip.json", ["H1", "HSJ"], h1(0.02, -0.02, 0.06, 0.4, {f"{x:.2f}": 1e-5 for x in L.SESOIS})); s, g, t = run(); C = s["families"]["C"]
assert "E1_siglip" in C["incluidos"] and "E2_vitsup" in C["incluidos"] and not C["excluidos"] and C["n_tests"] == 34, C
assert sorted(C["h4_tecnico_fuera_solo_prueba"]) == ["E1_siglip", "E2_vitsup"] and reg(t, "E1_siglip", "H4") == R14.TEC and reg(t, "E2_vitsup", "H4") == R14.TEC
r = t[(t.encoder == "E1_siglip") & (t.analisis == "A") & (t.fuente.astype(str) == "HSJ")].iloc[0]; assert r["familia_0.20"] == "C" and r["p_holm_0.20"] < .05
assert pd.isna(t[(t.encoder == "E1_siglip") & (t.analisis == "H4")]["p_holm_0.20"].iloc[0]) and ga(g, "A") == "GA1" and "E1_siglip" in g[(g.analisis == "A")].detalle.iloc[0] and ga(g, "P") == "GA2"
assert s["families"]["C-pisos"]["n_tests"] == 12 and s["families"]["D"]["n_tests"] == 6
print("[14] b12 fix: H4 missing (cap) / technical (fit) -> only the H4 test leaves family C (34 tests); encoder + H1/H2 stay; GA1 from H1 still possible")
write_base(); pd.DataFrame([dict(encoder=e, RE="slope", logOR=-3.0, se=.03, lo=-3.06, hi=-2.94, p=1e-10, status="ok") for e in ALL if e not in L.NEW_TRAINED]).to_csv(L.RES_DIR / "h4_by_encoder.csv", index=False)
s, g, t = run(); C = s["families"]["C"]; assert C["n_tests"] == 30 and sorted(C["incluidos"]) == sorted(L.NEW_TRAINED) and ga(g, "A") == "GA2" and ga(g, "P") == "GA2", C
print("[14] b12 fix: all 6 new-trained H4 missing -> family C = 30 tests (H1 x 3 + H2 x 2 per encoder); A and P GA unchanged")
print("ALL 16 TESTS PASSED")
