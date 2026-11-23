"""Regions, multiplicity and global conclusion (plan §7.1-§7.4, §6.1 cuts; REVISION2 BC1/I-a; Patricio 10:37 and 10:44).
Implements the plan literally. There is NO option to choose a delta, a cut or an encoder: DELTAS and CUTS are constants and all are
always computed and reported together.
Inputs : --stage2 stage2_primary.csv (06, variant=primary, items 8 and 24, suffix "") ; --poscontrol poscontrol.csv (07 poscontrol)
         --grouping equivalence_grouping.json (02) ; --pipeline pipeline_control.json (written by h4_prep_encoders.py --check) ; --out-dir DIR
Outputs: regions_table.csv, global_grid.csv, floors.csv, regions_report.md (in --out-dir)."""
import argparse, json, pathlib, numpy as np, pandas as pd
DELTAS = (0.05, 0.10, 0.20)
CUTS = ("0.3", "0.5", "0.7")
R0 = "R0_dinov2s"
TRAINED = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae"]
FLOORS = ["F1_rn50rand0", "F2_pixels"]
PRIMARY = TRAINED + FLOORS
NI = "no interpretable (control positivo)"; R1_ = "signo opuesto"; R2_ = "menor que δ (equivalente con margen δ)"
R3_ = "se mantiene"; R4_ = "atenuado"; R5_ = "inconcluso"; R5T = "inconcluso (técnico)"; POS24 = "positivo"
EPS = 1e-12
def region8(row, dl, pc_pass, is_floor):
    """§7.1, precedence: [BC1 no interpretable] -> 1 -> 2 -> 3 -> 4 -> 5."""
    if not is_floor and not pc_pass: return NI
    if row is None or row.get("status") != "ok": return R5T
    if row["ci95_hi"] < 0: return R1_
    if -dl < row["ci90_lo"] and row["ci90_hi"] < dl: return R2_
    if row["ci95_lo"] > 0 and row["delta_hat"] >= dl - EPS: return R3_
    if row["ci95_lo"] > 0: return R4_
    return R5_
def region24(row, pc_pass, is_floor):
    if not is_floor and not pc_pass: return NI
    if row is None or row.get("status") != "ok": return R5T
    if row["ci95_hi"] < 0: return R1_
    if row["ci95_lo"] > 0: return POS24
    return R5_
def holm(p):
    p = np.asarray(p, float); m = len(p)
    if m == 0: return p
    o = np.argsort(p, kind="stable"); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o): run = max(run, min(1.0, (m - i) * p[k])); adj[k] = run
    return adj
def family_holm(rows, members, dl, pc):
    """§7.3: members that passed the control (floors: always) and have a stage-2 result; 2 tests each -> 2k tests."""
    fam = [e for e in members if (e in FLOORS or pc.get(e, False)) and rows.get(e) is not None and rows[e].get("status") == "ok"]
    p = [rows[e][f"p_tost_{dl:.2f}"] for e in fam] + [rows[e]["p_opposite"] for e in fam]; a = holm(p); k = len(fam)
    return {e: dict(holm_tost=a[i], holm_opposite=a[k + i]) for i, e in enumerate(fam)}, 2 * k
def profile(labels):
    """labels: dict delta -> region. 'robusto: X' or 'dependiente del margen: X (δ=..); ...'"""
    vals = [labels[d] for d in DELTAS]
    if len(set(vals)) == 1: return f"robusto: {vals[0]}"
    parts = []
    for v in dict.fromkeys(vals): parts.append(f"{v} (δ = {', '.join(f'{d:.2f}' for d in DELTAS if labels[d] == v)})")
    return "dependiente del margen: " + "; ".join(parts)
TECH0 = "sin lectura: técnico (R0 inconcluso (técnico))"
def r0_reading24(reg24):
    """REVISION3b: the 24-item reference reading of R0 (no G grid at 24 items, §7.1); a technical R0 gives 'sin lectura: técnico', never a result."""
    return TECH0 if reg24 == R5T else reg24
def conclude(dl, cut, regs, adj, pc, pipeline_ok, grouping):
    """§7.4 for one (delta, cut). Returns (code, detail)."""
    if not pipeline_ok: return "G0", "falla el control de tubería"
    if not pc.get(R0, False): return "G0", "R0 falla el control positivo (BC1): se detiene todo"
    if regs[R0][dl] == R5T: return "G0", TECH0   # REVISION3b (IMPORTANT): a technical failure of R0 is never read as G1
    if regs[R0][dl] != R3_: return "G1", f"R0 en '{regs[R0][dl]}'"
    g = grouping["cuts"][cut]; disc = g["calibration"]["discriminates"]
    fails = [e for e in TRAINED[1:] if pc.get(e, False) and e in adj and (
        (regs[e][dl] == R1_ and adj[e]["holm_opposite"] < .05) or (regs[e][dl] == R2_ and adj[e]["holm_tost"] < .05))]
    if fails:
        where = {e: g["labels"].get(e) for e in fails} if disc else "sin grupos (calibración falla en este corte)"
        return "G2", f"fallan {fails}; grupos: {where}"
    all_pass = all(pc.get(e, False) for e in TRAINED)
    if all_pass and all(regs[e][dl] == R3_ for e in TRAINED):
        if disc and g["n_trained_groups"] >= 2: return "G3", f"{g['n_trained_groups']} grupos de entrenados"
        return "G3b", ("calibración falla" if not disc else "un solo grupo de entrenados")
    return "G4", ("algún entrenado no interpretable (control positivo)" if not all_pass else "inconcluso o mixto")
def grid_profile(G):
    """Declared rule (plan §7.4, 10:44): G is a dict (delta, cut) -> code over the 3x3 grid.
    robusta: Gk                     -> identical in all 9 cells
    depende solo del margen         -> within each delta all cuts agree, deltas differ
    depende solo del corte          -> within each cut all deltas agree, cuts differ
    depende del margen y del corte  -> otherwise (every cell is listed)."""
    cells = [G[(d, c)] for d in DELTAS for c in CUTS]
    if len(set(cells)) == 1: return f"robusta: {cells[0]}"
    by_d = all(len({G[(d, c)] for c in CUTS}) == 1 for d in DELTAS); by_c = all(len({G[(d, c)] for d in DELTAS}) == 1 for c in CUTS)
    if by_d: return "depende solo del margen: " + "; ".join(f"δ = {d:.2f}: {G[(d, CUTS[0])]}" for d in DELTAS)
    if by_c: return "depende solo del corte: " + "; ".join(f"corte {c}: {G[(DELTAS[0], c)]}" for c in CUTS)
    return "depende del margen y del corte: " + "; ".join(f"(δ = {d:.2f}, corte {c}): {G[(d, c)]}" for d in DELTAS for c in CUTS)
def floors_reading(dl, regs, adjf):
    r = [regs[f][dl] for f in FLOORS]
    if all(x == R3_ for x in r): return "la moderación no requiere una geometría aprendida"
    if all(f in adjf and ((regs[f][dl] == R1_ and adjf[f]["holm_opposite"] < .05) or (regs[f][dl] == R2_ and adjf[f]["holm_tost"] < .05)) for f in FLOORS):
        return "la moderación requiere alguna geometría aprendida"
    return "inconcluso"
SENS_FIT = "sensible al tipo de ajuste"
def s9_labels(s9, pc):
    """REVISION3 I1 + Patricio 12:12 (S9, descriptive): regions with the MIXED stage 1 (per experiment, first converged rung); an encoder whose region changes at
    any delta is labelled 'sensible al tipo de ajuste'. Regions do not depend on the cut, so 'any delta/cut' = 'any delta'."""
    s8 = {r["encoder"]: r for r in s9[s9["items"] == 8].to_dict("records")}
    return {e: {dl: region8(s8.get(e), dl, pc.get(e, False), e in FLOORS) for dl in DELTAS} for e in PRIMARY}
def analyse(s2, pc, pipeline_ok, grouping, s9=None):
    s8 = {r["encoder"]: r for r in s2[s2["items"] == 8].to_dict("records")}; s24 = {r["encoder"]: r for r in s2[s2["items"] == 24].to_dict("records")}
    regs = {e: {dl: region8(s8.get(e), dl, pc.get(e, False), e in FLOORS) for dl in DELTAS} for e in PRIMARY}
    rows, G, floors = [], {}, []
    adjs = {dl: family_holm(s8, TRAINED[1:], dl, pc) for dl in DELTAS}; adjf = {dl: family_holm(s8, FLOORS, dl, pc) for dl in DELTAS}
    for e in PRIMARY:
        r = s8.get(e) or {}; row = dict(encoder=e, control_positivo=("no es puerta (piso)" if e in FLOORS else ("pasa" if pc.get(e, False) else "NO pasa")))
        for dl in DELTAS: row[f"region_d{dl:.2f}"] = regs[e][dl]
        row["region_24"] = region24(s24.get(e), pc.get(e, False), e in FLOORS); row["perfil"] = profile(regs[e])
        for k in ("status", "delta_hat", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi", "p_opposite", "p_tost_0.05", "p_tost_0.10", "p_tost_0.20", "welch_ci95_lo", "welch_ci95_hi"): row[k] = r.get(k)
        for dl in DELTAS:
            a = (adjf if e in FLOORS else adjs)[dl][0].get(e, {}) if e != R0 else {}
            row[f"holm_tost_d{dl:.2f}"] = a.get("holm_tost"); row[f"holm_opposite_d{dl:.2f}"] = a.get("holm_opposite")
        rows.append(row)
    for dl in DELTAS:
        for c in CUTS: G[(dl, c)] = conclude(dl, c, regs, adjs[dl][0], pc, pipeline_ok, grouping)
        floors.append(dict(delta=dl, family_size=adjf[dl][1], lectura=floors_reading(dl, regs, adjf[dl][0])))
    grid = pd.DataFrame([dict(delta=d, cut=c, conclusion=G[(d, c)][0], detalle=G[(d, c)][1], trained_family_size=adjs[d][1]) for d in DELTAS for c in CUTS])
    if s9 is not None:
        r9 = s9_labels(s9, pc)
        for row in rows:
            e = row["encoder"]
            for dl in DELTAS: row[f"region_S9_d{dl:.2f}"] = r9[e][dl]
            row["tipo_de_ajuste"] = SENS_FIT if any(r9[e][dl] != regs[e][dl] for dl in DELTAS) else "estable"
            if row["tipo_de_ajuste"] == SENS_FIT: row["perfil"] += f"; {SENS_FIT} (S9, descriptivo)"
    fam = family_summary(s8, pc)
    for row in rows: row["familia_holm"] = fam["membership"][row["encoder"]]
    out = (pd.DataFrame(rows), grid, pd.DataFrame(floors), grid_profile({k: v[0] for k, v in G.items()}))
    analyse.families = fam; analyse.r0_24 = r0_reading24(next(r["region_24"] for r in rows if r["encoder"] == R0)); return out
def family_summary(s8, pc):
    """REVISION3 (2)/(3): k and excluded encoders of each Holm family (membership does not depend on delta)."""
    mem, fams = {}, {}
    for name, members in (("entrenados", TRAINED[1:]), ("pisos", FLOORS)):
        inc, exc = [], []
        for e in members:
            if e not in FLOORS and not pc.get(e, False): mem[e] = f"excluido de {name}: {NI}"; exc.append(e)
            elif s8.get(e) is None or s8[e].get("status") != "ok": mem[e] = f"excluido de {name}: {R5T}"; exc.append(e)
            else: mem[e] = f"familia {name}"; inc.append(e)
        fams[name] = dict(k=len(inc), n_tests=2 * len(inc), incluidos=inc, excluidos=exc)
    mem[R0] = "R0: referencia (no entra en Holm; G1)"
    return dict(membership=mem, families=fams)
def md_table(df):
    h = "| " + " | ".join(map(str, df.columns)) + " |"; sep = "|" + "---|" * len(df.columns)
    return "\n".join([h, sep] + ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])
def load_pc(path):
    t = pd.read_csv(path); t = t[t.term == "zd"]
    return {r.encoder: bool(r.gate_pass) if pd.notna(r.gate_pass) else False for r in t.itertuples()}  # missing/failed fit = not passed
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); [ap.add_argument(a, required=True) for a in ("--stage2", "--poscontrol", "--grouping", "--pipeline", "--out-dir")]
    ap.add_argument("--stage2-s9", default=None, help="stage-2 file containing variant S9_mixed; adds the descriptive 'sensible al tipo de ajuste' label"); a = ap.parse_args()
    s2 = pd.read_csv(a.stage2); s2 = s2[(s2.variant == "primary") & (s2.suffix.fillna("") == "")]
    pipeline_ok = bool(json.loads(pathlib.Path(a.pipeline).read_text())["ok"])  # JSON written by h4_prep_encoders.py --check (REVISION3 minor)
    grouping = json.loads(pathlib.Path(a.grouping).read_text())["grouping"]
    s9 = None
    if a.stage2_s9: s9 = pd.read_csv(a.stage2_s9); s9 = s9[(s9.variant == "S9_mixed") & (s9.suffix.fillna("") == "")]
    pc = load_pc(a.poscontrol); tab, grid, fl, gp = analyse(s2, pc, pipeline_ok, grouping, s9)
    o = pathlib.Path(a.out_dir); o.mkdir(parents=True, exist_ok=True)
    tab.to_csv(o / "regions_table.csv", index=False); grid.to_csv(o / "global_grid.csv", index=False); fl.to_csv(o / "floors.csv", index=False)
    r0 = tab[tab.encoder == R0].iloc[0]
    md = ["# Regiones y conclusión global (plan §7; generado por 08_regions.py)", "",
          f"- Control de tubería: {'PASA' if pipeline_ok else 'FALLA'}. Perfil del agrupamiento de geometría: {grouping['profile']}.",
          f"- Referencia descriptiva, no confirmatoria (I-b): Δ en dos etapas de R0 con 8 ítems = {r0.delta_hat} [IC 95 % {r0.ci95_lo}; {r0.ci95_hi}].",
          f"- Lectura de R0 con 24 ítems (REVISION3b): {analyse.r0_24}.", "",
          "## Tabla única (§7.2): región con δ = 0,05 / 0,10 / 0,20 y con 24 ítems", "", md_table(tab[[c for c in ["encoder", "control_positivo", "familia_holm", "region_d0.05", "region_d0.10", "region_d0.20", "region_24", "tipo_de_ajuste", "perfil"] if c in tab.columns]]), "",
          "## Familias Holm (§7.3; iguales con los tres δ)", "",
          *[f"- **{n}:** k = {f['k']} encoders, {f['n_tests']} pruebas por δ; incluidos: {', '.join(f['incluidos']) or 'ninguno'}; excluidos: {', '.join(f['excluidos']) or 'ninguno'}." for n, f in analyse.families["families"].items()], "",
          "## Conclusión global por (δ, corte)", "", md_table(grid), "", f"**Perfil global:** {gp}", "", "## Pisos", "", md_table(fl)]
    (o / "regions_report.md").write_text("\n".join(md)); print("\n".join(md))
