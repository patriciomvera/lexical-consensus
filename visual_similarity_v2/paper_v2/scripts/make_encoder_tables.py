"""BORRADOR (prototipo para el plan de la v2). Genera, solo desde archivos congelados de main,
las tablas por encoder (adultos, agentes CIFAR, comparación exploratoria) como .tex y un archivo
de macros con fuente única de verdad, y corre chequeos cruzados entre las tres partes.
Uso: python make_encoder_tables.py <checkout del repo en a945ed2>
No refit de modelos: solo lee salidas. Cada número tiene su archivo de origen en generated/fuentes.csv.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(sys.argv[1]); V2 = REPO / "visual_similarity_v2"
E = V2 / "encoder_equivalence_v2/encoder_equivalence"; A = E / "agents/out/results"
X = V2 / "agents_xtmem_v2/outputs"
OUT = Path(__file__).resolve().parents[1] / "generated"; OUT.mkdir(exist_ok=True)
ENC = ["R0_dinov2s","O1_dinov2b","R1_cvcl","E1_siglip","O2_openclip","E2_vitsup","E3_rn50sup","E4_mae","F1_rn50rand0","F2_pixels"]
VAR = ["E4s_mae_cls","F1_rn50rand1","F1_rn50rand2"]
NAME = {"R0_dinov2s":"DINOv2-small","O1_dinov2b":"DINOv2-base","R1_cvcl":"CVCL","E1_siglip":"SigLIP","O2_openclip":"OpenCLIP",
        "E2_vitsup":"Supervised ViT","E3_rn50sup":"Supervised ResNet-50","E4_mae":"MAE","F1_rn50rand0":"Random ResNet-50 (s0)",
        "F2_pixels":"Pixels","E4s_mae_cls":"MAE (CLS)","F1_rn50rand1":"Random ResNet-50 (s1)","F1_rn50rand2":"Random ResNet-50 (s2)"}
# rol en cada familia según los planes congelados
ROLE_H = {e: ("control (piso)" if e[:2] in ("F1","F2") else ("referencia" if e=="R0_dinov2s" else "entrenado")) for e in ENC}
ROLE_A = {e: ("control (piso)" if e[:2] in ("F1","F2") else ("descriptivo (v1)" if e in ("R0_dinov2s","R1_cvcl") else "confirmatorio")) for e in ENC}
src, macros, checks = [], {}, []
def M(key, val, f, d=2, lead=True):
    if isinstance(val,(float,np.floating)) and d==2 and 0 < abs(val) < 0.005: d = 3
    s = "--" if val is None or (isinstance(val,float) and np.isnan(val)) else (f"{val:.{d}f}" if isinstance(val,(float,np.floating)) else str(val))
    if not lead and s.startswith(("0.","-0.")): s = s.replace("0.", ".", 1)
    s = s.replace("-", "$-$", 1) if s.startswith("-") else s
    macros[key] = s; src.append((key, s, str(f.relative_to(REPO)))); return s
def chk(name, ok, detail=""): checks.append((name, bool(ok), detail))
ab = {"Apoyo":"S","No apoyo":"N","Inconcluso":"I","inconcluso":"I","inconcluso (técnico)":"T","menor que δ (equivalente con margen δ)":"E"}

# ---- adultos ----
fpc = E/"results/poscontrol.csv"; pc = pd.read_csv(fpc).query("term=='zd'").set_index("encoder")
fh4 = A/"h4_by_encoder.csv"; h4 = pd.read_csv(fh4).set_index("encoder")
frt = E/"results/regions/regions_table.csv"; rt = pd.read_csv(frt).set_index("encoder")
for e in ENC:
    r = pc.loc[e]; M(f"pcB@{e}", r.estimate, fpc); M(f"pcLo@{e}", r.zd_lo, fpc); M(f"pcHi@{e}", r.zd_hi, fpc)
    M(f"pcRE@{e}", "slope" if r.model=="poscontrol_slope" else "intercept-only", fpc)
    M(f"h4B@{e}", h4.loc[e,"logOR"], fh4); M(f"h4Lo@{e}", h4.loc[e,"lo"], fh4); M(f"h4Hi@{e}", h4.loc[e,"hi"], fh4)
    q = rt.loc[e]; M(f"ordD@{e}", q.delta_hat, frt, 3); M(f"ordLo@{e}", q.ci95_lo, frt, 3); M(f"ordHi@{e}", q.ci95_hi, frt, 3)
    M(f"ordReg@{e}", "/".join(ab.get(q[c], "?") for c in ["region_d0.05","region_d0.10","region_d0.20"]), frt)
    chk(f"h4 nobs/persons {e}", h4.loc[e,"nobs"]==172800 and h4.loc[e,"persons"]==520)
    chk(f"poscontrol nobs {e}", pc.loc[e,"nobs"]==57600)
# coherencia: exhibits/distance_effects.csv y tabla_agentes_personas.csv frente a las fuentes
de = pd.read_csv(V2/"exhibits/distance_effects.csv").set_index("encoder")
tp = pd.read_csv(A/"regions/tabla_agentes_personas.csv").set_index("encoder")
for e in ENC:
    chk(f"distance_effects adult = poscontrol {e}", abs(de.loc[e,"adult_beta"]-pc.loc[e,"estimate"])<1e-12)
    chk(f"tabla_agentes_personas poscontrol {e}", abs(tp.loc[e,"humano_beta_zd_control_positivo"]-pc.loc[e,"estimate"])<1e-12)
    chk(f"tabla_agentes_personas H4 {e}", abs(tp.loc[e,"H4_logOR"]-h4.loc[e,"logOR"])<1e-12)

# ---- agentes CIFAR ----
fr = A/"regions/regions_agents.csv"; reg = pd.read_csv(fr)
def regs(e, an, fu="-"):
    r = reg[(reg.encoder==e)&(reg.analisis==an)&(reg.fuente.astype(str)==fu)].iloc[0]
    return "/".join(ab.get(r[f"region_{s}"], "?") for s in ["0.10","0.20","0.30"])
for e in ENC:
    fT = A/f"T_{e}.json"; t = json.load(open(fT))
    for tier in ["native","far"]:
        M(f"c1{tier}@{e}", 100*t["c1_by_tier"]["centroid"][tier], fT, 1)
        lo, hi = t["c1_ci"]["centroid"][tier]; M(f"c1{tier}Lo@{e}", 100*lo, fT, 1); M(f"c1{tier}Hi@{e}", 100*hi, fT, 1)
    fD = A/f"D_{e}.json"; d = json.load(open(fD))["learners"]["centroid"]["model_ps"]
    M(f"dR@{e}", d["r_partial"], fD, 3 if abs(d["r_partial"]) < 0.05 else 2, False); M(f"dLo@{e}", d["r_ci_lo"], fD, 2, False); M(f"dHi@{e}", d["r_ci_hi"], fD, 2, False)
    chk(f"D n=100 {e}", d["n"]==100); chk(f"distance_effects agent = D {e}", abs(de.loc[e,"agent_r"]-d["r_partial"])<1e-12)
    M(f"dReg@{e}", regs(e,"D"), fr)
    fDa = A/f"Dall_{e}.json"; da = json.load(open(fDa))["test"]
    M(f"daR@{e}", da["r_partial"], fDa, 2, False); M(f"daReg@{e}", regs(e,"Dall"), fr); chk(f"Dall n=4950 {e}", da["n"]==4950)
    fH = A/f"H_{e}.json"; h = json.load(open(fH))
    for s, k in [("THINGS","t"),("HSJ","h"),("CIFAR10H","c")]:
        M(f"h1{k}@{e}", h["H1"][s]["rho"], fH, 2, False); M(f"h1{k}P@{e}", h["H1"][s]["mantel_p"], fH, 4, False)
        M(f"h1{k}Reg@{e}", regs(e,"A",s), fr)
    for s, k in [("THINGS","t"),("HSJ","h")]: M(f"h2{k}Reg@{e}", regs(e,"P",s), fr)
    chk(f"H1 n_pairs {e}", (h["H1"]["THINGS"]["n_pairs"], h["H1"]["HSJ"]["n_pairs"], h["H1"]["CIFAR10H"]["n_pairs"])==(3400,1390,45))

# ---- exploratorio (mismas imágenes) ----
fx = X/"one_word_items.csv"; it = pd.read_csv(fx)
dom = {"animals":"c3","vegetables":"c1","vehicles":"c2"}
c = it[(it.learner=="centroid") & it.condition.isin(["3sub","3bas","3sup"])].copy()
c["own"] = c.image.str[:2]==c.domain.map(dom); c = c[~c.own | c.image.str.contains("_sup")]
c["set"] = np.where(c.own, "in", "out")
ad = c[c.encoder=="R0_dinov2s"].groupby(["set","condition"])[["human_yes","human_n"]].sum()
for (s, cnd), r in ad.iterrows():
    M(f"xAdK@{s}{cnd}", int(r.human_yes), fx); M(f"xAdN@{s}{cnd}", int(r.human_n), fx); M(f"xAdPct@{s}{cnd}", 100*r.human_yes/r.human_n, fx, 2)
ag = c.groupby(["encoder","set","condition"]).accepted.agg(["sum","size"])
for (e, s, cnd), r in ag.iterrows():
    M(f"xK@{e}{s}{cnd}", int(r["sum"]), fx); M(f"xN@{e}{s}{cnd}", int(r["size"]), fx)
# chequeo cruzado: recuentos humanos del exploratorio = datos por ensayo de la parte humana congelada
fl = E/"data/h4_long_R0_dinov2s.csv"; hl = pd.read_csv(fl)
lab = {"three_subordinate":"3sub","three_basic":"3bas","three_superordinate":"3sup"}
hl["cond"] = hl.condition.replace(lab); hl = hl[hl.cond.isin(["3sub","3bas","3sup"])]
hl["set"] = np.where(hl.in_target==1, "in", "out"); hl = hl[(hl.set=="out") | (hl.level=="sup")]
hh = hl.groupby(["set","cond"]).selected.agg(["sum","size"])
for (s, cnd), r in hh.iterrows():
    chk(f"humanos exploratorio = h4_long ({s},{cnd})", r["sum"]==ad.loc[(s,cnd),"human_yes"] and r["size"]==ad.loc[(s,cnd),"human_n"], f"{r['sum']}/{r['size']}")
chk("h4_long usa dos grafías de condición (3sup y three_superordinate)", set(lab) <= set(pd.read_csv(fl, usecols=["condition"]).condition.unique()))
tc = pd.read_csv(V2/"exhibits/teaching_conditions.csv")
for _, r in tc.iterrows():
    s = "in" if r.candidate_set=="within" else "out"
    if r.encoder=="adult": ok = r.yes==ad.loc[(s,r.condition),"human_yes"] and r.total==ad.loc[(s,r.condition),"human_n"]
    else: ok = r.yes==ag.loc[(r.encoder,s,r.condition),"sum"] and r.total==ag.loc[(r.encoder,s,r.condition),"size"]
    chk(f"teaching_conditions.csv {r.encoder} {s} {r.condition}", ok)

# ---- huecos por encoder ----
gaps = []
for e in ENC+VAR:
    gaps.append(dict(encoder=e, adultos_pc=e in pc.index, adultos_primario=e in rt.index,
                     adultos_sens=e in set(pd.read_csv(E/"results/stage2_sensitivities.csv").encoder),
                     agentes_cifar=(A/f"D_{e}.json").exists(), exploratorio=e in set(it.encoder)))
pd.DataFrame(gaps).to_csv(OUT/"huecos_por_encoder.csv", index=False)

# ---- salidas ----
with open(OUT/"numbers_v2.tex","w") as f:
    f.write("% AUTO-GENERATED by scripts/make_encoder_tables.py (BORRADOR) -- do not edit\n")
    f.write("\\makeatletter\\newcommand{\\vnum}[1]{\\@ifundefined{vnum@#1}{\\textbf{??#1}}{\\csname vnum@#1\\endcsname}}\\makeatother\n")
    for k, v in sorted(macros.items()): f.write(f"\\expandafter\\def\\csname vnum@{k}\\endcsname{{{v}}}\n")
pd.DataFrame(src, columns=["macro","valor","archivo"]).to_csv(OUT/"fuentes.csv", index=False)
V = macros
import subprocess as _sp
_head = _sp.run(["git","-C",str(REPO),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
assert _head.startswith("a945ed2"), f"repo must be checked out at a945ed2, got {_head}"
NAMEP = {**NAME, "F1_rn50rand0":"Random ResNet-50 (s0)"}
RH = {e: ("Control" if e[:2] in ("F1","F2") else ("Reference" if e=="R0_dinov2s" else "Trained")) for e in ENC}
RA = {e: ("Control" if e[:2] in ("F1","F2") else ("Descriptive" if e in ("R0_dinov2s","R1_cvcl") else "Confirmatory")) for e in ENC}
def wrap(cols, head, rows, foot=None):
    out = [r"% AUTO-GENERATED by scripts/make_encoder_tables.py from frozen files @ a945ed2 -- do not edit",
           r"\setlength{\tabcolsep}{4pt}", r"\noindent\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + cols + "}", r"\toprule"] + head + [r"\midrule"]
    for k, r in enumerate(rows):
        if r == "MID": out.append(r"\midrule"); continue
        out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(out) + "\n"
def ci(b, lo, hi): return f"{V[b]} [{V[lo]}, {V[hi]}]"
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RH[e], ci(f"pcB@{e}", f"pcLo@{e}", f"pcHi@{e}") + (r"$^{a}$" if V[f"pcRE@{e}"]!="slope" else ""),
                 ci(f"h4B@{e}", f"h4Lo@{e}", f"h4Hi@{e}"), ci(f"ordD@{e}", f"ordLo@{e}", f"ordHi@{e}"), V[f"ordReg@{e}"]])
(OUT/"tab1_adults.tex").write_text(wrap("llcccc",
    [r"Encoder & Role & Positive control, 8 items & H4 replication, 24 items & Order moderation (primary) & Region \\",
     r" & & $\beta_{z(d)}$ [95\% CI] & $\beta_{z(d)}$ [95\% CI] & $\hat\Delta$ [95\% CI] & $\delta = .05/.10/.20$ \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e], f"{V[f'c1native@{e}']} [{V[f'c1nativeLo@{e}']}, {V[f'c1nativeHi@{e}']}]",
                 f"{V[f'c1far@{e}']} [{V[f'c1farLo@{e}']}, {V[f'c1farHi@{e}']}]",
                 ci(f"dR@{e}", f"dLo@{e}", f"dHi@{e}"), V[f"dReg@{e}"], V[f"daR@{e}"], V[f"daReg@{e}"]])
(OUT/"tab2a_agents_distance.tex").write_text(wrap("llcccccc",
    [r"Encoder & Role & C1 native (\%) & C1 far (\%) & D: partial $r$ [95\% CI] & Region D & D-all: partial $r$ & Region D-all \\",
     r" & & [95\% CI] & [95\% CI] & $n = 100$ pairs & SESOI $.10/.20/.30$ & $n = 4{,}950$ pairs & SESOI $.10/.20/.30$ \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e]] + [f"{V[f'h1{k}@{e}']} ({V[f'h1{k}Reg@{e}']})" for k in "thc"] + [(r"$<$.001" if V[f'h1cP@{e}'].startswith(".000") else V[f'h1cP@{e}']), V[f"h2tReg@{e}"], V[f"h2hReg@{e}"], r"n/a$^{b}$"])
(OUT/"tab2b_agents_human.tex").write_text(wrap("llccccccc",
    [r" & & \multicolumn{4}{c}{H1: Spearman $\rho$ (region)} & \multicolumn{3}{c}{H2: region} \\ \cmidrule(lr){3-6}\cmidrule(lr){7-9}",
     r"Encoder & Role & THINGS & HSJ & CIFAR-10H & CIFAR-10H Mantel $p$ & THINGS & HSJ & CIFAR-10H \\"], rows))
rows = [["Adults (selections / judgments)"] + [f"{V[f'xAdK@{s}{c}']}/{V[f'xAdN@{s}{c}']}" for s in ["in","out"] for c in ["3sub","3bas","3sup"]], "MID"]
for e in ENC + VAR:
    if e in ("F1_rn50rand0","E4s_mae_cls"): rows.append("MID")
    rows.append([NAMEP[e]] + [f"{V[f'xK@{e}{s}{c}']}/{V[f'xN@{e}{s}{c}']}" for s in ["in","out"] for c in ["3sub","3bas","3sup"]])
(OUT/"tab3_matched.tex").write_text(wrap("lcccccc",
    [r" & \multicolumn{3}{c}{Within-domain superordinates} & \multicolumn{3}{c}{Other-domain images} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Learner & Same subtype & Same basic type & Broad & Same subtype & Same basic type & Broad \\"], rows))

# ---- compact main-text tables (v2b): values only; dagger = not "support" at all three SESOI ----
DAG = r"$^{\dagger}$"
def flag(reg): return "" if reg == "S/S/S" else DAG
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e], V[f'c1native@{e}'], V[f'c1far@{e}'],
                 ci(f"dR@{e}", f"dLo@{e}", f"dHi@{e}") + flag(V[f"dReg@{e}"]), V[f"daR@{e}"] + flag(V[f"daReg@{e}"])])
(OUT/"tab_agents_compact.tex").write_text(wrap("llcccc",
    [r"Encoder & Role & Native (\%) & Far (\%) & D: partial $r$ [95\% CI] & D-all: partial $r$ \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e]] + [V[f'h1{k}@{e}'] + flag(V[f'h1{k}Reg@{e}']) for k in "thc"])
(OUT/"tab_h1_compact.tex").write_text(wrap("llccc",
    [r"Encoder & Role & THINGS & ImageNet-HSJ & CIFAR-10H \\"], rows))
for e in ENC:
    chk(f"compact D flag {e}", (flag(V[f"dReg@{e}"]) == "") == (V[f"dReg@{e}"] == "S/S/S"))
# ---- v2b table layout: estimate and interval in separate columns; interval kept in one \mbox; minus as \ensuremath{-} ----
def T(x): return x.replace("$-$", r"\ensuremath{-}")
def EC(b, lo, hi, sup=""): return [T(V[b]), r"\mbox{[" + T(V[lo]) + r",~" + T(V[hi]) + "]}" + sup]
def wrap2(cols, head, rows):
    out = [r"% AUTO-GENERATED by scripts/make_encoder_tables.py from frozen files @ a945ed2 -- do not edit (v2b layout)",
           r"\setlength{\tabcolsep}{3.5pt}", r"\noindent\adjustbox{max width=\linewidth}{%", r"\begin{tabular}{" + cols + "}", r"\toprule"] + head + [r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(r) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}}"]) + "\n"
EST = r"\multicolumn{1}{c}{Est.} & \multicolumn{1}{c}{95\% CI}"
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RH[e]] + EC(f"pcB@{e}", f"pcLo@{e}", f"pcHi@{e}", r"$^{a}$" if V[f"pcRE@{e}"] != "slope" else "")
                + EC(f"h4B@{e}", f"h4Lo@{e}", f"h4Hi@{e}") + EC(f"ordD@{e}", f"ordLo@{e}", f"ordHi@{e}") + [V[f"ordReg@{e}"]])
(OUT/"tab1_adults_v2b.tex").write_text(wrap2("llrlrlrlc",
    [r" & & \multicolumn{2}{c}{Positive control, 8 items} & \multicolumn{2}{c}{H4 replication, 24 items} & \multicolumn{2}{c}{Order moderation (primary)} & Region \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Encoder & Role & \multicolumn{2}{c}{$\beta_{z(d)}$} & \multicolumn{2}{c}{$\beta_{z(d)}$} & \multicolumn{2}{c}{$\hat\Delta$} & $\delta = .05/.10/.20$ \\",
     r" & & " + EST + " & " + EST + " & " + EST + r" & \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e]] + EC(f"c1native@{e}", f"c1nativeLo@{e}", f"c1nativeHi@{e}") + EC(f"c1far@{e}", f"c1farLo@{e}", f"c1farHi@{e}")
                + EC(f"dR@{e}", f"dLo@{e}", f"dHi@{e}") + [V[f"dReg@{e}"], T(V[f"daR@{e}"]), V[f"daReg@{e}"]])
(OUT/"tab2a_agents_distance_v2b.tex").write_text(wrap2("llrlrlrlcrc",
    [r" & & \multicolumn{2}{c}{C1 native (\%)} & \multicolumn{2}{c}{C1 far (\%)} & \multicolumn{2}{c}{D: partial $r$, 100 pairs} & Region D & D-all: $r$ & Region D-all \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Encoder & Role & " + EST + " & " + EST + " & " + EST + r" & SESOI $.10/.20/.30$ & 4{,}950 pairs & SESOI $.10/.20/.30$ \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e]] + sum([[V[f'h1{k}@{e}'], V[f'h1{k}Reg@{e}']] for k in "thc"], [])
                + [(r"$<$.001" if V[f'h1cP@{e}'].startswith(".000") else V[f'h1cP@{e}']), V[f"h2tReg@{e}"], V[f"h2hReg@{e}"], r"n/a$^{b}$"])
(OUT/"tab2b_agents_human_v2b.tex").write_text(wrap2("llrcrcrccccc",
    [r" & & \multicolumn{7}{c}{H1: Spearman $\rho$ and region} & \multicolumn{3}{c}{H2: region} \\ \cmidrule(lr){3-9}\cmidrule(lr){10-12}",
     r" & & \multicolumn{2}{c}{THINGS} & \multicolumn{2}{c}{ImageNet-HSJ} & \multicolumn{2}{c}{CIFAR-10H} & CIFAR-10H & & & \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
     r"Encoder & Role & $\rho$ & Region & $\rho$ & Region & $\rho$ & Region & Mantel $p$ & THINGS & HSJ & CIFAR-10H \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e], V[f'c1native@{e}'], V[f'c1far@{e}']] + EC(f"dR@{e}", f"dLo@{e}", f"dHi@{e}", flag(V[f"dReg@{e}"]))
                + [T(V[f"daR@{e}"]) + flag(V[f"daReg@{e}"])])
(OUT/"tab_agents_compact_v2b.tex").write_text(wrap2("llrrrlr",
    [r" & & & & \multicolumn{2}{c}{D: partial $r$} & D-all \\ \cmidrule(lr){5-6}",
     r"Encoder & Role & Native (\%) & Far (\%) & " + EST + r" & partial $r$ \\"], rows))
rows = []
for e in ENC:
    if e == "F1_rn50rand0": rows.append("MID")
    rows.append([NAMEP[e], RA[e]] + [V[f'h1{k}@{e}'] + flag(V[f'h1{k}Reg@{e}']) for k in "thc"])
(OUT/"tab_h1_compact_v2b.tex").write_text(wrap2("llrrr", [r"Encoder & Role & THINGS & ImageNet-HSJ & CIFAR-10H \\"], rows))
with open(OUT/"chequeos_cruzados.txt","w") as f:
    for n, ok, d in checks: f.write(f"{'OK ' if ok else 'FALLA'} {n} {d}\n")
print("macros", len(macros), "chequeos", len(checks), "fallas", sum(not ok for _, ok, _ in checks))
