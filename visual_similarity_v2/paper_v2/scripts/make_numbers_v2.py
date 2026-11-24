"""Macros de texto para el manuscrito v2 (prefijo \\Vt...). Fuente única: archivos congelados en el commit fijado.
Uso: python make_numbers_v2.py <checkout del repo en a945ed2> [/ruta/copia_freeze_del_parche]
Lee con `git show COMMIT:ruta` (no del árbol de trabajo). Escribe generated/numbers_v2_text.tex y generated/fuentes_v2_text.csv.
"""
import io, json, re, subprocess, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(sys.argv[1]); COMMIT = "a945ed2cbdd9365fe346b262763f0262eaf686ea"
OUT = Path(__file__).resolve().parents[1] / "generated"; OUT.mkdir(exist_ok=True)
EE = "visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence/"; AR = EE + "agents/out/results/"; XT = "visual_similarity_v2/agents_xtmem_v2/outputs/"
def raw(p, c=COMMIT): return subprocess.run(["git", "-C", str(REPO), "show", f"{c}:{p}"], capture_output=True, check=True).stdout
def csv(p): return pd.read_csv(io.BytesIO(raw(p)))
def js(p): return json.loads(raw(p))
M, SRC = {}, []
def put(name, val, src):
    assert re.fullmatch(r"V[tb][A-Za-z]+", name), name
    M[name] = val; SRC.append((name, val, src))
def f(x, d=2, lead=True, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    if not lead: s = s.replace("0.", ".", 1)
    return s.replace("-", r"\ensuremath{-}")
def thou(n): return f"{int(n):,}".replace(",", "{,}")
T8 = ["R0_dinov2s","O1_dinov2b","R1_cvcl","E1_siglip","O2_openclip","E2_vitsup","E3_rn50sup","E4_mae"]

# --- tortuga dentro de sesión (Paso 1) ---
p = EE + "data/h4_long_R0_dinov2s.csv"; hl = csv(p)
b = hl[hl.condition.isin(["3sup","three_superordinate"]) & (hl.image=="c3_sup2")]
w = b.pivot_table(index=["exp","subid"], columns="category", values="selected", aggfunc="first")
assert b.groupby(["exp","subid","category"]).size().max()==1 and w.notna().all().all()
a, v = w.animals, w.vehicles
for n, x in [("VtTurtleSessions", len(w)), ("VtTurtleAnimalsK", int(a.sum())), ("VtTurtleVehiclesK", int(v.sum())), ("VtTurtleVegK", int(w.vegetables.sum())),
             ("VtTurtleOnlyAnimals", int(((a==1)&(v==0)).sum())), ("VtTurtleOnlyVehicles", int(((a==0)&(v==1)).sum())),
             ("VtTurtleBoth", int(((a==1)&(v==1)).sum())), ("VtTurtleNeither", int(((a==0)&(v==0)).sum()))]:
    put(n, str(x), p + " (3sup, image c3_sup2, paired by exp x subid)")
put("VtPersons", str(hl.subid.nunique()), p)

# --- exploratorio (one_word_items) ---
p = XT + "one_word_items.csv"; it = csv(p); c = it[it.learner=="centroid"]
DOM = {"vegetables":"c1","vehicles":"c2","animals":"c3"}
c = c.assign(own=c.image.str[:2]==c.domain.map(DOM))
cc = c[c.condition.isin(["3sub","3bas","3sup"]) & (~c.own | c.image.str.contains("_sup"))]
ad = cc[cc.encoder=="R0_dinov2s"]
def adult(own, cond):
    x = ad[(ad.own==own)&(ad.condition==cond)]; return x.human_yes.sum(), x.human_n.sum()
k, n = adult(True, "3bas"); put("VtXSameBasicPct", f(100*k/n), p); put("VtXInDenom", thou(n), p)
k, n = adult(True, "3sup"); put("VtXBroadPct", f(100*k/n), p)
k, n = adult(False, "3sup"); put("VtXOutBroadPct", f(100*k/n), p); put("VtXOutDenom", thou(n), p)
ag = cc[cc.condition=="3sup"].groupby(["encoder","own"]).accepted.agg(["sum","size"])
put("VtXViTIn", str(int(ag.loc[("E2_vitsup",True),"sum"])), p); put("VtXDinoBIn", str(int(ag.loc[("O1_dinov2b",True),"sum"])), p)
put("VtXInN", str(int(ag.loc[("E2_vitsup",True),"size"])), p)
put("VtXViTOut", str(int(ag.loc[("E2_vitsup",False),"sum"])), p); put("VtXDinoBOut", str(int(ag.loc[("O1_dinov2b",False),"sum"])), p)
put("VtXOutN", str(int(ag.loc[("E2_vitsup",False),"size"])), p)
put("VtXPerCell", str(int(ad.human_n.iloc[0])), p)
p2 = XT + "broad_minus_narrow.csv"; bm = csv(p2); bm = bm[(bm.task=="one_word")&(bm.learner=="centroid")&bm.encoder.isin(T8)]
put("VtXJumpMin", f(100*bm.max_jump_any_fixed_threshold.min()), p2); put("VtXJumpMax", f(100*bm.max_jump_any_fixed_threshold.max()), p2)
put("VtXJumpAdult", f(100*bm.human_broad_minus_narrow.iloc[0]), p2)
e3 = cc[(cc.encoder=="E3_rn50sup")&(cc.condition=="3sup")&cc.own].groupby("domain").accepted.sum()
put("VtXEthreeAnimals", str(int(e3["animals"])), p); put("VtXEthreeVeg", str(int(e3["vegetables"])), p); put("VtXEthreeVeh", str(int(e3["vehicles"])), p)
rail = c[(c.encoder=="R0_dinov2s")&(c.condition=="3sup")&(c.image=="c2_sup2")&(c.domain=="vehicles")]
put("VtXRailcarK", str(int(rail.human_yes.iloc[0])), p)

# --- adultos: control positivo, H4, primario ---
p = EE + "results/poscontrol.csv"; pc = csv(p).query("term=='zd'").set_index("encoder")
for e, n in [("F1_rn50rand0","Rand"),("F2_pixels","Pix"),("E2_vitsup","ViT"),("R0_dinov2s","Ref"),("E4_mae","Mae")]:
    put(f"VtPc{n}B", f(pc.loc[e,"estimate"]), p); put(f"VtPc{n}Lo", f(pc.loc[e,"zd_lo"]), p); put(f"VtPc{n}Hi", f(pc.loc[e,"zd_hi"]), p)
put("VtPcNobs", thou(pc.loc["R0_dinov2s","nobs"]), p)
# v2b: macros for the integrated manuscript (additive; checks for the claims in the text)
put("VbPcCvclB", f(pc.loc["R1_cvcl","estimate"]), p); put("VbPcCvclLo", f(pc.loc["R1_cvcl","zd_lo"]), p); put("VbPcCvclHi", f(pc.loc["R1_cvcl","zd_hi"]), p)
assert pc.loc[T8,"estimate"].idxmin() == "E2_vitsup" and pc.loc[T8,"estimate"].idxmax() == "E4_mae"   # range "from ViT to MAE"
assert pc.loc["F1_rn50rand0","estimate"] < pc.loc["R1_cvcl","estimate"] and pc.loc["F1_rn50rand0","estimate"] < pc.loc["E4_mae","estimate"]
assert pc.loc["F1_rn50rand0","zd_hi"] < pc.loc["R1_cvcl","zd_lo"] and pc.loc["F1_rn50rand0","zd_hi"] < pc.loc["E4_mae","zd_lo"]  # CIs do not overlap
assert pc.loc["F1_rn50rand0","zd_hi"] < 0 and pc.loc["F2_pixels","zd_lo"] > 0
put("VbPcRandAbove", str(int((pc.loc[T8,"estimate"] > pc.loc["F1_rn50rand0","estimate"]).sum())), p)   # trained encoders with a shallower slope than the random net

assert (pc.loc[T8,"zd_hi"]<0).all()
put("VtPcTrainedNeg", str(int((pc.loc[T8,"zd_hi"]<0).sum())), p)
p = AR + "h4_by_encoder.csv"; h4 = csv(p).set_index("encoder"); put("VtHfourNobs", thou(h4.loc["R0_dinov2s","nobs"]), p)
ALL10 = T8 + ["F1_rn50rand0","F2_pixels"]; assert (h4.loc[ALL10,"hi"] < 0).all()
put("VbHfourAllNeg", str(int((h4.loc[ALL10,"hi"] < 0).sum())), p)
for e, n in [("F1_rn50rand0","Rand"),("R1_cvcl","Cvcl"),("E4_mae","Mae"),("F2_pixels","Pix")]: put(f"VbHfour{n}B", f(h4.loc[e,"logOR"]), p)

put("VtHfourPersons", str(int(h4.loc["R0_dinov2s","persons"])), p)
p = EE + "results/regions/regions_table.csv"; rt = csv(p).set_index("encoder")
put("VtOrdDelta", f(rt.loc["R0_dinov2s","delta_hat"],3), p); put("VtOrdLo", f(rt.loc["R0_dinov2s","ci95_lo"],3), p); put("VtOrdHi", f(rt.loc["R0_dinov2s","ci95_hi"],3), p)
p = EE + "results/regions/global_grid.csv"; gg = csv(p); col = [x for x in gg.columns if gg[x].astype(str).str.fullmatch(r"G\d\w?").any()][0]
put("VtGlobalCells", str(len(gg)), p); put("VtGlobalLabel", gg[col].iloc[0], p); assert gg[col].nunique()==1
p = EE + "results/stage2_primary.csv"; sp = csv(p); sp = sp[sp.encoder=="R0_dinov2s"].n_exp; assert sp.nunique()==1; put("VtNexp", str(int(sp.iloc[0])), p)

# --- agentes CIFAR ---
D = {e: js(AR + f"D_{e}.json")["learners"]["centroid"]["model_ps"] for e in T8 + ["F1_rn50rand0","F2_pixels"]}
reg = csv(AR + "regions/regions_agents.csv")
sup = [e for e in T8 if (reg[(reg.encoder==e)&(reg.analisis=="D")][["region_0.10","region_0.20","region_0.30"]]=="Apoyo").all(axis=1).iloc[0]]
put("VtDSupportN", str(len(sup)), AR + "regions/regions_agents.csv")
W = {i: w for i, w in enumerate("zero one two three four five six seven eight".split())}; put("VtDSupportWord", W[len(sup)], AR + "regions/regions_agents.csv")
rs = [D[e]["r_partial"] for e in sup]; put("VtDrMin", f(max(rs),2,False), AR + "D_*.json"); put("VtDrMax", f(min(rs),2,False), AR + "D_*.json")
for e, n in [("R1_cvcl","Cvcl"),("E4_mae","Mae")]:
    put(f"VtD{n}R", f(D[e]["r_partial"],3,False), AR + f"D_{e}.json"); put(f"VtD{n}Lo", f(D[e]["r_ci_lo"],2,False), AR + f"D_{e}.json"); put(f"VtD{n}Hi", f(D[e]["r_ci_hi"],2,False), AR + f"D_{e}.json")
put("VtDN", str(D["R0_dinov2s"]["n"]), AR + "D_R0_dinov2s.json"); put("VtDallN", thou(js(AR + "Dall_R0_dinov2s.json")["test"]["n"]), AR + "Dall_R0_dinov2s.json")
H = {e: js(AR + f"H_{e}.json")["H1"] for e in T8 + ["F1_rn50rand0","F2_pixels"]}
rh = [H[e][s]["rho"] for e in T8 for s in ["THINGS","HSJ","CIFAR10H"]]
put("VtHoneMin", f(min(rh),2,False), AR + "H_*.json"); put("VtHoneMax", f(max(rh),2,False), AR + "H_*.json")
for s, n in [("THINGS","Things"),("HSJ","Hsj"),("CIFAR10H","Cten")]:
    put(f"VtHoneRand{n}", f(H["F1_rn50rand0"][s]["rho"],2,False), AR + "H_F1_rn50rand0.json"); put(f"VtHonePix{n}", f(H["F2_pixels"][s]["rho"],2,False), AR + "H_F2_pixels.json")
put("VtHonePixCtenP", f(H["F2_pixels"]["CIFAR10H"]["mantel_p"],4,False), AR + "H_F2_pixels.json")
for s_, n in [("THINGS","Things"),("HSJ","Hsj"),("CIFAR10H","Cten")]:
    hr = H["F1_rn50rand0"][s_]; put(f"VbHoneRand{n}P", f(hr["mantel_p"],3,False), AR + "H_F1_rn50rand0.json")
    put(f"VbHoneRand{n}Lo", f(hr["ci_lo"],2,False), AR + "H_F1_rn50rand0.json"); put(f"VbHoneRand{n}Hi", f(hr["ci_hi"],2,False), AR + "H_F1_rn50rand0.json")
    assert hr["rho"] > 0 and hr["mantel_p"] < .05

put("VtHoneNThings", thou(H["R0_dinov2s"]["THINGS"]["n_pairs"]), AR + "H_R0_dinov2s.json"); put("VtHoneNHsj", thou(H["R0_dinov2s"]["HSJ"]["n_pairs"]), AR + "H_R0_dinov2s.json"); put("VtHoneNCten", str(H["R0_dinov2s"]["CIFAR10H"]["n_pairs"]), AR + "H_R0_dinov2s.json")
# controles de tubería
p = AR + "control_T_exact.json"; ct = js(p); put("VtCtrlOneObs", f"{ct['nonrandom_margin_max_abs_diff']:.15e}".replace("e-06", r"\times 10^{-6}"), p)
src = raw(EE + "agents/scripts/11_agent_sims.py").decode().splitlines()
line = [i for i, l in enumerate(src, 1) if "mx <= 1e-6 + 1e-12" in l]; assert line
put("VtCtrlOneLine", str(line[0]), EE + "agents/scripts/11_agent_sims.py")
p = AR + "control_c100_recode.json"; c2 = js(p); put("VtCtrlTwoMax", f"{c2['max_abs_diff']:.2e}".replace("e-07", r"\times 10^{-7}"), p); put("VtCtrlTwoPairs", thou(c2["n_pairs"]), p)
# parche y freeze
pub = raw(EE + "DIFF_agentes_propuesta.patch"); hp = hashlib.sha256(pub).hexdigest()
man = raw(EE + "FREEZE_agentes_v2.txt").decode().splitlines()
ml = [(i, l) for i, l in enumerate(man, 1) if l.endswith("DIFF_agentes_propuesta.patch")][0]; he = ml[1].split()[0]
put("VtPatchPub", hp[:8] + r"\ldots{}" + hp[-4:], EE + "DIFF_agentes_propuesta.patch"); put("VtPatchFrz", he[:8] + r"\ldots{}" + he[-4:], EE + f"FREEZE_agentes_v2.txt l.{ml[0]}")
put("VtPatchFrzLine", str(ml[0]), EE + "FREEZE_agentes_v2.txt")
if len(sys.argv) > 2:
    loc = Path(sys.argv[2]).read_bytes(); assert hashlib.sha256(loc).hexdigest() == he
    lines = loc.decode().splitlines(keepends=True)
    hdr = next(i for i in range(1, 20) if hashlib.sha256("".join(lines[i:]).encode()).hexdigest() == hp)
    put("VtPatchHdrLines", str(hdr), sys.argv[2] + " (header lines removed => published hash)")
put("VtFreezeTime", [l for l in man if l.startswith("time:")][0].split()[2], EE + "FREEZE_agentes_v2.txt l.2")
put("VtFreezeDate", [l for l in man if l.startswith("time:")][0].split()[1], EE + "FREEZE_agentes_v2.txt l.2")
xt = subprocess.run(["git","-C",str(REPO),"log","-1","--format=%ad","--date=format:%Y-%m-%d %H:%M","b3db91c"],capture_output=True,text=True,check=True).stdout.split()
put("VtXtDate", xt[0], "git log b3db91c"); put("VtXtTime", xt[1], "git log b3db91c")
# tabla de checkpoints (desde 01_embed.py)
emb = raw(EE + "scripts/01_embed.py").decode()
REV = dict(re.findall(r'"([\w./-]+)": "([0-9a-f]{40})"', emb))
CK = [("DINOv2-small (reference, v1)","facebook/dinov2-small","CLS, last layer"),("DINOv2-base","facebook/dinov2-base","CLS, last layer"),
      ("CVCL","wkvong/cvcl_s_dino_resnext50_embedding","ResNeXt-50 vision encoder"),("SigLIP","google/siglip-base-patch16-224","attention-pooled output"),
      ("OpenCLIP","laion/CLIP-ViT-B-32-laion2B-s34B-b79K","projected image embedding"),("Supervised ViT","google/vit-base-patch16-224","CLS, last layer"),
      ("Supervised ResNet-50","microsoft/resnet-50","global average pool"),("MAE","facebook/vit-mae-base","mean of patch tokens, mask ratio 0")]
rowsck = []
for name, repo, pool in CK:
    assert repo in REV and repo in emb, repo
    rowsck.append(f"{name} & \\texttt{{{repo.replace('_', chr(92)+'_')}}} & \\texttt{{{REV[repo][:8]}}} & {pool} \\\\")
rowsck.append("Random ResNet-50 (s0) & torchvision \\texttt{resnet50(weights=None)}, seed 0 & -- & global average pool \\\\")
rowsck.append("Pixels & none & -- & resized, center-cropped pixels \\\\")
assert "resnet50(weights=None)" in emb and "torch.manual_seed(seed)" in emb
(OUT/"tab_checkpoints.tex").write_text("% AUTO-GENERATED by scripts/make_numbers_v2.py from 01_embed.py @ a945ed2 -- do not edit\n\\setlength{\\tabcolsep}{4pt}\n\\noindent\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{llll}\n\\toprule\nEncoder & Checkpoint & Revision & Embedding \\\\\n\\midrule\n" + "\n".join(rowsck) + "\n\\bottomrule\n\\end{tabular}}\n")
with open(OUT/"numbers_v2_text.tex","w") as fh:
    fh.write(f"% AUTO-GENERATED by scripts/make_numbers_v2.py from {COMMIT} -- do not edit\n")
    for k, val in M.items(): fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
pd.DataFrame(SRC, columns=["macro","valor","fuente"]).to_csv(OUT/"fuentes_v2_text.csv", index=False)
print(len(M), "macros"); [print(k, v) for k, v in M.items()]
