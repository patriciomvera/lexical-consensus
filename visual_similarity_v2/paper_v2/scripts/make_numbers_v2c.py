"""Macros (\\Vc...) and post hoc tables for manuscript v2c. Sources: frozen files at commit a945ed2 (read with git show) and the
post hoc outputs in posthoc_v2c/out (scripts p01-p05, seeded, from the frozen per-trial data). Writes
manuscrito_v2c/generated/numbers_v2c.tex, generated/fuentes_v2c.csv, tables/tab_posthoc_v2c.tex and tables/tab_h2post_v2c.tex.
Usage: python make_numbers_v2c.py <repo checkout at a945ed2>"""
import io, json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(sys.argv[1]); COMMIT = "a945ed2cbdd9365fe346b262763f0262eaf686ea"
BASE = Path(__file__).resolve().parents[1]; import os; MS = BASE / os.environ.get("MS_DIR", "manuscrito_v2c"); PH = BASE / "posthoc_v2c"
EE = "visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence/"; AR = EE + "agents/out/results/"; XT = "visual_similarity_v2/agents_xtmem_v2/outputs/"
def raw(p): return subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:{p}"], capture_output=True, check=True).stdout
def csv(p): return pd.read_csv(io.BytesIO(raw(p)))
def js(p): return json.loads(raw(p))
M, SRC, CHECKS = {}, [], []
def put(name, val, src):
    assert re.fullmatch(r"Vc[A-Za-z]+", name) and name not in M, name
    M[name] = val; SRC.append((name, val, src))
def check(cond, what): CHECKS.append((what, bool(cond))); assert cond, what
def f(x, d=2, lead=True):
    s = f"{x:.{d}f}"
    if not lead: s = re.sub(r"^(-?)0\.", r"\1.", s)
    return s.replace("-", r"\ensuremath{-}")
T8 = ["R0_dinov2s","O1_dinov2b","R1_cvcl","E1_siglip","O2_openclip","E2_vitsup","E3_rn50sup","E4_mae"]; ALL = T8 + ["F1_rn50rand0","F2_pixels"]
NAME = {"R0_dinov2s":"DINOv2-small","O1_dinov2b":"DINOv2-base","R1_cvcl":"CVCL","E1_siglip":"SigLIP","O2_openclip":"OpenCLIP","E2_vitsup":"ViT (supervised)",
        "E3_rn50sup":"ResNet-50 (supervised)","E4_mae":"MAE","F1_rn50rand0":"Random ResNet-50","F2_pixels":"Pixels"}

# ---- H1 magnitude (A4) ----
H = {e: js(AR + f"H_{e}.json")["H1"] for e in ALL}
tr = [e for e in T8 if e != "E4_mae"]; rho = [H[e][s]["rho"] for e in tr for s in ("THINGS","HSJ")]
lo = [H[e][s]["ci_lo"] for e in tr for s in ("THINGS","HSJ")]
put("VcHoneTrMin", f(min(rho),2,False), AR + "H_*.json (trained except MAE; THINGS, HSJ)"); put("VcHoneTrMax", f(max(rho),2,False), AR + "H_*.json")
put("VcHoneTrLoMin", f(min(lo),2,False), AR + "H_*.json ci_lo")
put("VcHoneMaeThings", f(H["E4_mae"]["THINGS"]["rho"],2,False), AR + "H_E4_mae.json"); put("VcHoneMaeHsj", f(H["E4_mae"]["HSJ"]["rho"],2,False), AR + "H_E4_mae.json")
check(max(H["F1_rn50rand0"][s]["ci_hi"] for s in ("THINGS","HSJ")) < min(lo), "random-network upper CI limits below trained (non-MAE) lower limits")
check(all(H[e][s]["rho"] > 0 and H[e][s]["mantel_p"] < .05 for e in ALL for s in ("THINGS","HSJ")), "all H1 THINGS/HSJ in support region (rho>0, Mantel p<.05)")

# ---- Section 5 rules (A5) ----
p = XT + "broad_minus_narrow.csv"; bm = csv(p); ow = bm[(bm.task=="one_word") & bm.encoder.isin(T8)]
c = ow[ow.learner=="centroid"]; jmax = c.max_jump_any_fixed_threshold.max(); adult = c.human_broad_minus_narrow.iloc[0]
put("VcXSameSubPct", f(100*c.human_3sub.iloc[0]), p); put("VcXStep", f(100/12), p)
check(abs(jmax*12 - round(jmax*12)) < 1e-9 and round(jmax*12) == 8, "largest common-threshold gain = 8/12")
nxt = (round(jmax*12)+1)/12; put("VcXNextStep", f(100*nxt), p)
check(abs(100*c.human_3sup.iloc[0] - 100*c.human_3sub.iloc[0] - 100*adult) < 1e-9, "adult benchmark = 3sup - 3sub")
check(min(range(13), key=lambda k: abs(k/12 - adult)) == round(nxt*12), "next step is the attainable step closest to the adult value")
for lr, nm in [("kcentroid2","Kc"), ("knn3","Knn")]:
    x = ow[ow.learner==lr].max_jump_any_fixed_threshold
    put(f"VcXJump{nm}Min", f(100*x.min()), p); put(f"VcXJump{nm}Max", f(100*x.max()), p)
    check(x.max() <= adult, f"{lr}: common-threshold gain below the adult benchmark")

# ---- frozen structure, calibration ----
p = EE + "results/structure_primary.csv"; sp = csv(p); check(set(sp.re_used) == {"L4"}, "stage-1 common rung = L4 (1|person) for 8 and 24 items")

# ---- preprocessing lists (Appendix B) ----
spec = js(EE + "out/encoder_specs.json"); crop, full = [], []
for e in T8:
    pr = spec[e]["processor"]; s_ = json.dumps(pr)
    if pr.get("do_center_crop") and "shortest_edge" in json.dumps(pr.get("size", {})) and pr["size"]["shortest_edge"] == 256: crop.append(e)
    elif pr.get("image_processor_type") == "ConvNextImageProcessor" and abs(pr.get("crop_pct", 0) - .875) < 1e-9: crop.append(e)
    elif e == "O2_openclip": check(pr["size"]["shortest_edge"] == 224 and pr["crop_size"]["height"] == 224, "OpenCLIP 224/224"); 
    else: full.append(e)
check(set(crop) == {"R0_dinov2s","O1_dinov2b","E3_rn50sup"} and set(full) == {"R1_cvcl","E1_siglip","E2_vitsup","E4_mae"}, "preprocessing groups")
check("256" in spec["F1_rn50rand0"]["processor"]["resize"] and spec["F1_rn50rand0"]["processor"]["center_crop"] == 224, "random RN50 256/224")
put("VcCropList", "DINOv2-small, DINOv2-base, supervised ResNet-50, and the random ResNet-50", EE + "out/encoder_specs.json")
put("VcFullList", "CVCL, SigLIP, the supervised ViT, and MAE", EE + "out/encoder_specs.json")

# ---- H2 (B4) ----
h2 = pd.read_csv(PH / "out/p04_h2_estimates.csv"); mde = pd.read_csv(PH / "out/p04_h2_mde.csv").set_index("n")
rr = h2[(h2.encoder=="F1_rn50rand0") & (h2.source=="HSJ")].iloc[0]
put("VcHtwoRandHsjP", f(rr.p,3,False), "posthoc_v2c/out/p04_h2_estimates.csv (frozen H_F1_rn50rand0.json)")
for k, v in [("R", rr.r), ("Lo", rr.r_lo), ("Hi", rr.r_hi)]: put(f"VcHtwoRandHsj{k}", f(v,2,False), "p04_h2_estimates.csv")
check(rr.p < .05 and rr.r_lo < 0 < rr.r_hi, "random HSJ: HC3 p<.05 but Fisher-z CI of r includes 0")
put("VcMdeThings", f(mde.loc[61,"mde_r_80"],2,False), "p04_h2_mde.csv"); put("VcMdeHsj", f(mde.loc[32,"mde_r_80"],2,False), "p04_h2_mde.csv")
put("VcMdeHalfThings", f(mde.loc[61,"halfwidth_at_r0"],2,False), "p04_h2_mde.csv"); put("VcMdeHalfHsj", f(mde.loc[32,"halfwidth_at_r0"],2,False), "p04_h2_mde.csv")
reg = csv(AR + "regions/regions_agents.csv") if False else None
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{THINGS ($n = 61$)} & \multicolumn{3}{c}{ImageNet-HSJ ($n = 32$)} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Representation & $r$ [95\% CI] & $p$ & all pairs $r$ & $r$ [95\% CI] & $p$ & all pairs $r$ \\", r"\midrule"]
for e in ALL:
    row = [NAME[e]]
    for s in ("THINGS","HSJ"):
        x = h2[(h2.encoder==e)&(h2.source==s)].iloc[0]
        row += [f"{f(x.r,2,False)} [{f(x.r_lo,2,False)}, {f(x.r_hi,2,False)}]", f(x.p,3,False), f(x.r_all,2,False)]
    L.append(" & ".join(row) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
(MS / "tables/tab_h2post_v2c.tex").write_text("% AUTO-GENERATED by scripts/make_numbers_v2c.py -- do not edit\n" + "\n".join(L) + "\n")
put("VcHtwoAllThingsN", "3{,}400" if int(h2.n_all[h2.source=="THINGS"].iloc[0]) == 3400 else "?", "p04"); put("VcHtwoAllHsjN", "1{,}390" if int(h2.n_all[h2.source=="HSJ"].iloc[0]) == 1390 else "?", "p04")

# ---- B1 / B2 (adult post hoc models) ----
p02 = {e: pd.read_csv(PH / f"out/p02_{e}.csv") for e in ALL}; p03 = {e: pd.read_csv(PH / f"out/p03_{e}.csv") for e in ALL}
def row(d, m, re_=None):
    x = d[d.model == m] if re_ is None else d[(d.model == m) & (d.re == re_)]
    assert len(x) == 1, (m, re_); return x.iloc[0]
def ci(x): return f"{f(x.estimate)} [{f(x.lo)}, {f(x.hi)}]"
ex = {e: pd.read_csv(PH / f"data/ex_long_{e}.csv", usecols=["in_target","dist"]) for e in ALL}
TB = [r"\begin{tabular}{lcccccccc}", r"\toprule",
      r" & \multicolumn{2}{c}{SD of $d$} & \multicolumn{2}{c}{Distance slope (8 items)} & & \multicolumn{2}{c}{$\Delta$AIC vs.\ centroid} & \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-8}",
      r"Representation & 8 & 24 & Frozen model & By-image slopes & SD$_{\text{image}}$ & Mean & Nearest & $\chi^2_{\text{cond}}$ \\", r"\midrule"]
nci0 = 0; same_sign = 0; ok_img = 0; dmean_better = dmin_better = 0; condchi = []
for e in ALL:
    a, b = p02[e], p03[e]; fr = row(a, "pc_frozen_refit"); im = row(a, "pc_image_slope")
    b0 = row(b, "pc8_dist"); check(abs(b0.estimate - fr.estimate) < 1e-6, f"{e}: p02 and p03 refits agree")
    if im.converged_ok:
        ok_img += 1; same_sign += int(np.sign(im.estimate) == np.sign(fr.estimate)); nci0 += int(im.lo < 0 < im.hi)
        imt = ci(im) + ("" if im.re.startswith("(1 + zd | person) + (1 + zd | image)") else r"$^{b}$"); sdi = f(im.sd_image_slope)
    else: imt, sdi = "did not converge", "--"
    dA = row(b, "pc8_dmean").AIC - b0.AIC; dN = row(b, "pc8_dmin").AIC - b0.AIC
    dmean_better += int(dA < 0); dmin_better += int(dN < 0)
    cm = [row(b, f"sup_{k}").LR_cond_given_z for k in ("dist","dmin","dmean")]; condchi.append(min(cm))
    sdd8, sdd24 = ex[e].dist[ex[e].in_target == 1].std(), ex[e].dist.std()
    g3 = lambda v: re.sub(r"^0\.", ".", f"{v:.2g}" if v < .01 else f"{v:.3f}")
    TB.append(" & ".join([NAME[e], g3(sdd8), g3(sdd24), ci(fr) + ("" if fr.re.startswith("(1 + zd") else r"$^{a}$"), imt, sdi,
                          f"{dA:+.0f}".replace("-", r"\ensuremath{-}"), f"{dN:+.0f}".replace("-", r"\ensuremath{-}"), f"{min(cm):,.0f}".replace(",", "{,}")]) + r" \\")
TB += [r"\bottomrule", r"\end{tabular}"]
(MS / "tables/tab_posthoc_v2c.tex").write_text("% AUTO-GENERATED by scripts/make_numbers_v2c.py -- do not edit\n" + "\n".join(TB) + "\n")
W = "zero one two three four five six seven eight nine ten".split()
imr = {e: row(p02[e], "pc_image_slope") for e in ALL}; check(all(imr[e].converged_ok for e in ALL), "B1: all by-image slope models converged")
check(all(imr[e].re == "(1 + zd | person) + (1 + zd | image)" for e in ALL), "B1: richest rung I1 for all")
ss = [e for e in T8 if np.sign(imr[e].estimate) == np.sign(row(p02[e], "pc_frozen_refit").estimate)]
inc = [e for e in T8 if imr[e].lo < 0 < imr[e].hi]; exc = [e for e in T8 if imr[e].hi < 0]
check(len(inc) + len(exc) == len(T8) - sum(imr[e].lo > 0 for e in T8), "B1 partition")
put("VcImgTrSameSignWord", W[len(ss)], "p02_*.csv"); put("VcImgTrInclWord", W[len(inc)], "p02_*.csv"); put("VcImgTrExclWord", W[len(exc)], "p02_*.csv")
nm = [NAME[e] for e in inc]; put("VcImgTrInclList", (", ".join(nm[:-1]) + ", and " + nm[-1]) if len(nm) > 2 else " and ".join(nm), "p02_*.csv")
x = imr["F1_rn50rand0"]; put("VcImgRandB", f(x.estimate), "p02_F1"); put("VcImgRandLo", f(x.lo), "p02_F1"); put("VcImgRandHi", f(x.hi), "p02_F1")
put("VcImgOkN", str(ok_img), "p02_*.csv")
sds = [imr[e].sd_image_slope for e in ALL]; put("VcImgSdMin", f(min(sds),1), "p02_*.csv"); put("VcImgSdMax", f(max(sds),1), "p02_*.csv")
z0 = row(p02["R0_dinov2s"], "sup_only", "(1 | person) + (1 | image)"); put("VcSupFrozenDistChi", f(z0.LR_dist_given_cond), "p02_R0 sup_only frozen RE")
check(abs(z0.LR_dist_given_cond - row(p03["R0_dinov2s"], "sup_dist").LR_z_given_cond) < 1e-6, "sup-only frozen LR reproduced in p02 and p03")
check((h2.r_all < 0).all(), "H2 all-pairs r negative for every representation"); put("VcImgSameSignN", str(same_sign), "p02_*.csv"); put("VcImgCIzeroN", str(nci0), "p02_*.csv")
put("VcDmeanBetterN", str(dmean_better), "p03_*.csv"); put("VcDminBetterN", str(dmin_better), "p03_*.csv")
put("VcSupCondChiMinAll", f"{min(condchi):,.0f}".replace(",", "{,}"), "p03_*.csv (min over representations and readouts)")
r0 = p02["R0_dinov2s"]; im = row(r0, "pc_image_slope"); fr = row(r0, "pc_frozen_refit")
put("VcImgRzB", f(im.estimate), "p02_R0"); put("VcImgRzLo", f(im.lo), "p02_R0"); put("VcImgRzHi", f(im.hi), "p02_R0"); put("VcImgRzSd", f(im.sd_image_slope), "p02_R0")
so = row(r0, "sup_only", "(1 | person) + (1 + zd | image)"); put("VcSupImgB", f(so.estimate), "p02_R0 sup_only image slope")
put("VcSupImgLo", f(so.lo), "p02_R0"); put("VcSupImgHi", f(so.hi), "p02_R0"); put("VcSupImgChi", f(so.LR_dist_given_cond), "p02_R0"); put("VcSupImgP", f(so.p_dist_given_cond,2,False), "p02_R0")
put("VcSupImgCondChi", f"{so.LR_cond_given_dist:,.0f}".replace(",", "{,}"), "p02_R0")
b = p03["R0_dinov2s"]
for k, nm in [("dmean","Dmean"), ("dmin","Dmin")]:
    x = row(b, f"sup_{k}"); put(f"VcSup{nm}CondChi", f"{x.LR_cond_given_z:,.0f}".replace(",", "{,}"), "p03_R0"); put(f"VcSup{nm}DistChi", f(x.LR_z_given_cond), "p03_R0")
    y = row(b, f"pc8_{k}"); put(f"VcPc{nm}B", f(y.estimate), "p03_R0"); put(f"VcPc{nm}dAIC", f"{y.AIC - row(b,'pc8_dist').AIC:+.0f}".replace("-", r"\ensuremath{-}"), "p03_R0")
x = row(b, "sup_dist"); put("VcSupDistCondChi", f"{x.LR_cond_given_z:,.0f}".replace(",", "{,}"), "p03_R0")
cell_ok = [e for e in ALL if row(p02[e], "pc_cell_intercept").converged_ok]; put("VcCellOkN", str(len(cell_ok)), "p02_*.csv")
put("VcPosthocDate", "24 November 2026", "DESVIACIONES_posthoc_v2c.md")
# dispersion of teaching sets
dp = pd.read_csv(PH / "out/p05_support_dispersion.csv").pivot(index="encoder", columns="condition", values="disp_mean")
put("VcDispSub", f(dp.loc["R0_dinov2s","sub"],2,False), "p05_support_dispersion.csv"); put("VcDispSup", f(dp.loc["R0_dinov2s","sup"],2,False), "p05")
put("VcDispRandMax", f(dp.loc["F1_rn50rand0"].max(),3,False), "p05")
sm = pd.read_csv(PH / "out/p05_sup_distance_summary.csv").query("encoder=='R0_dinov2s'").set_index("condition")
for cnd, nm in [("sub","Sub"), ("sup","Sup")]: put(f"VcSupMean{nm}", f(sm.loc[cnd,"dist_mean"],2,False), "p05"); put(f"VcSupSd{nm}", f(sm.loc[cnd,"dist_std"],2,False), "p05")

out = MS / "generated"; out.mkdir(exist_ok=True)
(out / "numbers_v2c.tex").write_text("% AUTO-GENERATED by scripts/make_numbers_v2c.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
pd.DataFrame(SRC, columns=["macro","value","source"]).to_csv(out / "fuentes_v2c.csv", index=False)
print(f"{len(M)} macros, {len(CHECKS)} checks, {sum(not c for _, c in CHECKS)} failures")
for w, cc in CHECKS: print(("OK  " if cc else "FAIL"), w)
