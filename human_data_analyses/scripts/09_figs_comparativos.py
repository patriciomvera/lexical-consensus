"""Descriptive comparison figures (agent vs human). No fitting beyond what is in out/*.json except
binned means (H4) and lowess-free scatter. Figures not foreseen in the plan are labelled 'exploratory/descriptive'."""
import json, pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
B = pathlib.Path(__file__).resolve().parents[1]; D = B/"data"; O = B/"out"; F = B/"figs/comparativos"; F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":.6,"savefig.bbox":"tight"})
J = lambda n: json.load(open(O/n)); H12 = J("h1_h2_results.json"); H3 = J("h3_results.json"); H5 = J("h5_results.json"); H6 = J("h6_results.json"); DI = J("dispersion_results.json"); HO = J("holm_confirmatory.json")
def save(fig, name): [fig.savefig(F/f"{name}.{e}", dpi=300) for e in ("png","pdf")]; plt.close(fig)
def tag(ax, s): ax.figure.text(0.99, -0.02, s, ha="right", va="top", fontsize=6, style="italic", color="0.4")
P100 = pd.read_csv(D/"pairs_c100.csv"); P10 = pd.read_csv(D/"pairs_c10.csv"); S = pd.read_csv(D/"sampled_100.csv"); T = pd.read_csv(D/"tiers_c10.csv")
K = dict(c="0.25", s=4, alpha=.35, lw=0)
# (a)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, (src, df, col, key) in zip(axs, [("THINGS (CIFAR-100)", P100, "d_things", "THINGS"), ("ImageNet-HSJ (CIFAR-100)", P100, "d_hsj", "HSJ"), ("CIFAR-10H (CIFAR-10)", P10, "d_c10h", "CIFAR10H")]):
    d = df.dropna(subset=[col]); r = H12["H1"][key]
    ax.scatter(d[col], d.d_perc, **(K if len(d) > 100 else dict(c="0.2", s=12, lw=0)))
    ax.set_xlabel(f"Human distance ({key})"); ax.set_ylabel("Learner distance d_perc (DINOv2)")
    ax.set_title(f"{src}\nρ = {r['rho']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}], n = {r['n_pairs']}", fontsize=7.5)
tag(axs[-1], "H1 (confirmatory)"); fig.tight_layout(); save(fig, "a_h1_scatter_human_vs_dperc")
# (b)
fig, axs = plt.subplots(3, 2, figsize=(5.6, 7.2))
rows = [("THINGS", S, "d_things", "acc_pair", "100 sampled CIFAR-100 pairs"), ("HSJ", S, "d_hsj", "acc_pair", "100 sampled CIFAR-100 pairs"), ("CIFAR-10H", T, "d_c10h", "acc_tier", "10 CIFAR-10 tiers")]
for i, (src, df, col, acc, lab) in enumerate(rows):
    d = df.dropna(subset=[col, acc])
    for j, x in enumerate([col, "d_perc"]):
        ax = axs[i, j]; ax.scatter(d[x], d[acc], c="0.2", s=10, lw=0); rr = stats.spearmanr(d[x], d[acc])[0]
        ax.set_xlabel("Human distance (" + src + ")" if j == 0 else "d_perc (DINOv2)"); ax.set_ylabel("Learner accuracy")
        ax.set_title(f"{src}: {lab}\nn = {len(d)}, Spearman ρ = {rr:.2f}", fontsize=6.5)
tag(axs[0, 1], "descriptive; same pairs in both columns"); fig.tight_layout(); save(fig, "b_accuracy_vs_human_and_dperc")
# (c)
items = []  # (group, label, est, lo, hi, verdict)
for k, nm in [("THINGS", "THINGS"), ("HSJ", "HSJ"), ("CIFAR10H", "CIFAR-10H")]:
    r = H12["H1"][k]; items.append(("ρ (Spearman)", f"H1 {nm}", r["rho"], r["ci_lo"], r["ci_hi"], "Support"))
for k, c, nm, v in [("THINGS", "d_things", "THINGS", "Inconclusive"), ("HSJ", "d_hsj", "HSJ", "Outside rules")]:
    r = H12["H2"][k][c]; items.append(("partial r", f"H2 {nm} (β_human)", r["r_partial"], r["r_ci_lo"], r["r_ci_hi"], v))
for k in ["d_perc", "d_sem", "d_things", "d_hsj", "d_c10h"]:
    r = H3[f"tier_{k}"]; items.append(("ρ (Spearman)", f"H3 tiers {k} (n={r['n']})", r["rho"], r["ci_lo"], r["ci_hi"], "Descriptive"))
for k, nm in [("C100_glasgow", "C100"), ("C10_glasgow", "C10")]:
    r = H5[k]; items.append(("ρ (Spearman)", f"H5 AoA {nm} (n={r['n']}, partial)", r["rho_partial"], r["ci_lo"], r["ci_hi"], "Exploratory"))
t = H6["tests"]
for k, nm in [("b_c10_cvcl_vs_dino", "H6b C10 CVCL–DINOv2"), ("b_c100_cvcl_vs_dino", "H6b C100 CVCL–DINOv2"), ("c_c10_cvcl_vs_d_things", "H6c C10 CVCL–THINGS"), ("c_c10_cvcl_vs_d_hsj", "H6c C10 CVCL–HSJ"), ("c_c10_cvcl_vs_d_c10h", "H6c C10 CVCL–C10H"), ("c_c100_cvcl_vs_d_things", "H6c C100 CVCL–THINGS"), ("c_c100_cvcl_vs_d_hsj", "H6c C100 CVCL–HSJ")]:
    r = t[k]; items.append(("ρ (Spearman)", nm, r["rho"], r["ci_lo"], r["ci_hi"], "Exploratory"))
for c, nm in [("d_perc_cvcl", "H6a β_perc CVCL"), ("d_sem", "H6a β_sem")]:
    r = H6["a_ols"][c]; items.append(("partial r", f"{nm} (partial r)", r["r_partial"], r["r_ci_lo"], r["r_ci_hi"], "Inconsistent" if "perc" in nm else ""))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 5.2), gridspec_kw=dict(width_ratios=[3, 1]))
y = np.arange(len(items))[::-1]
for yy, (g, l, e, lo, hi, v) in zip(y, items):
    a1.errorbar(e, yy, xerr=[[e-lo], [hi-e]], fmt="o" if g.startswith("ρ") else "s", ms=3.5, c="0.15", elinewidth=.8, capsize=1.5)
    a1.text(1.08, yy, v, va="center", fontsize=6.5, transform=a1.get_yaxis_transform())
a1.set_yticks(y); a1.set_yticklabels([i[1] for i in items], fontsize=6.5); a1.axvline(0, c="0.5", lw=.6)
for s in (-.2, .2): a1.axvline(s, c="0.7", lw=.6, ls=":")
a1.set_xlabel("Estimate [95% CI]  (circles: Spearman ρ; squares: partial r)  — dotted: SESOI ±0.20"); a1.set_xlim(-1.05, 1.05)
h4 = HO["H4_est"]; S4 = {"Primary (24 items)": (-3.2325, -3.2916, -3.1735), "Excl. non-prereg. exp.": (-3.1731, -3.2381, -3.1082), "Target-category items only": (-3.1865, -3.3262, -3.0469)}
for i, (k, (e, lo, hi)) in enumerate(S4.items()): a2.errorbar(e, 2-i, xerr=[[e-lo], [hi-e]], fmt="D", ms=3.5, c="0.15", capsize=1.5)
a2.set_yticks([2, 1, 0]); a2.set_yticklabels(list(S4), fontsize=6.5); a2.set_xlabel("H4 log OR per SD distance"); a2.set_title("H4 (Support)", fontsize=7.5); a2.axvline(0, c="0.5", lw=.6); a2.set_xlim(-3.6, .3)
a2.yaxis.tick_right(); a2.set_ylim(-4, 6); fig.subplots_adjust(wspace=.9); tag(a2, "verdicts per frozen plan rules"); save(fig, "c_forest_all_H1_H6")
# (d)
L = pd.read_csv(D/"h4_long.csv"); fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
for ax, (nm, d) in zip(axs, [("All 24 test items (incl. other-category items)", L), ("Target-category items only (8 per trial)", L[L.in_target == 1])]):
    d = d.copy(); d["bin"] = pd.qcut(d.dist, 12, duplicates="drop"); g = d.groupby("bin", observed=True).agg(x=("dist", "mean"), p=("selected", "mean"), n=("selected", "size"))
    se = np.sqrt(g.p*(1-g.p)/g.n); ax.errorbar(g.x, g.p, yerr=1.96*se, fmt="o-", ms=3, c="0.15", lw=.8, capsize=1.5)
    if nm.startswith("All"):
        for it, mk, lab in [(1, "^", "target-category"), (0, "x", "other-category")]:
            dd = L[L.in_target == it]; ax.scatter(dd.groupby("image").dist.mean(), dd.groupby("image").selected.mean(), marker=mk, s=8, c="0.55", lw=.6, label=lab + " items (per image)")
        ax.legend(fontsize=6, frameon=False, loc="center left")
    ax.set_title(nm + f"\nn = {len(d):,} obs.", fontsize=7); ax.set_xlabel("DINOv2 distance to training exemplars"); ax.set_ylabel("Human selection rate (Lewis & Frank 2018)")
tag(axs[1], "descriptive: binned means (12 quantile bins), 95% binomial CI"); fig.tight_layout(); save(fig, "d_h4_generalization_curve")
# (e)
C10 = pd.read_csv(D/"h6_pairs_c10.csv"); C100 = P100.merge(pd.read_csv(D/"cvcl_dperc_c100.csv"), on=["a", "b"])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4))
for ax, (nm, d, k) in zip(axs[:2], [("CIFAR-10 (45 pairs)", C10, "b_c10_cvcl_vs_dino"), ("CIFAR-100 (4,950 pairs)", C100, "b_c100_cvcl_vs_dino")]):
    r = t[k]; ax.scatter(d.d_perc, d.d_perc_cvcl, **(K if len(d) > 100 else dict(c="0.2", s=12, lw=0)))
    ax.set_xlabel("d_perc DINOv2"); ax.set_ylabel("d_perc CVCL"); ax.set_title(f"{nm}\nρ = {r['rho']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]", fontsize=7.5)
tc, td = H6["d_tiers"]["c1_by_tier_cvcl"], H6["d_tiers"]["c1_by_tier_dino"]; tiers = list(td); x = np.arange(len(tiers))
axs[2].bar(x-.18, [td[k] for k in tiers], .36, color="0.3", label="DINOv2"); axs[2].bar(x+.18, [tc[k] for k in tiers], .36, color="0.7", label="CVCL")
axs[2].set_xticks(x); axs[2].set_xticklabels(tiers); axs[2].set_ylabel("Learner accuracy (CIFAR-10 tiers)"); axs[2].legend(fontsize=6, frameon=False)
axs[2].set_title(f"C100 mean pair acc.: DINOv2 {H6['base_acc_c100_dino_paper']:.2f}, CVCL {H6['base_acc_c100_cvcl']:.2f}", fontsize=6.5)
tag(axs[2], "H6 exploratory; CVCL is not human data"); fig.tight_layout(); save(fig, "e_h6_dino_vs_cvcl")
# (f)
d = P100.dropna(subset=["d_things", "d_hsj"]); rr = stats.spearmanr(d.d_things, d.d_hsj)[0]; ss = S.dropna(subset=["d_things", "d_hsj"])
fig, ax = plt.subplots(figsize=(3.2, 3)); ax.scatter(d.d_things, d.d_hsj, **K, label=f"all shared pairs (n = {len(d):,})")
ax.scatter(ss.d_things, ss.d_hsj, s=14, facecolor="none", edgecolor="k", lw=.7, label=f"sampled pairs in both (n = {len(ss)})")
ax.set_xlabel("THINGS distance"); ax.set_ylabel("ImageNet-HSJ distance"); ax.set_title(f"Agreement between human sources (CIFAR-100)\nSpearman ρ = {rr:.2f}", fontsize=7.5); ax.legend(fontsize=6, frameon=False)
tag(ax, "§4 dispersion (descriptive)"); fig.tight_layout(); save(fig, "f_agreement_things_vs_hsj")
# (g)
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.6))
for ax, (nm, f, k) in zip(axs, [("CIFAR-10", "h5_C10_glasgow.csv", "C10_glasgow"), ("CIFAR-100", "h5_C100_glasgow.csv", "C100_glasgow")]):
    d = pd.read_csv(D/f); r = H5[k]; ax.scatter(d.dispersion, d.aoa_glasgow, c="0.2", s=10, lw=0)
    if len(d) <= 10: [ax.annotate(w, (x_, y_), fontsize=5.5, xytext=(2, 2), textcoords="offset points") for w, x_, y_ in zip(d.word, d.dispersion, d.aoa_glasgow)]
    ax.set_xlabel("Within-class dispersion, DINOv2 (lower = more coherent)"); ax.set_ylabel("Glasgow AoA rating")
    ax.set_title(f"{nm}, n = {r['n']}: partial ρ (ctrl. log freq.) = {r['rho_partial']:.2f} [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]\nraw ρ = {r['rho_raw']:.2f}", fontsize=6.5)
tag(axs[1], "H5 exploratory (BH within H5)"); fig.tight_layout(); save(fig, "g_h5_aoa_vs_coherence")
print("ok", sorted(p.name for p in F.iterdir()))
