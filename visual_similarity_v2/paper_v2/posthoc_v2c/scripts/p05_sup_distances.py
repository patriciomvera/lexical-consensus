"""B3 (post hoc, v2c): distribution of the distances of the superordinate target-domain items by teaching condition (reference encoder),
with the centroid distance used in the paper and the nearest-exemplar distance, plus teaching-set dispersion for every representation.
Descriptive; reads the local per-trial tables of p01 (not redistributed). Deterministic (the horizontal jitter uses seed 42).
Usage: python p05_sup_distances.py <data_dir> <out_dir> <fig.pdf>"""
import sys, pathlib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); matplotlib.rcParams["pdf.fonttype"] = 42; matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
dd, od, fig = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
CM = {"one": "one", "three_subordinate": "sub", "3sub": "sub", "three_basic": "bas", "3bas": "bas", "three_superordinate": "sup", "3sup": "sup"}
ORDER = ["one", "sub", "bas", "sup"]; LAB = ["1 subordinate", "3 subordinate", "3 basic-level", "3 superordinate"]
IDS = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
summ, disp = [], []
for e in IDS:
    d = pd.read_csv(dd / f"ex_long_{e}.csv"); d["cond"] = d.condition.map(CM); assert d.cond.notna().all()
    p = d[(d.in_target == 1) & (d.level == "sup")]
    for c in ORDER:
        q = p[p.cond == c]
        summ.append(dict(encoder=e, condition=c, n=len(q), selected=q.selected.mean(), **{f"{v}_{s}": getattr(q[v], s)() for v in ("dist", "dmin", "dmean") for s in ("mean", "std", "min", "max")}))
    t = d.drop_duplicates(["exp", "subid", "trial"])
    for c in ORDER[1:]: disp.append(dict(encoder=e, condition=c, trials=int((t.cond == c).sum()), disp_mean=t[t.cond == c].disp.mean()))
pd.DataFrame(summ).to_csv(od / "p05_sup_distance_summary.csv", index=False); pd.DataFrame(disp).to_csv(od / "p05_support_dispersion.csv", index=False)
d = pd.read_csv(dd / "ex_long_R0_dinov2s.csv"); d["cond"] = d.condition.map(CM); p = d[(d.in_target == 1) & (d.level == "sup")]
rng = np.random.default_rng(42)
BOT, TOP = float(p[["dist", "dmin"]].min().min()), float(p[["dist", "dmin"]].max().max())
f, ax = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
for k, (v, title) in enumerate([("dist", "(a) Distance to the centroid (used in the paper)"), ("dmin", "(b) Distance to the nearest example (post hoc)")]):
    data = [p[p.cond == c][v].to_numpy() for c in ORDER]
    vp = ax[k].violinplot(data, positions=range(4), showextrema=False, widths=.8)
    for b in vp["bodies"]: b.set_facecolor("#9db7d5"); b.set_edgecolor("#3b5b82"); b.set_alpha(.7)
    for i, c in enumerate(ORDER):
        u = p[p.cond == c].groupby(["image", v]).size().reset_index(name="n")
        ax[k].scatter(i + rng.uniform(-.12, .12, len(u)), u[v], s=8 + u.n / 40, color="#1f3b5c", alpha=.75, lw=0)
        ax[k].plot([i - .3, i + .3], [data[i].mean()] * 2, color="#c0392b", lw=1.6)
    ax[k].set_xticks(range(4)); ax[k].set_xticklabels(LAB, fontsize=8); ax[k].set_title(title, fontsize=9)
    sel = [100 * p[p.cond == c].selected.mean() for c in ORDER]
    for i, s in enumerate(sel): ax[k].text(i, TOP + .02, f"{s:.1f}%", ha="center", fontsize=7.5)
ax[0].set_ylabel("Cosine distance (DINOv2-small)", fontsize=9); ax[0].set_ylim(BOT - .03, TOP + .07)
f.tight_layout(); f.savefig(fig); print(pd.DataFrame(summ).query("encoder=='R0_dinov2s'")[["condition", "selected", "dist_mean", "dist_std", "dmin_mean", "dmin_std"]].round(3))
print(pd.DataFrame(disp).pivot(index="encoder", columns="condition", values="disp_mean").round(3))
