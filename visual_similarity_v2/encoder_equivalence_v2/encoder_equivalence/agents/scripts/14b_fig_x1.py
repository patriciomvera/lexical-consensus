"""AGENTS addendum: Fig. x1 per encoder (DESCRIPTIVE; proposal §2, v1 main.tex l. 157): people (proportion selected vs distance,
8 target-category items, 10 equal-count bins of d of that encoder) next to the learner (centroid C1 vs d_perc of that encoder,
100 v1 pairs of D). No fitting. Reads the frozen human per-trial tables h4_long_<ID>.csv and agents/out/sims/D_<ID>_c1.csv.
Output: agents/out/results/regions/fig_x1_por_encoder.png"""
import sys, pathlib, numpy as np, pandas as pd, matplotlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ids = [e for e in L.P10 if (L.H_LONG / f"h4_long_{e}.csv").exists() and (L.SIM_DIR / f"D_{e}_c1.csv").exists()]
fig, axs = plt.subplots(len(ids), 2, figsize=(6, 1.8 * max(len(ids), 1)), squeeze=False)
for i, e in enumerate(ids):
    h = pd.read_csv(L.H_LONG / f"h4_long_{e}.csv", usecols=["in_target", "dist", "selected"]); h = h[h.in_target == 1]
    h["bin"] = pd.qcut(h.dist.rank(method="first"), 10, labels=False); b = h.groupby("bin").agg(d=("dist", "mean"), sel=("selected", "mean"))
    axs[i, 0].plot(b.d, b.sel, "o-", c="0.2", ms=3); axs[i, 0].set_ylabel(e, fontsize=7); axs[i, 0].set_xlabel("d (encoder), personas", fontsize=7)
    a = pd.read_csv(L.SIM_DIR / f"D_{e}_c1.csv"); a = a[a.learner == "centroid"]
    axs[i, 1].scatter(a.d_perc, a.c1_accuracy, s=6, c="0.2", lw=0); axs[i, 1].set_xlabel("d_perc (encoder), aprendiz", fontsize=7)
fig.text(0.99, 0.0, "descriptivo", ha="right", fontsize=6, style="italic"); fig.tight_layout()
o = L.RES_DIR / "regions"; o.mkdir(parents=True, exist_ok=True); fig.savefig(o / "fig_x1_por_encoder.png", dpi=200); print("wrote", o / "fig_x1_por_encoder.png")
