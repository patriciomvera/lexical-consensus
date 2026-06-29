"""
generate_figures.py — Phase 4, Step 10.

Produces the five pre-registered figures:

    fig_scatter_perc_vs_c1.png  — d_perc vs C1, colored by quadrant
    fig_scatter_sem_vs_c1.png   — d_sem  vs C1, colored by quadrant
    fig_partial_regression.png  — partial regression plots (perc and sem)
    fig_quadrant_boxplot.png    — C1 by quadrant (KEY FIGURE for decision)
    fig_heatmap_overlay.png     — d_perc (lower) / d_sem (upper) matrix

All figures are for the centroid learner unless otherwise noted.
The quadrant boxplot shows all 4 learners as separate panels.

Usage:
    python src/phase4/generate_figures.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import CIFAR100_CLASSES, get_run_dir

QUADRANT_COLORS = {
    "Q1": "#2196F3",   # blue — concordant-near
    "Q2": "#F44336",   # red  — concordant-far
    "Q3": "#FF9800",   # orange — dissociation-A
    "Q4": "#4CAF50",   # green  — dissociation-B
}
QUADRANT_LABELS = {
    "Q1": "Q1 concordant-near",
    "Q2": "Q2 concordant-far",
    "Q3": "Q3 perc-far, sem-near",
    "Q4": "Q4 perc-near, sem-far",
}
PRIMARY_LEARNER = "centroid"
LEARNER_NAMES   = ["centroid", "exemplar_knn_k3", "logreg", "random"]


def _scatter_vs_c1(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    out_path: Path,
    reg_params: dict | None = None,
) -> None:
    """d_X vs C1 scatter, colored by quadrant, with optional regression line."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for q, color in QUADRANT_COLORS.items():
        sub = df[df["quadrant"] == q]
        ax.scatter(sub[x_col], sub["c1_accuracy"],
                   color=color, alpha=0.65, s=28, label=QUADRANT_LABELS[q], zorder=3)

    if reg_params and not any(np.isnan(v) for v in reg_params.values()):
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        y_fit   = reg_params["intercept"] + reg_params["slope"] * x_range
        ax.plot(x_range, y_fit, "k--", linewidth=1.5, label="regression line", zorder=4)

    ax.axhline(1/3, color="grey", linestyle=":", linewidth=1, label="chance (1/3)")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("C1 accuracy (centroid learner)", fontsize=11)
    ax.set_title(f"{x_label} vs C1", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _partial_regression_plot(
    y: np.ndarray,
    d_perc: np.ndarray,
    d_sem: np.ndarray,
    out_path: Path,
) -> None:
    """Two-panel partial regression: y on d_perc | d_sem, y on d_sem | d_perc."""
    import statsmodels.api as sm

    X_both = sm.add_constant(np.column_stack([d_perc, d_sem]))
    model  = sm.OLS(y, X_both).fit()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, i, xlabel, title in [
        (axes[0], 1, "d_perc (residualized)", "Partial regression: d_perc → C1 | d_sem"),
        (axes[1], 2, "d_sem  (residualized)", "Partial regression: d_sem  → C1 | d_perc"),
    ]:
        # Residual of y regressed on the OTHER predictor
        other_idx = 2 if i == 1 else 1
        X_other = sm.add_constant(X_both[:, [0, other_idx]])
        resid_y = sm.OLS(y, X_other).fit().resid
        # Residual of predictor i regressed on the other predictor
        resid_x = sm.OLS(X_both[:, i], X_other).fit().resid

        ax.scatter(resid_x, resid_y, alpha=0.5, s=20, color="#555555")
        m_pr = sm.OLS(resid_y, sm.add_constant(resid_x)).fit()
        xr = np.linspace(resid_x.min(), resid_x.max(), 100)
        ax.plot(xr, m_pr.params[0] + m_pr.params[1] * xr, "r-", linewidth=2)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("C1 (residualized)", fontsize=10)
        ax.set_title(f"{title}\nβ={m_pr.params[1]:.4f}, p={m_pr.pvalues[1]:.3f}", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _quadrant_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    """4-panel boxplot: one panel per learner, x=quadrant, y=C1."""
    fig, axes = plt.subplots(1, len(LEARNER_NAMES), figsize=(14, 5), sharey=True)
    for ax, lname in zip(axes, LEARNER_NAMES):
        sub = df[df["learner"] == lname]
        data_by_q = [sub[sub["quadrant"] == q]["c1_accuracy"].values for q in ["Q1", "Q2", "Q3", "Q4"]]
        bp = ax.boxplot(data_by_q, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, q in zip(bp["boxes"], ["Q1", "Q2", "Q3", "Q4"]):
            patch.set_facecolor(QUADRANT_COLORS[q])
            patch.set_alpha(0.75)
        ax.axhline(1/3, color="grey", linestyle=":", linewidth=1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["Q1\nboth\nnear", "Q2\nboth\nfar",
                             "Q3\nperc-far\nsem-near", "Q4\nperc-near\nsem-far"],
                           fontsize=8)
        ax.set_title(lname, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("C1 accuracy", fontsize=11)

    fig.suptitle("C1 accuracy by quadrant (pre-registered key figure)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _heatmap_overlay(run_dir: Path, out_path: Path) -> None:
    """Lower-triangle = d_perc, upper-triangle = d_sem; sampled pairs highlighted."""
    perc_df = pd.read_csv(run_dir / "phase1" / "perceptual_distance_matrix.csv", index_col=0)
    sem_df  = pd.read_csv(run_dir / "phase1" / "semantic_distance_matrix.csv",  index_col=0)
    pairs_df = pd.read_csv(run_dir / "phase2" / "sampled_pairs.csv")

    d_perc = perc_df.values
    d_sem  = sem_df.values

    combined = np.zeros_like(d_perc)
    tri_lower = np.tril_indices(100, k=-1)
    tri_upper = np.triu_indices(100, k=1)
    combined[tri_lower] = d_perc[tri_lower]
    combined[tri_upper] = d_sem[tri_upper]
    np.fill_diagonal(combined, 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(combined, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="distance (lower=d_perc, upper=d_sem)")

    # Highlight sampled pairs
    class_to_idx = {cls: i for i, cls in enumerate(CIFAR100_CLASSES)}
    for _, row in pairs_df.iterrows():
        i = class_to_idx.get(row["class_u"], -1)
        j = class_to_idx.get(row["class_v"], -1)
        if i >= 0 and j >= 0:
            color = QUADRANT_COLORS.get(row["quadrant"], "white")
            ax.plot(j, i, ".", color=color, markersize=4, alpha=0.8)
            ax.plot(i, j, ".", color=color, markersize=4, alpha=0.8)

    ax.set_title(
        "Distance matrix: d_perc (lower triangle) / d_sem (upper triangle)\n"
        "Dots = sampled pairs (colored by quadrant)",
        fontsize=10,
    )
    ax.set_xlabel("class index (alphabetical)", fontsize=9)
    ax.set_ylabel("class index (alphabetical)", fontsize=9)
    ax.tick_params(labelsize=6)

    legend_patches = [
        mpatches.Patch(color=QUADRANT_COLORS[q], label=QUADRANT_LABELS[q])
        for q in ["Q1", "Q2", "Q3", "Q4"]
    ]
    ax.legend(handles=legend_patches, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 4 — generate figures"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)
    fig_dir = run_dir / "phase4"

    agg_path = run_dir / "phase3" / "exp_008_c1_results.csv"
    reg_path = run_dir / "phase4" / "regression_table.csv"

    if not agg_path.exists():
        raise FileNotFoundError(f"Missing: {agg_path}")

    df = pd.read_csv(agg_path)
    centroid_df = df[df["learner"] == PRIMARY_LEARNER].copy()

    reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
    reg_row = {}
    if len(reg_df) > 0:
        r = reg_df[reg_df["learner"] == PRIMARY_LEARNER]
        if len(r) > 0:
            # For scatter overlay: use Model P and Model S slopes
            reg_row = r.iloc[0].to_dict()

    import statsmodels.api as sm

    # Figure 1: d_perc vs C1
    params_p = None
    if reg_row:
        m_p = sm.OLS(
            centroid_df["c1_accuracy"].values,
            sm.add_constant(centroid_df["d_perc"].values),
        ).fit()
        params_p = {"intercept": float(m_p.params[0]), "slope": float(m_p.params[1])}

    f1 = fig_dir / "fig_scatter_perc_vs_c1.png"
    _scatter_vs_c1(centroid_df, "d_perc", "d_perc (1 − cosine similarity)", f1, params_p)
    print(f"[figures] {f1.name}")

    # Figure 2: d_sem vs C1
    params_s = None
    if reg_row:
        m_s = sm.OLS(
            centroid_df["c1_accuracy"].values,
            sm.add_constant(centroid_df["d_sem"].values),
        ).fit()
        params_s = {"intercept": float(m_s.params[0]), "slope": float(m_s.params[1])}

    f2 = fig_dir / "fig_scatter_sem_vs_c1.png"
    _scatter_vs_c1(centroid_df, "d_sem", "d_sem (1 − Wu-Palmer)", f2, params_s)
    print(f"[figures] {f2.name}")

    # Figure 3: Partial regression
    f3 = fig_dir / "fig_partial_regression.png"
    _partial_regression_plot(
        centroid_df["c1_accuracy"].values.astype(np.float64),
        centroid_df["d_perc"].values.astype(np.float64),
        centroid_df["d_sem"].values.astype(np.float64),
        f3,
    )
    print(f"[figures] {f3.name}")

    # Figure 4: Quadrant boxplot (key figure — all 4 learners)
    f4 = fig_dir / "fig_quadrant_boxplot.png"
    _quadrant_boxplot(df, f4)
    print(f"[figures] {f4.name}")

    # Figure 5: Heatmap overlay
    f5 = fig_dir / "fig_heatmap_overlay.png"
    _heatmap_overlay(run_dir, f5)
    print(f"[figures] {f5.name}")

    print(f"\nNext step:\n  python src/phase4/generate_report.py")


if __name__ == "__main__":
    main()
