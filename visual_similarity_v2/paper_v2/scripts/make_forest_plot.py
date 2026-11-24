"""Forest plot of the adult results by encoder (v2b manuscript), generated only from frozen files at a945ed2.
(a) positive-control distance slope beta_z(d) [95% CI], 8 within-domain items (results/poscontrol.csv, term zd);
(b) primary order-moderation slope difference Delta-hat [95% CI] (results/regions/regions_table.csv).
Rows ordered by beta_z(d), most negative first. Usage: python make_forest_plot.py <repo checkout at a945ed2>
"""
import io, subprocess, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5})
import matplotlib.pyplot as plt

REPO = Path(sys.argv[1]); COMMIT = "a945ed2"
EE = "visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence/"
head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout
assert head.startswith(COMMIT), head
def csv(rel): return pd.read_csv(io.BytesIO(subprocess.run(["git", "-C", str(REPO), "show", f"{COMMIT}:{rel}"], capture_output=True, check=True).stdout))
NAME = {"R0_dinov2s": "DINOv2-small", "O1_dinov2b": "DINOv2-base", "R1_cvcl": "CVCL", "E1_siglip": "SigLIP", "O2_openclip": "OpenCLIP",
        "E2_vitsup": "Supervised ViT", "E3_rn50sup": "Supervised ResNet-50", "E4_mae": "MAE", "F1_rn50rand0": "Random ResNet-50", "F2_pixels": "Pixels"}
ROLE = {e: ("control" if e[:2] in ("F1", "F2") else ("reference" if e == "R0_dinov2s" else "trained")) for e in NAME}
COL = {"reference": "#1f4e9c", "trained": "#333333", "control": "#c0571b"}
MRK = {"reference": "D", "trained": "o", "control": "s"}

pc = csv(EE + "results/poscontrol.csv").query("term=='zd'").set_index("encoder").loc[list(NAME)]
rt = csv(EE + "results/regions/regions_table.csv").set_index("encoder").loc[list(NAME)]
assert (pc.nobs == 57600).all()
assert (pc["items"] == 8).all()
assert set(csv(EE + "results/stage2_primary.csv").query("encoder=='R0_dinov2s'").n_exp) == {12}
order = pc.sort_values("estimate").index.tolist()            # most negative slope first
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True, gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.08})
ys = list(range(len(order)))[::-1]
for y, e in zip(ys, order):
    r, q, ro = pc.loc[e], rt.loc[e], ROLE[e]
    a1.errorbar(r.estimate, y, xerr=[[r.estimate - r.zd_lo], [r.zd_hi - r.estimate]], fmt=MRK[ro], color=COL[ro], ms=4.5, capsize=2, lw=1)
    a2.errorbar(q.delta_hat, y, xerr=[[q.delta_hat - q.ci95_lo], [q.ci95_hi - q.delta_hat]], fmt=MRK[ro], color=COL[ro], ms=4.5, capsize=2, lw=1)
import os
MARK = os.environ.get("FOREST_MARK", " (a)")   # v2c passes a superscript a (DeepReview: "(a)" collided with the panel label)
labels = [NAME[e] + (MARK if pc.loc[e, "model"] != "poscontrol_slope" else "") for e in order]
a1.set_yticks(ys); a1.set_yticklabels(labels)
for t, e in zip(a1.get_yticklabels(), order): t.set_color(COL[ROLE[e]])
for a in (a1, a2): a.axvline(0, color="0.6", lw=0.8, ls="--"); a.grid(axis="x", color="0.92", lw=0.6); a.set_axisbelow(True)
a1.set_xlabel(r"(a) Distance slope $\beta_{z(d)}$ [95% CI]" "\npositive control, 8 within-domain items")
a2.set_xlabel(r"(b) Order moderation $\hat{\Delta}$ [95% CI]" "\nprimary test, 12 experiments")
for d in (-0.20, -0.10, -0.05, 0.05, 0.10, 0.20): a2.axvline(d, color="0.85", lw=0.6, ls=":")
out = Path(__file__).resolve().parents[1] / "generated" / os.environ.get("FOREST_OUT", "fig_forest_adults.pdf")
fig.savefig(out, bbox_inches="tight"); print("wrote", out, "order:", order)
