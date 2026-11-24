"""B4 (post hoc, v2c): H2 estimates per representation from the frozen agent outputs, and the minimum detectable effect for H2.
Reads agents/out/results/H_<ID>.json at a945ed2 (no recomputation of any estimate). The minimum detectable partial correlation uses the
same Fisher-z approximation as the frozen intervals (se = 1/sqrt(n - 3 - k), k = 1 covariate, i.e. semantic distance; verified below
against the frozen interval of every encoder): two-sided alpha = .05, power .80. Deterministic (no random numbers).
Usage: python p04_h2_power.py <repo> <out_dir>"""
import sys, json, pathlib, math, numpy as np, pandas as pd
from scipy.stats import norm
repo, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
R = repo / "visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence/agents/out/results"
IDS = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
rows = []
for e in IDS:
    H = json.loads((R / f"H_{e}.json").read_text())
    for s in ("THINGS", "HSJ"):
        h = H["H2"][s]; n = h["n"]; se = 1 / math.sqrt(n - 4)
        lo, hi = math.tanh(math.atanh(h["r_partial"]) - norm.ppf(.975) * se), math.tanh(math.atanh(h["r_partial"]) + norm.ppf(.975) * se)
        assert abs(lo - h["r_ci_lo"]) < 5e-3 and abs(hi - h["r_ci_hi"]) < 5e-3, (e, s, lo, h["r_ci_lo"], hi, h["r_ci_hi"])  # frozen CI = Fisher z, n-4
        a = H["H2_Dall_desc"][s]
        rows.append(dict(encoder=e, source=s, n=n, beta=h["beta"], p=h["p"], r=h["r_partial"], r_lo=h["r_ci_lo"], r_hi=h["r_ci_hi"], p_sem=h["p_sem"],
                         n_all=a["n"], r_all=a["r_partial"], r_all_lo=a["r_ci_lo"], r_all_hi=a["r_ci_hi"]))
df = pd.DataFrame(rows); df.to_csv(out / "p04_h2_estimates.csv", index=False)
mde = []
for n in sorted(set(df.n)):
    se = 1 / math.sqrt(n - 4)
    mde.append(dict(n=n, mde_r_80=math.tanh((norm.ppf(.975) + norm.ppf(.80)) * se), halfwidth_at_r0=math.tanh(norm.ppf(.975) * se),
                    mde_r_80_alpha_holm36=math.tanh((norm.ppf(1 - .05 / 36 / 2) + norm.ppf(.80)) * se)))
pd.DataFrame(mde).to_csv(out / "p04_h2_mde.csv", index=False); print(df.round(3).to_string()); print(pd.DataFrame(mde).round(3))
