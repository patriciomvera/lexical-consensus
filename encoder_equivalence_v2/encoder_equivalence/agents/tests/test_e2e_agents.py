"""Checks the PLANTED effects after `run_agents.sh primary` on the synthetic world /tmp/synth_agents_e2e (make_synth_agents.py):
controls PASS, the DEAD encoder (E4_mae) fails the agents' positive control and becomes 'no interpretable', good encoders recover
H1 and H4, floors give rho ~ 0 and H4 ~ 0, D-todos and the regions/GA outputs exist; plus stats identity checks."""
import os, sys, json, pathlib, numpy as np, pandas as pd
T = pathlib.Path(os.environ.get("AGENTS_SYNTH_DIR", "/tmp/synth_agents_e2e")); os.environ["AGENTS_SYNTH_DIR"] = str(T)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts")); import lib_agents as L, importlib
R = T / "out/results"; rd = lambda n: json.loads((R / n).read_text())
GOOD = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup"]; DEAD = "E4_mae"; FLOORS = ["F1_rn50rand0", "F2_pixels"]
c1 = rd("control_T_exact.json"); assert c1["ok"] and c1["estricto"] and c1["propuesto_b1"]
c2 = rd("control_c100_recode.json"); assert c2["verdict"] == "PASA" and c2["max_abs_diff"] < 1e-4
print(f"[e2e] control 1: b1 and strict PASS (5,040 rows); control 2: PASA (max|Δ| {c2['max_abs_diff']:.1e}, planted 'GPU' noise 2e-6)")
for e in GOOD + FLOORS: assert rd(f"T_{e}.json")["positive_control"]["pass_"], e
assert not rd(f"T_{DEAD}.json")["positive_control"]["pass_"]; print("[e2e] positive control: 9 pass, planted dead encoder E4_mae fails")
for e in GOOD:
    h = rd(f"H_{e}.json")["H1"]
    for s in ("THINGS", "HSJ", "CIFAR10H"): assert h[s]["rho"] > 0.2 and h[s]["mantel_p"] < .05, (e, s, h[s]["rho"])
for e in FLOORS:
    h = rd(f"H_{e}.json")["H1"]
    for s in ("THINGS", "HSJ"): assert abs(h[s]["rho"]) < 0.15, (e, s, h[s]["rho"])
print("[e2e] H1: planted structure recovered in the 7 good encoders (rho > .2, Mantel p < .05); floors |rho| < .15 on 4,950 pairs")
h4 = pd.read_csv(R / "h4_by_encoder.csv").set_index("encoder"); assert (h4.status == "ok").all()
for e in GOOD: assert h4.logOR[e] < 0 and h4.p[e] < .05, e
for e in FLOORS: assert h4.lo[e] < 0 < h4.hi[e], e
print(f"[e2e] H4 (172,800 rows each): good logOR < 0, p < .05; floors' CI contains 0; fits {h4.seconds.min():.0f}-{h4.seconds.max():.0f} s; RE {dict(h4.RE.value_counts())}")
for e in GOOD + FLOORS + [DEAD]: assert rd(f"Dall_{e}.json")["status"] == "ok" and rd(f"D_{e}.json")["status"] == "ok"
assert abs(rd(f"D_{DEAD}.json")["test"]["beta_v1"]) < 0.05; print("[e2e] D and D-todos ok for 10 encoders; dead encoder beta_perc ~ 0")
reg = pd.read_csv(R / "regions/regions_agents.csv"); g = pd.read_csv(R / "regions/ga_grid.csv")
assert set(reg[reg.encoder == DEAD]["region_0.20"]) == {"no interpretable (control positivo de agentes)"}
s = rd("regions/agents_summary.json"); assert "E4_mae" in s["families"]["C"]["excluidos"] and s["controls"]["ok"]
assert (R / "regions/agents_regions_report.md").exists() and (R / "regions/tabla_agentes_personas.csv").exists()
assert not set(g.conclusion) & {"GA0", "GA2", "GA2b"}, set(g.conclusion)   # dead trained encoder -> never GA2/GA2b; controls ok -> never GA0
print("[e2e] regions: E4_mae 'no interpretable' and out of family C; GA cells:", dict(g.conclusion.value_counts()))
# stats identity: 12's spearman_boot_dist == stats_lib.spearman_boot
H = importlib.import_module("12_agent_h1h2h3"); rng = np.random.default_rng(3); x = rng.random(200); y = x + rng.normal(size=200)
a = H.SL.spearman_boot(x, y); b = H.spearman_boot_dist(x, y)[:3]; assert np.allclose(a, b, rtol=0, atol=0), (a, b)
print("[e2e] spearman_boot_dist identical to stats_lib.spearman_boot")
print("ALL E2E CHECKS PASSED")
