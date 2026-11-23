"""§3.5 pipeline controls on SYNTHETIC inputs: control 1 (T ledger vs v1 ledger; strict and b1 readings, perturbed ledgers) and
control 2 (R0 CPU re-encoding of CIFAR-100 vs the v1 matrix; PASA / PASA con desviación registrada / FALLA, thresholds of REVISION4 (5))."""
import os, sys, json, copy, shutil, pathlib, importlib, numpy as np, pandas as pd
T = pathlib.Path("/tmp/synth_agents_ctrl"); shutil.rmtree(T, ignore_errors=True); os.environ["AGENTS_SYNTH_DIR"] = str(T)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts")); import lib_agents as L; L.ensure_dirs()
S = importlib.import_module("11_agent_sims")
# ---- control 1
rng = np.random.default_rng(1); base = []
for ln in S.V1_LEARNERS_T:
    for ep in range(3):
        for pt in ["std_nway", "pool_b"]:
            base.append(dict(sub="x", tier="near", concept_type="disj", learner=ln, ways=3, shots=5, episode_id=ep, pool_type=pt, chance=1 / 3,
                             c1=float(rng.random()), c2=float(rng.random()), per_concept_c1={"a": 0.5}, margin=round(float(rng.random()), 6)))
assert any(r["learner"] == "random" for r in base), S.V1_LEARNERS_T
def mod(f):
    n = copy.deepcopy(base); f(n); return S.compare_ledgers(n, base)
c = S.compare_ledgers(copy.deepcopy(base), base); assert c["estricto"] and c["propuesto_b1"]
def m1(n): n[0]["margin"] += 5e-7
c = mod(m1); assert (not c["estricto"]) and c["propuesto_b1"]
def m2(n): n[0]["margin"] += 3e-6
c = mod(m2); assert not c["propuesto_b1"]
def m3(n): n[1]["c1"] += 1e-9
c = mod(m3); assert not c["propuesto_b1"] and c["nonrandom_mismatches"] == {"c1": 1}
def m4(n): [r.update(c1=r["c1"] + 0.01, margin=r["margin"] + 0.1) for r in n if r["learner"] == "random"]
c = mod(m4); assert (not c["estricto"]) and c["propuesto_b1"] and c["random_mean_c1_new"] != c["random_mean_c1_v1"]
def m5(n): n[2]["episode_id"] = 99
c = mod(m5); assert not c["propuesto_b1"] and not c["keys_identical"]
c = S.compare_ledgers(base[:-1], base); assert not c["propuesto_b1"] and not c["estricto"]
def m6(n): n[3]["per_concept_c1"] = {"a": 0.5000001}
c = mod(m6); assert not c["propuesto_b1"]
print("[control 1] identical -> both PASS; margin +5e-7 -> b1 PASS / strict FAIL; margin +3e-6, c1, per_concept, key or row-count change -> FAIL; random learner changed -> b1 PASS (reported)")
# control_T writes the file and applies CONTROL_T_RULE
(L.V1_LEDGER).parent.mkdir(parents=True, exist_ok=True); open(L.V1_LEDGER, "w").write("\n".join(json.dumps(r) for r in base))
n = copy.deepcopy(base); m1(n); S.CONTROL_T_RULE = "estricto"; assert S.control_T(n)["ok"] is False; S.CONTROL_T_RULE = "propuesto_b1"; assert S.control_T(n)["ok"] is True
assert json.loads((L.RES_DIR / "control_T_exact.json").read_text())["rule_used"] == "propuesto_b1"
print("[control 1] control_T honours CONTROL_T_RULE (estricto / propuesto_b1)")
# ---- control 2: three levels
names = sorted(f"c{i:02d}" for i in range(100)); X = rng.random((100, 8)); P = np.linalg.norm(X[:, None] - X[None], axis=2); P = P / P.max()
L.V1_PERC.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(P, index=names, columns=names).to_csv(L.V1_PERC)
L.v1_beta_perc = lambda: -0.2119
def ctrl(noise, beta):
    E = rng.normal(scale=1.0, size=P.shape); E = (E + E.T) / 2; E = E / np.abs(E).max() * noise; np.fill_diagonal(E, 0)
    S.dperc_c100 = lambda ID: (P + E, names); return S.control_c100({"test": {"beta_v1": beta}})
r = ctrl(5e-5, -0.30); assert r["verdict"] == "PASA", r          # beta is not part of PASA
r = ctrl(1e-3, -0.2119 + 0.004); assert r["verdict"] == "PASA con desviación registrada" and r["ok"], r
r = ctrl(1e-3, -0.2119 + 0.006); assert r["verdict"] == "FALLA" and not r["ok"], r
r = ctrl(3e-3, -0.2119); assert r["verdict"] == "FALLA", r
r = ctrl(1e-3, float("nan")); assert r["verdict"] == "FALLA", r
print("[control 2] max|Δ| 5e-5 -> PASA; 1e-3 & |Δβ| .004 -> PASA con desviación; |Δβ| .006 -> FALLA; max|Δ| 3e-3 -> FALLA; β missing -> FALLA")
print("ALL CONTROL TESTS PASSED")
