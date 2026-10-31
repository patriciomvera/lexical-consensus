"""Holm over the 6 confirmatory tests (H1 x3 Mantel, H2 x2 beta_hum, H4 zd).
H4 from out/h4_person_re.txt, model C1 (RE by person; DESV §26/§31). R prints p=0 (underflow, < 1e-300)."""
import json, re, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import holm
B = pathlib.Path(__file__).resolve().parents[1]; R = json.load(open(B / "out/h1_h2_results.json"))
H4J = [json.loads(l) for l in open(B / "out/h4_person_re.txt") if l.startswith('{"model"')]
c1 = next(r for r in H4J if r["model"].startswith("C1 "))
tests = {f"H1_{k}": R["H1"][k]["mantel_p"] for k in ("THINGS", "HSJ", "CIFAR10H")}
tests.update({"H2_THINGS": R["H2"]["THINGS"]["d_things"]["p"], "H2_HSJ": R["H2"]["HSJ"]["d_hsj"]["p"], "H4": float(c1["p"])})
adj = holm(list(tests.values())); out = {k: dict(p=p, p_holm=float(a)) for (k, p), a in zip(tests.items(), adj)}
out["H4_source"] = "out/h4_person_re.txt: " + c1["model"]; out["H4_est"] = dict(logOR=c1["logOR"], lo=c1["lo"], hi=c1["hi"])
json.dump(out, open(B / "out/holm_confirmatory.json", "w"), indent=1); print(json.dumps(out, indent=1))
