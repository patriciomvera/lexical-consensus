"""Holm over the 6 confirmatory tests (H1 x3 Mantel, H2 x2 beta_hum, H4 zd). H4 p from out/h4_glmm.txt (R prints p=0: < 1e-300)."""
import json, re, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from stats_lib import holm
B = pathlib.Path(__file__).resolve().parents[1]; R = json.load(open(B / "out/h1_h2_results.json"))
t = open(B / "out/h4_glmm.txt").read(); m = re.search(r"m1 zd: logOR=(\S+) CI\[(\S+), (\S+)\] p=(\S+)", t)
tests = {f"H1_{k}": R["H1"][k]["mantel_p"] for k in ("THINGS", "HSJ", "CIFAR10H")}
tests.update({"H2_THINGS": R["H2"]["THINGS"]["d_things"]["p"], "H2_HSJ": R["H2"]["HSJ"]["d_hsj"]["p"], "H4": float(m.group(4))})
adj = holm(list(tests.values())); out = {k: dict(p=p, p_holm=float(a)) for (k, p), a in zip(tests.items(), adj)}
out["H4_est"] = dict(logOR=float(m.group(1)), lo=float(m.group(2)), hi=float(m.group(3)))
json.dump(out, open(B / "out/holm_confirmatory.json", "w"), indent=1); print(json.dumps(out, indent=1))
