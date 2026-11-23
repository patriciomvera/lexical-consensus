"""SYNTHETIC unit tests of 05b_select_structure.py (stage-1 structure chosen ONCE for all encoders, by convergence only)."""
import importlib.util, pathlib, unittest, numpy as np, pandas as pd
S = pathlib.Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("sel", S / "05b_select_structure.py"); SEL = importlib.util.module_from_spec(sp); sp.loader.exec_module(SEL)
ORD = {e: ("1-3" if e in (1, 2, 5, 7, 10, 12) else "3-1") for e in range(1, 13)}
def pass1(first, beta_seed=0):
    """first[(enc, exp)] = first converging rung (None = none); pass-1 rows: failed rungs before it, converged at it."""
    rng = np.random.default_rng(beta_seed); rows = []
    for e in SEL.SELECTION_SET:
        for x in range(1, 13):
            f = first.get((e, x), "L1")
            for rg in SEL.RUNGS:
                ok = rg == f; rows.append(dict(encoder=e, items=8, variant="primary", suffix="", exp=x, order=ORD[x], rung=rg, converged=ok, beta=rng.normal(), se=.1))
                if ok: break
    return pd.DataFrame(rows)
def fixed(df, enc, rung, conv_fail=()):
    rows = [dict(encoder=enc, items=8, variant="primary", suffix="", exp=x, order=ORD[x], rung=rung, converged=x not in conv_fail, beta=0.0, se=.1) for x in range(1, 13)]
    return pd.concat([df, pd.DataFrame(rows)])
class T(unittest.TestCase):
    def test_all_L1(self):
        out, needs = SEL.select(pass1({})); self.assertEqual(needs, []); self.assertEqual(out[8]["common"], "L1")
    def test_one_hard_encoder_moves_everyone(self):
        """F2 converges only from L2 in exp 3: pass 1 asks for F2 at L2, pass 2 for every other encoder at L2; common = L2 for all,
        own rung stays L1 for the others (descriptive), S9 mixed = L1 everywhere except F2 exp 3."""
        df = pass1({("F2_pixels", 3): "L2"}); asked = []
        for it in range(5):
            out, needs = SEL.select(df)
            if not needs: break
            asked.append(needs)
            for e, _, rg in needs: df = fixed(df, e, rg)
        self.assertEqual(asked[0], [("F2_pixels", 8, "L2")]); self.assertEqual({n[0] for n in asked[1]}, set(SEL.SELECTION_SET) - {"F2_pixels"})
        self.assertEqual(out[8]["common"], "L2"); self.assertEqual(out[8]["own"]["R0_dinov2s"], "L1"); self.assertEqual(out[8]["own"]["F2_pixels"], "L2")
    def test_encoder_without_any_rung_is_left_out_and_still_evaluated(self):
        df = pass1({("E3_rn50sup", 7): "none"}); out, needs = SEL.select(df)
        self.assertEqual(out[8]["common"], "L1"); self.assertIn("E3_rn50sup", out[8]["excluded"]); self.assertIsNone(out[8]["own"]["E3_rn50sup"])
        rows = SEL.stage1_rows(out[8]["T"], "E3_rn50sup", 8, lambda x: "L1", "primary")
        self.assertEqual(sum(np.isnan(r["beta"]) for r in rows), 1)    # exp 7 missing -> inconcluso tecnico in stage 2
    def test_selection_ignores_beta(self):
        a, _ = SEL.select(pass1({("O2_openclip", 5): "L3"}, beta_seed=1)); b, _ = SEL.select(pass1({("O2_openclip", 5): "L3"}, beta_seed=99))
        self.assertEqual(a[8]["own"], b[8]["own"])
    def test_constant_and_no_option(self):
        self.assertEqual(SEL.STRUCTURE_SCOPE, "common"); src = (S / "05b_select_structure.py").read_text()
        self.assertNotIn("argparse", src); self.assertNotIn("--scope", src)
        self.assertNotIn("F0", SEL.SELECTION_SET); self.assertEqual(len(SEL.SELECTION_SET), 10)
    def test_REVISION3b_no_common_structure_stops(self):
        """Encoders converging only on disjoint rungs (the other 9: L1 is their only full rung, L2 fails in exp 4; F2: L2 is its only full
        rung, L1 fails in exp 4) -> no rung common to all: 05b exits 4, writes the DEVIATION file and a null common_rung, and writes NO stage-1 tables (no silent L4)."""
        import subprocess, tempfile, sys
        rows = []
        for e in SEL.SELECTION_SET:
            for x in range(1, 13):
                for rg in SEL.RUNGS:
                    if e == "F2_pixels": ok = (rg == "L2") if x == 4 else (rg in ("L1", "L2"))      # F2: L2 is its only full rung
                    else: ok = rg == "L1" or (rg == "L2" and x != 4)                               # others: L2 fails in exp 4 -> only L1
                    rows.append(dict(encoder=e, items=8, variant="primary", suffix="", exp=x, order=ORD[x], rung=rg, converged=ok, beta=0.0, se=.1))
        with tempfile.TemporaryDirectory(dir="/tmp") as d:
            f = pathlib.Path(d) / "s1.csv"; pd.DataFrame(rows).to_csv(f, index=False); pre = pathlib.Path(d) / "structure"
            r = subprocess.run([sys.executable, S / "05b_select_structure.py", pre, f], capture_output=True, text=True)
            self.assertEqual(r.returncode, 4, r.stdout + r.stderr); self.assertIn("DEVIATION", (pathlib.Path(d) / "structure_DEVIATION.txt").read_text())
            import json; self.assertIsNone(json.loads((pathlib.Path(d) / "structure_status.json").read_text())["8"]["common_rung"])
            self.assertFalse((pathlib.Path(d) / "structure_primary.csv").exists())
    def test_REVISION3b_all_encoders_left_out_is_no_structure(self):
        out, needs = SEL.select(pass1({(e, 1): "none" for e in SEL.SELECTION_SET})); self.assertEqual(needs, []); self.assertIsNone(out[8]["common"])
    def test_REVISION3b_rung8_stops_without_common_structure(self):
        """run_weekend.sh rung8: a null common_rung -> non-zero exit and a DEVIATION line in the log; never prints L4."""
        import subprocess, tempfile, json
        with tempfile.TemporaryDirectory(prefix="synth_rung8_", dir="/tmp") as d:
            res = pathlib.Path(d) / "w_results"; res.mkdir(); env = dict(__import__("os").environ, WEEKEND_SYNTH_DIR=d)
            (res / "structure_status.json").write_text(json.dumps({"8": {"common_rung": None}}))
            r = subprocess.run(["bash", S / "run_weekend.sh", "rung8"], capture_output=True, text=True, env=env)
            self.assertNotEqual(r.returncode, 0); self.assertNotIn("L4", r.stdout); self.assertIn("DEVIATION", (pathlib.Path(d) / "w_logs/run_weekend.log").read_text())
            (res / "structure_status.json").write_text(json.dumps({"8": {"common_rung": "L2"}}))
            r = subprocess.run(["bash", S / "run_weekend.sh", "rung8"], capture_output=True, text=True, env=env)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, "L2"))
if __name__ == "__main__": unittest.main(verbosity=2)
