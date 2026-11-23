"""SYNTHETIC unit tests of 06_stage2_permutation.py with KNOWN Delta (no data read)."""
import itertools, importlib.util, pathlib, unittest, numpy as np
S = pathlib.Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("st2", S / "06_stage2_permutation.py"); st2 = importlib.util.module_from_spec(sp); sp.loader.exec_module(st2)
IS31 = np.array([e not in (1, 2, 5, 7, 10, 12) for e in range(1, 13)])  # real order labels, exps 1..12
def brute(y, is31, d0=0.0, kind="two"):
    y = y - d0 * is31; t = y[is31].mean() - y[~is31].mean(); c = 0
    for comb in itertools.combinations(range(12), 6):
        A = np.zeros(12, bool); A[list(comb)] = True; ts = y[A].mean() - y[~A].mean()
        c += (abs(ts) >= abs(t) - 1e-12) if kind == "two" else (ts >= t - 1e-12) if kind == "ge" else (ts <= t + 1e-12)
    return c / 924
class T(unittest.TestCase):
    def test_enumeration(self):
        self.assertEqual(st2.COMBOS.shape, (924, 12)); self.assertTrue((st2.COMBOS.sum(1) == 6).all())
        self.assertEqual(len({tuple(r) for r in st2.COMBOS}), 924)
    def test_known_delta_sign_and_p(self):
        rng = np.random.default_rng(1); base = rng.normal(-0.8, 0.05, 12); y = base + 0.30 * IS31   # 3-1 slopes higher by 0.30
        r = st2.analyse(y, IS31); true = y[IS31].mean() - y[~IS31].mean()
        self.assertAlmostEqual(r["delta_hat"], true, 12); self.assertGreater(r["delta_hat"], 0.2)   # sign: 3-1 minus 1-3
        self.assertAlmostEqual(r["p_two_sided"], brute(y, IS31), 12); self.assertAlmostEqual(r["p_two_sided"], 2 / 924, 12)  # complete separation
        self.assertAlmostEqual(r["p_opposite"], brute(y, IS31, kind="le"), 12); self.assertAlmostEqual(r["p_positive"], brute(y, IS31, kind="ge"), 12)
        self.assertTrue(r["ci95_lo"] <= 0.30 <= r["ci95_hi"] and r["ci95_lo"] > 0)
        self.assertTrue(r["ci95_contiguous"] and not r["ci95_at_grid_edge"] and not r["ci90_at_grid_edge"])
        self.assertTrue(r["ci95_lo"] <= r["ci90_lo"] <= r["ci90_hi"] <= r["ci95_hi"])
    def test_ci_is_inversion(self):
        rng = np.random.default_rng(3); y = rng.normal(0, .2, 12) + 0.1 * IS31; r = st2.analyse(y, IS31)
        for b, inside in ((r["ci95_lo"], True), (r["ci95_hi"], True), (r["ci95_lo"] - .002, False), (r["ci95_hi"] + .002, False)):
            self.assertEqual(brute(y, IS31, b) > .05, inside, b)
    def test_tost_is_max_of_shifted(self):
        rng = np.random.default_rng(5); y = rng.normal(-.5, .01, 12) + 0.0 * IS31; r = st2.analyse(y, IS31)
        for dl in st2.DELTAS:
            self.assertAlmostEqual(r[f"p_tost_{dl:.2f}"], max(brute(y, IS31, -dl, "ge"), brute(y, IS31, dl, "le")), 12)
        self.assertLess(r["p_tost_0.05"], .05)   # null effect, tiny noise: equivalence at every delta
    def test_welch(self):
        from scipy import stats
        rng = np.random.default_rng(7); y = rng.normal(0, .3, 12); r = st2.analyse(y, IS31)
        self.assertAlmostEqual(r["welch_p"], stats.ttest_ind(y[IS31], y[~IS31], equal_var=False).pvalue, 12)
if __name__ == "__main__": unittest.main(verbosity=2)
