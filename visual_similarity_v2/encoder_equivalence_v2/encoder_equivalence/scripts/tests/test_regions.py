"""SYNTHETIC ground-truth unit tests of 08_regions.py and of the grouping in 02_equivalence.py (no data read).
Covers: every region reachable; the two worked examples of plan §7.1; BC1 (encoder failing the positive control: own region,
out of the Holm family, G3 impossible -> G4; R0 failing -> stop as G0); Holm; G0-G4 precedence; floors; delta x cut grid with
calibration passing at one cut and failing at another; no option to choose a delta or a cut."""
import sys, json, importlib.util, pathlib, unittest, numpy as np, pandas as pd
S = pathlib.Path(__file__).resolve().parents[1]
def load(name, f):
    sp = importlib.util.spec_from_file_location(name, S / f); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
R = load("reg", "08_regions.py"); argv = sys.argv; sys.argv = [argv[0]]; EQ = load("eq", "02_equivalence.py"); sys.argv = argv
def row(D, c95, c90, popp=.5, ptost=(.5, .5, .5), enc="X", items=8):
    return dict(encoder=enc, items=items, variant="primary", suffix="", status="ok", delta_hat=D, ci95_lo=c95[0], ci95_hi=c95[1], ci90_lo=c90[0], ci90_hi=c90[1],
                p_opposite=popp, **{"p_tost_0.05": ptost[0], "p_tost_0.10": ptost[1], "p_tost_0.20": ptost[2]}, welch_ci95_lo=np.nan, welch_ci95_hi=np.nan)
HOLDS = dict(D=0.35, c95=(0.25, 0.45), c90=(0.27, 0.43))          # se mantiene at every delta
def grouping(disc_by_cut, ngroups_by_cut, labels=None):
    lab = labels or {e: (1 if e in R.TRAINED[:4] else 2 if e in R.TRAINED else 3) for e in R.PRIMARY}
    return dict(profile="test", cuts={c: dict(labels=lab, calibration=dict(discriminates=disc_by_cut[c]), n_trained_groups=ngroups_by_cut[c]) for c in R.CUTS})
GOOD = grouping({c: True for c in R.CUTS}, {c: 2 for c in R.CUTS})
def s2(rows): return pd.DataFrame(rows)
class Regions(unittest.TestCase):
    def test_each_region_reachable(self):
        cases = {R.R1_: row(-.3, (-.5, -.1), (-.45, -.15)), R.R2_: row(.01, (-.03, .05), (-.02, .04)), R.R3_: row(**HOLDS),
                 R.R4_: row(.08, (.01, .15), (.02, .14)), R.R5_: row(.05, (-.2, .3), (-.15, .25))}
        for want, r in cases.items(): self.assertEqual(R.region8(r, .10, True, False), want)
        self.assertEqual(R.region8(dict(status="inconcluso_tecnico"), .10, True, False), R.R5T)
        self.assertEqual(R.region8(row(**HOLDS), .10, False, False), R.NI)            # BC1 precedes everything
        self.assertEqual(R.region8(row(**HOLDS), .10, False, True), R.R3_)            # floors: control is not a gate
        self.assertEqual(R.region24(row(.3, (.1, .5), (.15, .45)), True, False), R.POS24); self.assertEqual(R.region24(row(.3, (.1, .5), (.15, .45)), False, False), R.NI)
    def test_example1_section71(self):   # Delta = 0.05, CI95 [0.01, 0.09]
        r = row(.05, (.01, .09), (.017, .083)); g = {d: R.region8(r, d, True, False) for d in R.DELTAS}
        self.assertEqual(g, {.05: R.R3_, .10: R.R2_, .20: R.R2_}); self.assertTrue(R.profile(g).startswith("dependiente del margen"))
    def test_example2_section71(self):   # Delta = 0.15, CI95 [0.02, 0.28]; CI90 not inside +-0.20
        r = row(.15, (.02, .28), (.04, .26)); g = {d: R.region8(r, d, True, False) for d in R.DELTAS}
        self.assertEqual(g, {.05: R.R3_, .10: R.R3_, .20: R.R4_})
        self.assertEqual(R.profile(g), "dependiente del margen: se mantiene (δ = 0.05, 0.10); atenuado (δ = 0.20)")
    def test_holm(self):
        np.testing.assert_allclose(R.holm([.01, .04, .03, .005]), [.03, .06, .06, .02])
class Global(unittest.TestCase):
    def base(self, **over):
        rows = [row(**HOLDS, enc=e) for e in R.PRIMARY]; pc = {e: True for e in R.PRIMARY}
        return rows, pc
    def run_(self, rows, pc, grp=GOOD, ok=True): return R.analyse(s2(rows), pc, ok, grp)
    def test_G3_all_hold(self):
        tab, grid, fl, gp = self.run_(*self.base()); self.assertEqual(set(grid.conclusion), {"G3"}); self.assertEqual(gp, "robusta: G3")
        self.assertEqual(set(fl.lectura), {"la moderación no requiere una geometría aprendida"})
    def test_G0_pipeline(self):
        tab, grid, fl, gp = self.run_(*self.base(), ok=False); self.assertEqual(gp, "robusta: G0")
    def test_BC1_R0_fails_control_stops(self):
        rows, pc = self.base(); pc["R0_dinov2s"] = False; tab, grid, fl, gp = self.run_(rows, pc)
        self.assertEqual(gp, "robusta: G0"); self.assertTrue(all("R0 falla el control positivo" in d for d in grid.detalle))
    def test_REVISION3b_R0_technical_is_G0_not_G1(self):
        """R0 'inconcluso (técnico)' with 8 items -> G0 'sin lectura: técnico' in all 9 cells of the grid, never G1 (even if the others hold)."""
        rows, pc = self.base(); rows[0] = dict(encoder="R0_dinov2s", items=8, variant="primary", suffix="", status="inconcluso_tecnico")
        tab, grid, fl, gp = self.run_(rows, pc)
        self.assertEqual(set(grid.conclusion), {"G0"}); self.assertNotIn("G1", set(grid.conclusion)); self.assertEqual(gp, "robusta: G0")
        self.assertTrue(all(d == R.TECH0 for d in grid.detalle)); self.assertTrue(all(tab.set_index("encoder").loc["R0_dinov2s", f"region_d{d:.2f}"] == R.R5T for d in R.DELTAS))
        # same with the other encoders in mixed regions: still G0
        rows[1] = row(-.3, (-.5, -.1), (-.45, -.15), popp=.001, enc="O1_dinov2b"); tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(gp, "robusta: G0")
    def test_REVISION3b_R0_technical_at_24_only(self):
        """R0 technical only with 24 items: the 8-item grid is unchanged (G3) and the 24-item R0 reading is 'sin lectura: técnico'."""
        rows, pc = self.base(); rows.append(dict(encoder="R0_dinov2s", items=24, variant="primary", suffix="", status="inconcluso_tecnico"))
        tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(gp, "robusta: G3"); self.assertEqual(R.analyse.r0_24, R.TECH0)
        rows[-1] = dict(row(**HOLDS, enc="R0_dinov2s", items=24)); tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(R.analyse.r0_24, R.POS24)
    def test_BC1_trained_fails_control(self):
        rows, pc = self.base(); pc["E2_vitsup"] = False
        rows[R.PRIMARY.index("O1_dinov2b")] = row(.0, (-.02, .02), (-.015, .015), popp=.5, ptost=(.002, .002, .002), enc="O1_dinov2b")
        tab, grid, fl, gp = self.run_(rows, pc); t = tab.set_index("encoder")
        self.assertTrue(all(t.loc["E2_vitsup", f"region_d{d:.2f}"] == R.NI for d in R.DELTAS)); self.assertEqual(t.loc["E2_vitsup", "region_24"], R.NI)
        self.assertTrue(t.loc["E2_vitsup", ["holm_tost_d0.10", "holm_opposite_d0.10"]].isna().all())   # out of the Holm family
        self.assertEqual(set(grid.trained_family_size), {12})                                         # 2k with k = 6
        self.assertAlmostEqual(t.loc["O1_dinov2b", "holm_tost_d0.10"], .002 * 12)                     # smallest of 12 tests
        self.assertEqual(set(grid.conclusion), {"G2"})                                                # O1 "menor que delta" survives Holm
        rows[R.PRIMARY.index("O1_dinov2b")] = row(**HOLDS, enc="O1_dinov2b"); tab, grid, fl, gp = self.run_(rows, pc)
        self.assertEqual(gp, "robusta: G4"); self.assertTrue(all("no interpretable" in d for d in grid.detalle))   # G3 impossible -> G4
    def test_holm_family_counts(self):   # REVISION3 (2)/(3): k and excluded encoders shown
        rows, pc = self.base(); pc["E2_vitsup"] = False; rows[R.PRIMARY.index("E3_rn50sup")] = dict(encoder="E3_rn50sup", items=8, variant="primary", suffix="", status="inconcluso_tecnico")
        tab, grid, fl, gp = self.run_(rows, pc); f = R.analyse.families["families"]; t = tab.set_index("encoder")
        self.assertEqual((f["entrenados"]["k"], f["entrenados"]["n_tests"]), (5, 10)); self.assertEqual(sorted(f["entrenados"]["excluidos"]), ["E2_vitsup", "E3_rn50sup"])
        self.assertEqual(f["pisos"]["k"], 2); self.assertIn("no interpretable", t.loc["E2_vitsup", "familia_holm"]); self.assertIn("técnico", t.loc["E3_rn50sup", "familia_holm"])
        self.assertEqual(set(grid.trained_family_size), {10})
    def test_S9_fit_type_label(self):    # REVISION3 I1: S9 changes a region at some delta -> 'sensible al tipo de ajuste' (descriptive)
        rows, pc = self.base(); s9 = [dict(r, variant="S9_mixed") for r in rows]
        s9[R.PRIMARY.index("E1_siglip")] = dict(row(.15, (.02, .28), (.04, .26), enc="E1_siglip"), variant="S9_mixed")   # atenuado at 0.20
        tab, grid, fl, gp = R.analyse(s2(rows), pc, True, GOOD, s9=s2(s9)); t = tab.set_index("encoder")
        self.assertEqual(t.loc["E1_siglip", "tipo_de_ajuste"], R.SENS_FIT); self.assertIn(R.SENS_FIT, t.loc["E1_siglip", "perfil"])
        self.assertEqual(t.loc["O1_dinov2b", "tipo_de_ajuste"], "estable"); self.assertEqual(t.loc["E1_siglip", "region_d0.20"], R.R3_)   # primary unchanged
        self.assertEqual(gp, "robusta: G3")                                                                                            # G not affected by S9
    def test_G2_needs_holm(self):
        rows, pc = self.base(); rows[R.PRIMARY.index("E1_siglip")] = row(-.3, (-.5, -.1), (-.45, -.15), popp=.03, enc="E1_siglip")
        tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(gp, "robusta: G4")                    # region 1 but Holm(.03*14) > .05
        rows[R.PRIMARY.index("E1_siglip")] = row(-.3, (-.5, -.1), (-.45, -.15), popp=1 / 924, enc="E1_siglip")
        tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(gp, "robusta: G2")
    def test_G1_and_delta_dependence(self):
        rows, pc = self.base(); rows[0] = row(.15, (.02, .28), (.04, .26), enc="R0_dinov2s")
        tab, grid, fl, gp = self.run_(rows, pc)
        self.assertEqual(gp, "depende solo del margen: δ = 0.05: G3; δ = 0.10: G3; δ = 0.20: G1")
    def test_cut_dependent_calibration(self):
        grp = grouping({"0.3": False, "0.5": True, "0.7": True}, {"0.3": 3, "0.5": 2, "0.7": 2})
        tab, grid, fl, gp = self.run_(*self.base(), grp=grp)
        g = grid.set_index(["delta", "cut"]).conclusion
        self.assertTrue(all(g[(d, "0.3")] == "G3b" and g[(d, "0.5")] == "G3" and g[(d, "0.7")] == "G3" for d in R.DELTAS))
        self.assertEqual(gp, "depende solo del corte: corte 0.3: G3b; corte 0.5: G3; corte 0.7: G3")
        rows, pc = self.base(); rows[R.PRIMARY.index("E1_siglip")] = row(-.3, (-.5, -.1), (-.45, -.15), popp=1 / 924, enc="E1_siglip")
        tab, grid, fl, gp = self.run_(rows, pc, grp=grp); d = grid.set_index(["delta", "cut"]).detalle
        self.assertIn("sin grupos", d[(.10, "0.3")]); self.assertNotIn("sin grupos", d[(.10, "0.5")])   # m2 per cut
    def test_mixed_grid(self):
        grp = grouping({"0.3": False, "0.5": True, "0.7": True}, {"0.3": 3, "0.5": 2, "0.7": 1})
        rows, pc = self.base(); rows[0] = row(.15, (.02, .28), (.04, .26), enc="R0_dinov2s"); tab, grid, fl, gp = self.run_(rows, pc, grp=grp)
        self.assertTrue(gp.startswith("depende del margen y del corte")); self.assertEqual(gp.count("(δ ="), 9)
    def test_floors(self):
        rows, pc = self.base()
        for f in R.FLOORS: rows[R.PRIMARY.index(f)] = row(.0, (-.02, .02), (-.015, .015), ptost=(.001, .001, .001), enc=f)
        tab, grid, fl, gp = self.run_(rows, pc); self.assertEqual(set(fl.lectura), {"la moderación requiere alguna geometría aprendida"}); self.assertEqual(set(fl.family_size), {4})
    def test_no_choice_options(self):
        src = (S / "08_regions.py").read_text() + (S / "02_equivalence.py").read_text()
        for bad in ("--delta", "--cut", "--encoder", "--encoders", "--alpha"): self.assertNotIn(bad, src)
        self.assertEqual(R.DELTAS, (0.05, 0.10, 0.20)); self.assertEqual(R.CUTS, ("0.3", "0.5", "0.7")); self.assertEqual(EQ.CUTS, (0.3, 0.5, 0.7))
class Grouping02(unittest.TestCase):
    def test_calibration_passes_at_one_cut_fails_at_another(self):
        ids = EQ.GROUP_SET; n = len(ids); Rm = np.full((n, n), 0.0); np.fill_diagonal(Rm, 1)
        A = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip"]; Bg = ["E2_vitsup", "E3_rn50sup", "E4_mae"]
        def setr(a, b, v): i, j = ids.index(a), ids.index(b); Rm[i, j] = Rm[j, i] = v
        for g in (A, Bg):
            for a in g:
                for b in g:
                    if a != b: setr(a, b, .8)
        setr("R0_dinov2s", "O1_dinov2b", .6)                         # 1-r = 0.4: same group at 0.5/0.7, NOT at 0.3? (UPGMA)
        for a in ["O1_dinov2b"]:
            for b in A[2:]: setr(a, b, .6)
        for a in A:
            for b in Bg: setr(a, b, .1)
        out = EQ.group_all(ids, Rm); c = out["cuts"]
        self.assertFalse(c["0.3"]["calibration"]["discriminates"]); self.assertEqual(c["0.3"]["reading"], "sin grupos")
        self.assertTrue(c["0.5"]["calibration"]["discriminates"]); self.assertTrue(c["0.7"]["calibration"]["discriminates"])
        self.assertEqual(out["profile"], "depende del corte"); self.assertEqual(c["0.5"]["n_trained_groups"], 2)
        json.dumps(out, default=bool)
    def test_robust_grouping(self):
        ids = EQ.GROUP_SET; n = len(ids); Rm = np.eye(n)
        tr = [i for i, k in enumerate(ids) if k in EQ.TRAINED]
        for i in tr:
            for j in tr:
                if i != j: Rm[i, j] = .95
        out = EQ.group_all(ids, Rm); self.assertEqual(out["profile"], "robusto")
        self.assertTrue(all(v["calibration"]["discriminates"] and v["n_trained_groups"] == 1 for v in out["cuts"].values()))
if __name__ == "__main__": unittest.main(verbosity=2)
