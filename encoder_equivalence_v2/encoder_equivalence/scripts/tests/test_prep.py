"""SYNTHETIC test of h4_prep_encoders.py: (1) row-by-row identity with the v1 h4_prep.py loop (executed from its own source text
with the synthetic raw/key and the synthetic R0 embeddings), (2) G0 pipeline control PASS on identical data, FAIL on |diff| > 1e-5
and on reordered rows. Usage: python test_prep.py <synthetic_dir>"""
import sys, re, pathlib, subprocess, importlib.util, numpy as np, pandas as pd
S = pathlib.Path(__file__).resolve().parents[1]; D = pathlib.Path(sys.argv[1]); PY = sys.executable
V1 = pathlib.Path("/workspace/lexical-consensus-openmind/human_data_analyses/scripts/h4_prep.py").read_text()
loop = V1[V1.index("rows = []"):V1.index('d.to_csv(B / "data/h4_long.csv"')]
raw = pd.read_csv(D / "raw/all_raw_A_SYNTH.csv"); key = pd.read_csv(D / "raw/experiment_key_SYNTH.csv").iloc[:12]; key["exp"] = key.exp.astype(int)
z = np.load(D / "emb/emb_R0_dinov2s.npz"); emb = {str(n): v / np.linalg.norm(v) for n, v in zip(z["names"], z["emb"])}
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
ns = dict(raw=raw, key=key, emb=emb, re=re, np=np, pd=pd, TEST=TEST); exec(loop, ns); v1 = ns["d"]
(D / "v1like").mkdir(exist_ok=True); v1.to_csv(D / "v1like/h4_long_SYNTH_v1code.csv", index=False)
(D / "long").mkdir(exist_ok=True)
IDS = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels", "E4s_mae_cls", "F1_rn50rand1", "F1_rn50rand2"]
subprocess.run([PY, S / "h4_prep_encoders.py", "--raw", D / "raw/all_raw_A_SYNTH.csv", "--key", D / "raw/experiment_key_SYNTH.csv", "--emb-dir", D / "emb", "--out-dir", D / "long", *IDS], check=True)
new = pd.read_csv(D / "long/h4_long_R0_dinov2s.csv"); old = pd.read_csv(D / "v1like/h4_long_SYNTH_v1code.csv")
assert list(new.columns) == list(old.columns), (new.columns, old.columns)
assert new.drop(columns="dist").equals(old.drop(columns="dist")), "rows/order differ from v1 code"
md = float(np.abs(new.dist - old.dist).max()); print(f"[prep] identical rows/order/columns vs v1 loop; max|dist diff| = {md:.3g}"); assert md <= 1e-6
chk = lambda a, b: subprocess.run([PY, S / "h4_prep_encoders.py", "--check", a, b], capture_output=True, text=True)
r = chk(D / "long/h4_long_R0_dinov2s.csv", D / "v1like/h4_long_SYNTH_v1code.csv"); print("[G0 same]", r.stdout.strip()); assert r.returncode == 0
(D / "pipeline_control_SYNTH.log").write_text(r.stdout)
bad = old.copy(); bad.loc[5, "dist"] += 2e-5; bad.to_csv(D / "v1like/bad_tol.csv", index=False)
r = chk(D / "long/h4_long_R0_dinov2s.csv", D / "v1like/bad_tol.csv"); print("[G0 +2e-5]", r.stdout.strip()); assert r.returncode == 1
bad = old.iloc[::-1].reset_index(drop=True); bad.to_csv(D / "v1like/bad_order.csv", index=False)
r = chk(D / "long/h4_long_R0_dinov2s.csv", D / "v1like/bad_order.csv"); print("[G0 reordered]", r.stdout.strip()); assert r.returncode == 1
for ID in IDS: assert pd.read_csv(D / f"long/h4_long_{ID}.csv", usecols=["exp", "subid", "trial", "image"]).equals(new[["exp", "subid", "trial", "image"]])
print("[prep] all 13 sources have identical keys/order. OK")
# BC3 (REVISION3): near-ties. Inject numerical noise of 1e-8 (well below 1e-6 rounding of the rank check): must PASS.
# REVISION3b: the noise goes into the NEW file (the v1 file is fixed by its sha256; the near-tie groups and their width are measured on it)
rng = np.random.default_rng(0); noisy = new.copy(); noisy["dist"] = noisy.dist + rng.uniform(-1e-8, 1e-8, len(noisy)); noisy.to_csv(D / "v1like/noise_1e-8.csv", index=False)
r = chk(D / "v1like/noise_1e-8.csv", D / "v1like/h4_long_SYNTH_v1code.csv"); print("[G0 noise 1e-8]", r.stdout.strip()); assert r.returncode == 0
# near-ties of the v1 kind (same centroid, other image order: |gap| < 1e-6) broken in opposite directions: must PASS
u = np.sort(old.dist.unique()); a_, b_ = u[10], u[11]; nt = old.copy(); nt2 = new.copy()
nt.loc[nt.dist == b_, "dist"] = a_ + 3e-7; nt2.loc[nt2.dist == b_, "dist"] = a_ - 3e-7   # gap 3e-7 above in one file, below in the other
nt.to_csv(D / "v1like/neartie_v1.csv", index=False); nt2.to_csv(D / "v1like/neartie_new.csv", index=False)
r = chk(D / "v1like/neartie_new.csv", D / "v1like/neartie_v1.csv"); print("[G0 near-tie flipped at 3e-7]", r.stdout.strip()); assert r.returncode == 0
# rounding-boundary near-tie: two v1 values that are one near-tie (A = 0.3000004999, B = 0.3000004990; gap 9e-10) and, in the new file,
# A moves by 2e-10 to 0.3000005001. round(6) splits A from B in the new file only (0.300001 vs 0.300000) and would FAIL; clustered rule: PASS
bd = old.copy(); bd2 = new.copy(); vA, vB = u[50], u[51]
bd.loc[bd.dist == vA, "dist"] = 0.3000004999; bd.loc[bd.dist == vB, "dist"] = 0.3000004990
bd2.loc[bd2.dist == vA, "dist"] = 0.3000005001; bd2.loc[bd2.dist == vB, "dist"] = 0.3000004990
bd.to_csv(D / "v1like/boundary_v1.csv", index=False); bd2.to_csv(D / "v1like/boundary_new.csv", index=False)
rr = bool((bd.dist.round(6).rank(method="min").to_numpy() == bd2.dist.round(6).rank(method="min").to_numpy()).all()); assert rr is False
r = chk(D / "v1like/boundary_new.csv", D / "v1like/boundary_v1.csv")
print(f"[G0 rounding-boundary near-tie] round(6) rule: same_rank={rr} (would FAIL); clustered rule:", r.stdout.strip()); assert r.returncode == 0
# a REAL swap: two rows whose d differ by a real gap of 5e-6 (> 1e-6 rounding, < 1e-5 value tolerance) exchange their order between
# the files, so only the rank check can catch it: must FAIL
x = u[40]; i = old.index[old.dist == x][0]; j = old.index[old.dist == u[41]][0]
sv1 = old.copy(); sv1.loc[i, "dist"] = x; sv1.loc[j, "dist"] = x + 5e-6
snew = new.copy(); snew.loc[i, "dist"] = x + 5e-6; snew.loc[j, "dist"] = x
sv1.to_csv(D / "v1like/real_swap_v1.csv", index=False); snew.to_csv(D / "v1like/real_swap_new.csv", index=False)
r = chk(D / "v1like/real_swap_new.csv", D / "v1like/real_swap_v1.csv"); print("[G0 real swap, gap 5e-6, values within 1e-5]", r.stdout.strip()); assert r.returncode == 1 and "max_abs_diff=5e-06" in r.stdout
# REVISION3b (minor): chained near-ties. Five v1 values 4e-7 apart (each gap < 1e-6, group width 1.6e-6 >= 1e-6), identical in both
# files (so values and ranks agree): the width guard alone must FAIL G0; and the real v1-like case (width ~0) must report PASS
ch = old.copy(); base_ = 0.4
for q in range(5): ch.loc[ch.dist == u[60 + q], "dist"] = base_ + q * 4e-7
ch.to_csv(D / "v1like/chain_v1.csv", index=False); ch.to_csv(D / "v1like/chain_new.csv", index=False)
import importlib.util; sp = importlib.util.spec_from_file_location("hp", S / "h4_prep_encoders.py"); HP = importlib.util.module_from_spec(sp); sp.loader.exec_module(HP)
assert abs(HP.near_tie_max_width(ch.dist.to_numpy(float)) - 1.6e-6) < 1e-9 and HP.near_tie_max_width(old.dist.to_numpy(float)) < 1e-6
r = chk(D / "v1like/chain_new.csv", D / "v1like/chain_v1.csv"); print("[G0 chained near-ties, width 1.6e-6]", r.stdout.strip())
assert r.returncode == 1 and "same_rank_order(near-ties<1e-06 merged)=True" in r.stdout and "(<1e-06: False)" in r.stdout
r = subprocess.run([PY, S / "h4_prep_encoders.py", "--check", D / "long/h4_long_R0_dinov2s.csv", D / "v1like/h4_long_SYNTH_v1code.csv", D / "pipeline_control_SYNTH.json"], capture_output=True, text=True)
import json; js = json.loads((D / "pipeline_control_SYNTH.json").read_text()); assert js["ok"] is True and js["near_tie_groups_ok"] is True; print("[G0 json] ok=True written")
print("[prep] BC3 tests OK")
