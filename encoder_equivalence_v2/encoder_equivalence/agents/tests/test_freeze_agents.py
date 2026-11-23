"""15_freeze_agents.py WITHOUT writing the real manifest: a copy of the script whose ONLY change is MAN -> /tmp is run with --freeze
(reads and hashes the real files; writes nothing outside /tmp), then --verify (OK), a tampered manifest (MISMATCH, exit 1), refusal to
overwrite, refusal without a review file, and the synthetic-mode guard. FREEZE_agentes_v2.txt must not exist before or after."""
import os, sys, shutil, pathlib, subprocess
HERE = pathlib.Path(__file__).resolve().parents[1]; SRC = HERE / "scripts/15_freeze_agents.py"; REAL_MAN = HERE.parent / "FREEZE_agentes_v2.txt"
T = pathlib.Path("/tmp/synth_agents_freeze"); shutil.rmtree(T, ignore_errors=True); T.mkdir(); PY = sys.executable
assert not REAL_MAN.exists(), "the real agents manifest exists: this test must not run after the freeze"
s = SRC.read_text(); o = 'MAN = L.EE / "FREEZE_agentes_v2.txt"'; assert s.count(o) == 1
cp = HERE / "scripts" / "_tmp_freeze_copy_test.py"   # must sit next to lib_agents.py; removed at the end
s = s.replace(o, f'MAN = pathlib.Path("{T}/FREEZE_agentes_v2.txt")')
cp.write_text(s)
miss = [l for l in subprocess.run([PY, str(cp), "--list"], capture_output=True, text=True).stdout.splitlines() if l.startswith("MISSING")]
if miss:   # the real tree lacks files (22-11: v1 CIFAR-10 cache lost in the box restart): --freeze must refuse and name them
    r = subprocess.run([PY, str(cp), "--freeze", "--scripts-review", "/etc/hostname"], capture_output=True, text=True); assert r.returncode != 0 and "missing files" in r.stderr, r.stderr[-500:]
    print(f"[freeze] real tree has {len(miss)} missing file(s) -> --freeze refuses: {[m.split()[1] for m in miss]}")
    # happy path on a second copy that hashes only the files that exist (TEST ONLY; the real script requires all of them)
    s = s.replace("    return f\ndef rel(p):", "    return [x for x in f if pathlib.Path(x).exists()]\ndef rel(p):"); assert "return [x for x in f if" in s; cp.write_text(s)
env = {k: v for k, v in os.environ.items() if k != "AGENTS_SYNTH_DIR"}
def run(*a): return subprocess.run([PY, str(cp), *a], capture_output=True, text=True, env=env)
try:
    rev = T / "review_dummy.md"; rev.write_text("dummy review for the freeze test\n")
    r = run("--freeze"); assert r.returncode != 0 and not (T / "FREEZE_agentes_v2.txt").exists()
    r = run("--freeze", "--scripts-review", str(rev)); assert r.returncode == 0, r.stderr[-1500:]; man = (T / "FREEZE_agentes_v2.txt").read_text().splitlines()
    n = int([l for l in man if l.startswith("files:")][0].split()[1]); assert "human freeze at freeze time: VERIFY" in "\n".join(man) and "OK" in [l for l in man if l.startswith("human freeze")][0]
    print(f"[freeze] --freeze on the copy: manifest in /tmp with {n} files; human freeze verified inside")
    r = run("--verify"); assert r.returncode == 0 and "OK" in r.stdout, r.stdout; print("[freeze] --verify:", r.stdout.strip()[:120])
    lines = man[:]; i = next(k for k, l in enumerate(lines) if "agents/scripts/11_agent_sims.py" in l); lines[i] = "0" * 64 + lines[i][64:]
    (T / "FREEZE_agentes_v2.txt").write_text("\n".join(lines) + "\n"); r = run("--verify"); assert r.returncode == 1 and "MISMATCH" in r.stdout; print("[freeze] tampered hash -> MISMATCH, exit 1")
    r = run("--freeze", "--scripts-review", str(rev)); assert r.returncode != 0 and "never overwritten" in r.stderr; print("[freeze] refuses to overwrite an existing manifest")
    r = subprocess.run([PY, str(SRC), "--list"], capture_output=True, text=True, env=env | {"AGENTS_SYNTH_DIR": "/tmp/synth_agents_x"}); assert r.returncode != 0; print("[freeze] refuses in synthetic mode")
finally:
    cp.unlink(missing_ok=True)
assert not REAL_MAN.exists(); print("ALL FREEZE TESTS PASSED (real FREEZE_agentes_v2.txt not created)")
