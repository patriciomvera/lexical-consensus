"""Tests 09_freeze.py WITHOUT freezing: (1) --freeze refuses before 20-11-2026 / without a scripts review; (2) the manifest format
round-trips through --verify logic using a manifest written to /tmp (never to encoder_equivalence/)."""
import sys, subprocess, importlib.util, pathlib, tempfile
S = pathlib.Path(__file__).resolve().parents[1]
r = subprocess.run([sys.executable, S / "09_freeze.py", "--freeze"], capture_output=True, text=True); assert r.returncode != 0; print("[freeze] without review -> refused")
with tempfile.NamedTemporaryFile(suffix=".md") as f:
    r = subprocess.run([sys.executable, S / "09_freeze.py", "--freeze", "--scripts-review", f.name], capture_output=True, text=True)
    assert r.returncode != 0 and "before Friday" in r.stderr, r.stderr; print("[freeze] before 20-11 -> refused")
sp = importlib.util.spec_from_file_location("fz", S / "09_freeze.py"); fz = importlib.util.module_from_spec(sp); sp.loader.exec_module(fz)
assert not fz.MAN.exists(), "a real FREEZE manifest exists!"
tmp = pathlib.Path(tempfile.mkdtemp()) / "FREEZE_test.txt"; t = fz.table()
tmp.write_text("header\n\n" + "\n".join(f"{v}  {k}" for k, v in sorted(t.items())) + "\n"); fz.MAN = tmp
sys.argv = ["x", "--verify"]
try: exec(compile(open(S / "09_freeze.py").read().split('if __name__ == "__main__":')[1].replace("\n    ", "\n"), "v", "exec"), vars(fz))
except SystemExit as e: assert e.code == 0, e.code
print(f"[freeze] verify round-trip OK on {len(t)} files (manifest in /tmp only); real manifest exists: {fz.MAN.exists() and fz.MAN != tmp}")
