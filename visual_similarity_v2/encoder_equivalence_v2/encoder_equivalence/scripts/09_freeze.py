"""INTERNAL FREEZE of the encoder comparison (plan §10.2 step 4). DO NOT RUN until (a) Ariadna has approved the analysis scripts
and (b) not before Friday 20-11-2026 (decision relayed 18-11 10:44). Writes encoder_equivalence/FREEZE_encoders_v2.txt with the
time (America/Santiago, UTC-3) and the sha256 of: the plan, every file under scripts/ (incl. tests and README), encoder_specs.json,
all out/emb_*.npz and the review files. Refuses to overwrite an existing manifest.
  python 09_freeze.py --freeze --scripts-review <path to Ariadna's review of the scripts>
  python 09_freeze.py --verify          # Saturday step 0: recompute and compare; exit 1 on any mismatch"""
import sys, hashlib, pathlib, datetime, zoneinfo
B = pathlib.Path(__file__).resolve().parents[1]; ROOT = B.parent; MAN = B / "FREEZE_encoders_v2.txt"
PLAN = ROOT / "propuesta_comparacion_encoders_v2_2026-11-18.md"
def files():
    f = [PLAN, ROOT / "REVISION_ariadna_comparacion_encoders_v2_2026-11-18.md", ROOT / "REVISION2_ariadna_comparacion_encoders_v2_2026-11-18.md",
         ROOT / "REVISION3_ariadna_comparacion_encoders_v2_2026-11-18.md"]
    f += sorted(p for p in (B / "scripts").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    f += sorted(B.glob("propuesta_*snapshot*.md"))
    f += [B / "out/encoder_specs.json", B / "out/vision_params_and_checks.json", B / "out/rules_02_sha256_before_run.txt"] + sorted((B / "out").glob("emb_*.npz"))
    return f
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(extra=()): return {str(p.relative_to(ROOT)): sha(p) for p in list(files()) + list(extra)}
if __name__ == "__main__":
    if sys.argv[1:2] == ["--verify"]:
        want = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in MAN.read_text().splitlines() if len(l) > 66 and l[64:66] == "  "}
        bad = [k for k, v in want.items() if not (ROOT / k).exists() or sha(ROOT / k) != v]
        new = sorted(set(table()) - set(want))
        print(f"VERIFY {len(want)} files: {'OK' if not bad else 'MISMATCH ' + str(bad)}; files not in manifest: {new or 'none'}"); sys.exit(1 if bad else 0)
    if sys.argv[1:2] != ["--freeze"] or "--scripts-review" not in sys.argv: sys.exit(__doc__)
    rev = pathlib.Path(sys.argv[sys.argv.index("--scripts-review") + 1]); assert rev.exists(), f"missing scripts review {rev}"
    assert not MAN.exists(), "FREEZE manifest already exists; a freeze is never overwritten"
    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Santiago"))
    assert now.date() >= datetime.date(2026, 11, 20), "freeze not allowed before Friday 20-11-2026"
    t = table([rev]); lines = [f"FREEZE encoders v2 (arXiv 2611.14116) - internal freeze, NOT an external preregistration",
                               f"time: {now:%Y-%m-%d %H:%M:%S} America/Santiago (UTC{now:%z})", f"files: {len(t)}", ""]
    lines += [f"{v}  {k}" for k, v in sorted(t.items())]; MAN.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
