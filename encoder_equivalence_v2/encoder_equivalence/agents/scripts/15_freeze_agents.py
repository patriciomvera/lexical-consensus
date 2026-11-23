"""INTERNAL FREEZE of the AGENTS addendum (proposal §5: freeze by Fri 27-11-2026 at the latest). DO NOT RUN --freeze until
(a) Ariadna has reviewed the addendum scripts and given her go-ahead (written review file), (b) Patricio has approved, and
(c) the points of agents/ACLARACIONES_scripts.md section (b) are decided and written into the proposal / scripts.
Writes encoder_equivalence/FREEZE_agentes_v2.txt: time (America/Santiago) and sha256 of the proposal, REVISION4, the scripts review,
ACLARACIONES, every file under agents/scripts and agents/tests, the human-part manifest and the human outputs the addendum reads,
the v1 code loaded unchanged and the v1 data/caches used. Refuses to overwrite an existing manifest and refuses if the human
freeze does not verify. It never touches FREEZE_encoders_v2.txt.
  python 15_freeze_agents.py --freeze --scripts-review /ABS/PATH/review.md
  python 15_freeze_agents.py --verify      # run_agents.sh step 0; exit 1 on any mismatch or missing file
  python 15_freeze_agents.py --list        # print the file list (no hashing, no writing)
  python 15_freeze_agents.py --header      # print the declaration header that --freeze writes (REVISION5 C1/C5; no hashing, no writing)"""
import sys, re, hashlib, pathlib, datetime, zoneinfo, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
if L.SYNTH and "--allow-synth" not in sys.argv: sys.exit("15_freeze_agents.py works on the real files only (unset AGENTS_SYNTH_DIR)")
ROOT = L.ROOT; AG = L.AG; MAN = L.EE / "FREEZE_agentes_v2.txt"
PROPOSAL = ROOT / "propuesta_agentes_multi_encoder_v2_2026-11-18.md"
def files():
    f = [PROPOSAL, ROOT / "propuesta_agentes_bfc99fac_snapshot.md", ROOT / "REVISION4_ariadna_agentes_multi_encoder_2026-11-18.md", ROOT / "REVISION5_ariadna_scripts_agentes_2026-11-22.md", ROOT / "REVISION5b_ariadna_verificacion_2026-11-22.md",
         L.EE / "DIFF_agentes_propuesta.patch", AG / "ACLARACIONES_scripts.md", AG / "README_agentes.md"]
    f += sorted(p for d in ("scripts", "tests") for p in (AG / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    f += [L.EE / "FREEZE_encoders_v2.txt", L.EE / "FREEZE_encoders_v2_complemento.txt", L.EE / "results/poscontrol.csv",
          L.EE / "results/regions/regions_table.csv", L.EE / "out/equivalence_grouping.json", L.EE / "out/encoder_specs.json"]
    f += [L.EE / f"data/h4_long_{e}.csv" for e in L.P10] + [L.EE / f"out/emb_{e}.npz" for e in L.P10 + L.EXTRA]
    f += [L.E7B / "run_clean.py", L.E7 / "_shared.py", L.E8 / "_shared.py", L.E8 / "config/exp_008_config.yaml", L.E8 / "src/phase3/run_episodes.py",
          L.E8 / "src/phase4/regression_analysis.py", L.E8 / "src/phase4/dissociation_analysis.py", L.E8 / "src/phase2/classify_quadrants.py",
          L.HDA / "scripts/stats_lib.py", L.HDA / "scripts/h6_embed_cvcl.py", L.HDA / "scripts/05_h5.py", L.EE / "scripts/02_equivalence.py",
          L.EE / "scripts/01_embed.py"]
    f += [L.REPO / "src/agents/perception.py"]   # built the v1 R0 cache (cifar_all10.npz) on 22-11
    f += sorted((L.REPO / "src/learners").glob("*.py")) + sorted((L.REPO / "src/eval").glob("*.py"))
    f += [L.V1_C10_CACHE, L.V1_LEDGER, L.V1_PERC, L.V1_SEM, L.V1_SAMPLED, L.V1_REGTAB, L.PAIRS_C100, L.PAIRS_C10, L.SAMPLED_100, L.TIERS_C10,
          L.C10H_COUNTS, L.GLASGOW, L.SUB100, L.V1_H4_PERSON] + sorted(set(L.EXISTING.values()))
    return f
def header():
    """Declarations written at the top of the manifest (REVISION5 C1, C5 and the (b) declaration). Versions are read live."""
    import importlib.metadata as md
    v = {k: md.version(k) for k in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas", "huggingface_hub")}
    return ["# R0 / v1 CIFAR-10 cache (REVISION5 C1): " + rel(L.V1_C10_CACHE) + " sha256 9744d7cfd6ed2c284ec4797c5d27d87bc512e4a1bba5d25d0b8994b14668024d",
            "#   rebuilt on 22-11-2026 after the box restart with the v1 code (revision_runs/repo/src/agents/perception.py, unchanged) and",
            "#   " + ", ".join(f"{k} {x}" for k, x in v.items()) + " (CPU).",
            "#   facebook/dinov2-small revision ed25f3a31f01632728cabb09d1542f84ab7b0056: perception.py does NOT pin a revision",
            "#   (from_pretrained(DINO_MODEL)); on 22-11 the hub 'main' ref = ed25f3a3 = the revision pinned in 01_embed.py / 10_agent_embed.py.",
            "#   Independent verification by Ariadna (REVISION5, full precision): max|delta| 8.9e-8 on the 45 pairs vs pairs_c10.csv and",
            "#   5.6e-8 on the 10 tiers vs tiers_c10.csv.",
            "# (b) decisions (ACLARACIONES_scripts.md section (b), REVISION5): taken AFTER the human run and BEFORE any agent result;",
            "#   none uses human values (no human result was read to choose them).",
            "# COMPUTE (REVISION5 C5): primary ~13-28 core-h, ~3.5-7 h wall, 3 processes (<= 1.5 GB each), local CPU, no GPU/API, $0.",
            "#   Cap 14 h wall from the start of 'primary'. Rate (img/min) checked 15 min after the embeddings start; < 50 % of the",
            "#   measured rate -> logged and re-estimated. Over the cap: deviation logged; cut FIRST the sensitivities, THEN H4 per",
            "#   encoder; D-todos and the rest of 'primary' are never cut. Embeddings backed up outside /tmp (+SHA256SUMS) after step 3.",
            "# Patricio's confirmations (DONE): (1) b1, rebuilding the v1 R0 CIFAR-10 cache + recovering weights/CIFAR-10, approved 22-11-2026",
            "#   15:16 America/Santiago (message t218u); (2) freeze today and run on 23-11 at 08:30 (ahead of P1: freeze <= 27, run 28-29),",
            "#   approved 22-11-2026 17:59 America/Santiago (\"Sí, congela hoy cuando Ariadna dé el OK y parte mañana a las 08:30\").",
            "# Scripts review: REVISION5 (VB con condiciones) + REVISION5b_ariadna_verificacion_2026-11-22.md (OK para congelar, 16:48)."]
SHA_LINE = re.compile(r"^[0-9a-f]{64}  (.+)$")
def rel(p):
    p = pathlib.Path(p)
    try: return str(p.relative_to(ROOT))
    except ValueError: return str(p)
def absp(k): return pathlib.Path(k) if k.startswith("/") else ROOT / k
def sha(p): return L.sha256(p)
def table(extra=()): return {rel(p): sha(p) for p in list(files()) + list(extra)}
def human_verify():
    r = subprocess.run([sys.executable, str(L.EE / "scripts/09_freeze.py"), "--verify"], capture_output=True, text=True); return r.returncode == 0, r.stdout.strip()
if __name__ == "__main__":
    if "--list" in sys.argv:
        for p in files(): print(("OK  " if pathlib.Path(p).exists() else "MISSING  ") + rel(p))
        sys.exit(0)
    if "--header" in sys.argv:
        print("\n".join(header())); sys.exit(0)
    if sys.argv[1:2] == ["--verify"]:
        want = {m.group(1): l[:64] for l in MAN.read_text().splitlines() if (m := SHA_LINE.match(l))}
        bad = [k for k, v in want.items() if not absp(k).exists() or sha(absp(k)) != v]
        new = sorted(set(rel(p) for p in files() if pathlib.Path(p).exists()) - set(want))
        print(f"VERIFY agentes {len(want)} files: {'OK' if not bad else 'MISMATCH ' + str(bad)}; files not in manifest: {new or 'none'}"); sys.exit(1 if bad else 0)
    if sys.argv[1:2] != ["--freeze"] or "--scripts-review" not in sys.argv: sys.exit(__doc__)
    rev = pathlib.Path(sys.argv[sys.argv.index("--scripts-review") + 1]); assert rev.is_absolute() and rev.exists(), f"scripts review must be an existing absolute path: {rev}"
    assert not MAN.exists(), "FREEZE_agentes_v2.txt already exists; a freeze is never overwritten"
    missing = [rel(p) for p in files() if not pathlib.Path(p).exists()]; assert not missing, f"missing files: {missing}"
    okh, msg = human_verify(); assert okh, f"human freeze does not verify: {msg}"
    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Santiago")); t = table([rev])
    lines = ["FREEZE agentes v2 (arXiv 2611.14116, addendum to FREEZE_encoders_v2.txt) - internal freeze, NOT an external preregistration",
             f"time: {now:%Y-%m-%d %H:%M:%S} America/Santiago (UTC{now:%z})", f"human freeze at freeze time: {msg}", f"files: {len(t)}", ""] + header() + [""]
    lines += [f"{v}  {k}" for k, v in sorted(t.items())]; MAN.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
