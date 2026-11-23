"""Shared constants, paths and v1-code loaders for the AGENTS addendum (v2 of arXiv 2611.14116).
Proposal: propuesta_agentes_multi_encoder_v2_2026-11-18.md (sha256 633a5e7e...; REVISION4 incorporated). Nothing here runs anything.
SYNTHETIC MODE: if the environment variable AGENTS_SYNTH_DIR is set (it must start with /tmp/synth_agents), EVERY data path
(inputs and outputs) points inside that directory; only the v1 CODE (revision_runs/repo, stats_lib.py, frozen human scripts) is
read from its real location, unchanged. The tests use only this mode."""
import os, sys, json, pathlib, hashlib, importlib.util, ast
import numpy as np
ROOT = pathlib.Path("/workspace/lexical-consensus-openmind")
EE = ROOT / "encoder_equivalence"; AG = EE / "agents"; HDA = ROOT / "human_data_analyses"
REPO = ROOT / "revision_runs/repo"; E7 = REPO / "experiments/exp_007_concept_carving"; E7B = REPO / "experiments/exp_007b_extension_clean"
E8 = REPO / "experiments/exp_008_dissociation"; R8 = E8 / "results/exp_008_20260608_173255"
SYNTH = os.environ.get("AGENTS_SYNTH_DIR")
if SYNTH and not SYNTH.startswith("/tmp/synth_agents"):
    sys.exit("AGENTS_SYNTH_DIR must start with /tmp/synth_agents")
# ---------------- encoders (same IDs, representations and preprocessing as encoder_specs.json of the human part)
P10 = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
EXTRA = ["E4s_mae_cls", "F1_rn50rand1", "F1_rn50rand2"]
R0 = "R0_dinov2s"; KNOWN = ["R0_dinov2s", "R1_cvcl"]           # results known from v1 and H6: descriptive (§3.1)
NEW_TRAINED = ["O1_dinov2b", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae"]   # decide (k <= 6)
TRAINED = KNOWN + NEW_TRAINED; FLOORS = ["F1_rn50rand0", "F2_pixels"]
SESOIS = (0.10, 0.20, 0.30)       # |rho| and |partial r| scale (§3.2); NOT comparable with the human deltas
CUTS = ("0.3", "0.5", "0.7")      # geometry cuts of the main plan §6.1
H1_SOURCES = ("THINGS", "HSJ", "CIFAR10H"); H2_SOURCES = ("THINGS", "HSJ")
HCOL = {"THINGS": "d_things", "HSJ": "d_hsj", "CIFAR10H": "d_c10h"}
SETS = ("c100train", "c10pool", "c10test")       # + c10pool_seed<s> (T-sem 'full', s in 0..3)
TSEM_SEEDS = (0, 1, 2, 3)          # + 42 = the primary T
# ---------------- paths
def _p(real, rel):
    return pathlib.Path(SYNTH) / rel if SYNTH else pathlib.Path(real)
OUT = _p(AG / "out", "out"); EMB_DIR = OUT / "emb"; SIM_DIR = OUT / "sims"; RES_DIR = OUT / "results"
V1_C10_CACHE = _p(REPO / ".cache/embeddings_exp007/cifar_all10.npz", "v1/cifar_all10.npz")
V1_LEDGER = _p(HDA / "data/exp007b_clean_ledger.jsonl", "v1/exp007b_clean_ledger.jsonl")
V1_PERC = _p(R8 / "phase1/perceptual_distance_matrix.csv", "v1/perceptual_distance_matrix.csv")
V1_SEM = _p(R8 / "phase1/semantic_distance_matrix.csv", "v1/semantic_distance_matrix.csv")
V1_SAMPLED = _p(R8 / "phase2/sampled_pairs.csv", "v1/sampled_pairs.csv")
V1_REGTAB = _p(R8 / "phase4/regression_table.csv", "v1/regression_table.csv")
V1_TSEM_FULL_CACHE = lambda s: _p(ROOT / f"revision_runs/out/r04_seeds/cache_seed_{s}/cifar_all10.npz", f"v1/cache_seed_{s}/cifar_all10.npz")
PAIRS_C100 = _p(HDA / "data/pairs_c100.csv", "hda/pairs_c100.csv"); PAIRS_C10 = _p(HDA / "data/pairs_c10.csv", "hda/pairs_c10.csv")
SAMPLED_100 = _p(HDA / "data/sampled_100.csv", "hda/sampled_100.csv"); TIERS_C10 = _p(HDA / "data/tiers_c10.csv", "hda/tiers_c10.csv")
C10H_COUNTS = _p(ROOT / "human_data_scouting/cifar10h-counts.npy", "hda/cifar10h-counts.npy")
GLASGOW = _p(ROOT / "human_data_scouting/norms/glasgow_esm2.csv", "hda/glasgow_esm2.csv")
SUB100 = _p("/workspace/hda_cache/dino_c100_train_sub100.npz", "v1/dino_c100_train_sub100.npz")   # H5: the same 100 imgs/class (idx)
EXISTING = {  # embeddings that already exist and are reused, never re-encoded (§2)
    ("R0_dinov2s", "c10pool"): V1_C10_CACHE,
    ("R0_dinov2s", "c10test"): _p("/workspace/hda_cache/dino_c10_test.npz", "v1/dino_c10_test.npz"),
    ("R1_cvcl", "c100train"): _p("/workspace/hda_cache/cvcl_cifar100_train.npz", "v1/cvcl_cifar100_train.npz"),
    ("R1_cvcl", "c10pool"): _p("/workspace/hda_cache/cvcl_cifar_all10.npz", "v1/cvcl_cifar_all10.npz")}
# The r04 T-sem "full" caches of R0 (revision_runs/out/r04_seeds) were lost in the box restart of 22-11 (~12:34-14:31): R0 pools
# c10pool_seed<s> are now encoded by 10_agent_embed.py like every other encoder (ACLARACIONES (a)). V1_TSEM_FULL_CACHE kept for reference only.
# for _s in TSEM_SEEDS: EXISTING[("R0_dinov2s", f"c10pool_seed{_s}")] = V1_TSEM_FULL_CACHE(_s)
# human-part outputs (frozen run of 21-22 Nov) read by 14 and 13
H_LONG = _p(EE / "data", "human/data")                       # h4_long_<ID>.csv
H_POSCONTROL = _p(EE / "results/poscontrol.csv", "human/poscontrol.csv")
H_REGIONS = _p(EE / "results/regions/regions_table.csv", "human/regions_table.csv")
H_GROUPING = _p(EE / "out/equivalence_grouping.json", "human/equivalence_grouping.json")
V1_H4_PERSON = _p(HDA / "out/h4_person_re.txt", "v1/h4_person_re.txt")
# CIFAR images (only 10_agent_embed.py reads them; synthetic mode: uint8 arrays in <SYNTH>/cifar/*.npz)
CIFAR10_ROOT = REPO / ".cache/cifar10"; CIFAR100_ROOT = pathlib.Path("/workspace/hda_cache/cifar100")
def emb_path(ID, s):
    """Where the embeddings of encoder ID for image set s live (existing v1/H6 cache or the addendum's own output)."""
    return EXISTING.get((ID, s), EMB_DIR / f"{s}_{ID}.npz")
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def ensure_dirs():
    for d in (EMB_DIR, SIM_DIR, RES_DIR): d.mkdir(parents=True, exist_ok=True)
def write_json(path, obj):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(obj, indent=1, default=_default))
def _default(o):
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, pathlib.Path): return str(o)
    raise TypeError(type(o))
# ---------------- v1 code, loaded UNCHANGED (as h6_tiers.py / h6_c100.py do)
def _load(name, path, extra_path=()):
    for p in (str(REPO), *map(str, extra_path)):
        if p not in sys.path: sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def run_clean():       # exp_007b_extension_clean/run_clean.py (T)
    return _load("rc_agents", E7B / "run_clean.py")
def run_episodes():    # exp_008 phase3/run_episodes.py (D)
    return _load("re8_agents", E8 / "src/phase3/run_episodes.py", [E8])
def regression_v1():   # exp_008 phase4/regression_analysis.py (Model PS, raw d, classical OLS: the v1 beta_perc = -0.2119 scale)
    return _load("reg8_agents", E8 / "src/phase4/regression_analysis.py", [E8])
def dissociation_v1(): # exp_008 phase4/dissociation_analysis.py (_apply_rule = rules A/B/C)
    return _load("dis8_agents", E8 / "src/phase4/dissociation_analysis.py", [E8])
def stats_lib():       # human_data_analyses/scripts/stats_lib.py (spearman_boot, mantel, ols_hc3, holm, bh)
    return _load("stats_lib_agents", HDA / "scripts/stats_lib.py")
def v1_functions(path, names):
    """Extract top-level defs/assignments `names` from a v1 SCRIPT whose module level has side effects (e.g. 05_h5.py writes
    files), and exec ONLY those nodes. The extracted source is byte-identical to the v1 file (tested)."""
    src = pathlib.Path(path).read_text(); tree = ast.parse(src); keep = []
    for node in tree.body:
        nm = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if nm in names: keep.append(ast.get_source_segment(src, node))
    ns = {}; exec("import numpy as np, pandas as pd\nfrom scipy import stats\n" + "\n".join(keep), ns); return ns, keep
def v1_beta_perc():
    import pandas as pd
    t = pd.read_csv(V1_REGTAB); return float(t[t.learner == "centroid"].iloc[0]["PS_beta_perc"])
