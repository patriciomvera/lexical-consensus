"""Re-run experiments/exp_007b_extension_clean/run_clean.py unchanged except for seeds and output paths.
mode 'episodes': change run_clean.MASTER_SEED only (episode sampling + learner seeds); the 200-image-per-class pool
                 (exp_007 _shared, rng 42, cache .cache/embeddings_exp007/cifar_all10.npz) is held fixed.
mode 'full'    : additionally re-draw the 200-image-per-class pool with default_rng(seed) (exp_007 _shared.MASTER_SEED)
                 and re-encode it with DINOv2-small (cache in revision_analyses/out/r04_seeds/cache_seed_<s>/).
Output: revision_analyses/out/r04_seeds/<mode>_seed_<s>/ledger_c2_homogeneous.jsonl (committed ledger is not touched).
Usage: python r04_run_clean_seed.py <seed> [episodes|full]"""
import sys, importlib.util, pathlib, torch
torch.set_num_threads(1)
seed = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "episodes"
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("rc", "experiments/exp_007b_extension_clean/run_clean.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
out = pathlib.Path("revision_analyses/out/r04_seeds") / f"{mode}_seed_{seed}"; out.mkdir(parents=True, exist_ok=True)
rc.MASTER_SEED = seed; rc.RESULTS_DIR = out
if mode == "full":
    rc._shared007.MASTER_SEED = seed
    rc._shared007.EMB_CACHE_DIR = (pathlib.Path("revision_analyses/out/r04_seeds") / f"cache_seed_{seed}").resolve()
rc.main()
