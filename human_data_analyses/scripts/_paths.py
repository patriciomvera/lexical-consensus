"""Local paths. Override with environment variables; defaults assume this layout:
  <LC_ROOT>/human_data_analyses   (this repository)
  <LC_ROOT>/human_data_scouting   (third-party raw data, fetched by data/download_data.sh)
  <LC_ROOT>/main_repo             (main project repository: PerceptionLayer, exp_007/exp_008 caches)
  <HDA_CACHE>                     (embedding caches, CIFAR images, HSJ units)"""
import os, pathlib
HDA = pathlib.Path(__file__).resolve().parents[1]
ROOT = pathlib.Path(os.environ.get("LC_ROOT", HDA.parent))
SCOUT = ROOT / "human_data_scouting"
MAIN_REPO = pathlib.Path(os.environ.get("MAIN_REPO", ROOT / "main_repo"))
CACHE = pathlib.Path(os.environ.get("HDA_CACHE", ROOT / "hda_cache"))
XTMEM = pathlib.Path(os.environ.get("XTMEM_DIR", ROOT / "XTMEM"))
