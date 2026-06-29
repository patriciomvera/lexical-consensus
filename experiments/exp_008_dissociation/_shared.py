"""
_shared.py — paths, config, embedding cache, and CIFAR-100 metadata for exp_008.

All phase scripts import from here. Mirrors the _shared.py convention of exp_007.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# On Windows the default console encoding is cp1252, which can't represent
# Unicode math/arrow characters used in print statements.  Force UTF-8 for
# all stdout/stderr so every phase script works without per-file changes.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

EXP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]   # exported for phase scripts that need src.*
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_BASE      = EXP_DIR / "results"
CACHE_DIR         = PROJECT_ROOT / ".cache"
EMB_CACHE_DIR     = CACHE_DIR / "embeddings_exp008"
CONFIG_PATH       = EXP_DIR / "config" / "exp_008_config.yaml"
WORDNET_MAP_PATH  = EXP_DIR / "config" / "wordnet_mappings.json"

# CIFAR-100 fine-grained class names (torchvision alphabetical order, indices 0-99).
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]
assert len(CIFAR100_CLASSES) == 100, f"Expected 100 classes, got {len(CIFAR100_CLASSES)}"

CLASS_TO_IDX = {c: i for i, c in enumerate(CIFAR100_CLASSES)}

# CIFAR-100 superclass structure (20 superclasses × 5 subclasses).
# Used in phase 3 to exclude same-superclass classes when sampling contrasts.
SUPERCLASSES: dict[str, list[str]] = {
    "aquatic_mammals":               ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish":                          ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers":                       ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers":               ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables":          ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices":  ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture":           ["bed", "chair", "couch", "table", "wardrobe"],
    "insects":                       ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores":              ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man_made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes":  ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores":["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals":                ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non_insect_invertebrates":      ["crab", "lobster", "snail", "spider", "worm"],
    "people":                        ["baby", "boy", "girl", "man", "woman"],
    "reptiles":                      ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals":                 ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees":                         ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1":                    ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2":                    ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# Reverse: class -> superclass name
CLASS_TO_SUPERCLASS: dict[str, str] = {
    cls: sc for sc, classes in SUPERCLASSES.items() for cls in classes
}

# Carroll labels for 3-way episodes (pre-registered, matches exp_007 convention)
CARROLL_LABELS = ["vorpal", "slithy", "mimsy"]


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_wordnet_mappings() -> dict[str, str]:
    with open(WORDNET_MAP_PATH) as f:
        data = json.load(f)
    # Strip the _notes key if present
    return {k: v for k, v in data.items() if not k.startswith("_")}


# ── Run directory management ──────────────────────────────────────────────────

def create_run_dir(timestamp: str) -> Path:
    """Create a timestamped run directory with all phase subdirs."""
    run_dir = RESULTS_BASE / f"exp_008_{timestamp}"
    for subdir in ["phase1/raw", "phase2/raw", "phase3/raw", "phase4/raw"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    print(f"[run_dir] created {run_dir}")
    return run_dir


def find_latest_run_dir() -> Path:
    # Prefer the explicit pointer written by embed_cifar100.py
    ptr = RESULTS_BASE / "last_run.txt"
    if ptr.exists():
        p = Path(ptr.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    # Fall back to lexicographic sort (timestamp format = sorted order)
    dirs = sorted(RESULTS_BASE.glob("exp_008_2*"))
    if not dirs:
        raise FileNotFoundError(
            f"No run directories found under {RESULTS_BASE}. "
            "Run embed_cifar100.py first."
        )
    return dirs[-1]


def get_run_dir(arg: str | None = None) -> Path:
    if arg:
        p = Path(arg)
        if not p.exists():
            raise FileNotFoundError(f"Specified run_dir does not exist: {p}")
        return p
    return find_latest_run_dir()


# ── Embedding cache ───────────────────────────────────────────────────────────

def load_embeddings() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CIFAR-100 embeddings from cache.

    Returns:
        emb:         (N, 384) float32 — L2-normalized DINOv2 embeddings
        labels:      (N,) int         — CIFAR-100 class index 0-99
        class_names: list[str]        — 100 class names, index matches label value
    """
    cache_path = EMB_CACHE_DIR / "cifar100.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Embedding cache not found: {cache_path}\n"
            "Run 'python src/phase1/embed_cifar100.py' first."
        )
    data = np.load(cache_path, allow_pickle=False)
    emb         = data["emb"]                        # (N, 384)
    labels      = data["labels"].astype(int)         # (N,)
    class_names = [str(n) for n in data["class_names"]]   # (100,)
    print(f"[cache] loaded {len(emb)} embeddings ({len(class_names)} classes) "
          f"from {cache_path.name}")
    return emb, labels, class_names


def build_episode_pool(
    emb: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    pool_per_class: int = 200,
    seed: int = 42,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Sample `pool_per_class` embeddings per class from the cache.

    Returns:
        embeddings:  {hash -> (384,) embedding} for all sampled images
        h2c:         {hash -> class_name}
    """
    rng = np.random.default_rng(seed)
    embeddings: dict[str, np.ndarray] = {}
    h2c: dict[str, str] = {}
    for idx, cls in enumerate(class_names):
        mask = np.where(labels == idx)[0]
        chosen = rng.choice(mask, size=min(pool_per_class, len(mask)), replace=False)
        for j, gi in enumerate(chosen):
            h = f"{cls}_{j:03d}"
            embeddings[h] = emb[gi]
            h2c[h] = cls
    print(f"[pool] built episode pool: {len(embeddings)} images "
          f"({pool_per_class}/class × {len(class_names)} classes)")
    return embeddings, h2c
