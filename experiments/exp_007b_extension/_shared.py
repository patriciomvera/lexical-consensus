"""
_shared.py — configuration for exp_007b_extension.

Reuses embeddings and learners from exp_007; adds tier thresholds and pool config.
All thresholds are derived from exp_007b data — see inline derivations.

Tier boundaries (inter_member_distance):
    NEAR  : d < FAR_THRESHOLD         — within-superordinate pairs
    MID   : FAR_THRESHOLD <= d < FAR_MEDIAN — cross-boundary, sub-maximal
    FAR   : d >= FAR_MEDIAN           — loaded from exp_007b (do not re-run)
    NATIVE: single-category           — loaded from exp_007b (do not re-run)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
EXT_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = EXT_DIR.parents[1]   # experiments/exp_007b_extension -> experiments -> project root
PARENT_EXP   = PROJECT_ROOT / "experiments" / "exp_007_concept_carving"

# EXT_DIR MUST be first in sys.path so `from _shared import ...` resolves to THIS
# file, not the parent's _shared.py. PROJECT_ROOT and PARENT_EXP are appended.
if str(EXT_DIR) in sys.path:
    sys.path.remove(str(EXT_DIR))
sys.path.insert(0, str(EXT_DIR))
for _p in [str(PROJECT_ROOT), str(PARENT_EXP)]:
    if _p not in sys.path:
        sys.path.append(_p)

EXP007_RESULTS = PROJECT_ROOT / "results" / "exp_007"
RESULTS_DIR    = PROJECT_ROOT / "results" / "exp_007b_extension"

# ─── Tier thresholds ──────────────────────────────────────────────────────────
# FAR_THRESHOLD: minimum inter_member_distance in exp_007b disjunctive set.
#   zorb = bird+dog: dist(centroid_bird, centroid_dog) = 0.4767
#   All inter_member_distances in exp_007b: zorb=0.4767, plonk=0.6492,
#   gleth=0.6783, quax=0.7144, mivor=0.7343.
FAR_THRESHOLD = 0.4767

# FAR_MEDIAN: median of the 4 "clean" far-disjunctive inter_member_distances
#   (excluding zorb which is partially degenerate):
#   plonk=0.6492, gleth=0.6783, quax=0.7144, mivor=0.7343
#   → sorted: [0.6492, 0.6783, 0.7144, 0.7343] → median = (0.6783+0.7144)/2
FAR_MEDIAN = 0.6964

# NATIVE_MEDIAN: median of native NS silhouettes from exp_007b.
#   NS values (all 10 native): 0.0783,0.1238,0.1927,0.1945,0.2273,
#                               0.2404,0.2613,0.2758,0.3085,0.3117
#   median = (0.2273 + 0.2404) / 2 = 0.2339 ≈ 0.234 (established in exp_007b)
NATIVE_MEDIAN = 0.234

# ─── Validity gate ────────────────────────────────────────────────────────────
GATE_NMI_MAX = 0.80   # exclude near-native (NMI >= 0.80 => concept ≈ single native cat)

# ─── Far-disjunctive concepts reused from exp_007b ────────────────────────────
# zorb (bird+dog) is excluded as partially degenerate (lowest NS, bird is weak cluster).
CLEAN_FAR_LABELS = frozenset({"gleth", "mivor", "plonk", "quax"})

# ─── Pool variant config ──────────────────────────────────────────────────────
POOL_TYPES = ["small", "medium", "large", "hard", "oov_25", "oov_50"]
PRIMARY_POOL = "hard"

POOL_SIZES = {
    "small":  5,    # 1 correct + 4 random from other episode concepts
    "medium": 10,   # 1 correct + 9 random
    "large":  25,   # 1 correct + 24 random
    "hard":   10,   # 1 correct + 9 nearest-neighbor distractors
    "oov_25": 10,   # 1 correct + ~7 random + ~2 OOV
    "oov_50": 10,   # 1 correct + ~5 random + ~5 OOV
}

# OOV count = how many of the (pool_size - 1) distractors are OOV
OOV_COUNTS = {
    "oov_25": 2,  # 2/9 ≈ 22% OOV (rounds to ~25%)
    "oov_50": 5,  # 5/9 ≈ 56% OOV (rounds to ~50%)
}

# Offset added to the per-trial RNG seed to ensure pool types get independent streams.
POOL_TYPE_SEED_OFFSET = {
    "small":  100_000,
    "medium": 200_000,
    "large":  300_000,
    "hard":   400_000,
    "oov_25": 500_000,
    "oov_50": 600_000,
}

# ─── Episode grid ─────────────────────────────────────────────────────────────
WAYS            = [3, 5]
SHOTS           = [1, 3, 5, 10, 20]
N_EPISODES      = 30
N_QUERY_PER_CONCEPT = 50
N_RETRIEVAL_TRIALS  = 20
MASTER_SEED     = 42

# Full learner roster (same as exp_007b)
ROSTER = (
    ["centroid"]
    + [f"kcentroid_k{k}" for k in [1, 2, 3]]
    + [f"exemplar_knn_k{k}" for k in [1, 3, 5]]
    + ["logreg", "linsvm", "random"]
)

REFERENCE_LEARNER = "centroid"
EXEMPLAR_LEARNER  = "exemplar_knn_k3"   # the primary episodic learner for paired tests
FIG_SHOTS = 5

# ─── Concept tokens (distinct from exp_007b disjunctive tokens) ───────────────
# exp_007b used: zorb, gleth, mivor, plonk, quax, sn-ark, blore, wovel, frabb, grellow
NEAR_TOKENS = ["tromble", "varble", "glintz", "snomph", "breftle", "cworch", "phliggle"]
MID_TOKENS  = ["brumble", "strenk", "worblix", "snoffin", "gremble", "tulgorp", "fibblet"]

# ─── Near/mid candidate pairs ─────────────────────────────────────────────────
# These are the CIFAR-10 category pairs evaluated by the concept gate.
# Derived from the full 10x10 inter-category distance matrix.
# Only pairwise-DISJOINT subsets are kept for episode construction.

# Near candidates: d < FAR_THRESHOLD (0.4767)
NEAR_CANDIDATES = [
    # (cat_a, cat_b, approx_distance)  — all within-superordinate or cross-boundary near pairs
    ("cat",        "dog",        0.2795),
    ("bird",       "deer",       0.3979),
    ("automobile", "truck",      0.3985),
    ("deer",       "horse",      0.4178),
    ("cat",        "frog",       0.4489),
    ("bird",       "frog",       0.4617),
    ("bird",       "cat",        0.4703),
]

# Mid candidates: FAR_THRESHOLD <= d < FAR_MEDIAN (0.4767 to 0.6964)
MID_CANDIDATES = [
    ("deer",       "dog",        0.5294),
    ("airplane",   "ship",       0.5305),
    ("airplane",   "automobile", 0.5551),
    ("deer",       "frog",       0.5735),
    ("dog",        "horse",      0.5754),
    ("ship",       "truck",      0.5814),
    ("automobile", "ship",       0.5820),
    ("bird",       "horse",      0.6167),
    ("cat",        "horse",      0.6170),
    ("airplane",   "truck",      0.6169),
]


def build_native_centroid_dict(
    embeddings: dict,
    records,
) -> dict[str, np.ndarray]:
    """Returns {category -> L2-normalised centroid} for all categories in records.

    Used to compute inter_member_distance without re-touching DINOv2.
    """
    from experiments.exp_007_concept_carving._shared import hash_to_category
    h2c = hash_to_category(records)
    cats = sorted({r.object_category for r in records})
    out: dict[str, np.ndarray] = {}
    for cat in cats:
        vecs = np.asarray(
            [embeddings[h] for h in embeddings if h2c[h] == cat], dtype=np.float64
        )
        m = vecs.mean(axis=0)
        n = np.linalg.norm(m)
        out[cat] = m / n if n > 0 else m
    return out
