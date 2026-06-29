"""
compute_semantic_distances.py — Step 3 of Phase 1.

Computes the 100×100 pairwise semantic distance matrix between CIFAR-100
classes using Wu-Palmer similarity (wup_similarity) on WordNet synsets:

    d_sem(u, v) = 1 - wup_similarity(synset_u, synset_v)

Wu-Palmer is preferred over path_similarity because it normalizes by tree
depth, avoiding artifacts from uneven WordNet hierarchy depth.

Mappings are loaded from config/wordnet_mappings.json. Each synset is
validated at runtime; if a specified synset name is not found, the script
falls back to wordnet.synsets(class_name)[0] and logs the substitution.

If wup_similarity returns None (no common ancestor — rare in practice),
the distance is set to 1.0 (maximum dissimilarity).

Outputs:
    <run_dir>/phase1/semantic_distance_matrix.csv    — 100×100 distance
    <run_dir>/phase1/wordnet_mappings_used.json      — frozen copy of mappings

Usage:
    python src/phase1/compute_semantic_distances.py [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import CIFAR100_CLASSES, get_run_dir, load_wordnet_mappings


def _ensure_nltk_wordnet() -> None:
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        # Trigger a small access to confirm it's downloaded
        list(wn.synsets("dog"))[:1]
    except LookupError:
        print("[wordnet] downloading WordNet corpus ...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def _resolve_synsets(
    mappings: dict[str, str],
    class_names: list[str],
) -> tuple[dict[str, object], dict[str, str]]:
    """Validate each synset name; fall back to first synset if not found.

    Returns:
        resolved:  {class_name: Synset}
        used_map:  {class_name: synset_name_actually_used}  (for logging)
    """
    from nltk.corpus import wordnet as wn

    resolved: dict[str, object] = {}
    used_map: dict[str, str] = {}
    fallbacks = []

    for cls in class_names:
        if cls not in mappings:
            raise KeyError(f"No WordNet mapping for class '{cls}'. "
                           "Add it to config/wordnet_mappings.json.")
        synset_name = mappings[cls]
        synset = None
        try:
            synset = wn.synset(synset_name)
        except Exception:
            pass

        if synset is None:
            # Fall back to first synset for the class name (or a simplified form)
            query = cls.replace("_", " ").split()[0] if "_" in cls else cls
            candidates = wn.synsets(query)
            if not candidates:
                candidates = wn.synsets(cls)
            if not candidates:
                raise RuntimeError(
                    f"No WordNet synsets found for class '{cls}' "
                    f"(tried synset '{synset_name}' and query '{query}')."
                )
            synset = candidates[0]
            fallback_name = synset.name()
            fallbacks.append((cls, synset_name, fallback_name))
            synset_name = fallback_name

        resolved[cls] = synset
        used_map[cls] = synset_name

    if fallbacks:
        print(f"[wordnet] {len(fallbacks)} synset fallback(s):")
        for cls, attempted, actual in fallbacks:
            print(f"  {cls:20s}  tried={attempted:30s}  used={actual}")

    return resolved, used_map


def compute_semantic_distance_matrix(
    resolved: dict[str, object],
    class_names: list[str],
) -> np.ndarray:
    """100×100 Wu-Palmer semantic distance matrix."""
    n = len(class_names)
    dist_matrix = np.zeros((n, n), dtype=np.float64)

    none_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            syn_i = resolved[class_names[i]]
            syn_j = resolved[class_names[j]]
            sim = syn_i.wup_similarity(syn_j)
            if sim is None:
                d = 1.0
                none_count += 1
            else:
                d = 1.0 - float(sim)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    if none_count > 0:
        print(f"[wordnet] {none_count} pairs had no common ancestor "
              f"(wup=None) — set to d=1.0")

    return dist_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="exp_008 Phase 1 — compute semantic distance matrix"
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    run_dir = get_run_dir(args.run_dir)

    print("[semantic] ensuring NLTK WordNet is available ...")
    _ensure_nltk_wordnet()

    print("[semantic] loading WordNet mappings ...")
    mappings = load_wordnet_mappings()

    print("[semantic] resolving synsets ...")
    resolved, used_map = _resolve_synsets(mappings, CIFAR100_CLASSES)

    # Freeze the actual mappings used (may differ from config due to fallbacks)
    frozen_path = run_dir / "phase1" / "wordnet_mappings_used.json"
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(used_map, f, indent=2, sort_keys=True)
    print(f"[semantic] frozen mappings -> {frozen_path}")

    print("[semantic] computing 100×100 Wu-Palmer distance matrix ...")
    dist_matrix = compute_semantic_distance_matrix(resolved, CIFAR100_CLASSES)
    np.fill_diagonal(dist_matrix, 0.0)

    assert dist_matrix.shape == (100, 100)
    assert np.allclose(dist_matrix, dist_matrix.T, atol=1e-6)
    print(f"  d_sem range: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]")
    print(f"  d_sem mean (off-diagonal): {dist_matrix[dist_matrix > 0].mean():.4f}")

    out_path = run_dir / "phase1" / "semantic_distance_matrix.csv"
    df = pd.DataFrame(dist_matrix, index=CIFAR100_CLASSES, columns=CIFAR100_CLASSES)
    df.to_csv(out_path)
    print(f"[semantic] saved -> {out_path}")

    print("\nNext step:\n  python src/phase1/correlation_diagnostic.py")
    print("  *** STOP after reviewing the correlation report (§9.4 checkpoint) ***")


if __name__ == "__main__":
    main()
