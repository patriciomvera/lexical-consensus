"""
concepts.py
-----------
Concepts as predicates over (object identity, attributes) — the independent
variable of exp_007 (plan §2 Axis A, §3, §4).

A `Concept` binds a Carroll label to an EXTENSION: the set of images that fall
under it. The whole experiment turns on HOW that extension relates to the
native structure of the frozen DINOv2 space:

    NATIVE        — extension == one native category (e.g. slithy = frog).
                    High NS, NMI_native ~ 1. The Round-1 setup; the
                    non-triviality control (H2).
    DISJUNCTIVE   — extension == union of perceptually DISTANT native
                    categories (e.g. zorb = {frog, ship}). The centroid lands
                    between clusters. Low NS, partial NMI_native. Primary,
                    cheapest payoff — no new data, just a relabeling over
                    cached embeddings (plan §3).
    CROSS_CUTTING — extension == an attribute orthogonal to identity
                    (e.g. zorb = red things, across shapes). Low NS, low
                    NMI_native by construction. Needs the factorial
                    shapes×colors set (plan §4, §9).
    SYNTHETIC_NOVEL — textures/shapes DINOv2 underrepresents. Weak control.

Each image is described by an `ImageRecord` carrying its native object category
and (where applicable) attributes, so descriptors.py and the H4 identity-cheating
detector can be computed without re-touching pixels.

CRITICAL VALIDITY RULES (plan §3) — enforced by the builders, not optional:
  * Disjunctive unions must be ARBITRARY and perceptually DISTANT. Pair
    selection uses the inter-cluster distances measured in exp_000; never build
    a semantically coherent union ({frog, deer} = "animals") because "animal"
    may be a latent DINOv2 dimension and would not test carving.
  * Build several disjunctive concepts spanning a RANGE of native separability
    (near -> far) to trace the H1 collapse curve. Minimum 5 (plan §3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class ConceptType(str, Enum):
    NATIVE = "native"
    DISJUNCTIVE = "disjunctive"
    CROSS_CUTTING = "cross_cutting"
    SYNTHETIC_NOVEL = "synthetic_novel"


@dataclass(frozen=True)
class ImageRecord:
    """One image's identity in the experiment, independent of any learner.

    The embedding is referenced by hash (same convention as the rest of the
    project, e.g. "frog_007"); the actual 384-d vector lives in the shared
    embedding cache so concepts stay lightweight and serializable.
    """
    img_hash: str
    object_category: str               # native category (e.g. "frog", "circle")
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"color": "red"}


@dataclass
class Concept:
    """A Carroll label plus the extension that defines its meaning.

    `member_categories` records which native categories (or attribute values)
    the extension is built from — needed by metrics.c2_retrieval to compute the
    valid set X_ℓ (a `zorb = {frog, ship}` query accepts a frog OR a ship) and
    by descriptors.NMI_native.
    """
    concept_id: int
    label: str                          # Carroll word (vorpal, slithy, ...)
    concept_type: ConceptType
    member_categories: tuple[str, ...]  # native categories / attribute values
    image_hashes: tuple[str, ...]       # full extension (support + query pool)
    predicate_desc: str = ""            # human-readable definition, for logs


# ---------------------------------------------------------------------------
# Builders — each returns a list[Concept] given the image inventory + cache.
# Scaffolding: signatures and contracts only.
# ---------------------------------------------------------------------------

def build_native_concepts(
    records: list[ImageRecord],
    category_to_label: dict[str, str],
) -> list[Concept]:
    """One concept per native object category (plan §2 NATIVE).

    Establishes H2 (no learner dissociation on native concepts) and replicates
    the Round-1 result with full statistics. Each concept's extension is exactly
    the images of one native category, so NS is high and NMI_native ~ 1 by
    construction.

    Args:
        records:           every image's identity (object_category drives the
                           grouping).
        category_to_label: native category -> Carroll label. Determines which
                           categories become concepts and their concept_id order
                           (dict insertion order).
    """
    concepts: list[Concept] = []
    for concept_id, (category, label) in enumerate(category_to_label.items()):
        hashes = tuple(r.img_hash for r in records if r.object_category == category)
        if not hashes:
            raise ValueError(f"build_native_concepts: no images for category '{category}'.")
        concepts.append(Concept(
            concept_id=concept_id,
            label=label,
            concept_type=ConceptType.NATIVE,
            member_categories=(category,),
            image_hashes=hashes,
            predicate_desc=f"native: object_category == '{category}'",
        ))
    return concepts


# Semantically-empty invented tokens for disjunctive concepts (NOT Carroll
# words — those name the native categories). Kept meaningless on purpose: a
# disjunctive concept's meaning must come only from its extension.
_DISJUNCTIVE_TOKENS = ["zorb", "gleth", "mivor", "plonk", "quax",
                       "sn-ark", "blore", "wovel", "frabb", "grellow"]


def build_disjunctive_concepts(
    records: list[ImageRecord],
    embeddings: dict[str, np.ndarray],
    n_concepts: int = 5,
    *,
    distance_matrix: np.ndarray,
    category_order: list[str],
    arity: int = 2,
    seed: int = 0,
) -> list[Concept]:
    """Unions of perceptually DISTANT native categories (plan §3, directives 4-5).

    Construction:
      1. Compute each category's individual native separability NS (silhouette
         vs the rest).
      2. Greedy DISJOINT MATCHING scored by `distance × min(member NS)`:
         repeatedly take the highest-scoring pair whose categories are both
         still unused. The `distance` factor makes unions perceptually distant
         (arbitrary, not a latent "animal" cluster); the `min(member NS)` factor
         pairs WELL-SEPARATED categories together, so a union's low NS comes from
         the disjunction itself rather than from an already-degenerate member —
         and consequently isolates any weak categories (e.g. bird, dog in
         DINOv2/CIFAR) into their own pair. The matching stays pairwise
         category-DISJOINT so no image belongs to two concepts in an episode
         (otherwise labels are ambiguous) and partitions all
         `arity * n_concepts` categories.
      3. Order concepts by union NS ASCENDING so concept_id 0 is the hardest
         (lowest-NS) carving target — the x-axis of the H1 collapse curve.

    Selection vs. native (directive 5) is verified by the caller, which compares
    the disjunctive set's NS / compactness to the native distribution (median,
    not min — a single degenerate native category like bird would make the min a
    misleading reference). `arity` defaults to 2 so a 5-pair matching partitions
    all 10 CIFAR categories.
    """
    from .descriptors import ns_silhouette, compactness as _compactness

    if arity != 2:
        raise NotImplementedError("build_disjunctive_concepts currently supports arity=2.")
    needed = arity * n_concepts
    if len(category_order) < needed:
        raise ValueError(
            f"need {needed} categories for {n_concepts} disjoint pairs, "
            f"got {len(category_order)}."
        )

    h2c: dict[str, str] = {r.img_hash: r.object_category for r in records}
    cat_hashes: dict[str, list[str]] = {c: [] for c in category_order}
    for h, c in h2c.items():
        if c in cat_hashes:
            cat_hashes[c].append(h)

    # 1. Per-category individual NS (clamped at 0 so a near-degenerate category
    #    cannot produce a misleadingly large product via a negative value).
    all_emb = {h: embeddings[h] for h in h2c}
    cat_ns: dict[str, float] = {}
    for cat in category_order:
        ext = np.asarray([all_emb[h] for h in cat_hashes[cat]], dtype=np.float64)
        oth = np.asarray([all_emb[h] for h in all_emb if h2c[h] != cat], dtype=np.float64)
        cat_ns[cat] = max(0.0, ns_silhouette(ext, oth))

    # 2. Greedy disjoint matching by distance * min(member NS).
    def score(i, j):
        return distance_matrix[i, j] * min(cat_ns[category_order[i]], cat_ns[category_order[j]])

    pairs_by_score = sorted(
        ((i, j) for i in range(len(category_order)) for j in range(i + 1, len(category_order))),
        key=lambda ij: score(*ij), reverse=True,
    )
    used: set[int] = set()
    matched: list[tuple[str, str]] = []
    for i, j in pairs_by_score:
        if len(matched) == n_concepts:
            break
        if i in used or j in used:
            continue
        used.update((i, j))
        matched.append((category_order[i], category_order[j]))
    if len(matched) < n_concepts:
        raise ValueError("could not form enough disjoint pairs — check category count.")

    # 2. Descriptors per union, to order by NS (hardest first).
    all_emb = {h: embeddings[h] for h in h2c}
    scored = []
    for (ca, cb) in matched:
        members = cat_hashes[ca] + cat_hashes[cb]
        ext = np.asarray([all_emb[h] for h in members], dtype=np.float64)
        others = [h for h in all_emb if h2c[h] not in (ca, cb)]
        oth = np.asarray([all_emb[h] for h in others], dtype=np.float64)
        ns = ns_silhouette(ext, oth)
        scored.append((ns, _compactness(ext), ca, cb, tuple(members)))
    scored.sort(key=lambda t: t[0])     # NS ascending => concept_id 0 hardest

    concepts: list[Concept] = []
    for cid, (ns, comp, ca, cb, members) in enumerate(scored):
        concepts.append(Concept(
            concept_id=cid,
            label=_DISJUNCTIVE_TOKENS[cid % len(_DISJUNCTIVE_TOKENS)],
            concept_type=ConceptType.DISJUNCTIVE,
            member_categories=(ca, cb),
            image_hashes=members,
            predicate_desc=f"disjunctive: object in {{{ca}, {cb}}} "
                           f"(NS={ns:.3f}, compactness={comp:.3f})",
        ))
    return concepts


def build_cross_cutting_concepts(
    records: list[ImageRecord],
    attribute: str = "color",
) -> list[Concept]:
    """One concept per attribute VALUE, cutting across object identity.

    Only valid after the substrate diagnostic (descriptors / plan §4) confirms
    DINOv2 organizes the synthetic set primarily by SHAPE, not by the chosen
    attribute — otherwise the attribute is not actually cross-cutting on this
    set and the test is void.
    """
    raise NotImplementedError


def chance_level(concept_type: ConceptType, n_concepts: int) -> float:
    """Explicit chance level for a config (plan §6).

    For the macro-averaged N-way naming metric, a label-blind classifier scores
    1 / n_concepts in expectation regardless of concept type (each concept's
    recall is 1/n_concepts under uniform random prediction). Disjunctive /
    cross-cutting sets keep the same nominal chance because the metric is
    macro-averaged; what changes across types is the achievable accuracy, not
    the floor. Kept as a function so per-type adjustments stay in one place.
    """
    if n_concepts <= 0:
        raise ValueError("chance_level: n_concepts must be positive.")
    return 1.0 / n_concepts
