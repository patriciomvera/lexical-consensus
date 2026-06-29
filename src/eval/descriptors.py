"""
descriptors.py
--------------
Concept-level "carving" descriptors, computed in the FROZEN space BEFORE any
learning (plan §5). These are the x-axis of the central figure (Accuracy vs
NS(c)) and the legibility numbers that tell the reader whether a concept is
"just a rename of a native category" or a genuinely new region.

All four are functions of a concept's extension E_c (its embeddings) and, where
relevant, the native object-category labels of every image:

    NS_silhouette(c)        — cosine silhouette of E_c vs. the rest. Principal
                              native-separability measure. High for native,
                              low for disjunctive / cross-cutting.
    compactness(c)          — mean pairwise cosine distance WITHIN E_c. Low
                              compactness => multimodal extension.
    centroid_lands_outside  — bool: is the nearest neighbor of E_c's centroid an
                              intruder (non-member) rather than a member? Direct
                              "centroid in no-man's-land" diagnostic (plan §3).
    NMI_native(c)           — normalized mutual information between the concept
                              assignment and the native object category over all
                              images. ~1 => label is a rename of a native
                              category; ~0 => the concept cuts across clusters.

Plus the §4 SUBSTRATE DIAGNOSTIC for the cross-cutting set: cluster the frozen
embeddings and compare NMI(clusters, shape) vs NMI(clusters, color). The
cross-cutting test is only valid if NMI(clusters, shape) >> NMI(clusters, color).
"""

from __future__ import annotations

import numpy as np


def ns_silhouette(extension_emb: np.ndarray, other_emb: np.ndarray) -> float:
    """Cosine silhouette of a concept's extension against everything else.

    Principal NS(c). Returns a value in [-1, 1]; higher = more natively
    separable. Computed as the mean over extension samples of the two-group
    (member vs non-member) cosine silhouette.
    """
    from sklearn.metrics import silhouette_samples

    extension_emb = np.asarray(extension_emb, dtype=np.float64)
    other_emb = np.asarray(other_emb, dtype=np.float64)
    if len(extension_emb) < 1 or len(other_emb) < 1:
        return float("nan")
    X = np.vstack([extension_emb, other_emb])
    labels = np.r_[np.ones(len(extension_emb)), np.zeros(len(other_emb))]
    sil = silhouette_samples(X, labels, metric="cosine")
    return float(sil[: len(extension_emb)].mean())


def compactness(extension_emb: np.ndarray) -> float:
    """Mean pairwise cosine distance within the extension.

    Low value = tight single cluster; high value = spread / multimodal. The
    cheap multimodality signal that complements NS.
    """
    X = np.asarray(extension_emb, dtype=np.float64)
    if len(X) < 2:
        return 0.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    sims = Xn @ Xn.T
    iu = np.triu_indices(len(X), k=1)
    return float((1.0 - sims[iu]).mean())


def centroid_lands_outside(
    extension_emb: np.ndarray,
    other_emb: np.ndarray,
) -> bool:
    """True iff the nearest image to E_c's centroid is a NON-member.

    The literal "the centroid sits between the real modes" diagnostic for
    disjunctive concepts (plan §3): the unit-norm mean of a multimodal
    extension can land closer to an intruder than to any of its own members.
    """
    extension_emb = np.asarray(extension_emb, dtype=np.float64)
    other_emb = np.asarray(other_emb, dtype=np.float64)
    if len(extension_emb) == 0 or len(other_emb) == 0:
        return False
    centroid = extension_emb.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    nearest_member = (extension_emb @ centroid).max()
    nearest_intruder = (other_emb @ centroid).max()
    return bool(nearest_intruder > nearest_member)


def nmi_native(
    concept_ids: np.ndarray,
    object_categories: np.ndarray,
) -> float:
    """Normalized mutual information between concept assignment and native
    object category over all images (plan §5).

    The headline legibility number: ~1 means the label is essentially a rename
    of a native category; ~0 means the concept cuts across native clusters.
    """
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(
        np.asarray(object_categories), np.asarray(concept_ids)
    ))


def inter_member_distance(
    member_categories: tuple,
    native_centroid_dict: dict,
) -> float:
    """Cosine distance between the native centroids of a concept's member categories.

    For a 2-member disjunctive concept {A, B}:
        inter_member_distance = 1 − cosine_similarity(centroid_A, centroid_B)
    For N members: mean pairwise cosine distance across all (i, j) member pairs.

    Uses the pre-computed centroid dict (category -> L2-normalised centroid vector).
    This is a lookup, not a recomputation — the caller supplies the dict built
    once from the embedding cache.
    """
    cats = list(member_categories)
    if len(cats) < 2:
        return 0.0
    pairs = [(i, j) for i in range(len(cats)) for j in range(i + 1, len(cats))]
    distances = []
    for i, j in pairs:
        ci = np.asarray(native_centroid_dict[cats[i]], dtype=np.float64)
        cj = np.asarray(native_centroid_dict[cats[j]], dtype=np.float64)
        ni, nj = np.linalg.norm(ci), np.linalg.norm(cj)
        if ni > 0:
            ci = ci / ni
        if nj > 0:
            cj = cj / nj
        distances.append(1.0 - float(np.dot(ci, cj)))
    return float(np.mean(distances))


def substrate_diagnostic(
    embeddings: np.ndarray,
    shape_labels: np.ndarray,
    color_labels: np.ndarray,
    *,
    n_clusters: int,
    seed: int = 0,
) -> dict:
    """Cross-cutting validity gate (plan §4).

    k-means the frozen embeddings, then report
        {"nmi_shape": ..., "nmi_color": ..., "valid_cross_cutting": bool}
    where valid_cross_cutting is True iff the embeddings organize primarily by
    shape (nmi_shape >> nmi_color). If False, color is NOT cross-cutting on this
    set and the harness must pick a different attribute or report and adjust.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(np.asarray(embeddings, dtype=np.float64))
    nmi_shape = float(normalized_mutual_info_score(np.asarray(shape_labels), clusters))
    nmi_color = float(normalized_mutual_info_score(np.asarray(color_labels), clusters))
    return {
        "nmi_shape": nmi_shape,
        "nmi_color": nmi_color,
        "valid_cross_cutting": nmi_shape > nmi_color,
    }
