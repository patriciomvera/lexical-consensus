"""
kcentroid.py
------------
Multi-anchor prototype learner: K sub-centroids per concept, found by
intra-concept k-means.

This is the THEORETICAL BRIDGE of exp_007 (plan §2, Axis B). A single
centroid cannot represent a multimodal concept (disjunction of perceptually
distant categories); K sub-centroids can — IF the concept is allowed multiple
anchors. Showing that the centroid -> k-centroid step recovers disjunctive
concepts is direct evidence that the "fix" is multi-anchor memory, which is
the prototype -> episodic transition in Episodic Grounding terms.

score(emb, concept) = max over the concept's K sub-centroids of cosine
similarity (an OR over modes), so an embedding near ANY mode of the concept
scores high.
"""

from __future__ import annotations

import numpy as np

from .base import Learner


class KCentroidLearner(Learner):
    """K sub-centroids per concept (intra-concept k-means), OR-max scoring."""

    def __init__(self, n_concepts: int, *, k: int = 2, seed: int = 0, **kwargs) -> None:
        """
        Args:
            k: Sub-centroids per concept (plan suggests K in {2, 3}). If a
               concept has fewer than k seeds, fall back to n_seeds clusters.
        """
        super().__init__(n_concepts, seed=seed, **kwargs)
        self.k = k
        # subcentroids[c] -> (k_c, dim) array of unit-norm anchors, or None if
        # concept c had no seeds this episode.
        self._subcentroids: list[np.ndarray | None] | None = None

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        seed_emb = np.asarray(seed_emb, dtype=np.float64)
        seed_concept_ids = np.asarray(seed_concept_ids)
        subs: list[np.ndarray | None] = []
        for c in range(self.n_concepts):
            members = seed_emb[seed_concept_ids == c]
            if len(members) == 0:
                subs.append(None)
                continue
            # Can't ask for more clusters than points; with <= k seeds each seed
            # is its own anchor (degenerates to exemplar storage for that concept).
            k_c = min(self.k, len(members))
            if k_c == len(members):
                centers = members
            else:
                km = KMeans(n_clusters=k_c, random_state=self.seed, n_init=10)
                km.fit(members)
                centers = km.cluster_centers_
            norms = np.linalg.norm(centers, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            subs.append(centers / norms)
        self._subcentroids = subs
        self._is_fit = True

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        n = emb2d.shape[0]
        out = np.full((n, self.n_concepts), -np.inf, dtype=np.float64)
        for c, centers in enumerate(self._subcentroids):
            if centers is None:
                continue
            # OR over modes: an embedding near ANY sub-centroid scores high.
            out[:, c] = (emb2d @ centers.T).max(axis=1)
        return out[0] if was_1d else out

    def config(self) -> dict:
        return {**super().config(), "k": self.k}
