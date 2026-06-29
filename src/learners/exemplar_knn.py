"""
exemplar_knn.py
---------------
Pure-episodic learner: store every seed embedding, classify by the k nearest.

This is the opposite pole from CentroidLearner on the prototype -> episodic
continuum (plan §2, Axis B). It keeps the full support set in memory and makes
no prototype assumption, so it is predicted to survive low-NS concepts where
the centroid collapses. The prototype-failure gap

    Acc(exemplar_knn) - Acc(centroid)

is the headline H1 quantity (plan §1, §6), reported with a paired bootstrap CI.

Distance is cosine; ties broken toward the nearest neighbor. score() returns a
per-concept vote-or-similarity vector so margin/AUROC remain well-defined.
"""

from __future__ import annotations

import numpy as np

from .base import Learner


class ExemplarKNN(Learner):
    """k-nearest-neighbor classifier over stored seed embeddings (cosine)."""

    def __init__(self, n_concepts: int, *, k: int = 1, weighted: bool = True,
                 seed: int = 0, **kwargs) -> None:
        """
        Args:
            k:        Neighbors to poll (plan sweeps k in {1, 3, 5}).
            weighted: If True (default), each neighbor votes with its cosine
                      similarity (distance-weighted soft k-NN); if False, each
                      neighbor casts an equal unit vote (classic majority k-NN).
                      For k=1 the two are identical. Distance weighting keeps the
                      score continuous, which matters for the margin diagnostic.
        """
        super().__init__(n_concepts, seed=seed, **kwargs)
        self.k = k
        self.weighted = weighted
        self._mem_emb: np.ndarray | None = None        # (n_seed, dim)
        self._mem_concept: np.ndarray | None = None     # (n_seed,)

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        # Pure episodic: just store the support set. No training.
        self._mem_emb = np.asarray(seed_emb, dtype=np.float64).copy()
        self._mem_concept = np.asarray(seed_concept_ids).copy()
        self._is_fit = True

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        sims = emb2d @ self._mem_emb.T                  # (n, n_seed) cosine sims
        k = min(self.k, sims.shape[1])
        # Indices of the k nearest exemplars per query (unordered top-k).
        nn_idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]

        out = np.full((emb2d.shape[0], self.n_concepts), -np.inf, dtype=np.float64)
        for i in range(emb2d.shape[0]):
            neigh = nn_idx[i]
            neigh_concepts = self._mem_concept[neigh]
            # Per concept, aggregate over its neighbors: SUM of similarities
            # (distance-weighted) or COUNT (majority vote). Soft weighting keeps
            # the score continuous for the margin diagnostic. Concepts absent
            # from the k neighbors keep -inf and can never win.
            weights = sims[i, neigh] if self.weighted else np.ones_like(neigh, dtype=np.float64)
            for c in np.unique(neigh_concepts):
                out[i, c] = weights[neigh_concepts == c].sum()
        return out[0] if was_1d else out

    def config(self) -> dict:
        return {**super().config(), "k": self.k, "weighted": self.weighted}
