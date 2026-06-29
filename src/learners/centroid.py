"""
centroid.py
-----------
Pure-prototype learner: one L2-normalized centroid per concept.

This is the SAME mechanism as src.agents.lexicon.LexiconLayer, re-expressed
behind the exp_007 Learner interface so it can be compared head-to-head with
episodic and parametric learners on identical episodes. It is the learner the
Round-1 reviewer called "a pointer to a pre-existing cluster": exp_007 tests
exactly where that characterization holds (native concepts) and where it
breaks (disjunctive / cross-cutting concepts, plan §1 H1).

Prediction: centroid accuracy decreases monotonically as native separability
NS(c) drops, because a single centroid of a multimodal extension lands in
no-man's-land between the modes (plan §3, "multimodal true-centroid
diagnostic").
"""

from __future__ import annotations

import numpy as np

from .base import Learner


class CentroidLearner(Learner):
    """Nearest-centroid classifier in cosine space (one anchor per concept)."""

    def __init__(self, n_concepts: int, *, seed: int = 0, **kwargs) -> None:
        super().__init__(n_concepts, seed=seed, **kwargs)
        # centroids[c] is the unit-norm mean embedding of concept c's seeds,
        # or a NaN row if the concept had no seeds this episode.
        self._centroids: np.ndarray | None = None  # (n_concepts, dim)

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        seed_emb = np.asarray(seed_emb, dtype=np.float64)
        seed_concept_ids = np.asarray(seed_concept_ids)
        dim = seed_emb.shape[1]
        centroids = np.full((self.n_concepts, dim), np.nan, dtype=np.float64)
        for c in range(self.n_concepts):
            members = seed_emb[seed_concept_ids == c]
            if len(members) == 0:
                continue                                  # stays NaN -> -inf in score()
            mean = members.mean(axis=0)
            norm = np.linalg.norm(mean)
            centroids[c] = mean / norm if norm > 0 else mean
        self._centroids = centroids
        self._is_fit = True

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        # Cosine similarity == dot product (both sides unit-norm). NaN centroids
        # propagate to NaN, which we then send to -inf so they can never win.
        sims = emb2d @ self._centroids.T               # (n, n_concepts)
        sims = np.where(np.isnan(sims), -np.inf, sims)
        return sims[0] if was_1d else sims
