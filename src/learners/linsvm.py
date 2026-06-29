"""
linsvm.py
---------
Reviewer-requested baseline: linear SVM, one-vs-rest (plan §2, Axis B).

Same role as logreg — a strong linear decoder — but a max-margin objective
rather than a likelihood one. Including both guards against the criticism that
the chosen linear baseline was weak: if BOTH linear probes recover a concept
the centroid misses, the result is robustly about the centroid, not about a
particular linear fitting procedure.

Wraps sklearn.svm.LinearSVC in a one-vs-rest configuration. score() returns the
one-vs-rest decision_function margins.
"""

from __future__ import annotations

import numpy as np

from .base import Learner


class LinSVMLearner(Learner):
    """One-vs-rest linear SVM over frozen embeddings."""

    def __init__(self, n_concepts: int, *, C: float = 1.0, seed: int = 0, **kwargs) -> None:
        super().__init__(n_concepts, seed=seed, **kwargs)
        self.C = C
        self._clf = None        # sklearn LinearSVC (OvR), built in fit()

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        from sklearn.svm import LinearSVC

        seed_emb = np.asarray(seed_emb, dtype=np.float64)
        seed_concept_ids = np.asarray(seed_concept_ids)
        classes = np.unique(seed_concept_ids)
        if len(classes) < 2:
            raise ValueError(
                f"LinSVMLearner.fit needs >= 2 concepts in the support set, "
                f"got {len(classes)}."
            )
        # LinearSVC is inherently one-vs-rest for multiclass.
        self._clf = LinearSVC(C=self.C, random_state=self.seed)
        self._clf.fit(seed_emb, seed_concept_ids)
        self._is_fit = True

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        df = self._clf.decision_function(emb2d)
        if df.ndim == 1:                                # binary case -> (n,)
            df = np.column_stack([-df, df])
        out = np.full((emb2d.shape[0], self.n_concepts), -np.inf, dtype=np.float64)
        out[:, self._clf.classes_] = df
        return out[0] if was_1d else out

    def config(self) -> dict:
        return {**super().config(), "C": self.C}
