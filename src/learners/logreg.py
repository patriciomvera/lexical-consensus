"""
logreg.py
---------
Reviewer-requested baseline: multinomial logistic regression over frozen
embeddings (plan §2, Axis B).

A linear probe is the standard "how much is linearly decodable from the frozen
features" baseline. On native concepts it should match everyone (H2). Its
behavior on disjunctive / cross-cutting concepts is itself informative: a
linear boundary CAN carve a half-space union in 384-d, so if logreg recovers
disjunctive concepts the dissociation story is about the centroid specifically,
not about linear separability in general (plan §1 result table, "both hold"
row).

Wraps sklearn.linear_model.LogisticRegression. score() returns decision-function
logits (not probabilities) so margins are on a comparable scale to the other
learners.
"""

from __future__ import annotations

import numpy as np

from .base import Learner


class LogRegLearner(Learner):
    """Multinomial logistic regression probe (one linear model, softmax head)."""

    def __init__(self, n_concepts: int, *, C: float = 1.0, seed: int = 0, **kwargs) -> None:
        """
        Args:
            C: Inverse L2 regularization strength (sklearn convention).
        """
        super().__init__(n_concepts, seed=seed, **kwargs)
        self.C = C
        self._clf = None        # sklearn LogisticRegression, built in fit()
        self._classes: np.ndarray | None = None   # concept_ids the clf can emit

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        seed_emb = np.asarray(seed_emb, dtype=np.float64)
        seed_concept_ids = np.asarray(seed_concept_ids)
        self._classes = np.unique(seed_concept_ids)
        if len(self._classes) < 2:
            # No silent failure: a probe needs >= 2 classes to define a boundary.
            raise ValueError(
                f"LogRegLearner.fit needs >= 2 concepts in the support set, "
                f"got {len(self._classes)}."
            )
        self._clf = LogisticRegression(C=self.C, random_state=self.seed, max_iter=1000)
        self._clf.fit(seed_emb, seed_concept_ids)
        self._is_fit = True

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        df = self._clf.decision_function(emb2d)
        if df.ndim == 1:                                # binary case -> (n,)
            df = np.column_stack([-df, df])             # logits for [class0, class1]
        # Expand to full n_concepts width; concepts never seen in support get
        # -inf so they can never win (keeps the (n_concepts,) contract).
        out = np.full((emb2d.shape[0], self.n_concepts), -np.inf, dtype=np.float64)
        out[:, self._clf.classes_] = df
        return out[0] if was_1d else out

    def config(self) -> dict:
        return {**super().config(), "C": self.C}
