"""
random_baseline.py
------------------
Chance floor: ignore the embedding, assign a label at random (plan §2, §6).

This pins the explicit chance level for every (N-way, concept-type) config so
that "above chance" claims (plan §1 H3) have a concrete, episode-matched
reference rather than the nominal 1/N. It draws from the empirical concept
prior in the support set so that, for unbalanced disjunctive concepts, the
floor reflects the actual class frequencies the macro-averaged metric will see.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .base import Learner


class RandomLearner(Learner):
    """Embedding-blind random classifier, drawing from the support-set prior."""

    def __init__(self, n_concepts: int, *, seed: int = 0, **kwargs) -> None:
        super().__init__(n_concepts, seed=seed, **kwargs)
        self._prior: np.ndarray | None = None  # (n_concepts,) sampling weights

    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        seed_concept_ids = np.asarray(seed_concept_ids)
        counts = np.bincount(seed_concept_ids, minlength=self.n_concepts).astype(np.float64)
        if counts.sum() == 0:
            counts = np.ones(self.n_concepts)
        self._prior = counts / counts.sum()
        self._is_fit = True

    def _row_noise(self, row: np.ndarray) -> np.ndarray:
        """Gumbel noise that is a PURE FUNCTION of (row content, learner seed).

        Making the noise deterministic per input is what lets predict() == argmax
        (score()) across separate calls (the Learner contract), while still being
        embedding-BLIND in the sense that matters: the noise does not depend on
        the row's concept, only on its bytes. Different episodes pass different
        seeds, so the same image still gets different random labels across
        episodes.
        """
        digest = hashlib.blake2b(row.tobytes(), digest_size=8, key=str(self.seed).encode())
        sub = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
        u = sub.random(self.n_concepts)
        return -np.log(-np.log(u + 1e-12) + 1e-12)

    def score(self, emb: np.ndarray) -> np.ndarray:
        self._check_fit()
        emb2d, was_1d = self._as_2d(np.asarray(emb, dtype=np.float64))
        # Gumbel-max trick: argmax(log(prior) + Gumbel noise) ~ Categorical(prior).
        logits = np.log(self._prior + 1e-12)
        out = np.vstack([logits + self._row_noise(row) for row in emb2d])
        return out[0] if was_1d else out
