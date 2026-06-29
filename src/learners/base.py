"""
base.py
-------
The common learner interface for exp_007 (plan §2, §7).

Every lexical mechanism the experiment compares — prototype centroid,
k-centroid, episodic k-NN, logreg, linear SVM, random floor — is a `Learner`.
Fixing one interface is what makes the dissociation test fair: the SAME
seeded episodes, the SAME embeddings, the SAME scoring contract; only the
mechanism changes.

A concept is identified by an integer `concept_id` in [0, n_concepts).
The mapping concept_id <-> Carroll label and concept_id <-> extension lives
in src.eval.concepts, NOT here — a learner only ever sees integer ids so it
cannot accidentally exploit label semantics.

Design notes
------------
- Embeddings are frozen DINOv2 vectors, already L2-normalized (see
  src.agents.perception). Learners must NOT re-encode images; they operate on
  precomputed 384-d vectors only (plan §10: all learning is CPU over cached
  embeddings).
- `score()` is the single source of truth: `predict()` is argmax over
  `score()`, and margin / AUROC metrics (plan §6) are computed from the score
  vector. Subclasses that produce distances rather than similarities must
  convert so that HIGHER score == MORE likely (sim, not distance).
- No silent failures (CLAUDE.md coding principle 2): calling predict/score
  before fit raises, it does not return a default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Learner(ABC):
    """Abstract base for every exp_007 lexical learner.

    Subclasses store whatever internal state they need in `fit()` and must
    honor the score-is-similarity contract described in the module docstring.
    """

    def __init__(self, n_concepts: int, *, seed: int = 0, **kwargs) -> None:
        """
        Args:
            n_concepts: Number of concepts in this episode (the "ways", N).
            seed:       Per-learner RNG seed. Logged for reproducibility
                        (CLAUDE.md principle 3). Even deterministic learners
                        accept it so the harness can pass one uniformly.
            **kwargs:   Learner-specific hyperparameters (e.g. K for
                        kcentroid, k for k-NN, C for SVM). Subclasses pop what
                        they use and should reject unknown keys loudly.
        """
        self.n_concepts = n_concepts
        self.seed = seed
        self._is_fit = False

    # -- training ---------------------------------------------------------

    @abstractmethod
    def fit(self, seed_emb: np.ndarray, seed_concept_ids: np.ndarray) -> None:
        """Learn the lexicon from the support set.

        Args:
            seed_emb:        (n_seed, dim) float array of frozen embeddings.
            seed_concept_ids: (n_seed,) int array in [0, n_concepts).
        """
        raise NotImplementedError

    # -- naming (Condition 1, image -> label) -----------------------------

    @abstractmethod
    def score(self, emb: np.ndarray) -> np.ndarray:
        """Per-concept similarity/logit vector for one or many embeddings.

        Args:
            emb: (dim,) or (n, dim) float array.

        Returns:
            (n_concepts,) for a single embedding, or (n, n_concepts) for a
            batch. Higher == more likely. Used directly by metrics.margin and
            metrics. oov_auroc.
        """
        raise NotImplementedError

    def predict(self, emb: np.ndarray) -> np.ndarray:
        """Argmax over `score()`. Returns int concept_id(s).

        Single embedding -> scalar int; batch -> (n,) int array.
        Concrete subclasses normally inherit this unchanged.
        """
        self._check_fit()
        s = self.score(emb)
        if s.ndim == 1:
            return int(np.argmax(s))
        return np.argmax(s, axis=1)

    # -- inverse grounding (Condition 2, label -> image) ------------------

    def retrieve(self, concept_id: int, candidate_emb: np.ndarray) -> int:
        """Given a concept id, pick which candidate embedding is that concept.

        Default implementation: the candidate whose score for `concept_id` is
        highest. For disjunctive concepts the valid set X_ℓ is the union of
        member categories — that bookkeeping lives in the metrics layer
        (plan §6), not here.

        Args:
            concept_id:    Target concept.
            candidate_emb: (n_candidates, dim) float array.

        Returns:
            Index into candidate_emb of the best match.
        """
        self._check_fit()
        s = self.score(candidate_emb)            # (n_candidates, n_concepts)
        return int(np.argmax(s[:, concept_id]))

    # -- helpers ----------------------------------------------------------

    def _check_fit(self) -> None:
        if not self._is_fit:
            raise RuntimeError(
                f"{type(self).__name__}.predict/score/retrieve called before fit()."
            )

    @staticmethod
    def _as_2d(emb: np.ndarray) -> tuple[np.ndarray, bool]:
        """Promote a (dim,) vector to (1, dim); report whether it was 1-D."""
        if emb.ndim == 1:
            return emb[None, :], True
        return emb, False

    @property
    def name(self) -> str:
        return type(self).__name__

    def config(self) -> dict:
        """Reproducibility record (seed + hyperparameters). Logged per episode."""
        return {"learner": self.name, "n_concepts": self.n_concepts, "seed": self.seed}
