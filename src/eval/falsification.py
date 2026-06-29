"""
falsification.py
----------------
Falsification controls (plan §6, reusing exp_002 logic), applied to the
disjunctive and cross-cutting sets to confirm the H1 dissociation is a real
property of concept type and not an artifact.

Controls
--------
random_label    — shuffle the concept assignment across all images; any learner
                  should collapse to chance. Catches metrics that score high on
                  noise.
permuted_binding — keep each image's concept count but permute WHICH label binds
                  to WHICH extension; tests that success depends on the genuine
                  perception<->label binding, not on marginal statistics.
oov_probe       — present out-of-vocabulary queries; a properly grounded learner
                  should give them low max-score (feeds metrics.oov_auroc).

Each returns a transformed (concept_ids / binding) the harness re-runs the full
learner set on, so the controls live on the SAME episodes and statistics as the
real conditions.
"""

from __future__ import annotations

import numpy as np


def random_label(
    concept_ids: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return a fully shuffled copy of the concept assignment (chance control)."""
    raise NotImplementedError


def permuted_binding(
    concept_ids: np.ndarray,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Permute the label<->extension binding while preserving class sizes
    (binding control)."""
    raise NotImplementedError


def oov_probe(
    in_vocab_hashes: list[str],
    oov_hashes: list[str],
) -> dict:
    """Package an in-vocab vs OOV query set for metrics.oov_auroc. Run the OOV
    falsification on cross-cutting / disjunctive concepts too, not just native
    (plan §6)."""
    raise NotImplementedError
