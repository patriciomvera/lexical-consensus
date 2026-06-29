"""
test_learner_interface.py — interface-contract tests for src/learners.

exp_007's whole comparison rests on every learner honoring ONE interface
(base.Learner), so the dissociation test is fair across mechanisms. These tests
pin the contract every concrete learner must satisfy:

  1. predict() == argmax(score()) — predict is derived, never independent.
  2. score() shape is (n_concepts,) for one embedding and (n, n_concepts) for a
     batch, with HIGHER == more likely.
  3. predict/score/retrieve before fit() raises (no silent defaults —
     CLAUDE.md principle 2).
  4. retrieve(concept_id, cands) returns a valid index into cands.

They are written against the registry so a newly added learner is covered
automatically. Skipped until the learners are implemented (currently scaffolds
that raise NotImplementedError).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.learners import LEARNER_REGISTRY  # noqa: E402


@pytest.fixture
def toy_episode():
    """Tiny separable 3-way / 2-shot episode in a 4-d space, fixed seed."""
    rng = np.random.default_rng(0)
    dim, n_concepts, shots = 4, 3, 2
    centers = np.eye(n_concepts, dim)
    seed_emb = np.repeat(centers, shots, axis=0) + 0.01 * rng.standard_normal((n_concepts * shots, dim))
    seed_emb /= np.linalg.norm(seed_emb, axis=1, keepdims=True)
    seed_ids = np.repeat(np.arange(n_concepts), shots)
    return dim, n_concepts, seed_emb, seed_ids


def _instantiate(name: str, n_concepts: int):
    """Resolve a 'module:Class' registry entry and construct it. TODO: factor
    into a shared helper once run_exp007.build_learner exists."""
    module_path, cls_name = LEARNER_REGISTRY[name].split(":")
    mod = __import__(module_path, fromlist=[cls_name])
    return getattr(mod, cls_name)(n_concepts, seed=0)


@pytest.mark.parametrize("name", list(LEARNER_REGISTRY))
def test_predict_is_argmax_of_score(name, toy_episode):
    dim, n_concepts, seed_emb, seed_ids = toy_episode
    learner = _instantiate(name, n_concepts)
    learner.fit(seed_emb, seed_ids)
    s = learner.score(seed_emb[0])
    assert s.shape == (n_concepts,)
    assert learner.predict(seed_emb[0]) == int(np.argmax(s))


@pytest.mark.parametrize("name", list(LEARNER_REGISTRY))
def test_raises_before_fit(name, toy_episode):
    _, n_concepts, seed_emb, _ = toy_episode
    learner = _instantiate(name, n_concepts)
    with pytest.raises(RuntimeError):
        learner.predict(seed_emb[0])


@pytest.mark.parametrize("name", list(LEARNER_REGISTRY))
def test_batch_score_shape(name, toy_episode):
    _, n_concepts, seed_emb, seed_ids = toy_episode
    learner = _instantiate(name, n_concepts)
    learner.fit(seed_emb, seed_ids)
    s = learner.score(seed_emb)                      # batch
    assert s.shape == (len(seed_emb), n_concepts)
    # predict() on a batch agrees with argmax of the batch scores.
    np.testing.assert_array_equal(learner.predict(seed_emb), np.argmax(s, axis=1))


@pytest.mark.parametrize("name", list(LEARNER_REGISTRY))
def test_retrieve_returns_valid_index(name, toy_episode):
    _, n_concepts, seed_emb, seed_ids = toy_episode
    learner = _instantiate(name, n_concepts)
    learner.fit(seed_emb, seed_ids)
    idx = learner.retrieve(0, seed_emb)
    assert 0 <= idx < len(seed_emb)


@pytest.mark.parametrize("name", ["centroid", "kcentroid", "exemplar_knn", "logreg", "linsvm"])
def test_non_random_learners_separate_clean_clusters(name, toy_episode):
    """On a trivially separable toy episode, every non-chance learner should
    name its own training points correctly. (random is excluded — it ignores
    the embedding by design.)"""
    _, n_concepts, seed_emb, seed_ids = toy_episode
    learner = _instantiate(name, n_concepts)
    learner.fit(seed_emb, seed_ids)
    np.testing.assert_array_equal(learner.predict(seed_emb), seed_ids)
