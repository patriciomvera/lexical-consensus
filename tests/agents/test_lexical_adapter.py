"""
test_lexical_adapter.py — Unit tests for src/agents/lexical_adapter.py

LexicalAdapter is the first gradient-trained component in the project. These
tests verify the four invariants that downstream experiments (exp_005b and on)
rely on:

  1. forward(x) is always unit-normed — centroid math assumes this.
  2. Initialization is near identity — initial inter-agent geometry must be
     close to what the gradient-free experiments saw, otherwise the H1 test
     is contaminated by initialization noise.
  3. Gradients flow through W — without this the experiment cannot train W
     at all and Condition B would silently degenerate to Condition A.
  4. save/load is a true round-trip — drift metrics depend on W_initial
     surviving serialization unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.lexical_adapter import LexicalAdapter


# ─── Output norm ──────────────────────────────────────────────────────────────

def test_output_is_unit_norm_single_vector():
    """forward(x) for a (dim,) input returns a unit-normed (dim,) vector."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384)
    x = torch.randn(384)
    z = adapter(x)
    assert z.shape == (384,)
    assert abs(float(torch.norm(z).item()) - 1.0) < 1e-5


def test_output_is_unit_norm_batched():
    """forward(x) for a (batch, dim) input returns (batch, dim), each row unit-normed."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384)
    x = torch.randn(8, 384)
    z = adapter(x)
    assert z.shape == (8, 384)
    norms = torch.norm(z, dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)


def test_output_is_unit_norm_even_when_input_is_not():
    """Input does not need to be unit-normed; output is normalized regardless."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=64)
    x = torch.randn(64) * 17.0
    z = adapter(x)
    assert abs(float(torch.norm(z).item()) - 1.0) < 1e-5


# ─── Initialization near identity ─────────────────────────────────────────────

def test_init_frobenius_distance_to_identity_is_small():
    """
    With epsilon=0.01 and dim=384, expected ||W-I||_F is approximately
    epsilon * sqrt(dim * dim) = 0.01 * 384 = 3.84. Allow some slack.
    """
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384, epsilon=0.01)
    dist = adapter.frobenius_distance_to_identity()
    # Expected value ~3.84; reject anything indicating misinitialization
    assert 2.0 < dist < 6.0


def test_init_drift_from_initial_is_zero():
    """At construction time the adapter has not been updated, so ||W - W_0||_F = 0."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=128, epsilon=0.01)
    assert adapter.frobenius_drift_from_initial() == pytest.approx(0.0, abs=1e-7)


def test_init_acts_approximately_as_identity_on_normed_input():
    """
    For an x that is already unit normed, z = normalize(W @ x) should be
    close to x at init because W is close to I.
    """
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384, epsilon=0.01)
    x = torch.randn(384)
    x = x / torch.norm(x)
    z = adapter(x)
    cos_sim = float(torch.dot(x, z).item())
    # Not exactly 1.0, but very close (epsilon * noise barely moves direction)
    assert cos_sim > 0.95


# ─── Gradient flow ────────────────────────────────────────────────────────────

def test_gradient_flows_through_W():
    """
    A scalar loss built from forward(x) must produce a non-zero gradient on W.
    If W.grad is None or all-zero, optimizer.step() in exp_005b would do
    nothing and Condition B would silently equal Condition A.
    """
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384, epsilon=0.01)
    x = torch.randn(384)
    x = x / torch.norm(x)
    target = torch.randn(384)
    target = target / torch.norm(target)

    z = adapter(x)
    loss = 1.0 - torch.dot(z, target)  # cosine distance
    loss.backward()

    assert adapter.W.grad is not None
    assert float(adapter.W.grad.abs().sum().item()) > 0.0


def test_adam_step_actually_changes_W():
    """
    Running one Adam step on a non-trivial loss must change W and produce
    nonzero drift_from_initial. This is the end-to-end sanity check that
    exp_005b's update rule will work.
    """
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384, epsilon=0.01)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    x = torch.randn(384)
    x = x / torch.norm(x)
    target = torch.randn(384)
    target = target / torch.norm(target)

    z_before = adapter(x).detach().clone()

    optimizer.zero_grad()
    loss = 1.0 - torch.dot(adapter(x), target)
    loss.backward()
    optimizer.step()

    z_after = adapter(x).detach()

    # z must have moved at least a little toward the target
    cos_before = float(torch.dot(z_before, target).item())
    cos_after  = float(torch.dot(z_after,  target).item())
    assert cos_after > cos_before

    # And drift_from_initial must be strictly positive after one step
    assert adapter.frobenius_drift_from_initial() > 0.0


# ─── save / load round-trip ───────────────────────────────────────────────────

def test_save_load_roundtrip_preserves_W(tmp_path):
    """W and W_initial must survive save -> load unchanged (bit-for-bit on CPU)."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=64, epsilon=0.01)
    # Mutate W so that W != W_initial — tests that both buffers round-trip.
    with torch.no_grad():
        adapter.W.add_(0.05)
    path = tmp_path / "adapter.pt"
    adapter.save(path)

    loaded = LexicalAdapter.load(path)

    assert torch.allclose(loaded.W,         adapter.W,         atol=0.0)
    assert torch.allclose(loaded.W_initial, adapter.W_initial, atol=0.0)
    assert loaded.dim     == adapter.dim
    assert loaded.epsilon == adapter.epsilon


def test_save_load_preserves_drift_metric(tmp_path):
    """After mutating W, the drift metric must survive save/load."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=64, epsilon=0.01)
    with torch.no_grad():
        adapter.W.add_(0.05)
    drift_before = adapter.frobenius_drift_from_initial()
    assert drift_before > 0.0

    path = tmp_path / "adapter.pt"
    adapter.save(path)
    loaded = LexicalAdapter.load(path)

    drift_after = loaded.frobenius_drift_from_initial()
    assert abs(drift_after - drift_before) < 1e-6


# ─── NumPy convenience ────────────────────────────────────────────────────────

def test_project_numpy_matches_forward():
    """project_numpy(x) and forward(torch.from_numpy(x)) must agree."""
    torch.manual_seed(0)
    adapter = LexicalAdapter(dim=384, epsilon=0.01)
    x_np = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    x_np = x_np / np.linalg.norm(x_np)

    z_np = adapter.project_numpy(x_np)
    z_th = adapter(torch.from_numpy(x_np)).detach().numpy()

    assert z_np.shape == (384,)
    assert np.allclose(z_np, z_th, atol=1e-6)
    assert abs(float(np.linalg.norm(z_np)) - 1.0) < 1e-5
