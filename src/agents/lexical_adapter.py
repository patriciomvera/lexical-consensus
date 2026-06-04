"""
lexical_adapter.py
------------------
LexicalAdapter — agent-specific trainable projection over frozen DINOv2 embeddings.

This is the first gradient-trained component introduced in the Lexical Consensus
project. Until exp_005b the architecture was deliberately gradient-free: DINOv2
was frozen, centroids were updated by simple averaging, and consensus feedback
only changed which images contributed to a centroid — never how an agent
transformed perceptual space.

exp_005 confirmed that under the gradient-free architecture, consensus feedback
does not substantially reshape inter-agent representational geometry: the only
remaining variation between agents was the floor imposed by their disjoint seed
sets, and feedback could not push past it. The cause was architectural rather
than empirical — there was no mechanism for representational rewriting.

exp_005b reintroduces such a mechanism via a per-agent trainable linear adapter:

    z_i = normalize(W_i @ x)

where
    x   — frozen, agent-independent DINOv2 embedding (384-dim, unit norm)
    W_i — agent-specific projection matrix (384 x 384), trained from consensus
          ACCEPT / REJECT events
    z_i — projected embedding that the agent uses to build centroids and to
          classify new images

The adapter is initialized as the identity plus small Gaussian noise:

    W_i = I + epsilon * N(0, 1),   epsilon = 0.01

This guarantees that
  (a) all agents start with nearly identical geometry (the identity), so any
      observed inter-agent drift is attributable to the experimental conditions
      rather than to initialization, and
  (b) there is a small initial gap between agents that consensus feedback can
      potentially close — without it, the H1 test would be trivially satisfied.

The forward pass is the only place where W enters classification; downstream
code in exp_005b treats z_i exactly as it previously treated x. This keeps the
new component isolated and easy to ablate (Condition A in exp_005b freezes W
and therefore reduces to the exp_005 setup).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LexicalAdapter(nn.Module):
    """
    Agent-specific trainable projection over frozen DINOv2 embeddings.

    Allows consensus feedback to reshape an agent's categorical geometry
    without modifying the underlying perceptual encoder. The encoder remains
    frozen and shared across agents; only W differs per agent and only W
    is updated by gradient descent.

    This component enables a proper operational test of weak Sapir-Whorf in
    the Lexical Consensus system: shared lexical feedback can pull W toward
    the geometry implied by the Carroll labels, while raw DINOv2 distances
    remain invariant.
    """

    def __init__(self, dim: int = 384, epsilon: float = 0.01) -> None:
        super().__init__()
        # W = I + epsilon * N(0, 1). The identity component keeps the initial
        # projection close to a no-op; the Gaussian perturbation gives each
        # agent a unique starting geometry so that inter-agent alignment is
        # a non-trivial quantity to measure.
        init = torch.eye(dim) + epsilon * torch.randn(dim, dim)
        self.W = nn.Parameter(init)
        self.dim = dim
        self.epsilon = epsilon

        # Snapshot of the initial W. Used by drift metrics
        # (||W_t - W_0||_F) without coupling to whatever optimizer or
        # checkpointing strategy the experiment runner chooses. Stored as a
        # buffer so it travels with the module on save / load and is not
        # treated as a learnable parameter.
        self.register_buffer("W_initial", self.W.detach().clone())

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a frozen DINOv2 embedding into the agent's lexical space.

        Args:
            x: tensor of shape (dim,) or (batch, dim). Expected to be unit
               normed (this is how PerceptionLayer returns embeddings) but
               nothing here enforces it — the output is always normalized.

        Returns:
            z = normalize(W @ x), same shape as x, last dim unit-normed.
        """
        # Support both single-vector and batched inputs. matmul handles
        # both (dim,) @ (dim, dim) → (dim,) and (batch, dim) @ (dim, dim) → (batch, dim)
        # when we transpose W on the right.
        if x.dim() == 1:
            z = self.W @ x
        else:
            z = x @ self.W.T
        return F.normalize(z, dim=-1)

    # ------------------------------------------------------------------
    # Geometry diagnostics
    # ------------------------------------------------------------------

    def frobenius_distance_to_identity(self) -> float:
        """
        ||W - I||_F. Measures how far the current projection has departed from
        a no-op. Near zero at init (only the epsilon * N(0,1) noise), grows as
        feedback reshapes W. Used as a regularizer target in exp_005b's loss
        and as a per-round diagnostic in adapter_drift.csv.
        """
        eye = torch.eye(self.dim, device=self.W.device, dtype=self.W.dtype)
        return float(torch.norm(self.W - eye).item())

    def frobenius_drift_from_initial(self) -> float:
        """
        ||W_t - W_0||_F. Cumulative drift from this adapter's own initialization,
        independent of how noisy the init happened to be. This is what
        adapter_drift.csv tracks for the temporal plot.
        """
        return float(torch.norm(self.W - self.W_initial).item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """
        Save W, W_initial, and the construction args to a .pt file. Round-trips
        through `LexicalAdapter.load(path)`.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "dim":       self.dim,
                "epsilon":   self.epsilon,
                "W":         self.W.detach().cpu(),
                "W_initial": self.W_initial.detach().cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LexicalAdapter":
        """
        Load an adapter previously written by `save()`. The reconstructed
        module has the same dim, epsilon, current W, and initial W as the
        original — so drift metrics remain comparable across runs.
        """
        # weights_only=False because the checkpoint contains Python ints
        # (dim, epsilon) alongside tensors. The file is project-internal.
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        adapter = cls(dim=int(blob["dim"]), epsilon=float(blob["epsilon"]))
        with torch.no_grad():
            adapter.W.copy_(blob["W"])
            adapter.W_initial.copy_(blob["W_initial"])
        return adapter

    # ------------------------------------------------------------------
    # NumPy convenience (for centroid math that lives outside autograd)
    # ------------------------------------------------------------------

    def project_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the current W to a numpy embedding and return a unit-normed
        numpy vector. Useful for building centroids in projected space without
        ever entering autograd — centroids are not differentiable targets in
        exp_005b, only references for the per-image gradient step.
        """
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32))
            z = self.forward(t).cpu().numpy()
        return z

    def __repr__(self) -> str:
        return (
            f"LexicalAdapter(dim={self.dim}, epsilon={self.epsilon}, "
            f"||W-I||_F={self.frobenius_distance_to_identity():.4f}, "
            f"||W-W0||_F={self.frobenius_drift_from_initial():.4f})"
        )
