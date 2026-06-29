"""
generate.py
-----------
Synthetic shapes×colors generator for the cross-cutting condition (exp_007c,
plan §4 and §9).

Cross-cutting concepts need a FACTORIAL set where object identity (shape) and an
attribute (color) vary independently, so that "color" is genuinely orthogonal to
"shape". CIFAR-10 cannot provide this; we synthesize it.

Design
------
  shapes  = {circle, square, triangle, star}     (4 identities)
  colors  = {red, green, blue, yellow}           (4 attribute values)
  => 16 factorial cells, ~200 images each, 224×224 RGB.

Per image, randomize (so neither shape nor color is trivially memorizable from
position/size): position jitter, scale (0.4–0.8 of frame), rotation, mild
background noise/texture, and slight color jitter WITHIN the color's hue band
(color is a band, not one RGB triple).

Outputs
-------
  PNG files under data/synthetic_shapes/images/
  manifest.csv with columns: image_id, path, shape, color, split

Determinism (plan §9, CLAUDE.md principle 3): one fixed master seed drives every
random choice and is written into the manifest header / a sidecar so the set is
exactly reproducible. Pure PIL + numpy, no GPU (plan §10).

After generation, embeddings are extracted ONCE with DINOv2 and cached to the
same .npy path convention the other datasets use — and the §4 substrate
diagnostic (descriptors.substrate_diagnostic) must pass before exp_007c is
considered valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHAPES = ("circle", "square", "triangle", "star")
COLORS = ("red", "green", "blue", "yellow")

IMG_SIZE = 224
N_PER_CELL = 200
MASTER_SEED = 42

# Hue bands (approximate RGB centers); color jitter samples within a band so
# "red" is a region of color space, not a single triple.
COLOR_BANDS: dict[str, tuple[int, int, int]] = {
    "red":    (220, 40, 40),
    "green":  (40, 200, 60),
    "blue":   (50, 90, 220),
    "yellow": (230, 210, 50),
}

OUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUT_DIR / "images"
MANIFEST_PATH = OUT_DIR / "manifest.csv"


@dataclass(frozen=True)
class ImageSpec:
    """Resolved randomized parameters for one synthetic image (logged so any
    image can be regenerated from its row)."""
    image_id: str
    shape: str
    color: str
    cx: float
    cy: float
    scale: float
    rotation_deg: float
    rgb: tuple[int, int, int]
    split: str


def draw_shape(spec: ImageSpec) -> "object":
    """Render one ImageSpec to a 224×224 RGB PIL image (PIL.ImageDraw).

    Scaffolding: signature only. Implementation draws the shape at (cx, cy)
    with the given scale/rotation/rgb over a mildly noisy background.
    """
    raise NotImplementedError


def build_specs(seed: int = MASTER_SEED) -> list[ImageSpec]:
    """Deterministically enumerate all factorial cells × N_PER_CELL image specs,
    assigning a reproducible train/query split per cell."""
    raise NotImplementedError


def main(seed: int = MASTER_SEED) -> None:
    """Generate every image + write manifest.csv. Idempotent given the seed."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
