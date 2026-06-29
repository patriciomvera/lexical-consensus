"""
embed_cifar100.py — Step 1 of Phase 1.

Downloads (if needed) and encodes all CIFAR-100 training images using
DINOv2-small (same checkpoint as exp_007). Saves embeddings to
.cache/embeddings_exp008/cifar100.npz:
    emb:         (50000, 384) float32  L2-normalized CLS embeddings
    labels:      (50000,) int32        CIFAR-100 class index 0-99
    class_names: (100,) str            class name for each label index

Uses torchvision with transform=None (raw PIL images) and processes them in
batches via the DINOv2 AutoImageProcessor — the same pipeline as PerceptionLayer
but batched for GPU throughput. Local CPU runs are slower (~hours) — use Modal.

Also creates the timestamped run directory and writes a last_run.txt pointer
so subsequent phase scripts find it automatically.

Usage:
    python src/phase1/embed_cifar100.py [--run-dir PATH] [--device cpu|cuda|auto]
    python src/phase1/embed_cifar100.py --force-reembed
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared import CACHE_DIR, EMB_CACHE_DIR, RESULTS_BASE, create_run_dir, load_config


DINO_MODEL = "facebook/dinov2-small"
BATCH_SIZE = 128


def embed_all(cache_path: Path, device_str: str = "auto") -> None:
    """Encode all CIFAR-100 train images with DINOv2-small; write cache."""
    import torch
    import torch.nn.functional as F
    import torchvision
    from transformers import AutoImageProcessor, AutoModel

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"[embed] device = {device}")

    cifar_root = str(CACHE_DIR / "cifar100")
    print("[embed] loading CIFAR-100 train split (transform=None -> raw PIL) ...")
    # transform=None so we get PIL images; AutoImageProcessor handles resizing/normalization.
    dataset = torchvision.datasets.CIFAR100(
        root=cifar_root, train=True, download=True, transform=None,
    )
    n           = len(dataset)
    class_names = dataset.classes
    print(f"[embed] {n} images, {len(class_names)} classes")

    print(f"[embed] loading DINOv2-small ({DINO_MODEL}) ...")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model     = AutoModel.from_pretrained(DINO_MODEL).to(device)
    model.eval()

    dim = 384
    emb_matrix  = np.zeros((n, dim), dtype=np.float32)
    label_array = np.zeros(n, dtype=np.int32)

    with torch.no_grad():
        for start in range(0, n, BATCH_SIZE):
            end       = min(start + BATCH_SIZE, n)
            pil_imgs  = [dataset[i][0] for i in range(start, end)]
            lbls      = [dataset[i][1] for i in range(start, end)]
            inputs    = processor(images=pil_imgs, return_tensors="pt").to(device)
            outputs   = model(**inputs)
            cls_tok   = outputs.last_hidden_state[:, 0, :]          # (B, 384)
            normed    = F.normalize(cls_tok, dim=-1).cpu().numpy()
            emb_matrix[start:end]  = normed
            label_array[start:end] = lbls

            if (start // BATCH_SIZE) % 20 == 0:
                pct = 100.0 * end / n
                print(f"  {end:>6}/{n}  ({pct:.0f}%)", flush=True)

    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        emb=emb_matrix,
        labels=label_array,
        class_names=np.array(class_names),
    )
    print(f"[embed] saved {n} embeddings -> {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp_008 Phase 1 — embed CIFAR-100")
    parser.add_argument("--run-dir", default=None,
                        help="Explicit run directory. Created if not provided.")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for DINOv2 inference (default: auto).")
    parser.add_argument("--force-reembed", action="store_true",
                        help="Re-embed even if cache exists.")
    args = parser.parse_args()

    # Project root must be on sys.path for transformers model download cache
    project_root = Path(__file__).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    config     = load_config()
    cache_path = EMB_CACHE_DIR / "cifar100.npz"

    if cache_path.exists() and not args.force_reembed:
        print(f"[embed] cache already exists: {cache_path}")
        print(f"  (use --force-reembed to rebuild)")
    else:
        embed_all(cache_path, device_str=args.device)

    # Create (or reuse) the run directory
    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir)
        for sub in ["phase1/raw", "phase2/raw", "phase3/raw", "phase4/raw"]:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
    else:
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = create_run_dir(ts)

    ptr = RESULTS_BASE / "last_run.txt"
    ptr.write_text(str(run_dir), encoding="utf-8")
    print(f"[run_dir] pointer written -> {ptr}")
    print(f"\nRun directory: {run_dir}")
    print(f"Next step:\n  python src/phase1/compute_perceptual_distances.py")


if __name__ == "__main__":
    main()
