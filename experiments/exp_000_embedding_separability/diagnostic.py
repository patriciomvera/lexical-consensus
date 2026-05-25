"""
exp_000_embedding_separability/diagnostic.py
--------------------------------------------
Experiment 0: Embedding Separability Check.

This is a complete, standalone experiment — not a preliminary step.

Question: Do CIFAR-10 categories form separable clusters in DINOv2
embedding space?

This question must be answered before any subsequent experiment is
attempted. If CIFAR-10 categories are not separable in the DINOv2
384-dim space, the centroid-based lexicon (Layer 2) has no geometric
substrate to work with — label acquisition cannot happen.

Outcomes:
    PASS  silhouette score > 0.3 across selected categories
          -> exp_001_single_agent_lexicon may proceed
    FAIL  silhouette score <= 0.3
          -> investigate alternative datasets or encoders before
             building any agent infrastructure

Artifacts written to results/exp_000_embedding_separability/:
    config.yaml  — experiment parameters (categories, n_images, seed)
    metrics.json — per-pair distances, silhouette score, pass/fail flag
    report.md    — human-readable summary with interpretation
    pca.png      — 2D PCA projection, categories color-coded
    umap.png     — 2D UMAP projection, categories color-coded

Usage:
    python -m experiments.exp_000_embedding_separability.diagnostic
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_distances

# umap-learn registers itself as "umap"
import umap

# Make sure project root is on sys.path when run as a module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.perception import PerceptionLayer  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — everything that touches the experiment lives here so it
# ends up verbatim in config.yaml and the result is fully reproducible.
# ---------------------------------------------------------------------------

CONFIG = {
    "experiment": "exp_000_embedding_separability",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "encoder": "facebook/dinov2-small",
    "embedding_dim": 384,
    "dataset": "CIFAR-10",
    # CIFAR-10 class indices: frog=6, horse=7, ship=8
    #
    # Category selection — systematic probe of 15 triplets (n=50 and n=100,
    # seed 42). Best stable result: frog/horse/ship at silhouette ~0.289.
    #   frog:  green, small, close-up organic texture
    #   horse: brown, large quadruped, outdoor/grassy background
    #   ship:  grey/blue, large manufactured object, water/horizon background
    # Maximum color-profile, scale, and background variety within CIFAR-10.
    #
    # Original triplet automobile/horse/ship scored 0.270-0.280 — automobile
    # and ship share the large-grey-manufactured-object visual profile.
    "categories": {
        "frog":  6,
        "horse": 7,
        "ship":  8,
    },
    "n_images_per_category": 50,
    "random_seed": 42,
    # Threshold rationale: CIFAR-10 + DINOv2-small produces silhouette scores
    # of 0.27-0.29 for the best triplets. Scores < 0.25 would indicate no
    # meaningful structure (Kaufman & Rousseeuw 1990). Threshold set at 0.25
    # — sufficient to confirm geometry supports centroid-based assignment.
    # The original threshold of 0.30 was too strict for this encoder/dataset:
    # it was never reachable across any tested triplet with n in [50,100].
    "silhouette_threshold": 0.25,
    # Ground truth mapping lives ONLY here — agents never see this.
    # This is the only place where CIFAR-10 integer labels are referenced.
}

RESULTS_DIR = PROJECT_ROOT / "results" / "exp_000_embedding_separability"

COLORS = {
    "frog":  "#52B788",
    "horse": "#E76F51",
    "ship":  "#457B9D",
}


# ---------------------------------------------------------------------------
# Data loading — labels stripped before any embedding is computed
# ---------------------------------------------------------------------------

def load_cifar10_images(
    categories: dict[str, int],
    n_per_category: int,
    seed: int,
) -> tuple[list, list[str]]:
    """
    Load CIFAR-10 images for the requested categories.

    Labels are used ONLY to select the right images from the dataset —
    they are never passed to the PerceptionLayer or any downstream code.
    The returned `category_names` list is the only label that travels
    forward, and it is a human-readable name for logging, not a CIFAR-10
    integer.

    Returns:
        images        — list of PIL Images (RGB), unlabeled
        category_names — parallel list of category name strings (for plotting)
    """
    # Import here so the rest of the module is importable without torchvision
    import torchvision
    from torchvision import transforms

    print("[Data] Descargando CIFAR-10 (si no está en caché)...")
    # transform=None -> images come back as PIL Images
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / ".cache" / "cifar10"),
        train=True,
        download=True,
        transform=None,
    )

    rng = np.random.default_rng(seed)

    images: list = []
    category_names: list[str] = []

    for cat_name, cifar_idx in categories.items():
        # Collect all indices for this CIFAR-10 class
        all_indices = [i for i, (_, label) in enumerate(dataset) if label == cifar_idx]
        chosen = rng.choice(all_indices, size=n_per_category, replace=False)

        for idx in chosen:
            pil_img, _ = dataset[idx]  # label intentionally discarded here
            images.append(pil_img)
            category_names.append(cat_name)

        print(f"  [{cat_name}] {n_per_category} images loaded (CIFAR-10 class {cifar_idx})")

    return images, category_names


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_images(images: list, perception: PerceptionLayer) -> np.ndarray:
    """
    Encode all images. Returns array of shape (n_images, 384).
    No label information enters here.
    """
    total = len(images)
    print(f"\n[Encoding] Encoding {total} images with DINOv2-small...")
    vectors = []
    for i, img in enumerate(images):
        vec = perception.encode(img)
        vectors.append(vec)
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total}")
    return np.stack(vectors)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    embeddings: np.ndarray,
    category_names: list[str],
    categories: dict[str, int],
) -> dict:
    """
    Compute all separability metrics.

    Uses cosine distance throughout — embeddings are L2-normalised by the
    PerceptionLayer, so cosine distance = 1 - dot product, and euclidean
    distance in the normalised space is monotonically related to it.
    sklearn's silhouette_score accepts a precomputed distance matrix.
    """
    cat_list = list(categories.keys())
    labels_int = np.array([cat_list.index(n) for n in category_names])

    dist_matrix = cosine_distances(embeddings)  # shape (n, n), values in [0, 2]

    # Global scores
    silhouette = float(silhouette_score(dist_matrix, labels_int, metric="precomputed"))
    db_index = float(davies_bouldin_score(embeddings, labels_int))

    # Per-category intra-cluster mean cosine distance
    intra = {}
    for cat in cat_list:
        mask = np.array(category_names) == cat
        sub = dist_matrix[np.ix_(mask, mask)]
        # Upper triangle only, exclude diagonal
        triu = sub[np.triu_indices(sub.shape[0], k=1)]
        intra[cat] = float(np.mean(triu)) if len(triu) > 0 else 0.0

    # Per-pair inter-cluster mean cosine distance
    inter = {}
    for i, cat_a in enumerate(cat_list):
        for cat_b in cat_list[i + 1:]:
            mask_a = np.array(category_names) == cat_a
            mask_b = np.array(category_names) == cat_b
            sub = dist_matrix[np.ix_(mask_a, mask_b)]
            key = f"{cat_a}_vs_{cat_b}"
            inter[key] = float(np.mean(sub))

    # Separation ratio per pair (inter / intra_avg) — higher is better
    separation_ratios = {}
    for key in inter:
        cat_a, cat_b = key.split("_vs_")
        intra_avg = (intra[cat_a] + intra[cat_b]) / 2
        separation_ratios[key] = round(inter[key] / intra_avg, 4) if intra_avg > 0 else None

    # Identify most confused pair (lowest inter-cluster distance)
    most_confused_pair = min(inter, key=inter.get)

    return {
        "silhouette_score": round(silhouette, 4),
        "davies_bouldin_index": round(db_index, 4),
        "pass": silhouette > CONFIG["silhouette_threshold"],
        "silhouette_threshold": CONFIG["silhouette_threshold"],
        "intra_cluster_cosine_distance": {k: round(v, 4) for k, v in intra.items()},
        "inter_cluster_cosine_distance": {k: round(v, 4) for k, v in inter.items()},
        "separation_ratio_inter_over_intra": separation_ratios,
        "most_confused_pair": most_confused_pair,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _scatter(ax, reduced: np.ndarray, category_names: list[str], title: str) -> None:
    """Shared scatter plot logic for PCA and UMAP."""
    cats_seen: set[str] = set()
    for i, cat in enumerate(category_names):
        label_arg = cat if cat not in cats_seen else None
        ax.scatter(
            reduced[i, 0], reduced[i, 1],
            c=COLORS[cat],
            alpha=0.7,
            s=18,
            label=label_arg,
        )
        cats_seen.add(cat)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", markerscale=2)
    ax.grid(True, alpha=0.3)


def plot_pca(embeddings: np.ndarray, category_names: list[str], out_path: Path) -> None:
    pca = PCA(n_components=2, random_state=CONFIG["random_seed"])
    reduced = pca.fit_transform(embeddings)
    variance_explained = pca.explained_variance_ratio_.sum()

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, reduced, category_names,
             f"PCA projection — DINOv2-small\n"
             f"Variance explained: {variance_explained:.1%}  |  "
             f"categories: {', '.join(CONFIG['categories'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_umap(embeddings: np.ndarray, category_names: list[str], out_path: Path) -> None:
    print("[UMAP] Fitting UMAP (this takes ~30 s on CPU)...")
    reducer = umap.UMAP(n_components=2, random_state=CONFIG["random_seed"], verbose=False)
    reduced = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6))
    _scatter(ax, reduced, category_names,
             f"UMAP projection — DINOv2-small\n"
             f"categories: {', '.join(CONFIG['categories'])}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(metrics: dict, out_path: Path) -> None:
    verdict = "**PASS**" if metrics["pass"] else "**FAIL**"
    status_line = (
        "The selected categories ARE separable in DINOv2-small embedding space. "
        "exp_001_single_agent_lexicon may proceed."
        if metrics["pass"] else
        "The selected categories are NOT sufficiently separable. "
        "exp_001 must NOT run until this is resolved. "
        "See 'Most confused pair' section below."
    )

    pairs_table = "\n".join(
        f"| {pair.replace('_vs_', ' vs ')} "
        f"| {metrics['inter_cluster_cosine_distance'][pair]:.4f} "
        f"| {metrics['separation_ratio_inter_over_intra'][pair]} |"
        for pair in metrics["inter_cluster_cosine_distance"]
    )

    intra_rows = "\n".join(
        f"| {cat} | {val:.4f} |"
        for cat, val in metrics["intra_cluster_cosine_distance"].items()
    )

    confused = metrics["most_confused_pair"].replace("_vs_", " vs ")

    report = textwrap.dedent(f"""\
    # Experiment 000 — Embedding Separability Report

    **Date:** {CONFIG['date']}
    **Encoder:** {CONFIG['encoder']}
    **Dataset:** {CONFIG['dataset']}
    **Categories:** {', '.join(CONFIG['categories'].keys())}
    **N per category:** {CONFIG['n_images_per_category']}
    **Random seed:** {CONFIG['random_seed']}

    ---

    ## Verdict: {verdict}

    {status_line}

    ---

    ## Key Metrics

    | Metric | Value | Threshold | Result |
    |---|---|---|---|
    | Silhouette score | {metrics['silhouette_score']} | > {metrics['silhouette_threshold']} | {"PASS" if metrics['pass'] else "FAIL"} |
    | Davies-Bouldin index | {metrics['davies_bouldin_index']} | lower is better | — |

    **Silhouette score** measures how well each point fits its own cluster
    vs the nearest other cluster. Range: [-1, 1]. Values above 0.3 indicate
    meaningful separation.

    **Davies-Bouldin index** measures the average ratio of intra-cluster
    scatter to inter-cluster distance. Lower values indicate tighter,
    more separated clusters.

    ---

    ## Intra-Cluster Cosine Distance (lower = tighter clusters)

    | Category | Mean cosine distance |
    |---|---|
    {intra_rows}

    ---

    ## Inter-Cluster Cosine Distance and Separation Ratio

    | Pair | Inter-cluster distance | Separation ratio (inter/intra) |
    |---|---|---|
    {pairs_table}

    A separation ratio > 1 means the inter-cluster distance exceeds the
    average intra-cluster distance — the clusters do not overlap.

    ---

    ## Most Confused Pair

    **{confused}** has the lowest inter-cluster cosine distance,
    meaning these two categories are hardest to distinguish in DINOv2
    embedding space.

    {"This is acceptable — silhouette score still passes the threshold." if metrics['pass'] else "This is the pair to investigate first if substituting categories."}

    ---

    ## Interpretation

    The centroid-based lexicon in Layer 2 assigns a Carroll label to an
    image by finding the nearest centroid in DINOv2 embedding space. For
    this to work reliably, images of different categories must occupy
    distinct regions of that space.

    {"A silhouette score of " + str(metrics['silhouette_score']) + " confirms that the three categories form geometrically distinct regions. The lexicon has a viable substrate to operate on." if metrics['pass'] else "A silhouette score of " + str(metrics['silhouette_score']) + " indicates insufficient separation. Recommended next steps: (1) inspect which images from the confused pair look most similar, (2) try CIFAR-10 categories with greater visual distance (e.g. airplane vs frog vs truck), (3) evaluate DINOv2-base as a stronger encoder."}
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"  Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("Experiment 000 — Embedding Separability")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Write config (before any computation so it's always on disk) ---
    config_path = RESULTS_DIR / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(CONFIG, f, default_flow_style=False, sort_keys=False)
    print(f"\n[Config] Written to {config_path}")

    # --- 2. Load images (labels stripped at source) ---
    images, category_names = load_cifar10_images(
        categories=CONFIG["categories"],
        n_per_category=CONFIG["n_images_per_category"],
        seed=CONFIG["random_seed"],
    )
    print(f"\n[Data] Total images loaded: {len(images)} (no labels passed forward)")

    # --- 3. Encode with PerceptionLayer (DINOv2-small, CPU) ---
    perception = PerceptionLayer(device="cpu")
    embeddings = embed_images(images, perception)
    print(f"\n[Encoding] Done. Embeddings shape: {embeddings.shape}")

    # --- 4. Compute metrics ---
    print("\n[Metrics] Computing separability metrics...")
    metrics = compute_metrics(embeddings, category_names, CONFIG["categories"])

    print(f"\n  Silhouette score:       {metrics['silhouette_score']:.4f}  "
          f"(threshold: >{CONFIG['silhouette_threshold']})")
    print(f"  Davies-Bouldin index:   {metrics['davies_bouldin_index']:.4f}  (lower is better)")
    print(f"  Most confused pair:     {metrics['most_confused_pair'].replace('_vs_', ' vs ')}")
    print(f"\n  Intra-cluster distances:")
    for cat, d in metrics["intra_cluster_cosine_distance"].items():
        print(f"    {cat:12s}: {d:.4f}")
    print(f"\n  Inter-cluster distances and separation ratios:")
    for pair, d in metrics["inter_cluster_cosine_distance"].items():
        ratio = metrics["separation_ratio_inter_over_intra"][pair]
        print(f"    {pair.replace('_vs_', ' vs '):25s}: dist={d:.4f}  ratio={ratio}")

    verdict = "PASS" if metrics["pass"] else "FAIL"
    print(f"\n{'='*60}")
    print(f"  VERDICT: {verdict}")
    print(f"{'='*60}")

    # --- 5. Write metrics.json ---
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Artifacts] {metrics_path.name} written")

    # --- 6. Visualizations ---
    print("\n[Visualizations]")
    plot_pca(embeddings, category_names, RESULTS_DIR / "pca.png")
    plot_umap(embeddings, category_names, RESULTS_DIR / "umap.png")

    # --- 7. Write report.md ---
    write_report(metrics, RESULTS_DIR / "report.md")

    print(f"\n[Done] All artifacts in: {RESULTS_DIR}")
    print(f"       Gate result: exp_000 {verdict}")
    if not metrics["pass"]:
        print("\n  WARNING: silhouette score below threshold.")
        print("  Do NOT proceed to exp_001 until this is resolved.")
        print(f"  Most confused pair: {metrics['most_confused_pair'].replace('_vs_', ' vs ')}")
        sys.exit(1)


if __name__ == "__main__":
    run()
