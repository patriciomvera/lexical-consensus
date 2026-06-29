"""
modal_app.py — Modal cloud deployment for exp_008.

The embedding step (60K images × DINOv2-small) benefits from a T4/A10G GPU.
Everything else (distances, quadrants, episodes, analysis) runs on CPU and
can be called locally after the cache is built.

Usage:
    modal run modal_app.py::embed_cifar100
    modal run modal_app.py::run_all_phases   # embed + distances + report
"""

from __future__ import annotations

import modal

app = modal.App("exp-008-dissociation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision",
        "transformers", "accelerate",
        "nltk", "scipy", "statsmodels",
        "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn",
        "pyyaml",
    )
    .run_commands(
        # Download WordNet corpus inside the image
        "python -c \"import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')\""
    )
)

# Mount the local experiment directory so scripts can import _shared and src/
exp_mount    = modal.Mount.from_local_dir(".", remote_path="/root/exp_008")
project_mount = modal.Mount.from_local_dir("../../", remote_path="/root/project")

RESULTS_VOL = modal.Volume.from_name("exp008-results", create_if_missing=True)
CACHE_VOL   = modal.Volume.from_name("exp008-cache",   create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    mounts=[exp_mount, project_mount],
    volumes={"/root/results": RESULTS_VOL, "/root/cache": CACHE_VOL},
)
def embed_cifar100() -> str:
    """Embed all 60K CIFAR-100 train images using DINOv2-small on GPU."""
    import subprocess
    result = subprocess.run(
        ["python", "/root/exp_008/src/phase1/embed_cifar100.py"],
        capture_output=True, text=True,
        env={"PYTHONPATH": "/root/project", "HOME": "/root"},
    )
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout


@app.function(
    image=image,
    timeout=7200,
    mounts=[exp_mount, project_mount],
    volumes={"/root/results": RESULTS_VOL, "/root/cache": CACHE_VOL},
)
def run_all_phases() -> str:
    """Run the full exp_008 pipeline (except figures/report — run those locally)."""
    import subprocess

    steps = [
        ["python", "/root/exp_008/src/phase1/compute_perceptual_distances.py"],
        ["python", "/root/exp_008/src/phase1/compute_semantic_distances.py"],
        ["python", "/root/exp_008/src/phase1/correlation_diagnostic.py"],
        ["python", "/root/exp_008/src/phase2/classify_quadrants.py"],
        ["python", "/root/exp_008/src/phase2/sample_pairs.py"],
        ["python", "/root/exp_008/src/phase3/run_episodes.py"],
        ["python", "/root/exp_008/src/phase4/regression_analysis.py"],
        ["python", "/root/exp_008/src/phase4/dissociation_analysis.py"],
    ]
    env = {"PYTHONPATH": "/root/project", "HOME": "/root"}
    out = []
    for cmd in steps:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out.append(r.stdout)
        print(r.stdout)
        if r.returncode != 0:
            raise RuntimeError(f"FAILED: {' '.join(cmd)}\n{r.stderr}")
    return "\n".join(out)
