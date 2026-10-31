"""H5: DINOv2 (repo PerceptionLayer model+processor, CLS, L2) embeddings of a seeded subset of CIFAR-100 TRAIN images
(100 per class, numpy seed 42) for within-class dispersion. Batched call of the same processor/model as PerceptionLayer.encode."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, pathlib, numpy as np, torch, torchvision, torch.nn.functional as F
REPO = pathlib.Path(_P.MAIN_REPO); sys.path.insert(0, str(REPO))
from src.agents.perception import PerceptionLayer
torch.set_num_threads(4)
ds = torchvision.datasets.CIFAR100(root=f"{_P.CACHE}/cifar100", train=True, download=False)
t = np.array(ds.targets); rng = np.random.default_rng(42)
idx = np.concatenate([rng.choice(np.where(t == c)[0], 100, replace=False) for c in range(100)])
p = PerceptionLayer(device="cpu"); p._load() if getattr(p, "_model", None) is None else None
out = []
with torch.no_grad():
    for i in range(0, len(idx), 50):
        ims = [ds[int(j)][0].convert("RGB") for j in idx[i:i + 50]]
        x = p._processor(images=ims, return_tensors="pt")
        out.append(F.normalize(p._model(**x).last_hidden_state[:, 0, :], dim=-1).numpy())
        if i % 1000 == 0: print(i, flush=True)
# check equivalence with the single-image path
assert np.allclose(out[0][0], p.encode(ds[int(idx[0])][0]), atol=1e-4)
np.savez(f"{_P.CACHE}/dino_c100_train_sub100.npz", emb=np.concatenate(out), labels=t[idx], idx=idx, class_names=np.array(ds.classes))
print("saved")
