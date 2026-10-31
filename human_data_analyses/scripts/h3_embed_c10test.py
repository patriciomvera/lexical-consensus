"""H3 complement: DINOv2 embeddings (repo PerceptionLayer, unchanged) of the 10,000 CIFAR-10 test images (CIFAR-10H set)."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, pickle, pathlib, numpy as np, torch
from PIL import Image
REPO = pathlib.Path(_P.MAIN_REPO); sys.path.insert(0, str(REPO))
from src.agents.perception import PerceptionLayer
torch.set_num_threads(8)
tb = pickle.load(open(REPO / ".cache/cifar10/cifar-10-batches-py/test_batch", "rb"), encoding="latin1")
X = tb["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); lab = np.array(tb["labels"])
p = PerceptionLayer(device="cpu"); ck = pathlib.Path(f"{_P.CACHE}/dino_c10_test_partial.npy")
vec = list(np.load(ck)) if ck.exists() else []  # resume from checkpoint (added after an interruption killed the first run)
for i in range(len(vec), len(X)):
    vec.append(p.encode(Image.fromarray(X[i])))
    if (i + 1) % 250 == 0: np.save(ck, np.asarray(vec, dtype=np.float32)); print(i + 1, flush=True)
np.savez(f"{_P.CACHE}/dino_c10_test.npz", emb=np.asarray(vec, dtype=np.float32), labels=lab)
print("saved")
