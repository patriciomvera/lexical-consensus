"""H6: encode CIFAR images with the CVCL vision encoder (Vong et al. 2024, HF wkvong/cvcl_s_dino_resnext50_embedding, MIT).
Read-only download of the checkpoint via hf_hub_download. Architecture = torchvision resnext50_32x4d with fc 2048->512,
weights = state_dict keys 'model.image_embed.model.*' (multimodal-baby/multimodal/multimodal.py::VisionEncoder).
Preprocessing = MultiModalLitModel.load_model preprocess (Resize 224x224 bicubic, ToTensor, ImageNet Normalize).
Output embeddings are L2-normalised (CVCL encode_image normalises; plan H6 also requires L2).
Outputs:
  cache/cvcl_cifar100_train.npz  (emb 50000x512, labels, class_names)  -- same format as exp_008 cifar100.npz
  cache/cvcl_cifar_all10.npz     (emb 2000x512, hashes, categories)    -- same images/order as exp_007 cifar_all10 cache
"""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, pickle, pathlib, numpy as np, torch, torchvision, importlib.util
from torchvision import transforms
from huggingface_hub import hf_hub_download
torch.set_num_threads(8); torch.manual_seed(0)
ROOT = pathlib.Path(_P.ROOT); CACHE = pathlib.Path(f"{_P.CACHE}")
REPO = ROOT / "revision_runs/repo"
class _U(pickle.Unpickler):
    def find_class(s, m, n):
        try: return super().find_class(m, n)
        except Exception: return type(n, (), {"__init__": lambda self, *a, **k: None, "__setstate__": lambda self, st: None})
class _P: Unpickler = _U; load = pickle.load
def load_cvcl_vision():
    n = "cvcl_s_dino_resnext50_embedding"
    ck = torch.load(hf_hub_download(repo_id="wkvong/" + n, filename=n + ".ckpt"), map_location="cpu", weights_only=False, pickle_module=_P)
    pre = "model.image_embed.model."
    sd = {k[len(pre):]: v for k, v in ck["state_dict"].items() if k.startswith(pre)}
    m = torchvision.models.resnext50_32x4d(weights=None); m.fc = torch.nn.Linear(2048, 512)
    missing = m.load_state_dict(sd, strict=True); m.eval(); return m
PRE = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                          transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
@torch.no_grad()
def encode(model, pil_list, bs=64):
    out = []
    for i in range(0, len(pil_list), bs):
        x = torch.stack([PRE(p.convert("RGB")) for p in pil_list[i:i + bs]])
        out.append(torch.nn.functional.normalize(model(x), dim=-1).numpy())
        if (i // bs) % 50 == 0: print(f"  {i + len(x)}/{len(pil_list)}", flush=True)
    return np.concatenate(out).astype(np.float32)
if __name__ == "__main__":
    model = load_cvcl_vision(); print("CVCL vision encoder loaded (strict)")
    which = sys.argv[1:] or ["c10", "c100"]
    if "c10" in which:
        spec = importlib.util.spec_from_file_location("s7", REPO / "experiments/exp_007_concept_carving/_shared.py")
        sys.path.insert(0, str(REPO)); s7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s7)
        ds = torchvision.datasets.CIFAR10(root=str(REPO / ".cache/cifar10"), train=True, download=False, transform=None)
        rng = np.random.default_rng(s7.MASTER_SEED); hashes, cats, imgs = [], [], []
        targets = np.array(ds.targets)
        for cat, info in s7.DATASET_CATEGORIES["cifar_all10"].items():
            idxs = [int(i) for i in np.where(targets == info["cifar_idx"])[0]]
            chosen = rng.choice(idxs, size=s7.N_IMAGES_PER_CATEGORY, replace=False)
            for j, gi in enumerate(chosen):
                hashes.append(f"{cat}_{j:03d}"); cats.append(cat); imgs.append(ds[int(gi)][0])
        ref = np.load(REPO / ".cache/embeddings_exp007/cifar_all10.npz")
        assert list(ref["hashes"]) == hashes, "image order differs from exp_007 cache"
        np.savez(CACHE / "cvcl_cifar_all10.npz", emb=encode(model, imgs), hashes=np.array(hashes), categories=np.array(cats))
        print("saved cvcl_cifar_all10.npz")
    if "c100" in which:
        ds = torchvision.datasets.CIFAR100(root=str(CACHE / "cifar100"), train=True, download=False, transform=None)
        imgs = [ds[i][0] for i in range(len(ds))]
        np.savez(CACHE / "cvcl_cifar100_train.npz", emb=encode(model, imgs), labels=np.array(ds.targets, dtype=np.int32), class_names=np.array(ds.classes))
        print("saved cvcl_cifar100_train.npz")
