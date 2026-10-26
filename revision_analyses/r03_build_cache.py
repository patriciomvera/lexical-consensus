"""Build .cache/embeddings_exp007/cifar_all10.npz with the repo's own code path
(experiments/exp_007_concept_carving/_shared.py::load_or_build_embeddings, torchvision CIFAR10, rng(42), DINOv2-small CPU).
CIFAR-10 archive: HF mirror MIT-OL-AI-D/cifar-10-python, MD5 c58f30108f718f92721af3b95e74349a (= torchvision's checked MD5)."""
import sys, time, importlib.util, torch
torch.set_num_threads(8)
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("s", "experiments/exp_007_concept_carving/_shared.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
t = time.time(); emb, rec = m.load_or_build_embeddings("cifar_all10"); print(len(emb), "embeddings", round(time.time() - t), "s")
