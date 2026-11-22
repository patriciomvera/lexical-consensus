"""Vision-only parameter counts per encoder and pipeline check R0 vs v1 PerceptionLayer (45 stimuli only; no participant data)."""
import sys, gc, json, importlib.util, pathlib, numpy as np, torch, torchvision, transformers as T
from PIL import Image
sys.argv = ["x"]; exec(open(pathlib.Path(__file__).parent / "01_embed.py").read().split("ALL = [")[0])
out = {}
def n(m): return int(sum(p.numel() for p in m.parameters()))
for ID, repo, cls, kw in [("R0_dinov2s", "facebook/dinov2-small", T.AutoModel, {}), ("O1_dinov2b", "facebook/dinov2-base", T.AutoModel, {}),
                          ("E1_siglip", "google/siglip-base-patch16-224", T.SiglipVisionModel, {}), ("O2_openclip", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", T.CLIPVisionModelWithProjection, {}),
                          ("E2_vitsup", "google/vit-base-patch16-224", T.ViTModel, {"add_pooling_layer": False}), ("E3_rn50sup", "microsoft/resnet-50", T.ResNetModel, {}),
                          ("E4_mae", "facebook/vit-mae-base", T.ViTMAEModel, {"mask_ratio": 0.0})]:
    m = cls.from_pretrained(dl(repo), **kw); out[ID] = n(m); del m; gc.collect()
spec = importlib.util.spec_from_file_location("h6", "/workspace/lexical-consensus-openmind/human_data_analyses/scripts/h6_embed_cvcl.py")
h6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h6); m = h6.load_cvcl_vision(); out["R1_cvcl"] = n(m); del m; gc.collect()
m = torchvision.models.resnet50(weights=None); m.fc = torch.nn.Identity(); out["F1_rn50rand"] = n(m); out["F2_pixels"] = 0
sys.path.insert(0, "/workspace/lexical-consensus-openmind/revision_runs/repo"); from src.agents.perception import PerceptionLayer
z = np.load(OUT / "emb_R0_dinov2s.npz"); pl = PerceptionLayer(device="cpu")
v1 = np.stack([pl.encode(Image.open(IMG / f"{nm}.jpg").convert("RGB")) for nm in z["names"]]); v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
D1, D2 = 1 - v1 @ v1.T, 1 - z["emb"] @ z["emb"].T
chk = dict(max_abs_emb=float(np.abs(v1 - z["emb"]).max()), max_abs_pairwise_d=float(np.abs(D1 - D2).max()))
json.dump(dict(vision_params=out, R0_vs_v1_PerceptionLayer=chk), open(OUT / "vision_params_and_checks.json", "w"), indent=1); print(out, chk)
