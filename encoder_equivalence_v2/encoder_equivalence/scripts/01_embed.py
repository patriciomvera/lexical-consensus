"""Encoder equivalence table, step 1: embed ONLY the 45 Lewis & Frank (2018) stimulus images (c{1,2,3}_*.jpg, 720x720).
No participant/trial data is read. One model is loaded at a time and freed afterwards (RAM is limited).
Output: out/emb_<ID>.npz (emb = L2-normalised float32 [45, D], names) and out/encoder_specs.json (preprocessing/pooling per encoder).
Usage: python 01_embed.py [ID ...]   (default: all)"""
import sys, gc, json, pathlib, importlib.util, numpy as np, torch, torchvision, torch.nn.functional as F
from PIL import Image
from huggingface_hub import snapshot_download
torch.set_num_threads(4)
B = pathlib.Path(__file__).resolve().parents[1]; OUT = B / "out"
IMG = pathlib.Path("/tmp/XTMEM/experiments/image_stim/images")
FILES = sorted(IMG.glob("c*.jpg")); NAMES = [f.stem for f in FILES]; assert len(FILES) == 45
IMS = [Image.open(f).convert("RGB") for f in FILES]
REV = {  # revisions verified 2026-11-18 via HF API
    "facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
    "wkvong/cvcl_s_dino_resnext50_embedding": "f50eaa0c50a6076a5190b1dd52aeeb6c3e747045",
    "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
    "google/vit-base-patch16-224": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
    "microsoft/resnet-50": "34c2154c194f829b11125337b98c8f5f9965ff19",
    "facebook/vit-mae-base": "25b184bea5538bf5c4c852c79d221195fdd2778d",
    "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "1a25a446712ba5ee05982a381eed697ef9b435cf"}
def dl(repo):
    return snapshot_download(repo, revision=REV[repo], allow_patterns=["*.json", "model.safetensors", "*.ckpt"])
def proc_spec(p):
    d = p.to_dict(); keep = ["image_processor_type", "do_resize", "size", "resample", "do_center_crop", "crop_size", "do_rescale",
                             "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb", "crop_pct"]
    return {k: d.get(k) for k in keep if k in d}
def l2(x): return F.normalize(x.float(), dim=-1).numpy().astype(np.float32)
SPECS = {}
def hf_vision(ID, repo, model_cls, pool, extra=None):
    from transformers import AutoImageProcessor
    path = dl(repo); p = AutoImageProcessor.from_pretrained(path); m = model_cls.from_pretrained(path, **(extra or {})).eval()
    x = p(images=IMS, return_tensors="pt")
    with torch.no_grad():
        e1 = pool(m, x); e2 = pool(m, x)
    assert torch.allclose(e1, e2, atol=1e-6), f"{ID}: non-deterministic embeddings"
    SPECS[ID] = dict(checkpoint=repo, revision=REV[repo], model_class=model_cls.__name__, processor=proc_spec(p), dim=int(e1.shape[-1]), **(extra and {"model_kwargs": extra} or {}))
    del m; gc.collect(); return l2(e1)
def run(ID):
    import transformers as T
    if ID == "R0_dinov2s":
        e = hf_vision(ID, "facebook/dinov2-small", T.AutoModel, lambda m, x: m(**x).last_hidden_state[:, 0]); SPECS[ID]["pooling"] = "CLS token of last_hidden_state (post-layernorm), last layer"
    elif ID == "O1_dinov2b":
        e = hf_vision(ID, "facebook/dinov2-base", T.AutoModel, lambda m, x: m(**x).last_hidden_state[:, 0]); SPECS[ID]["pooling"] = "CLS token of last_hidden_state (post-layernorm), last layer"
    elif ID == "E1_siglip":
        def pool(m, x):
            o = m(**x); return o.pooler_output
        e = hf_vision(ID, "google/siglip-base-patch16-224", T.SiglipVisionModel, pool); SPECS[ID]["pooling"] = "pooler_output of SiglipVisionModel (MAP attention-pooling head), last layer = image features used by get_image_features"
    elif ID == "E2_vitsup":
        e = hf_vision(ID, "google/vit-base-patch16-224", T.ViTModel, lambda m, x: m(**x).last_hidden_state[:, 0], {"add_pooling_layer": False}); SPECS[ID]["pooling"] = "CLS token of last_hidden_state (post-layernorm), last layer; classifier head not used"
    elif ID == "E3_rn50sup":
        e = hf_vision(ID, "microsoft/resnet-50", T.ResNetModel, lambda m, x: m(**x).pooler_output.flatten(1)); SPECS[ID]["pooling"] = "pooler_output = global average pool of last stage (2048); fc not used"
    elif ID in ("E4_mae", "E4s_mae_cls"):
        mean = ID == "E4_mae"
        NOISE = torch.arange(196, dtype=torch.float32).expand(len(IMS), 196)  # identity patch order: with mask_ratio=0 the random shuffle only adds ~5e-6 float noise
        pool = (lambda m, x: m(**x, noise=NOISE).last_hidden_state[:, 1:].mean(1)) if mean else (lambda m, x: m(**x, noise=NOISE).last_hidden_state[:, 0])
        e = hf_vision(ID, "facebook/vit-mae-base", T.ViTMAEModel, pool, {"mask_ratio": 0.0})
        SPECS[ID]["pooling"] = ("mean of patch tokens" if mean else "CLS token") + " of last_hidden_state (post-layernorm), mask_ratio=0.0 (config default 0.75); fixed identity noise so patch order is deterministic"
    elif ID == "O2_openclip":
        e = hf_vision(ID, "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", T.CLIPVisionModelWithProjection, lambda m, x: m(**x).image_embeds); SPECS[ID]["pooling"] = "image_embeds (CLS -> post-layernorm -> visual projection), last layer"
    elif ID == "R1_cvcl":
        spec = importlib.util.spec_from_file_location("h6", "/workspace/lexical-consensus-openmind/human_data_analyses/scripts/h6_embed_cvcl.py")
        h6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h6); m = h6.load_cvcl_vision()
        with torch.no_grad(): e = l2(m(torch.stack([h6.PRE(i) for i in IMS])))
        SPECS[ID] = dict(checkpoint="wkvong/cvcl_s_dino_resnext50_embedding", revision=REV["wkvong/cvcl_s_dino_resnext50_embedding"] + " (HF cache; loaded by h6_embed_cvcl.load_cvcl_vision)",
                         model_class="torchvision resnext50_32x4d, fc 2048->512", processor=dict(resize="224x224 bicubic (no aspect-preserving resize)", center_crop=None, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]),
                         pooling="fc output (512) after global average pool, as in H6", dim=512); del m; gc.collect()
    elif ID.startswith("F1_rn50rand"):
        seed = int(ID[-1]); torch.manual_seed(seed); m = torchvision.models.resnet50(weights=None); m.fc = torch.nn.Identity(); m.eval()
        pre = torchvision.transforms.Compose([torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC), torchvision.transforms.CenterCrop(224),
                                              torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        with torch.no_grad(): e = l2(m(torch.stack([pre(i) for i in IMS])))
        SPECS[ID] = dict(checkpoint=f"torchvision resnet50(weights=None), torch.manual_seed({seed})", revision=f"torchvision {torchvision.__version__}", model_class="ResNet-50 random init (eval mode)",
                         processor=dict(resize="shorter side 256 bicubic", center_crop=224, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]), pooling="global average pool (2048), fc removed", dim=2048); del m; gc.collect()
    elif ID == "F2_pixels":
        pre = torchvision.transforms.Compose([torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC), torchvision.transforms.CenterCrop(224),
                                              torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = torch.stack([pre(i) for i in IMS]); x = F.interpolate(x, size=(32, 32), mode="area"); e = l2(x.flatten(1))
        SPECS[ID] = dict(checkpoint="none (pixels)", revision="-", model_class="identity", processor=dict(resize="shorter side 256 bicubic", center_crop=224, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], then="area downsample to 32x32"),
                         pooling="flattened RGB 3x32x32 = 3072", dim=3072)
    else: raise ValueError(ID)
    SPECS[ID]["metric"] = "cosine distance 1 - cos on L2-normalised vectors"
    np.savez(OUT / f"emb_{ID}.npz", emb=e, names=np.array(NAMES)); print(ID, e.shape, flush=True)
ALL = ["R0_dinov2s", "R1_cvcl", "E1_siglip", "E2_vitsup", "E3_rn50sup", "E4_mae", "E4s_mae_cls", "F1_rn50rand0", "F1_rn50rand1", "F1_rn50rand2", "F2_pixels", "O1_dinov2b", "O2_openclip"]
if __name__ == "__main__":
    sp = OUT / "encoder_specs.json"; old = json.loads(sp.read_text()) if sp.exists() else {}
    for ID in (sys.argv[1:] or ALL):
        try: run(ID)
        except Exception as ex: print("FAILED", ID, repr(ex), flush=True); SPECS[ID] = {"FAILED": repr(ex)}
        old.update(SPECS); sp.write_text(json.dumps(old, indent=1, default=str)); gc.collect()
