"""S6 (plan §5.4): the same 10 primary sources embedded with COMMON preprocessing: shorter side 256 bicubic, centre crop 224,
then each model's OWN normalisation (image_mean/image_std of its processor; ImageNet for R1/F1/F2). Same checkpoints, revisions
and pooling as 01_embed.py (imported, not copied). Only the 45 stimulus images are read; no participant data.
Output: <out_dir>/emb_<ID>_s6common.npz (+ encoder_specs_s6common.json). Default out_dir = encoder_equivalence/out.
Usage: python 01b_embed_common.py [--out-dir DIR] [ID ...]   (default: the 10 primary IDs)"""
import sys, json, pathlib, importlib.util, shutil, numpy as np, torch, torchvision
from torchvision import transforms as TT
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("embed01", HERE / "01_embed.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
PRIMARY = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def common(mean, std):
    return TT.Compose([TT.Resize(256, interpolation=TT.InterpolationMode.BICUBIC), TT.CenterCrop(224), TT.ToTensor(), TT.Normalize(mean, std)])
def hf_vision_common(ID, repo, model_cls, pool, extra=None):
    from transformers import AutoImageProcessor
    path = E.dl(repo); p = AutoImageProcessor.from_pretrained(path); m = model_cls.from_pretrained(path, **(extra or {})).eval()
    mean, std = list(p.image_mean), list(p.image_std); x = {"pixel_values": torch.stack([common(mean, std)(i) for i in E.IMS])}
    with torch.no_grad():
        e1 = pool(m, x); e2 = pool(m, x)
    assert torch.allclose(e1, e2, atol=1e-6), f"{ID}: non-deterministic embeddings"
    E.SPECS[ID] = dict(checkpoint=repo, revision=E.REV[repo], model_class=model_cls.__name__, dim=int(e1.shape[-1]),
                       processor=dict(resize="shorter side 256 bicubic", center_crop=224, image_mean=mean, image_std=std, note="S6 common preprocessing"))
    del m; return E.l2(e1)
def run_cvcl_common():
    spec = importlib.util.spec_from_file_location("h6", "/workspace/lexical-consensus-openmind/human_data_analyses/scripts/h6_embed_cvcl.py")
    h6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h6); m = h6.load_cvcl_vision()
    with torch.no_grad(): e = E.l2(m(torch.stack([common(*IMNET)(i) for i in E.IMS])))
    E.SPECS["R1_cvcl"] = dict(checkpoint="wkvong/cvcl_s_dino_resnext50_embedding", processor=dict(resize="shorter side 256 bicubic", center_crop=224, image_mean=IMNET[0], image_std=IMNET[1], note="S6 common preprocessing"), pooling="as 01_embed", dim=512)
    return e
if __name__ == "__main__":
    a = sys.argv[1:]; out = E.OUT
    if a[:1] == ["--out-dir"]: out = pathlib.Path(a[1]); a = a[2:]
    out.mkdir(parents=True, exist_ok=True); tmp = out / "_s6tmp"; tmp.mkdir(exist_ok=True); E.OUT = tmp; E.hf_vision = hf_vision_common
    specs = {}
    for ID in (a or PRIMARY):
        if ID == "R1_cvcl":
            e = run_cvcl_common(); np.savez(tmp / f"emb_{ID}.npz", emb=e, names=np.array(E.NAMES))
        else:
            E.run(ID)  # F1/F2 already use exactly the common preprocessing in 01_embed (ImageNet normalisation)
        shutil.move(tmp / f"emb_{ID}.npz", out / f"emb_{ID}_s6common.npz"); specs[ID] = E.SPECS.get(ID); print("S6", ID, flush=True)
    tmp.rmdir(); (out / "encoder_specs_s6common.json").write_text(json.dumps(specs, indent=1, default=str))
