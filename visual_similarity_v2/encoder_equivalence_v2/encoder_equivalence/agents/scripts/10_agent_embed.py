"""AGENTS addendum, step 1: embeddings of CIFAR images per encoder (proposal §2). Produces ONLY npz + sha256: no centroid,
distance or accuracy is computed here. RUN ONLY AFTER FREEZE_agentes_v2.txt exists (run_agents.sh checks it).
Same checkpoints/revisions, preprocessing and pooling as encoder_equivalence/scripts/01_embed.py (encoder_specs.json); the
equivalence is checked first on the 45 Lewis & Frank stimuli against the frozen out/emb_<ID>.npz (--parity).
Image sets (proposal §2):
  c100train      CIFAR-100 train, 50,000, torchvision order            -> emb, labels, class_names    (D, D-todos, H1/H2 CIFAR-100, H5)
  c10pool        CIFAR-10 pool, 2,000 = 200/class, exp_007 rule rng(42) -> emb, hashes, categories     (T, H1 CIFAR-10H, H3)
  c10test        CIFAR-10 test, 10,000                                  -> emb, labels                 (H3 per-image complement only)
  c10pool_seed<s> pool re-drawn with rng(s), s in 0..3 (T-sem 'full', optional sensitivity)
Existing caches are reused and never re-encoded (lib_agents.EXISTING: R0 pool/test/T-sem pools, R1 CIFAR-100 and pool).
E4_mae also writes E4s_mae_cls from the same forward pass (proposal §4.2).
Usage: python 10_agent_embed.py --encoder ID --set SET      |   python 10_agent_embed.py --parity ID
Output: agents/out/emb/<SET>_<ID>.npz and one line per file in agents/out/emb/embed_manifest.jsonl (sha256, n, dim, seconds)."""
import os, sys, gc, json, time, argparse, pathlib, importlib.util, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import lib_agents as L
import torch, torchvision, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get("AGENT_TORCH_THREADS", "2")))
BS = 64; CHUNK = 2048
# ---- encoders: literal copy of the per-ID branches of 01_embed.py, with the image list passed per batch instead of the global IMS
REV = {"facebook/dinov2-small": "ed25f3a31f01632728cabb09d1542f84ab7b0056", "wkvong/cvcl_s_dino_resnext50_embedding": "f50eaa0c50a6076a5190b1dd52aeeb6c3e747045",
       "google/siglip-base-patch16-224": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", "google/vit-base-patch16-224": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
       "microsoft/resnet-50": "34c2154c194f829b11125337b98c8f5f9965ff19", "facebook/vit-mae-base": "25b184bea5538bf5c4c852c79d221195fdd2778d",
       "facebook/dinov2-base": "f9e44c814b77203eaa57a6bdbbd535f21ede1415", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "1a25a446712ba5ee05982a381eed697ef9b435cf"}
def l2(x): return F.normalize(x.float(), dim=-1).numpy().astype(np.float32)
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
def _tv_pre():
    T = torchvision.transforms
    return T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(*IMNET)])
def make_encoder(ID):
    """Returns f(list of PIL RGB) -> {ID: L2 float32 [n, D]} (E4_mae also returns E4s_mae_cls)."""
    if ID in ("R0_dinov2s", "O1_dinov2b", "E1_siglip", "E2_vitsup", "E3_rn50sup", "E4_mae", "E4s_mae_cls", "O2_openclip"):
        import transformers as T
        from transformers import AutoImageProcessor
        from huggingface_hub import snapshot_download
        repo, cls, pool, extra = {
            "R0_dinov2s": ("facebook/dinov2-small", T.AutoModel, lambda m, x, n: {"R0_dinov2s": m(**x).last_hidden_state[:, 0]}, None),
            "O1_dinov2b": ("facebook/dinov2-base", T.AutoModel, lambda m, x, n: {"O1_dinov2b": m(**x).last_hidden_state[:, 0]}, None),
            "E1_siglip": ("google/siglip-base-patch16-224", T.SiglipVisionModel, lambda m, x, n: {"E1_siglip": m(**x).pooler_output}, None),
            "E2_vitsup": ("google/vit-base-patch16-224", T.ViTModel, lambda m, x, n: {"E2_vitsup": m(**x).last_hidden_state[:, 0]}, {"add_pooling_layer": False}),
            "E3_rn50sup": ("microsoft/resnet-50", T.ResNetModel, lambda m, x, n: {"E3_rn50sup": m(**x).pooler_output.flatten(1)}, None),
            "E4_mae": ("facebook/vit-mae-base", T.ViTMAEModel, None, {"mask_ratio": 0.0}),
            "E4s_mae_cls": ("facebook/vit-mae-base", T.ViTMAEModel, None, {"mask_ratio": 0.0}),
            "O2_openclip": ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", T.CLIPVisionModelWithProjection, lambda m, x, n: {"O2_openclip": m(**x).image_embeds}, None)}[ID]
        if ID in ("E4_mae", "E4s_mae_cls"):
            def pool(m, x, n):  # identity patch order, as 01_embed.py (NOISE = arange(196) expanded to the batch)
                h = m(**x, noise=torch.arange(196, dtype=torch.float32).expand(n, 196)).last_hidden_state
                return {"E4_mae": h[:, 1:].mean(1), "E4s_mae_cls": h[:, 0]}
        path = snapshot_download(repo, revision=REV[repo], allow_patterns=["*.json", "model.safetensors", "*.ckpt"])
        p = AutoImageProcessor.from_pretrained(path); m = cls.from_pretrained(path, **(extra or {})).eval()
        def f(ims):
            x = p(images=ims, return_tensors="pt")
            with torch.no_grad(): return {k: l2(v) for k, v in pool(m, x, len(ims)).items()}
        return f
    if ID == "R1_cvcl":
        spec = importlib.util.spec_from_file_location("h6", str(L.HDA / "scripts/h6_embed_cvcl.py")); h6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(h6)
        m = h6.load_cvcl_vision()
        def f(ims):
            with torch.no_grad(): return {ID: l2(m(torch.stack([h6.PRE(i) for i in ims])))}
        return f
    if ID.startswith("F1_rn50rand"):
        torch.manual_seed(int(ID[-1])); m = torchvision.models.resnet50(weights=None); m.fc = torch.nn.Identity(); m.eval(); pre = _tv_pre()
        def f(ims):
            with torch.no_grad(): return {ID: l2(m(torch.stack([pre(i) for i in ims])))}
        return f
    if ID == "F2_pixels":
        pre = _tv_pre()
        def f(ims):
            x = torch.stack([pre(i) for i in ims]); x = F.interpolate(x, size=(32, 32), mode="area"); return {ID: l2(x.flatten(1))}
        return f
    raise ValueError(ID)
# ---- images
def _synth_set(name):
    z = np.load(pathlib.Path(L.SYNTH) / "cifar" / f"{name}.npz", allow_pickle=False)
    from PIL import Image
    return [Image.fromarray(a).convert("RGB") for a in z["x"]], z["y"].astype(int), [str(c) for c in z["class_names"]]
def load_images(s):
    """(PIL list, metadata dict to store in the npz). The c10pool rule is the exp_007 _shared.load_or_build_embeddings rule, as h6_embed_cvcl.py."""
    from PIL import Image
    if s == "c100train":
        if L.SYNTH: ims, y, cn = _synth_set("c100train")
        else:
            ds = torchvision.datasets.CIFAR100(root=str(L.CIFAR100_ROOT), train=True, download=False, transform=None)
            ims = [ds[i][0] for i in range(len(ds))]; y = np.array(ds.targets); cn = list(ds.classes)
        return ims, dict(labels=np.asarray(y, dtype=np.int32), class_names=np.array(cn))
    if s == "c10test":
        if L.SYNTH: ims, y, _ = _synth_set("c10test")
        else:
            ds = torchvision.datasets.CIFAR10(root=str(L.CIFAR10_ROOT), train=False, download=False, transform=None)
            ims = [ds[i][0] for i in range(len(ds))]; y = np.array(ds.targets)
        return ims, dict(labels=np.asarray(y, dtype=np.int64))
    if s == "c10pool" or s.startswith("c10pool_seed"):
        seed = 42 if s == "c10pool" else int(s[len("c10pool_seed"):])
        s7 = L._load("s7_agents", L.E7 / "_shared.py")
        if L.SYNTH: ims_all, targets, _ = _synth_set("c10train")
        else:
            ds = torchvision.datasets.CIFAR10(root=str(L.CIFAR10_ROOT), train=True, download=False, transform=None); targets = np.array(ds.targets)
        rng = np.random.default_rng(seed); hashes, cats, ims = [], [], []
        for cat, info in s7.DATASET_CATEGORIES["cifar_all10"].items():
            idxs = [int(i) for i in np.where(targets == info["cifar_idx"])[0]]
            chosen = rng.choice(idxs, size=s7.N_IMAGES_PER_CATEGORY, replace=False)
            for j, gi in enumerate(chosen):
                hashes.append(f"{cat}_{j:03d}"); cats.append(cat); ims.append(ims_all[int(gi)] if L.SYNTH else ds[int(gi)][0])
        if seed == 42:
            ref = np.load(L.V1_C10_CACHE)
            assert [str(h) for h in ref["hashes"]] == hashes and [str(c) for c in ref["categories"]] == cats, "pool order differs from exp_007 cache"
        return ims, dict(hashes=np.array(hashes), categories=np.array(cats))
    raise ValueError(s)
def encode_all(f, ims, partdir, log=print):
    """Batches of BS; checkpoint every CHUNK images in partdir (resumable). Returns {ID: array}."""
    partdir.mkdir(parents=True, exist_ok=True); t0 = time.time(); n = len(ims)
    for start in range(0, n, CHUNK):
        pf = partdir / f"part_{start:06d}.npz"
        if pf.exists(): continue
        acc = {}
        for i in range(start, min(start + CHUNK, n), BS):
            for k, v in f(ims[i:min(i + BS, start + CHUNK, n)]).items(): acc.setdefault(k, []).append(v)
        np.savez(pf, **{k: np.concatenate(v) for k, v in acc.items()})
        done = min(start + CHUNK, n); el = time.time() - t0; log(f"  {done}/{n} images, {60 * done / max(el, 1e-9):.0f} img/min (this session)")
    parts = sorted(partdir.glob("part_*.npz")); out = {}
    for pf in parts:
        z = np.load(pf)
        for k in z.files: out.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v) for k, v in out.items()}; assert all(len(v) == n for v in out.values()), "incomplete parts"
    return out
def manifest_line(path, ID, s, n, dim, secs):
    rec = dict(file=str(path.name), encoder=ID, set=s, n=int(n), dim=int(dim), sha256=L.sha256(path), seconds=round(secs, 1),
               torch=torch.__version__, torchvision=torchvision.__version__, threads=torch.get_num_threads(), time=time.strftime("%F %T %Z"))
    with open(L.EMB_DIR / "embed_manifest.jsonl", "a") as fh: fh.write(json.dumps(rec) + "\n")
    return rec
def run_set(ID, s, limit=None):
    L.ensure_dirs()
    if (ID, s) in L.EXISTING: print(f"{ID} {s}: existing cache reused, not re-encoded: {L.EXISTING[(ID, s)]}"); return None
    out = L.emb_path(ID, s)
    if out.exists(): print(f"{ID} {s}: already done ({out})"); return None
    t0 = time.time(); ims, meta = load_images(s)
    if limit:
        assert L.SYNTH, "--limit is for synthetic tests only"; ims = ims[:limit]; meta = {k: v[:limit] for k, v in meta.items() if k != "class_names"} | ({"class_names": meta["class_names"]} if "class_names" in meta else {})
    f = make_encoder(ID); E = encode_all(f, ims, L.EMB_DIR / f".parts_{s}_{ID}", log=lambda m: print(f"[{ID} {s}]{m}", flush=True)); secs = time.time() - t0; recs = []
    for k, v in E.items():
        p = L.emb_path(k, s)
        if p.exists() or (k, s) in L.EXISTING: continue
        np.savez(p, emb=v, **meta); recs.append(manifest_line(p, k, s, len(v), v.shape[1], secs)); print(k, s, v.shape, "->", p, flush=True)
    import shutil; shutil.rmtree(L.EMB_DIR / f".parts_{s}_{ID}", ignore_errors=True); return recs
# ---- parity on the 45 Lewis & Frank stimuli (images only; no participant data)
LF_DIR = pathlib.Path(L.SYNTH) / "lf_stimuli" if L.SYNTH else pathlib.Path("/tmp/XTMEM/experiments/image_stim/images")
FROZEN_EMB = pathlib.Path(L.SYNTH) / "human/out" if L.SYNTH else L.EE / "out"
PARITY_TOL = 1e-5
def parity(ID):
    from PIL import Image
    files = sorted(LF_DIR.glob("c*.jpg")); assert len(files) == 45, f"{len(files)} L&F stimuli"
    ims = [Image.open(x).convert("RGB") for x in files]; names = [x.stem for x in files]
    got = make_encoder(ID)(ims); res = {}
    for k, v in got.items():
        ref = FROZEN_EMB / f"emb_{k}.npz"
        if not ref.exists(): continue
        z = np.load(ref); assert [str(n) for n in z["names"]] == names
        d = float(np.abs(v - z["emb"]).max()); res[k] = dict(max_abs_diff=d, tol=PARITY_TOL, ok=bool(d <= PARITY_TOL), frozen_sha256=L.sha256(ref))
    ok = bool(res) and all(r["ok"] for r in res.values()); L.write_json(L.RES_DIR / f"parity_{ID}.json", dict(encoder=ID, ok=ok, detail=res))
    print(f"PARITY {ID}: {'PASS' if ok else 'FAIL'} {res}"); return ok
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--encoder"); ap.add_argument("--set"); ap.add_argument("--parity"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.parity: sys.exit(0 if parity(a.parity) else 1)
    assert a.encoder in L.P10 + ["F1_rn50rand1", "F1_rn50rand2"], f"unknown encoder {a.encoder} (E4s comes from the E4_mae pass)"
    assert a.set in L.SETS or a.set.startswith("c10pool_seed"), a.set
    run_set(a.encoder, a.set, a.limit)
