"""10_agent_embed.py on SYNTHETIC images only (random-noise PNG/JPG; no CIFAR, no L&F stimulus is opened):
 (1) parity: a COPY of the frozen 01_embed.py (only IMG/OUT paths replaced, written to /tmp) embeds 45 synthetic 'stimuli';
     10_agent_embed.make_encoder must reproduce them within 1e-5 (F2, F1 and, if the HF cache has them, R0 and E4/E4s real weights);
     a perturbed frozen file must FAIL parity;
 (2) c10pool selection = exp_007 rule rng(42) (independent re-implementation; checked pixel by pixel) and the seed-42 order assert;
 (3) chunked/resumable encoding gives identical arrays; manifest has sha256; EXISTING caches are never re-encoded;
 (4) throughput on noise images (img/s per encoder at AGENT_TORCH_THREADS) -> used for the cost estimate."""
import os, sys, json, time, shutil, pathlib, subprocess, importlib, numpy as np
T = pathlib.Path("/tmp/synth_agents_embed"); shutil.rmtree(T, ignore_errors=True); T.mkdir(parents=True); os.environ["AGENTS_SYNTH_DIR"] = str(T)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("AGENT_TORCH_THREADS", "2")
HERE = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE / "scripts")); import lib_agents as L; L.ensure_dirs()
from PIL import Image
rng = np.random.default_rng(7); PY = sys.executable; REAL = [x for x in sys.argv[1:] if not x.startswith("-")] or ["R0_dinov2s", "E4_mae"]
# ---- (1) parity
lf = T / "lf_stimuli"; lf.mkdir(); (T / "human/out").mkdir(parents=True)
for c in (1, 2, 3):
    for k in range(15): Image.fromarray(rng.integers(0, 256, (720, 720, 3), dtype=np.uint8)).save(lf / f"c{c}_{k:02d}.jpg", quality=95)
src = (L.EE / "scripts/01_embed.py").read_text()
assert 'IMG = pathlib.Path("/tmp/XTMEM/experiments/image_stim/images")' in src and 'OUT = B / "out"' in src
pat = src.replace('IMG = pathlib.Path("/tmp/XTMEM/experiments/image_stim/images")', f'IMG = pathlib.Path("{lf}")').replace('OUT = B / "out"', f'OUT = pathlib.Path("{T}/human/out")'); (T / "01_embed_patched.py").write_text(pat)
def hf_ok(repo):
    try:
        from huggingface_hub import snapshot_download; E = importlib.import_module("10_agent_embed"); snapshot_download(repo, revision=E.REV[repo], allow_patterns=["*.json", "model.safetensors"], local_files_only=True); return True
    except Exception: return False
REPOS = {"R0_dinov2s": "facebook/dinov2-small", "E4_mae": "facebook/vit-mae-base", "O1_dinov2b": "facebook/dinov2-base", "E2_vitsup": "google/vit-base-patch16-224", "E1_siglip": "google/siglip-base-patch16-224", "E3_rn50sup": "microsoft/resnet-50", "O2_openclip": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"}
ids = ["F2_pixels", "F1_rn50rand0"] + [i for i in REAL if (i in REPOS and hf_ok(REPOS[i])) or i == "R1_cvcl"]   # R1: .ckpt via hf_hub_download (h6_embed_cvcl)
for i in REAL:
    if i not in ids: print(f"[embed] SKIP {i}: real weights not in the local HF cache (pinned revision) -> parity of {i} NOT tested here (run_agents.sh step 1 still checks it)")
r = subprocess.run([PY, str(T / "01_embed_patched.py"), *ids] + (["E4s_mae_cls"] if "E4_mae" in ids else []), capture_output=True, text=True, env=os.environ | {"PYTHONPATH": ""}); assert r.returncode == 0, r.stderr[-2000:]
assert "FAILED" not in r.stdout, r.stdout
E = importlib.import_module("10_agent_embed")
for i in ids:
    assert E.parity(i), i
    d = json.loads((L.RES_DIR / f"parity_{i}.json").read_text()); print(f"[embed] parity {i}: PASS, max|Δ| = {max(v['max_abs_diff'] for v in d['detail'].values()):.2e} on {list(d['detail'])}")
z = dict(np.load(T / "human/out/emb_F2_pixels.npz")); z["emb"] = z["emb"].copy(); z["emb"][3, 5] += 1e-3; np.savez(T / "human/out/emb_F2_pixels.npz", **z)
assert not E.parity("F2_pixels"); print("[embed] perturbed frozen embedding (+1e-3) -> parity FAIL (run_agents.sh stops)")
# ---- (2) pool selection
s7 = L._load("s7_t", L.E7 / "_shared.py"); cats = list(s7.DATASET_CATEGORIES["cifar_all10"]); npc = s7.N_IMAGES_PER_CATEGORY
cd = T / "cifar"; cd.mkdir(); n10 = 10 * (npc + 20); y10 = rng.permutation(np.repeat(np.arange(10), npc + 20)); x10 = rng.integers(0, 256, (n10, 32, 32, 3), dtype=np.uint8)
np.savez(cd / "c10train.npz", x=x10, y=y10, class_names=np.array(sorted(cats)))
np.savez(cd / "c10test.npz", x=x10[:70], y=y10[:70], class_names=np.array(sorted(cats)))
y100 = np.repeat(np.arange(100), 3); np.savez(cd / "c100train.npz", x=rng.integers(0, 256, (300, 32, 32, 3), dtype=np.uint8), y=y100, class_names=np.array([f"k{i:02d}" for i in range(100)]))
exp = []; g = np.random.default_rng(42)
for c, info in s7.DATASET_CATEGORIES["cifar_all10"].items(): exp += [int(i) for i in g.choice(np.where(y10 == info["cifar_idx"])[0], size=npc, replace=False)]
L.V1_C10_CACHE.parent.mkdir(parents=True, exist_ok=True); np.savez(L.V1_C10_CACHE, emb=np.zeros((10 * npc, 2)), hashes=np.array([f"{c}_{j:03d}" for c in cats for j in range(npc)]), categories=np.array([c for c in cats for _ in range(npc)]))
ims, meta = E.load_images("c10pool"); assert len(ims) == 10 * npc and all(np.array_equal(np.asarray(a), x10[i]) for a, i in zip(ims, exp))
ims0, _ = E.load_images("c10pool_seed0"); assert not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(ims0, ims))
np.savez(L.V1_C10_CACHE, emb=np.zeros((10 * npc, 2)), hashes=np.array([f"{c}_{j:03d}" for c in cats[::-1] for j in range(npc)]), categories=np.array([c for c in cats[::-1] for _ in range(npc)]))
try: E.load_images("c10pool"); raise SystemExit("order assert did not fire")
except AssertionError: pass
print(f"[embed] c10pool = exp_007 rule rng(42), {10 * npc} images, pixel-identical to an independent re-implementation; seed 0 differs; wrong v1 order -> assert")
# ---- (3) chunking / resume / manifest / EXISTING
E.CHUNK = 128; f = E.make_encoder("F2_pixels"); ims, _ = E.load_images("c10test"); pd_ = T / "parts_t"
a = E.encode_all(f, ims, pd_, log=lambda *_: None); shutil.rmtree(pd_); E.encode_all(f, ims[:0] + ims, pd_, log=lambda *_: None)
(pd_ / "part_000000.npz").unlink(); b = E.encode_all(f, ims, pd_, log=lambda *_: None); assert np.array_equal(a["F2_pixels"], b["F2_pixels"]) and np.array_equal(a["F2_pixels"], f(ims)["F2_pixels"])
E.CHUNK = 2048; recs = E.run_set("F2_pixels", "c100train"); z = np.load(L.emb_path("F2_pixels", "c100train")); assert z["emb"].shape == (300, 3072) and list(z["labels"]) == list(y100)
m = [json.loads(l) for l in open(L.EMB_DIR / "embed_manifest.jsonl")]; assert m[-1]["sha256"] == L.sha256(L.emb_path("F2_pixels", "c100train"))
assert E.run_set("R0_dinov2s", "c10pool") is None and E.run_set("R1_cvcl", "c100train") is None
print("[embed] chunked + resumed encoding identical to one pass; npz + manifest sha256 OK; EXISTING caches (R0 pool, R1 CIFAR-100) not re-encoded")
# ---- (4) throughput on noise images (32x32 upsampled by each processor, as CIFAR)
tp = {}; noise = [Image.fromarray(x) for x in x10[:128]]
for i in ids + ["F1_rn50rand0"]:
    if i in tp: continue
    f = E.make_encoder(i); f(noise[:8]); t0 = time.time(); f(noise[:64]); f(noise[64:128]); tp[i] = 128 / (time.time() - t0)
L.write_json(T / "throughput.json", dict(threads=int(os.environ["AGENT_TORCH_THREADS"]), img_per_s=tp)); print("[embed] throughput img/s:", {k: round(v, 1) for k, v in tp.items()})
print("ALL EMBED TESTS PASSED")
