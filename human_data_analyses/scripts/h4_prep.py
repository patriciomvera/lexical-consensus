"""H4 prep: reconstruct the 24-item test array per trial (identical fixed list in all 12 experiment scripts, js `test_pics`;
only the display order is shuffled) and compute DINOv2 distance of each test item to the mean of the trainPics embeddings."""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, re, pathlib, numpy as np, pandas as pd, torch
from PIL import Image
REPO = pathlib.Path(_P.MAIN_REPO); sys.path.insert(0, str(REPO))
from src.agents.perception import PerceptionLayer
B = pathlib.Path(_P.HDA); X = B / "data/xtmem"; IMG = pathlib.Path(f"{_P.XTMEM}/experiments/image_stim/images")
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
torch.set_num_threads(4); p = PerceptionLayer(device="cpu")
emb = {f.stem: p.encode(Image.open(f).convert("RGB")) for f in sorted(IMG.glob("c*.jpg"))}
emb = {k: v / np.linalg.norm(v) for k, v in emb.items()}
raw = pd.read_csv(X / "all_raw_A.csv"); key = pd.read_csv(X / "experiment_key.csv").iloc[:12]; key["exp"] = key.exp.astype(int)
rows = []
for _, r in raw.iterrows():
    for t in range(1, 13):
        tp = re.split(r"[,;\- ]+", str(r[f"trainPics_T{t}"]).strip()); sel = set(re.split(r"[,;\- ]+", str(r[f"selected_T{t}"]).strip())) if pd.notna(r[f"selected_T{t}"]) else set()
        assert all(x in emb for x in tp), tp
        m = np.mean([emb[x] for x in tp], 0); m /= np.linalg.norm(m)
        cat = r[f"category_T{t}"]; cnum = {"animals": 3, "vehicles": 2}.get(cat, 1)
        for it in TEST:
            rows.append(dict(exp=int(r.exp), subid=int(r.subids), trial=t, category=cat, condition=r[f"condition_T{t}"], image=it,
                             in_target=int(int(it[1]) == cnum), level=it[3:6], selected=int(it in sel), dist=1 - float(emb[it] @ m)))
d = pd.DataFrame(rows).merge(key[["exp", "exp_recoded", "preregistered"]], on="exp")
d.to_csv(B / "data/h4_long.csv", index=False)
print(d.shape, d.groupby(["in_target", "level"]).selected.mean().round(3).to_dict(), d.condition.value_counts().to_dict())
