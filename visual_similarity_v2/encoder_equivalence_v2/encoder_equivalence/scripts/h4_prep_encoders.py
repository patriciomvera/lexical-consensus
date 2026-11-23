"""H4 prep for the encoder comparison (v2 of arXiv 2611.14116). Identical to human_data_analyses/scripts/h4_prep.py except
that embeddings come from encoder_equivalence/out/emb_<ID>.npz (01_embed.py) instead of PerceptionLayer.
Same 24-item TEST list, same centroid (mean of L2 train embeddings, renormalised), d = 1 - cos. Same row order as h4_prep.py.
Usage:
  python h4_prep_encoders.py --raw <all_raw_A.csv> --key <experiment_key.csv> --emb-dir <dir with emb_*.npz> --out-dir <dir> [--suffix ""] ID [ID ...]
  python h4_prep_encoders.py --check <h4_long_R0_dinov2s.csv> <v1 h4_long.csv> [pipeline_control.json]   # G0: |d diff| <= 1e-5 and same rank order with near-ties (< 1e-6) merged (BC3)
The weekend run writes data to <out-dir>/h4_long_<ID><suffix>.csv. Participant data are read ONLY in the weekend run, after the freeze."""
import argparse, re, sys, pathlib, numpy as np, pandas as pd
TEST = [f"c{c}_{l}{i}" for c in (1, 2, 3) for l, k in (("sub", 2), ("bas", 2), ("sup", 4)) for i in range(1, k + 1)]
SPLIT = r"[,;\- ]+"
def build(raw, key, emb):
    rows = []
    for _, r in raw.iterrows():
        for t in range(1, 13):
            tp = re.split(SPLIT, str(r[f"trainPics_T{t}"]).strip())
            sel = set(re.split(SPLIT, str(r[f"selected_T{t}"]).strip())) if pd.notna(r[f"selected_T{t}"]) else set()
            assert all(x in emb for x in tp), tp
            m = np.mean([emb[x] for x in tp], 0); m /= np.linalg.norm(m)
            cat = r[f"category_T{t}"]; cnum = {"animals": 3, "vehicles": 2}.get(cat, 1)
            for it in TEST:
                rows.append(dict(exp=int(r.exp), subid=int(r.subids), trial=t, category=cat, condition=r[f"condition_T{t}"], image=it,
                                 in_target=int(int(it[1]) == cnum), level=it[3:6], selected=int(it in sel), dist=1 - float(emb[it] @ m)))
    return pd.DataFrame(rows).merge(key[["exp", "exp_recoded", "preregistered"]], on="exp")
def load_emb(path):
    z = np.load(path)  # read BY IMAGE NAME (names array), float32 as v1 PerceptionLayer; per-vector normalisation exactly as h4_prep.py
    return {str(n): v / np.linalg.norm(v) for n, v in zip(z["names"], z["emb"].astype(np.float32))}
NEAR_TIE = 1e-6  # BC3 (REVISION3): in v1, 207 of 494 gaps between consecutive d values are < 1e-6 (same centroid computed with another
                 # image order); the smallest real gap is 4.6e-6 (Ariadna's measurement). Values closer than 1e-6 are one near-tie cluster.
def rank_order_ok(new, v1, near_tie=NEAR_TIE):
    """Same rank order up to near-ties: cluster the v1 values (single linkage, consecutive gaps < near_tie); the new values must keep
    the clusters strictly ordered (max of cluster k < min of cluster k+1). Equivalent to Ariadna's 'round to 6 decimals' rule except at
    rounding boundaries, where round(6) would split a near-tie (e.g. ...4999 vs ...5001) and stop G0 for nothing."""
    o = np.argsort(v1, kind="stable"); vs = v1[o]; ns = new[o]
    cl = np.concatenate([[0], np.cumsum(np.diff(vs) >= near_tie)])
    k = cl.max() + 1; mx = np.full(k, -np.inf); mn = np.full(k, np.inf)
    np.maximum.at(mx, cl, ns); np.minimum.at(mn, cl, ns)
    return bool((mx[:-1] < mn[1:]).all())
def near_tie_max_width(v1, near_tie=NEAR_TIE):
    """REVISION3b (minor): width (max - min of the v1 values) of the widest near-tie group. Single linkage can chain; a group whose
    width reaches near_tie is no longer 'one value computed twice' and the check must not merge it (measured on v1: 6.0e-8, 288 groups)."""
    vs = np.sort(v1); cl = np.concatenate([[0], np.cumsum(np.diff(vs) >= near_tie)]); k = cl.max() + 1
    lo = np.full(k, np.inf); hi = np.full(k, -np.inf); np.minimum.at(lo, cl, vs); np.maximum.at(hi, cl, vs)
    return float((hi - lo).max())
def check(new_csv, v1_csv, tol=1e-5, json_out=None):
    a = pd.read_csv(new_csv, usecols=["exp", "subid", "trial", "image", "dist"]); b = pd.read_csv(v1_csv, usecols=["exp", "subid", "trial", "image", "dist"])
    same_rows = a[["exp", "subid", "trial", "image"]].equals(b[["exp", "subid", "trial", "image"]])
    diff = float(np.abs(a.dist.to_numpy() - b.dist.to_numpy()).max()) if same_rows else float("nan")
    width = near_tie_max_width(b.dist.to_numpy(float)); no_chain = width < NEAR_TIE   # REVISION3b: guard against chained near-tie groups
    same_rank = same_rows and rank_order_ok(a.dist.to_numpy(float), b.dist.to_numpy(float))
    ok = bool(same_rows and diff <= tol and same_rank and no_chain)
    print(f"PIPELINE_CONTROL rows_identical={same_rows} max_abs_diff={diff:.3g} tol={tol} same_rank_order(near-ties<{NEAR_TIE:g} merged)={same_rank} "
          f"near_tie_group_max_width={width:.3g} (<{NEAR_TIE:g}: {no_chain}) -> {'PASS' if ok else 'FAIL (G0: stop)'}")
    if json_out:
        import json; pathlib.Path(json_out).write_text(json.dumps(dict(ok=ok, rows_identical=bool(same_rows), max_abs_diff=diff, tol=tol, near_tie=NEAR_TIE, same_rank_order=bool(same_rank),
                                                       near_tie_group_max_width=width, near_tie_groups_ok=bool(no_chain))))
    return ok
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        sys.exit(0 if check(sys.argv[2], sys.argv[3], json_out=sys.argv[4] if len(sys.argv) > 4 else None) else 1)
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--emb-dir", required=True); ap.add_argument("--out-dir", required=True); ap.add_argument("--suffix", default=""); ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    raw = pd.read_csv(a.raw); key = pd.read_csv(a.key).iloc[:12]; key["exp"] = key.exp.astype(int)
    for ID in a.ids:
        d = build(raw, key, load_emb(pathlib.Path(a.emb_dir) / f"emb_{ID}{a.suffix}.npz"))
        out = pathlib.Path(a.out_dir) / f"h4_long_{ID}{a.suffix}.csv"; d.to_csv(out, index=False); print(ID, d.shape, "->", out)
