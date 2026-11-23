"""Selection of the stage-1 random-effects structure (plan §5.2; Patricio 18-11 12:12 option B, 12:14 option (1) = ONE common structure).
Depends ONLY on convergence (column `converged` of 05_twostage_stage1.R; criterion in lib_glmm.R / plan §5.2), never on beta, Delta or p.
STRUCTURE_SCOPE is the single pre-declared constant of the plan: "common" (confirmed by Patricio 18-11 12:14). There is no runtime option.
  common     : the richest rung of LADDER that converges in all 12 experiments for EVERY encoder of SELECTION_SET (the 10 primary
               encoders, floors included; F0 has no zd and is NOT included), chosen separately for 8 and for 24 items. Encoders with no
               fully converging rung at all are left out of the selection and are evaluated at the common rung anyway (missing -> inconcluso tecnico).
  per_encoder: descriptive only (each encoder's own richest fully converging rung).
Descriptive outputs (never enter the regions): per-encoder rung, and S9 = the "mixed" fit (per experiment, the first converged rung of pass 1).
Usage: python 05b_select_structure.py <out_prefix> <stage1_ladder.csv> [more stage-1 csv ...]
Exit 0 = selection complete (writes <out_prefix>_primary.csv, _per_encoder.csv, _S9_mixed.csv, _structure.csv, _status.json)
Exit 3 = fits missing (writes <out_prefix>_needs.csv with encoder, items, rung): run 05 with fit=<rung> for those and call again.
Exit 4 = NO common structure for some item set (REVISION3b): writes <out_prefix>_status.json (common_rung null) and <out_prefix>_DEVIATION.txt
         and NO stage-1 tables. There is no silent fallback to L4 or to anything else: the run stops and the deviation is logged (plan §5.2)."""
import sys, json, pandas as pd
STRUCTURE_SCOPE = "common"          # pre-declared constant (plan §5.2 and §10.4). "per_encoder" would be the only alternative.
RUNGS = ["L1", "L2", "L3", "L4"]
SELECTION_SET = ["R0_dinov2s", "O1_dinov2b", "R1_cvcl", "E1_siglip", "O2_openclip", "E2_vitsup", "E3_rn50sup", "E4_mae", "F1_rn50rand0", "F2_pixels"]
EXPS = list(range(1, 13))
def status(T, enc, exp, rung):
    r = T.get((enc, exp, rung)); return None if r is None else bool(r["converged"])
def richest(T, encs):
    """('ok', rung) | ('need', [(enc, rung)]) | ('none', None). An empty encoder set has no common rung (REVISION3b: never a vacuous L1)."""
    if not encs: return "none", None
    for rg in RUNGS:
        st = [status(T, e, x, rg) for e in encs for x in EXPS]
        if any(s is False for s in st): continue
        if any(s is None for s in st): return "need", sorted({(e, rg) for e in encs for x in EXPS if status(T, e, x, rg) is None})
        return "ok", rg
    return "none", None
def select(df):
    df = df[df.variant == "primary"]; out = {}; needs = []
    for items, g in df.groupby("items"):
        T = {}
        for r in g.to_dict("records"): T.setdefault((r["encoder"], int(r["exp"]), r["rung"]), r)
        encs = [e for e in SELECTION_SET if e in set(g.encoder)]
        own = {}
        for e in encs:
            k, v = richest(T, [e]); own[e] = v if k == "ok" else (None if k == "none" else "?")
            if k == "need": needs += [(e, items, rg) for _, rg in v]
        if needs and any(n[1] == items for n in needs): out[items] = dict(T=T, own=own, common=None); continue
        elig = [e for e in encs if own[e] is not None]
        k, v = richest(T, elig)
        if k == "need": needs += [(e, items, rg) for e, rg in v]; out[items] = dict(T=T, own=own, common=None); continue
        common = v if k == "ok" else None
        if common:   # encoders left out of the selection are still evaluated at the common rung
            for e in encs:
                if e not in elig and any(status(T, e, x, common) is None for x in EXPS): needs.append((e, items, common))
        out[items] = dict(T=T, own=own, common=common, eligible=elig, excluded=[e for e in encs if e not in elig])
    return out, sorted(set(needs))
def stage1_rows(T, enc, items, rung_of_exp, variant):
    rows = []
    for x in EXPS:
        rg = rung_of_exp(x); r = T.get((enc, x, rg)) if rg else None; ok = r is not None and bool(r["converged"])
        any_r = next((T[k] for k in T if k[0] == enc and k[1] == x), None)
        rows.append(dict(encoder=enc, items=items, variant=variant, suffix="", exp=x, order=any_r["order"] if any_r else None,
                         beta=r["beta"] if ok else float("nan"), se=r["se"] if ok else float("nan"), re_used=rg if ok else "failed"))
    return rows
if __name__ == "__main__":
    pre = sys.argv[1]; df = pd.concat([pd.read_csv(f) for f in sys.argv[2:]]); df["suffix"] = df["suffix"].fillna("")
    out, needs = select(df)
    if needs:
        pd.DataFrame(needs, columns=["encoder", "items", "rung"]).to_csv(f"{pre}_needs.csv", index=False); print("NEEDS", needs); sys.exit(3)
    none = {items: o for items, o in out.items() if o["common"] is None}
    if none:   # REVISION3b: no common rung (e.g. every encoder left out, or encoders converging only on disjoint rungs) -> stop, never a silent L4
        st = {str(i): dict(scope=STRUCTURE_SCOPE, common_rung=o["common"], excluded_from_selection=o.get("excluded", []), own_rung=o["own"]) for i, o in out.items()}
        open(f"{pre}_status.json", "w").write(json.dumps(st, indent=1))
        msg = "".join(f"DEVIATION (plan §5.2): no common stage-1 structure for {i} items; eligible={o.get('eligible')}; own rungs={o['own']}. Stop: no fallback.\n" for i, o in none.items())
        open(f"{pre}_DEVIATION.txt", "w").write(msg); print(msg, end=""); sys.exit(4)
    prim, per, mix, struct, st = [], [], [], [], {}
    for items, o in out.items():
        T, own, common = o["T"], o["own"], o["common"]
        for e in own:
            first = {x: next((rg for rg in RUNGS if status(T, e, x, rg)), None) for x in EXPS}
            prim += stage1_rows(T, e, items, (lambda x, c=(common if STRUCTURE_SCOPE == "common" else own[e]): c), "primary")
            per += stage1_rows(T, e, items, (lambda x, c=own[e]: c), "per_encoder_rung")
            mix += stage1_rows(T, e, items, (lambda x, f=first: f[x]), "S9_mixed")
            struct.append(dict(encoder=e, items=items, scope=STRUCTURE_SCOPE, common_rung=common, own_rung=own[e],
                               in_selection=e in o["eligible"], **{f"first_rung_exp{x}": first[x] for x in EXPS}))
        st[str(items)] = dict(scope=STRUCTURE_SCOPE, common_rung=common, excluded_from_selection=o["excluded"], own_rung=own)
    pd.DataFrame(prim).to_csv(f"{pre}_primary.csv", index=False); pd.DataFrame(per).to_csv(f"{pre}_per_encoder.csv", index=False)
    pd.DataFrame(mix).to_csv(f"{pre}_S9_mixed.csv", index=False); pd.DataFrame(struct).to_csv(f"{pre}_structure.csv", index=False)
    open(f"{pre}_status.json", "w").write(json.dumps(st, indent=1)); print(json.dumps(st, indent=1))
