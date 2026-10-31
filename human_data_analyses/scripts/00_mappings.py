"""Versioned class mappings CIFAR -> THINGS / ImageNet-HSJ, written BEFORE any human distance is computed.
Rule (plan, Decisiones fijadas #2): ambiguous classes take the sense shown by CIFAR images; every mapping documented.
THINGS (0-based rows of concepts-metadata_things.tsv == rows of spose_embedding_66d_sorted.txt):
  PRIMARY  = scouting mapping used to write the plan (cifar100_things_match.json, 79 classes)
             + sense resolution (mouse->mouse1 rodent, tank->tank1 military, sweet_pepper->bell pepper)
             + 4 THINGS concepts whose WordNet synset IS the CIFAR synset but were missed by the string search
               (bicycle->bike, telephone->phone, wardrobe->closet, streetcar->trolley).
             Classes sharing one THINGS concept (maple/oak/willow -> tree) keep the mapping but the pairs among them are excluded.
  SENS_SCOUT  = scouting 79 only (pepper -> pepper2 = pepper.n.04 Capsicum).
  SENS_STRICT = drop hypernym-only maps (aquarium_fish->fish, maple/oak/willow->tree, pickup_truck->truck)
                and keyboard (THINGS keyboard is a musical instrument per THINGS category); add hyponym maps
                cattle->{cow,bull} (mean of the two), ray->stingray.
HSJ: ILSVRC-2012 wnids in the hyponym closure of the corrected CIFAR synset; a wnid claimed by >1 class goes to the most
     specific class (deepest synset); exact ties are dropped. CIFAR-10 uses C10 synsets in cifar_wn.py.
"""
import sys, pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parent)); import _paths as _P
import sys, json, csv, glob, gzip, pathlib, collections
ROOT = pathlib.Path(_P.ROOT); SC = ROOT / "human_data_scouting"; OUT = ROOT / "human_data_analyses/data"
sys.path.insert(0, str(SC / "scripts")); import cifar_wn as W
from nltk.corpus import wordnet as wn
rows = list(csv.reader(open(SC / "concepts-metadata_things.tsv"), delimiter="\t"))[1:]
scout = json.load(open(SC / "cifar100_things_match.json"))
def info(i): return {"row": i, "word": rows[i][0], "uniqueID": rows[i][1], "synset": rows[i][5], "things_def": rows[i][7][:120], "things_category": rows[i][19]}
prim, notes = {}, {}
for c, v in scout.items():
    if v is None: continue
    if len(v) == 1: prim[c] = [v[0][0]]; notes[c] = "scouting"
prim["mouse"] = [1026]; notes["mouse"] = "ambigua: mouse1 (roedor, mouse.n.01) por imágenes CIFAR; se descarta mouse2 (dispositivo)"
prim["tank"] = [1620]; notes["tank"] = "ambigua: tank1 (vehículo militar, tank.n.01) por imágenes CIFAR; se descarta tank2 (depósito)"
prim["sweet_pepper"] = [117]; notes["sweet_pepper"] = "ambigua: bell pepper (bell_pepper.n.02 = pimiento dulce) por imágenes CIFAR; pepper1 (especia) descartado; pepper2 (Capsicum) = sensibilidad"
for c, i in [("bicycle", 123), ("telephone", 1161), ("wardrobe", 360), ("streetcar", 1715)]:
    assert wn.synset(rows[i][5]) == wn.synset(W.C100[c]); prim[c] = [i]; notes[c] = f"mismo synset WordNet ({W.C100[c]}); omitido por la búsqueda de texto del scouting"
for c in ["aquarium_fish", "maple_tree", "oak_tree", "willow_tree", "pickup_truck"]: notes[c] += " (hiperónimo; excluido en SENS_STRICT)"
scout79 = {c: [v[0][0]] for c, v in scout.items() if v is not None and len(v) == 1}
scout79.update({"mouse": [1026], "tank": [1620], "sweet_pepper": [1151]})
strict = {c: v for c, v in prim.items() if c not in {"aquarium_fish", "maple_tree", "oak_tree", "willow_tree", "pickup_truck", "keyboard"}}
strict.update({"cattle": [426, 214], "ray": [1556]})
c10 = {"airplane": [10], "automobile": [267], "bird": [128], "cat": [284], "deer": [476], "dog": [495], "frog": [644], "horse": [798], "ship": [1421], "truck": [1720]}
for k, (i,) in c10.items(): assert wn.synset(rows[i][5]) == wn.synset(W.C10[k]), k
things = {"PRIMARY": prim, "SENS_SCOUT": scout79, "SENS_STRICT": strict, "C10": c10}
json.dump({"rule": __doc__, "maps": things, "notes": notes,
           "details": {n: {c: [info(i) for i in v] for c, v in m.items()} for n, m in things.items()}},
          open(OUT / "map_things_v1.json", "w"), indent=1, ensure_ascii=False)
print({k: len(v) for k, v in things.items()})
# ---- HSJ: parse trials once
trials = []
for f in sorted(glob.glob(f"{_P.CACHE}/hsj/*/unit_*.json")):
    d = json.load(open(f)); split = f.split("/")[-2]
    for u in d["data"]:
        for s in u["subunits"]:
            if s["kind"] != "rank:8rank2": continue
            it = {x["kind"]: x["detail"].split("/")[1] for x in s["interactions"] if "/" in x["detail"]}
            if "behavior:rank_0" not in it or "behavior:rank_1" not in it: continue
            q = it["content:query"]; refs = [it[f"content:reference_{j}"] for j in range(8)]
            trials.append((split, u["unit_id"], q, "|".join(refs), it["behavior:rank_0"], it["behavior:rank_1"]))
with gzip.open(OUT / "hsj_trials.csv.gz", "wt") as g:
    w = csv.writer(g); w.writerow(["split", "unit", "q", "refs", "r0", "r1"]); w.writerows(trials)
wnids = sorted({t[2] for t in trials} | {r for t in trials for r in t[3].split("|")})
print("HSJ trials", len(trials), "wnids", len(wnids))
def syn_of(w): return wn.synset_from_pos_and_offset("n", int(w[1:]))
def hsj_map(classes):
    claim = collections.defaultdict(list)
    for c, s in classes.items():
        S = wn.synset(s)
        for w in wnids:
            t = syn_of(w)
            if t == S or S in t.closure(lambda x: x.hypernyms() + x.instance_hypernyms()): claim[w].append((S.max_depth(), c))
    out, dropped = collections.defaultdict(list), {}
    for w, cl in claim.items():
        cl.sort(reverse=True)
        if len(cl) > 1 and cl[0][0] == cl[1][0]: dropped[w] = [c for _, c in cl]; continue
        out[cl[0][1]].append(w)
    return {c: sorted(v) for c, v in out.items()}, dropped
hp, dp = hsj_map(W.C100); horig, dorig = hsj_map({c: s for c, s in W.m.items() if not c.startswith("_")}) if hasattr(W, "m") else ({}, {})
h10, d10 = hsj_map(W.C10)
json.dump({"rule": "hyponym closure of CIFAR synset; multi-claimed wnid -> deepest class; ties dropped",
           "C100_corrected": hp, "C100_dropped_ties": dp, "C100_original_mapping": horig, "C10": h10, "C10_dropped_ties": d10},
          open(OUT / "map_hsj_v1.json", "w"), indent=1)
print("HSJ C100 classes", len(hp), "C10 classes", len(h10), sorted(h10), "ties", dp, d10)
