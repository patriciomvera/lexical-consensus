# NMI_native Audit — exp_007b_extension gate

## Finding

**NMI_native = 0.357 is a mathematical constant** for any balanced 2-from-10
category union under the formula used in `_concept_ns_nmi()` in
`experiments/exp_007b_extension/run_ext.py`. The gate condition
`NMI_native < GATE_NMI_MAX = 0.80` is **trivially satisfied** for every
disjunctive pair and adds zero filtering power.

---

## Formula being used

```python
# from run_ext.py _concept_ns_nmi():
all_concept_ids = np.array([0 if h2c[h] in (cat_a, cat_b) else 1
                             for h in all_hashes])   # binary: in-union=0, other=1
all_native_cats = np.array([h2c[h] for h in all_hashes])  # 10-way native label
nmi = normalized_mutual_info_score(all_native_cats, all_concept_ids)
```

This computes NMI between:
- `all_concept_ids`: **binary** label (0 = the two union members, 1 = the 8 other categories)
  over all 2000 images in the cifar_all10 cache.
- `all_native_cats`: **10-way** native CIFAR category label.

---

## Why 0.357 is constant for any 2-from-10 balanced union

Dataset: 10 categories × 200 images = 2000 images.
For a union of any 2 categories (e.g., cat+dog):

```
P(in_union) = 400/2000 = 0.20
P(not_in_union) = 1600/2000 = 0.80

H(concept_membership) = -0.20 * log2(0.20) - 0.80 * log2(0.80) = 0.7219 bits
H(native_category_10way) = log2(10) = 3.3219 bits  [uniform over 10 categories]

H(concept_membership | native_category) = 0
  [knowing the native category perfectly determines union membership]

MI(concept_membership, native_category) = H(concept_membership) = 0.7219 bits

NMI_avg = 2 * MI / (H(concept) + H(native))
        = 2 * 0.7219 / (0.7219 + 3.3219)
        = 1.4439 / 4.0439
        = 0.3570
```

**The formula depends only on the ratio 2/10 and the equal-size assumption.**
Any choice of 2 categories from 10 equal-sized categories gives exactly 0.357.

---

## Verification across three pairs

| pair | expected (formula) | observed (concept_gate.csv) |
|---|---|---|
| cat + dog (imd=0.280, near) | 0.357 | 0.357 |
| airplane + ship (imd=0.531, mid) | 0.357 | 0.357 |
| horse + ship (imd=0.714, far, from 007b) | 0.357 | 0.357 |

All 17 concept pairs in concept_gate.csv confirm: NMI_native = 0.357 identically.

---

## What the NMI is NOT computing

The intended diagnostic ("~1 => label is a rename of a native category; ~0 => cuts
across clusters", from `descriptors.py` docstring) assumes NMI is computed in an
**episode context** where:
- `concept_ids` = local episode IDs (0, 1, 2 for 3-way)
- `object_categories` = the 3 (or n_ways) native categories present in the episode

In that computation:
- **3-way native episode** (each concept = one category): NMI = 1.0 ✓
- **3-way disjunctive episode** (each concept = 2 categories): NMI ≈ 0.760

That difference is discriminative and matches the docstring's description.

The gate formula (binary over all 2000 images) is a different quantity entirely.
It cannot be computed in episode context because episodes are not yet constructed
during gate evaluation, so a pre-episode binary formulation was used — but this
formulation is not discriminative across pairs.

**NMI_native computed within a native episode:**
```
# 3-way native (slithy, mimsy, vorpal):
H(concept_ids) = log2(3) = 1.585 bits
H(native_cats) = log2(3) = 1.585 bits  [3 categories in episode]
MI = 1.585 bits, NMI = 1.0
```

**NMI_native computed within a disjunctive episode:**
```
# 3-way near-disjunctive (tromble=cat+dog, varble=bird+deer, glintz=auto+truck):
H(concept_ids) = log2(3) = 1.585 bits
H(native_cats) = log2(6) = 2.585 bits  [6 native categories in episode]
H(concept_ids | native_cats) = 0  [native cat -> concept is deterministic]
MI = 1.585 bits
NMI = 2 * 1.585 / (1.585 + 2.585) = 0.760
```

Native NMI = 1.0 vs disjunctive NMI ≈ 0.760 — discriminative but not applicable
as a per-pair pre-episode gate criterion.

---

## Impact on gate conclusions

The effective gate criteria are:

1. `dist_lo <= inter_member_distance < dist_hi` — **discriminative** ✓
2. `ns_silhouette < NATIVE_MEDIAN (0.234)` — **discriminative** ✓
3. `nmi_native < 0.80` — **trivially satisfied** (value = 0.357 always), adds **no filtering**

All 17 concept pairs (7 near, 10 mid) pass condition 3 automatically. Removing
NMI_native from the gate logic would produce identical gate output.

---

## Recommendations

1. **Do not cite NMI_native as a gate criterion** in any paper claim. The gate
   reduces to: imd in tier range AND NS_silhouette < NATIVE_MEDIAN.

2. **Rename** the metric in the gate CSV column. The column currently labeled
   `nmi_native` in `concept_gate.csv` is more accurately:
   `binary_membership_nmi` or `membership_category_nmi`. It is not the
   episode-level NMI that descriptors.py documents.

3. **For future use**: if a pre-episode NMI diagnostic is needed, an alternative
   is: compute NMI between (which member category) and (native label) **within
   the extension only**, over just the union images. For any 2-member union from
   distinct categories this is always 1.0 (member label perfectly predicts
   native cat), so this is also non-discriminative for arity-2 unions. A
   meaningful pre-episode discriminator for "is this concept essentially a native
   category?" is simply: is one of the member categories 90%+ of the extension?
   For all 2-category balanced unions here, each member is exactly 50% — so no
   pair is near-native in this sense.

4. **Positive conclusion**: the gate's imd + NS criteria are sufficient and
   scientifically well-founded. All 17 candidates passed the gate for the right
   reasons (low NS confirms the extension is not natively separable; imd confirms
   the members are at the intended geometric distance). The NMI column is
   uninformative but not harmful.

---

*Audit completed: 2026-06-06. Checked against `concept_gate.csv` (17 rows,
all NMI_native = 0.357) and mathematical derivation.*
