# exp_007b_extension_clean — Homogeneous Re-run & Report
**Reviewer defense** (stated explicitly, as required by spec §0):

> The goal of this experiment is not to characterize DINOv2. The goal is to
> measure what kinds of lexical concepts an agent can acquire given a frozen
> perceptual substrate. The acquisition gradient documented below is an
> empirical boundary on lexical learning induced by that substrate — not a
> visualization of the encoder's internal cluster structure. The distinction
> matters because the finding is about *language-learning capacity*, not
> *encoder quality*.

---

## 0. Homogeneous pool fix

The original extension (exp_007b_extension) evaluated near/mid tiers under
hard and random pools, while far/native rows were loaded from exp_007/ledger.jsonl
under std_nway pools. Cross-tier C2 comparisons in that report therefore
confounded pool construction with tier distance. This re-run eliminates the
confound by running all four tiers under all seven pool constructions at
identical ways=3, shots=5, 30 episodes.

---

## 1. C1 Acquisition Gradient (primary finding)

C1 (naming accuracy) is pool-type-independent. Values below use std_nway rows; other pool types give identical C1 values.

| tier | centroid C1 | exemplar C1 | logreg C1 | random C1 | chance |
|---|---|---|---|---|---|
| native | 0.943 | 0.932 | 0.949 | 0.346 | 0.333 |
| near | 0.847 | 0.833 | 0.843 | 0.344 | 0.333 |
| mid | 0.654 | 0.650 | 0.641 | 0.329 | 0.333 |
| far | 0.530 | 0.529 | 0.535 | 0.342 | 0.333 |

**C1 gradient is monotonic**: native > near > mid > far > chance.

All learners show comparable C1 within each tier — no learner dissociation on
C1. Naming accuracy is determined by the concept's geometric relationship to
DINOv2's native clusters, not by the learning mechanism.

---

## 2. C2 Retrieval Gradient (hard pool)

| tier | centroid C2 | exemplar C2 | logreg C2 | gap (ex-cen) | supported? |
|---|---|---|---|---|---|
| native | 0.465 [0.415,0.514] | 0.497 [0.446,0.548] | 0.616 | +0.032 | YES |
| near | 0.306 [0.268,0.342] | 0.380 [0.338,0.419] | 0.448 | +0.074 | YES |
| mid | 0.274 [0.245,0.305] | 0.341 [0.308,0.374] | 0.387 | +0.066 | YES |
| far | 0.304 [0.274,0.335] | 0.352 [0.319,0.386] | 0.404 | +0.048 | YES |

---

## 3. Gap vs Native (is the C2 episodic advantage disjunctive-specific?)

This test answers: does the exemplar-centroid C2 gap differ significantly between disjunctive and native tiers?

Under each pool type, the gap_vs_native is:
  gap_vs_native = gap_disjunctive - gap_native
  supported = CI_low > 0 (disjunctive gap significantly LARGER than native)

| tier | pool | gap | native_gap | gap_vs_native (95% CI) | significant? |
|---|---|---|---|---|---|
| native | std_nway | 0.049 | 0.049 | -0.000 [-0.025,0.024] | no |
| near | std_nway | 0.078 | 0.049 | 0.029 [0.002,0.053] | YES |
| mid | std_nway | 0.066 | 0.049 | 0.016 [-0.012,0.043] | no |
| far | std_nway | 0.064 | 0.049 | 0.014 [-0.014,0.041] | no |
| native | small | 0.059 | 0.059 | -0.000 [-0.026,0.026] | no |
| near | small | 0.093 | 0.059 | 0.034 [0.005,0.061] | YES |
| mid | small | 0.072 | 0.059 | 0.013 [-0.016,0.042] | no |
| far | small | 0.043 | 0.059 | -0.017 [-0.044,0.010] | no |
| native | medium | 0.075 | 0.075 | -0.000 [-0.028,0.028] | no |
| near | medium | 0.104 | 0.075 | 0.029 [-0.006,0.061] | no |
| mid | medium | 0.089 | 0.075 | 0.013 [-0.017,0.042] | no |
| far | medium | 0.019 | 0.075 | -0.057 [-0.082,-0.033] | no |
| native | large | 0.093 | 0.093 | -0.000 [-0.031,0.030] | no |
| near | large | 0.102 | 0.093 | 0.009 [-0.023,0.041] | no |
| mid | large | 0.077 | 0.093 | -0.017 [-0.047,0.016] | no |
| far | large | 0.017 | 0.093 | -0.077 [-0.101,-0.052] | no |
| native | hard | 0.032 | 0.032 | -0.000 [-0.037,0.035] | no |
| near | hard | 0.074 | 0.032 | 0.042 [0.007,0.077] | YES |
| mid | hard | 0.066 | 0.032 | 0.035 [0.003,0.068] | YES |
| far | hard | 0.048 | 0.032 | 0.017 [-0.016,0.051] | no |
| native | oov_25 | 0.039 | 0.039 | -0.000 [-0.021,0.019] | no |
| near | oov_25 | 0.070 | 0.039 | 0.030 [0.005,0.054] | YES |
| mid | oov_25 | 0.045 | 0.039 | 0.005 [-0.019,0.031] | no |
| far | oov_25 | 0.011 | 0.039 | -0.029 [-0.050,-0.007] | no |
| native | oov_50 | 0.023 | 0.023 | -0.000 [-0.023,0.022] | no |
| near | oov_50 | 0.046 | 0.023 | 0.022 [-0.004,0.050] | no |
| mid | oov_50 | 0.019 | 0.023 | -0.004 [-0.028,0.021] | no |
| far | oov_50 | 0.003 | 0.023 | -0.020 [-0.039,-0.001] | no |

---

## 4. Pool Sensitivity (C2 gap across pool types, near tier)

| pool_type | gap (exemplar-centroid) | ci | supported? |
|---|---|---|---|
| std_nway | 0.078 | [0.062,0.095] | YES |
| small | 0.093 | [0.073,0.113] | YES |
| medium | 0.104 | [0.078,0.131] | YES |
| large | 0.102 | [0.076,0.127] | YES |
| hard | 0.074 | [0.051,0.098] | YES |
| oov_25 | 0.070 | [0.050,0.089] | YES |
| oov_50 | 0.046 | [0.023,0.067] | YES |

---

## 5. Interpretation

**C1 gradient (main finding):** C1 naming accuracy follows a monotonic gradient aligned with perceptual coherence. Native concepts (each a single CIFAR category) are learned most accurately. Near-disjunctive concepts (cat+dog, bird+deer, automobile+truck — within-superordinate pairs) achieve C1 near native performance. Mid-disjunctive (cross-superordinate pairs, imd ≈ 0.53–0.62) show further degradation. Far-disjunctive (imd ≈ 0.65–0.73) collapse toward chance. No learner dissociation appears on C1: all mechanisms show the same gradient, confirming that naming accuracy reflects concept geometry rather than learning mechanism choice.

**C2 gradient and episodic advantage:** The exemplar-centroid gap in C2 is present across all disjunctive tiers under hard-pool conditions. Whether this gap is significantly larger than the native-tier gap (i.e., whether the episodic advantage is disjunctive-specific) is reported in Section 3 above — see the 'significant?' column in the gap_vs_native table.

**Linear baselines:** LogReg and LinSVM consistently achieve high C2 under hard pools, outperforming both centroid and exemplar in many tiers. This indicates that the frozen DINOv2 space contains linearly decodable structure that prototype and exemplar mechanisms do not fully exploit. We report all learners to map the mechanism space.

**Pool sensitivity (near tier):** The C2 episodic gap is present across all seven pool constructions (std_nway, small, medium, large, hard, oov_25, oov_50). The gap is largest in random pools (small/medium/large) and smallest in oov_50. This confirms the finding is not an artifact of pool construction.

**Honest baseline statement:** The centroid learner is interpretable and sufficient for native acquisition. Exemplar memory improves over centroid on C2; logistic regression improves further. No single learner is optimal across all conditions.

---

## 6. C1/C2 Distinction

- **C1 (naming):** image -> label accuracy. Measures whether the agent can assign the correct Carroll word to a new image. Reflects acquisition as compatibility between the concept's extension and the frozen perceptual geometry.
- **C2 (retrieval):** label -> image accuracy (inverse grounding). Measures whether the agent can identify a valid instance of a concept given only its label. Reflects memory fidelity and the ability to reconstruct the extension from the learned label.
- These two directions are NOT redundant: C1 can succeed while C2 fails (a centroid that names correctly may not retrieve correctly for multimodal concepts), and vice versa. Both conditions are necessary for a complete evaluation of grounded word learning.

---

## 7. Reproducibility

- Master seed: 42
- Embeddings: .cache/embeddings_exp007/cifar_all10.npz (DINOv2-small, frozen)
- Concepts: fixed Carroll labels from exp_007b_extension (not re-selected)
- Pool conditions: all 7 types run independently for all 4 tiers
- Platform: CPU-only
- Results dir: results/exp_007b_extension_clean/
