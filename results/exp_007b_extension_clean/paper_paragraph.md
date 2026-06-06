# Paper Paragraph — exp_007b_extension_clean

*Ready-to-paste after reviewing the numbers below. Fill brackets are replaced.*

---

These results suggest that lexical acquisition over frozen perception is not
all-or-nothing. Naming accuracy (C1) follows a graded pattern aligned with
perceptual coherence: labels for native categories are easiest
(C1 ≈ 0.943), labels for coherent within-superordinate overextensions
remain highly learnable (C1 ≈ 0.847), and labels for cross-superordinate
mid-range disjunctions degrade further (C1 ≈ 0.654), while labels for
far-disjunctive concepts collapse toward chance
(C1 ≈ 0.530, chance = 0.333). This supports the view that early
lexical acquisition is constrained by perceptual coherence rather than arbitrary
set membership, paralleling the developmental trajectory in which children first
generalize within perceptually coherent regions before refining category boundaries.

In contrast, retrieval (C2) exposes a different capacity: exemplar-based memory
consistently outperforms compressed prototypes when recovering valid instances
from a learned label (hard-pool gap: near ≈ +0.074,
mid ≈ +0.066, native ≈ +0.032). Under homogeneous pool
conditions, this C2 advantage IS specific to disjunctive concepts (CI of gap_disj - gap_native excludes zero).
Linear discriminative baselines (LogReg, LinSVM) remain stronger under hard
candidate pools, indicating that the frozen embedding space contains recoverable
structure beyond what prototype or exemplar mechanisms exploit alone.

Together, C1 and C2 are not redundant: each direction exposes a distinct
dimension of the lexical mapping. C1 measures acquisition as concept-geometry
compatibility; C2 measures retrieval as memory fidelity. Both are necessary for
a complete evaluation of grounded word learning.

---

*Key numbers for inline reference:*
- C1 gradient (centroid, std_nway): native=0.943 / near=0.847 / mid=0.654 / far=0.530 / chance=0.333
- C2 hard gap (exemplar-centroid): near=+0.074 / mid=+0.066 / native=+0.032
- C2 gap disjunctive-specific: IS specific to disjunctive concepts
