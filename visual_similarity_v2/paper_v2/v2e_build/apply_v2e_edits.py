r"""v2e manuscript = v2d manuscript (snapshot base_main_v2d.tex, sha256 9129ab0e...) + P. M. Vera's final decisions of 24-11-2026.
Usage: python apply_v2e_edits.py <base.tex> <out main.tex>"""
import sys
src = open(sys.argv[1]).read(); s = src
def R(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100]); s = s.replace(old, new)
# (2) start of 7.1 Conclusions
R(r"In the planned within-domain models, greater visual distance was associated with less word extension for all eight trained encoders and for the randomly initialized network. "
  r"When the distance slope was allowed to vary across images, \VcImgTrSameSignWord{} of the eight trained encoders retained a negative estimate, and \VcImgTrExclWord{} had 95\% intervals entirely below zero.",
  r"In the planned within-domain models, greater cosine distance from the test image to the centroid of the teaching examples was associated with less word extension for all eight trained encoders and for the randomly initialized network. "
  r"In post hoc models that allowed the distance slope to vary across images, \VcImgTrSameSignWord{} of the eight trained encoders retained a negative estimate, and \VcImgTrExclWord{} had 95\% intervals entirely below zero (Appendix~\ref{app:posthoc}).")
# (1) agents sentence
R(r"After broad teaching, both agents (supervised ViT and DINOv2-base) accepted \VtXViTIn{} of \VtXInN{} within-domain images, agreeing on \VdAgreeWord{} and differing on \VdDifferWord.",
  r"After broad teaching, the agents using supervised ViT and DINOv2-base each accepted \VtXViTIn{} of \VtXInN{} within-domain images, agreeing on \VdAgreeWord{} and differing on \VdDifferWord.")
for frag in [r"Outside the taught domain, supervised ViT accepted none of the \VtXOutN{} word--image pairings, whereas DINOv2-base accepted \VtXDinoBOut.",
             r"After three examples of the same subtype, adults selected other members of the broader category on only \VcXSameSubPct\% of judgments.",
             r"The central example is a single photograph (\texttt{c3\_sup2}, a turtle).",
             r"A direct test would ask whether training on category relations improves transfer to novel items beyond what a learner can achieve by adapting to the visual dispersion of the same teaching examples.",
             r"A next step is to train agents to use relations among categories to infer how broadly a novel word applies."]:
    assert src.count(frag) == 1 and s.count(frag) == 1, frag
p = lambda t, k: t[t.index(k):t.index("\n", t.index(k))]
for k in ["The central example is a single photograph", "A next step is to train agents"]: assert p(src, k) == p(s, k), k
assert r"\label{app:posthoc}" in open(sys.argv[1].replace("v2e_build/base_main_v2d.tex", "v2c_build/appendix_posthoc_v2c.tex")).read()
assert "Use of AI" not in s
open(sys.argv[2], "w").write(s); print("v2e edits applied; guards passed")
