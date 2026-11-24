r"""v2d manuscript = v2c manuscript (snapshot base_main_v2c.tex, sha256 cd88ad2f...) + P. M. Vera's decisions of 24-11-2026
on four sentences of the Conclusions and Future Work. Each edit replaces one exact, unique string; the rest of his texts is asserted unchanged.
Usage: python apply_v2d_edits.py <base.tex> <out main.tex>"""
import sys, re
src = open(sys.argv[1]).read(); s = src
def R(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100]); s = s.replace(old, new)
R(r"\input{generated/numbers_v2c.tex}" + "\n", r"\input{generated/numbers_v2c.tex}" + "\n" + r"\input{generated/numbers_v2d.tex}" + "\n")
# (1) Conclusions: P. M. Vera's new two sentences replace his first sentence; the qualifying footnote is removed
i = s.index(r"Visual distance predicted adult word extension across all eight trained encoders, extending the original finding beyond DINOv2.\footnote{")
j = s.index("} In an exploratory comparison, presenting adults and agents", i)
s = s[:i] + (r"In the planned within-domain models, greater visual distance was associated with less word extension for all eight trained encoders and for the randomly initialized network. "
             r"When the distance slope was allowed to vary across images, \VcImgTrSameSignWord{} of the eight trained encoders retained a negative estimate, and \VcImgTrExclWord{} had 95\% intervals entirely below zero.") + s[j+1:]
# (1b) Abstract says the same thing
R(r"the slope stayed negative for \VcImgTrSameSignWord{} of the eight trained encoders, but its interval included zero for \VcImgTrInclWord{} of them, including the reference encoder.",
  r"\VcImgTrSameSignWord{} of the eight trained encoders retained a negative estimate, and \VcImgTrExclWord{} had 95\% intervals entirely below zero; the interval of the reference encoder included zero.")
# (1c) editorial Conclusions paragraph: drop the clause now redundant with (1)
R(r"did not account for the effect of composition in adults either, and with by-image random slopes the adult distance slope was uncertain for \VcImgTrInclList.",
  r"did not account for the effect of composition in adults either.")
# (2) baseline: same-subtype condition (1.06%), the one that yields the 76.43-point contrast; 1.83% stays in Table 4 and Section 5
R(r"After three examples of the same basic type, adults selected other members of the broader category on only \VtXSameBasicPct\% of judgments.",
  r"After three examples of the same subtype, adults selected other members of the broader category on only \VcXSameSubPct\% of judgments.")
# (3) category boundaries in terms of images
R(r"In the broad teaching condition, supervised ViT and DINOv2-base each accepted \VtXViTIn{} of \VtXInN{} within-domain candidates. Yet ViT accepted none of the \VtXOutN{} outside-domain combinations, whereas DINOv2-base accepted \VtXDinoBOut. Thus, two agents can attain the same within-domain score while drawing substantially different category boundaries.",
  r"After broad teaching, both agents (supervised ViT and DINOv2-base) accepted \VtXViTIn{} of \VtXInN{} within-domain images, agreeing on \VdAgreeWord{} and differing on \VdDifferWord. Outside the taught domain, supervised ViT accepted none of the \VtXOutN{} word--image pairings, whereas DINOv2-base accepted \VtXDinoBOut.")
# (4) Future Work: one sentence added at the end of P. M. Vera's paragraph
R(r"provide evidence that learned relations help the agent interpret what a word refers to." + "\n",
  r"provide evidence that learned relations help the agent interpret what a word refers to. A direct test would ask whether training on category relations improves transfer to novel items beyond what a learner can achieve by adapting to the visual dispersion of the same teaching examples." + "\n")
# guards: the rest of P. M. Vera's texts is unchanged
for frag in [r"The central example is a single photograph (\texttt{c3\_sup2}, a turtle).", r"The two agents thus departed from the adult pattern in opposite directions.",
             r"The adult pattern is consistent with an inference about the intended scope of a novel word. Whether adults arrive at that inference through the same processes as any of these agents remains open.",
             r"After three examples drawn from different basic types within one domain, that rate rose to \VtXBroadPct\%, while selection outside the domain remained at \VtXOutBroadPct\%.",
             r"A next step is to train agents to use relations among categories to infer how broadly a novel word applies.",
             r"This distinction can be tested by holding the visual encoder and training images fixed"]:
    assert src.count(frag) == 1 and s.count(frag) == 1, frag
p = lambda t, k: t[t.index(k):t.index("\n", t.index(k))]
k = "The central example is a single photograph"; assert p(src, k) == p(s, k), "turtle paragraph changed"
assert r"Table~\ref{tab:v2matched} shows extension by teaching condition. After three examples of the same basic type" in s, "Section 5 keeps 1.83%"
assert "Visual distance predicted adult word extension" not in s and "substantially different category boundaries" not in s
open(sys.argv[2], "w").write(s); print("v2d edits applied; guards passed")
