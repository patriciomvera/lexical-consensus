r"""v2c manuscript = v2b manuscript (snapshot base_main_v2b.tex, sha256 e1e7a25f...) + the edits below, approved by P. M. Vera on
24-11-2026 (groups A and B of the review changes; group C as future work). Each edit replaces one exact, unique string.
Texts by P. M. Vera (turtle Results sentence, Conclusions paragraph, Future Work paragraph) are not modified; the script asserts it.
Usage: python apply_v2c_edits.py <base.tex> <out main.tex>"""
import sys, re
src = open(sys.argv[1]).read(); s = src
def R(old, new, n=1):
    global s
    c = s.count(old); assert c == n, (c, old[:120]); s = s.replace(old, new)

# ---------- preamble: v2c macros ----------
R(r"\input{generated/numbers_v2_text.tex}" + "\n", r"\input{generated/numbers_v2_text.tex}" + "\n" + r"\input{generated/numbers_v2c.tex}" + "\n")

# ---------- A3 + A4 + A1: abstract ----------
R(r"Whether presentation order changes this slope, the frozen primary question, remained inconclusive ($\hat\Delta = \VtOrdDelta$, 95\% CI [\VtOrdLo,~\VtOrdHi]).",
  r"The original study found the suspicious-coincidence effect to depend on whether the one-example trial came before or after the three-example trials, an order that varied between its experiments; whether this order also changes the distance slope was therefore the frozen primary question, tested at the level of the experiments, and it remained inconclusive ($\hat\Delta = \VtOrdDelta$, 95\% CI [\VtOrdLo,~\VtOrdHi]).")
R(r"Every trained encoder's geometry correlated with human similarity measures from three datasets, but so did the untrained network's, so this alignment does not show what the encoders learned.",
  r"Every trained encoder's geometry correlated with human similarity measures from three datasets, and so did the untrained network's, although far more weakly on the two similarity datasets (Spearman $\rho = \VtHoneRandThings$ and \VtHoneRandHsj, against \VcHoneTrMin{} to \VcHoneTrMax{} for the trained encoders other than MAE); the frozen support rule, which required only a positive correlation, did not separate them.")
R(r"(at most \VtXJumpMax{} vs.\ \VtXJumpAdult{} percentage points)",
  r"(at most \VtXJumpMax{} vs.\ \VtXJumpAdult{} percentage points; with 12 candidate images, a gap of about one image)")
R(r"Visual geometry predicts part of word extension across architectures, even without training, but does not capture the effect of example composition.",
  r"Cosine distance to the centroid of the teaching examples, computed on final-layer global embeddings, predicts part of word extension across architectures, even without training, but does not capture the effect of example composition; in post hoc checks, distances to the individual examples did not capture it either.")

R(r"raw pixels showed no negative association (\VtPcPixB).",
  r"raw pixels showed no negative association (\VtPcPixB). In post hoc models that let the distance slope vary across images, the slope stayed negative for \VcImgTrSameSignWord{} of the eight trained encoders, but its interval included zero for \VcImgTrInclWord{} of them, including the reference encoder.")
# ---------- A1 + A3: introduction ----------
R(r"The pattern is consistent across architectures: visual distance is associated with part of adult extension,",
  r"The pattern is consistent across architectures: distance to the centroid of the teaching examples, measured in each encoder's final-layer global embedding, is associated with part of adult extension,")
R(r"and the frozen primary test of whether presentation order changes the distance slope was inconclusive." + "\n",
  r"and the frozen primary test of whether presentation order changes the distance slope was inconclusive. Our conclusions concern this readout of the encoders' geometry, not visual geometry in general: readouts that use the individual examples, their dispersion, or intermediate layers could behave differently, and we examine exemplar-based distances only post hoc." + "\n")
R(r"Finally, in an exploratory comparison, we gave artificial learners the same teaching and test images that adults saw." + "\n",
  r"Finally, in an exploratory comparison, we gave artificial learners the same teaching and test images that adults saw." + "\n\n" +
  r"Presentation order is the primary question of the multi-encoder analysis for a specific reason. \textcite{lewis2018} found that the suspicious-coincidence effect depended on whether the one-example trial came before or after the three-example trials, an order that varied between their 12 experiments. In an exploratory reanalysis with the reference encoder (Section~\ref{sec:results}), a model fitted to individual decisions suggested that order also changed the slope of visual distance. That model treats \OrdNobs{} decisions as independent evidence about a factor that varied across only 12 experiments. The frozen multi-encoder plan therefore made order moderation its primary question and tested it at the experiment level, asking whether the result survived inference at the level at which order was assigned and whether it depended on which encoder defined distance; the association of distance with extension, already supported for the reference encoder, served as a positive control." + "\n")

# ---------- 2.2 / 2.3: definitions ----------
R(r"The distance is defined entirely by the images, independently of participants' responses.",
  r"The distance is defined entirely by the images, independently of participants' responses. Because the centroid is renormalized, $d$ does not use how dispersed the teaching examples are; Appendix~\ref{app:posthoc} compares it post hoc with distances to the individual examples.")
R(r"over all 24 items. Random effects are grouped by person,",
  r"over all 24 items. The level is the item's taxonomic level within its own domain (subordinate, basic, or superordinate, from the stimulus file name), so out-of-domain items keep their within-domain level; the image factor indexes the 24 stimulus slots (e.g., \texttt{c3\_sup2}), each a single photograph used in all experiments (12 levels in the superordinate-only model). Random effects are grouped by person,")
R(r"summarized by a global rule across margins and geometry cutoffs. We also refitted",
  r"summarized by a global rule across margins and geometry cutoffs; Appendix~\ref{app:rules} gives the formal definitions. Distance was standardized within each representation and item set, so each slope is in log odds per standard deviation of that representation's own distance (raw standard deviations in Appendix Table~\ref{tab:posthoc}). We also refitted")
R(r"\subsection{Results}" + "\n\n" + r"\paragraph{Distance across representations.}",
  r"\subsection{Results}" + "\n" + r"\label{sec:results}" + "\n\n" + r"\paragraph{Distance across representations.}")
R(r"and MAE, whereas pixels gave \VtPcPixB{} [\VtPcPixLo,~\VtPcPixHi]. The association therefore",
  r"and MAE, whereas pixels gave \VtPcPixB{} [\VtPcPixLo,~\VtPcPixHi]. Because each slope is per standard deviation of that representation's distance, and the MAE model omits the by-person distance slope (omitting a random slope tends to pull a conditional logit slope toward zero), the ranking of slopes across representations is approximate. The association therefore")

# ---------- 2.4: minor + 24-item interaction + post hoc order ----------
R(r"(range \DistSupMin--\DistSupMax) (\DistOneSup, \DistSubSup, \DistBasSup, and \DistSupSup, respectively)",
  r"(range \DistSupMin--\DistSupMax: \DistOneSup, \DistSubSup, \DistBasSup, and \DistSupSup, respectively; distributions in Appendix Figure~\ref{fig:supdist})")
R(r"Note that this model uses all 24 items, whereas the curves in Figure~\ref{fig:x1}a show only the 8 target-domain items.",
  r"Note that this model uses all 24 items, whereas the curves in Figure~\ref{fig:x1}a show only the 8 target-domain items. Its condition-specific slopes are therefore driven largely by the contrast between the taught domain and the other domains, whose items stayed near zero while within-domain selection rose after broad teaching; they are not the slopes of the within-domain curves in Figure~\ref{fig:x1}a.")
R(r"so moderation of the distance slope by order depended on the teaching condition." + "\n",
  r"so moderation of the distance slope by order depended on the teaching condition. These likelihood-ratio tests treat individual decisions as independent evidence about order, which varied only between the 12 experiments, and the model has no experiment-level random effects; the experiment-level primary test, designed for this problem, was inconclusive, and the post hoc $p$ values do not override it." + "\n\n" +
  r"\paragraph{Post hoc checks after review (exploratory).} Three checks were added after the analyses and two external reviews were complete (Appendix~\ref{app:posthoc}). Adding by-image random slopes for distance to the positive-control model (all ten fits converged) kept the slope negative for \VcImgTrSameSignWord{} of the eight trained encoders but widened every interval: it excluded zero for \VcImgTrExclWord{} and included it for \VcImgTrInclList{} (DINOv2-small: \VcImgRzB{} [\VcImgRzLo,~\VcImgRzHi], with a standard deviation of \VcImgRzSd{} for the image slopes). Under this model the reference encoder would not have passed the positive control; the random network kept an interval below zero (\VcImgRandB{} [\VcImgRandLo,~\VcImgRandHi]). In the superordinate-only model, distance no longer improved fit given condition once its slope could vary across images ($\chi^2(1) = \VcSupImgChi$, $p = \VcSupImgP$). Models with an intercept for each image--condition cell did not converge for any representation. Distances to the individual teaching examples did not absorb the effect of composition: in the superordinate-only model, teaching condition still improved fit given the mean-exemplar distance ($\chi^2(3) = \VcSupDmeanCondChi$ for DINOv2-small; at least \VcSupCondChiMinAll{} for every representation), and given the nearest-exemplar distance ($\chi^2(3) = \VcSupDminCondChi$). Across the eight within-domain items, the mean-exemplar distance fitted better than the centroid distance for \VcDmeanBetterN{} of the ten representations and the nearest-exemplar distance for \VcDminBetterN{} (AIC)." + "\n")

# ---------- figures 1 and 2 ----------
R(r"\includegraphics[width=\linewidth]{figs/fig_forest_adults.pdf}", r"\includegraphics[width=\linewidth]{figs/fig_forest_adults_v2c.pdf}")
R(r"controls. (a) after MAE: fitted with random intercepts only,", r"controls. $^{a}$MAE: fitted with random intercepts only,")
R(r"by teaching condition and test level, against DINOv2 distance to the training examples;", r"by teaching condition and test level, against DINOv2-small distance to the training examples;")
R(r"against the distance between member categories. The vertical axes", r"against the DINOv2-small distance between member categories. The vertical axes")

# ---------- section 3 ----------
R(r"Images were preprocessed with the model's default Hugging Face image processor:",
  r"For DINOv2-small, images were preprocessed with the model's default Hugging Face image processor:")
R(r"We did not test alternative preprocessing choices (e.g., other interpolation methods or crops).",
  r"The other representations used their own preprocessing (Appendix~\ref{app:checkpoints}), which on square CIFAR images keeps either the full frame or its central 87.5\% per side. We did not test alternative preprocessing choices (e.g., other interpolation methods or crops).")
R(r"and were tested on naming held-out images.", r"and were tested on naming held-out images (naming accuracy, macro-averaged over words, is labeled C1 in figures and tables).")
R(r"(perceptual $\beta = \WnCorrBeta$, semantic $p = \WnCorrPsem$, versus $\WnFrozenBeta$ and \WnFrozenPsem{} with the original mapping;",
  r"(perceptual slope $\beta = \WnCorrBeta$ and semantic-distance $p = \WnCorrPsem$, versus $\beta = \WnFrozenBeta$ and $p = \WnFrozenPsem$ with the original mapping;")
R(r"(D-all; a sensitivity analysis, since the pairs share categories).",
  r"(D-all; a sensitivity analysis, since the pairs share categories). The same 100 pairs, selected with DINOv2-small distances, were used for every representation, each with its own distances, so the quadrants are balanced only for DINOv2-small.")
R(r"and .30 \parencite{schuirmann1987,lakens2017}.",
  r"and .30 \parencite{schuirmann1987,lakens2017}, with the rules of the frozen plan (Appendix~\ref{app:rules}): support requires an effect in the predicted direction with unadjusted $p < .05$ and does not depend on the SESOI, which enters only the no-support region.")
R(r"so only the links tested in Section~\ref{sec:link} are read against people.",
  r"so only the links tested in Section~\ref{sec:link} are read against people; T and D characterize each representation as a substrate for the learner (how separable its categories are for a prototype learner) rather than test a claim about people.")
R(r"both did in the descriptive D-all analysis, whose pairs share classes and are not independent.",
  r"both did in the descriptive D-all analysis, whose pairs share classes and are not independent (its regions treat the \VtDallN{} pairs as independent and are descriptive only).")

# ---------- section 4: H1 magnitude, H2 labels, H5, H6 ----------
R(r"Because the calibration across stimulus sets failed at every cutoff, the frozen global profile for H1 was ``holds for all, without evidence of a geometric difference''; these alignments do not show that the encoders differ in a human-relevant way.",
  r"The frozen rule asks only for a positive correlation with Mantel $p < .05$, so it does not distinguish small from large alignments. On THINGS and ImageNet-HSJ, the trained encoders other than MAE reached $\rho$ from \VcHoneTrMin{} to \VcHoneTrMax{} (MAE: \VcHoneMaeThings{} and \VcHoneMaeHsj), against \VtHoneRandThings{} and \VtHoneRandHsj{} for the untrained network, whose upper interval limits (\VbHoneRandThingsHi{} and \VbHoneRandHsjHi) lie below the lowest lower limit among those trained encoders (\VcHoneTrLoMin). The trained encoders were thus far better aligned than the untrained network, even though the binary rule placed all of them in the support region. The frozen global profile for H1, ``holds for all, without evidence of a geometric difference,'' reflects that binary rule and the failure of the geometry calibration at every cutoff (a control shared a geometry group with a trained encoder; Appendix~\ref{app:rules}); it does not mean that the encoders and the untrained network are equally aligned.")
R(r"Support requires $\rho > 0$ and Mantel $p < .05$ (10{,}000 permutations).",
  r"Support requires $\rho > 0$ and Mantel $p < .05$ (10{,}000 permutations), whatever the size of $\rho$.")
R(r"Because DINOv2-small was inconclusive on THINGS, the multi-encoder comparison for H2 remained descriptive (Appendix Table~\ref{tab:v2agentsH}); its decision regions are unadjusted and therefore classify the ImageNet-HSJ estimate for DINOv2-small as support (Appendix~\ref{app:deviations}).",
  r"Because DINOv2-small was inconclusive on THINGS, the multi-encoder comparison for H2 remained descriptive. Its per-encoder regions are unadjusted and test the coefficient (with HC3 standard errors), so they label the DINOv2-small ImageNet-HSJ estimate as support (unadjusted $p = \HTwoHsjP$), whereas the reference analysis above is Holm-adjusted and inconclusive; the two labels answer different questions and both are reported. The same unadjusted rule labels the untrained network as support on ImageNet-HSJ ($p = \VcHtwoRandHsjP$), although the interval of its partial correlation includes zero ($r = \VcHtwoRandHsjR$ [\VcHtwoRandHsjLo,~\VcHtwoRandHsjHi]); with \HTwoHsjN{} pairs such labels are fragile. With \HTwoThingsN{} and \HTwoHsjN{} pairs, the smallest partial correlation detectable with 80\% power at $\alpha = .05$ is \VcMdeThings{} and \VcMdeHsj, respectively. Estimates for every representation, and a descriptive analysis on all mapped pairs, are in Appendix Table~\ref{tab:h2post}.")
R(r"not predict naming accuracy ($\beta = \HSixB$ [\HSixLo,~\HSixHi], $p = \HSixP$),",
  r"not predict naming accuracy ($\beta = \HSixB$ [\HSixLo,~\HSixHi], $p = \HSixP$; the multi-encoder dissociation analysis D for CVCL in Table~\ref{tab:agents} is a separate computation, reported as a partial correlation, $r = \VtDCvclR$),")
R(r"following the preprocessing in the released CVCL code; the 512-dimensional outputs were L2-normalized.",
  r"following the preprocessing in the released CVCL code; the 512-dimensional outputs were L2-normalized. Unlike the DINOv2-small pipeline, which keeps the central 224 of 256 pixels per side of the upsampled image, this preprocessing keeps the full frame, so the field of view also differs between the two encoders.")
R(r"which is inconsistent with the pre-specified criterion (Figure~\ref{fig:h6}).",
  r"which is inconsistent with the pre-specified criterion (Benjamini--Hochberg correction over the \HSixBHTests{} H6 tests; Figure~\ref{fig:h6}).")
R(r"in CIFAR-100 (partial $\rho = \HFiveHundredRho$,", r"in CIFAR-100 (partial $\rho$ controlling log word frequency $= \HFiveHundredRho$,")

# ---------- section 5: decision rules (A5) ----------
R(r"Adults contribute selections in \VtXPerCell{} sessions per word--image pairing; each agent makes one deterministic decision per pairing." + "\n",
  r"Adults contribute selections in \VtXPerCell{} sessions per word--image pairing; each agent makes one deterministic decision per pairing." + "\n\n" +
  r"\paragraph{Decision rules.} The centroid score of an image is its cosine similarity to the normalized centroid of the teaching images ($1 - d$). With three teaching images $s_1, s_2, s_3$ and a candidate $x$, the learner was refitted on each set of three images that leaves out one teaching image and scored the image left out; $x$ was accepted if at least one teaching image scored at or below $x$ (tolerance $10^{-6}$), that is, if $x$ was at least as typical as the least typical teaching image. This support-dependent rule is the one written in the protocol; it uses no human response or category label, can accept any number of the 24 images, and lets a dispersed teaching set broaden acceptance by itself. With one teaching image it is undefined, so one-example trials were not used. Separately, for each encoder we computed the largest gain attainable with one \emph{common threshold}: the 12 within-domain superordinate candidates (four per domain) were scored after broad and after same-subtype teaching, a single threshold on the centroid score was applied to both conditions and all three domains, and the gain is the maximum, over all thresholds, of the fraction accepted after broad teaching minus the fraction after same-subtype teaching, with no constraint on other-domain acceptance. It is a diagnostic upper bound, not a chosen operating point. The adult benchmark is the same contrast, \VtXBroadPct\% minus \VcXSameSubPct\% (same-subtype teaching). With 12 candidates, rates move in steps of 1/12 (\VcXStep{} points): the largest encoder gain, \VtXJumpMax{} points (8 of 12), is one candidate below 9 of 12 (\VcXNextStep{} points), the step closest to the adult value. The same bound with the two other deterministic learners of the protocol ranged from \VcXJumpKcMin{} to \VcXJumpKcMax{} points (two centroids) and from \VcXJumpKnnMin{} to \VcXJumpKnnMax{} points (weighted 3-nearest-neighbor exemplar learner) across the trained encoders. A three-word forced-choice version, in which each image receives one of three taught words, is in the released outputs. We did not try calibrated thresholds, normalized scores, or threshold-free measures." + "\n")
R(r"but ViT accepted \VtXViTOut{} of the \VtXOutN{} other-domain pairings and DINOv2-base accepted \VtXDinoBOut.",
  r"but ViT accepted \VtXViTOut{} of the \VtXOutN{} other-domain pairings and DINOv2-base accepted \VtXDinoBOut; under the same rule, the two encoders operate at very different false-alarm rates.")
R(r"descriptions are visual descriptions of the photographs, not category labels used in the task.",
  r"descriptions are visual descriptions of the photographs, not category labels used in the task. DINOv2-s, DINOv2-b, ViT, and RN50: DINOv2-small, DINOv2-base, supervised ViT, and supervised ResNet-50.")
R(r"Adults: selections in the same \VtTurtleSessions{} sessions under each word. A = accepted; R = rejected.",
  r"Adults: selections in the same \VtTurtleSessions{} sessions under each word. A = accepted; R = rejected. Abbreviations as in Table~\ref{tab:v2images}. DINOv2-base accepted \VtXDinoBOut{} of \VtXOutN{} other-domain pairings after broad teaching, so its acceptance of the turtle under the vehicle word partly reflects a permissive operating point of the rule for this encoder.")

# ---------- discussion ----------
R(r"Taken together, these results suggest that a shared, partly human-aligned similarity space describes part of the organization of generalization in both systems.",
  r"Taken together, these results suggest that distances in these spaces describe part of the organization of generalization in both systems. For the learners this is close to mechanical, since they operate on the same distances, and the direct test of whether human distances predict learner difficulty (H2) was inconclusive, so the link between the two systems remains indirect.")
R(r"and alignment with human similarity did not separate the trained encoders from the untrained network.",
  r"and the binary H1 rule did not separate the trained encoders from the untrained network, although the trained encoders' correlations with THINGS and ImageNet-HSJ were several times larger. We did not test which properties of an untrained convolutional network, absent from raw pixels, produce its negative slope.")
R(r"the distance-to-centroid measure we used does not, by itself, account for these differences between conditions.",
  r"the distance-to-centroid measure we used does not, by itself, account for these differences between conditions, and in post hoc checks neither did the mean or nearest distance to the individual examples (Appendix~\ref{app:posthoc}). These checks do not exhaust readouts of the same geometry: models that scale generalization by the dispersion of the examples \parencite{tenenbaum2001} remain untested.")
R(r"In our exploratory post hoc models, terms involving experiment-level trial order improved fit,",
  r"In our exploratory post hoc models, terms involving experiment-level trial order improved fit in observation-level tests that ignore the clustering of decisions within experiments,")

R(r"it added to taxonomic level and teaching condition within the taught domain.",
  r"it added to taxonomic level and teaching condition within the taught domain, although its interval included zero once the slope could vary across images (post hoc).")
R(r"gave a negative distance slope, so the association cannot be attributed only to what the encoders learned;",
  r"gave a negative distance slope, so the association cannot be attributed only to what the encoders learned (with by-image random slopes, post hoc, the slopes varied widely across images and the interval included zero for \VcImgTrInclList);")
R(r"and within superordinate items both condition and distance improved fit given the other;",
  r"and within superordinate items both condition and distance improved fit given the other (distance no longer did once its slope could vary across images);")
# ---------- limitations ----------
R(r"only one final-layer global embedding per encoder was used.",
  r"only one final-layer global embedding per encoder was used, read out as the cosine distance to the normalized centroid. Intermediate layers and patch tokens, other metrics (e.g., Mahalanobis or whitened Euclidean distance), exemplar models of categorization such as the generalized context model \parencite{nosofsky1986}, and a randomly initialized vision transformer as a second untrained control were not examined; the post hoc exemplar distances of Appendix~\ref{app:posthoc} are only a first check.")
R(r"The human mixed models include random intercepts by image but no by-image random slopes for distance, so heterogeneity across images in sensitivity to distance was not estimated.",
  r"The planned human mixed models include random intercepts by image (24 stimulus slots) but no by-image random slopes for distance; in a post hoc check with by-image slopes, the image slopes varied widely and the interval of the average slope included zero for \VcImgTrInclList{} (Appendix~\ref{app:posthoc}). For comparability across encoders, the stage-1 models of the primary test used the structure that converged for every encoder, which had by-person random intercepts only. The dissociation pairs were selected with DINOv2-small, and the WordNet sense correction was checked only with DINOv2-small. Preprocessing pipelines differ in field of view on CIFAR images (Appendix~\ref{app:checkpoints}).")
R(r"and the alignment of encoder geometry with human similarity (H1) did not separate trained encoders from the controls.",
  r"and the binary H1 rule did not separate trained encoders from the controls, despite large differences in the size of the correlations.")
R(r"Optional sensitivity analyses (MAE with CLS pooling, random-network seeds) were not run.",
  r"The optional variants (MAE with CLS pooling, random-network seeds 1 and 2) were not run in the CIFAR analyses; in adults they entered only the sensitivity analyses of the primary test, and they are reported in the exploratory comparison on the same images (Table~\ref{tab:v2matched}).")

# ---------- conclusions: footnote and editorial paragraph (P. M. Vera's paragraph untouched) ----------
R(r"\footnote{This statement rests on the positive-control models of the eight within-domain test images (Figure~\ref{fig:forest}a). A randomly initialized ResNet-50 also gave a negative slope (\VtPcRandB{} [\VtPcRandLo,~\VtPcRandHi]), so the association is not attributed to what the encoders learned, and the frozen primary test of whether presentation order changes the slope was inconclusive ($\hat\Delta = \VtOrdDelta$ [\VtOrdLo,~\VtOrdHi]).}",
  r"\footnote{This statement rests on the positive-control models of the eight within-domain test images (Figure~\ref{fig:forest}a), with distance measured as the cosine distance to the centroid of the teaching examples in final-layer global embeddings. A randomly initialized ResNet-50 also gave a negative slope (\VtPcRandB{} [\VtPcRandLo,~\VtPcRandHi]), so the association is not attributed to what the encoders learned. In post hoc models that let the slope vary across images, its interval included zero for \VcImgTrInclList{} (Appendix~\ref{app:posthoc}). The frozen primary test of order moderation is summarized in the next paragraph.}")
R(r"because the geometry calibration failed at every cutoff, these alignments do not show that the encoders differ in a human-relevant way.",
  r"the binary support rule did not separate the encoders from the controls, although on THINGS and ImageNet-HSJ the trained encoders other than MAE correlated far more strongly with human judgments ($\rho$ from \VcHoneTrMin{} to \VcHoneTrMax) than the untrained network.")
R(r"the largest attainable gain ranged from" + "\n" if False else r"\VtXJumpMin{} to \VtXJumpMax{} percentage points across the eight trained encoders, against \VtXJumpAdult{} points for adults." + "\n",
  r"\VtXJumpMin{} to \VtXJumpMax{} percentage points across the eight trained encoders, against \VtXJumpAdult{} points for adults (same-subtype to broad teaching); the largest value is one candidate in twelve below the next attainable step. Post hoc, distances to the individual teaching examples did not account for the effect of composition in adults either, and with by-image random slopes the adult distance slope was uncertain for \VcImgTrInclList." + "\n")

# ---------- future work: editorial paragraph after P. M. Vera's text ----------
R(r"provide evidence that learned relations help the agent interpret what a word refers to." + "\n",
  r"provide evidence that learned relations help the agent interpret what a word refers to." + "\n\n" +
  r"% Editorial paragraph (review changes approved by P. M. Vera, 24-11-2026); P. M. Vera's Future Work text above is unchanged." + "\n" +
  r"Other directions follow from the limits of the present readout: intermediate layers and patch tokens, other metrics (Mahalanobis or whitened Euclidean distances), exemplar models of categorization such as the generalized context model \parencite{nosofsky1986}, dispersion-sensitive accounts of generalization \parencite{tenenbaum2001}, and a randomly initialized vision transformer as a second untrained control." + "\n")

# ---------- code and data availability: URL break ----------
R(r"(CC BY 4.0) and \url{https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data-cifar10h}",
  r"(CC BY 4.0) and\newline \url{https://huggingface.co/datasets/MrPatoVera/lexical-consensus-data-cifar10h}")

# ---------- appendix A ----------
R(r"multi-encoder analyses; the released deviation logs document every step.",
  r"multi-encoder analyses; the released deviation logs document every step. Hypotheses are numbered as in the first plan: H1 (alignment with human distances), H2 (human distances and learner difficulty), H3 (a descriptive comparison of the CIFAR-10 tiers with human distances, not reported here), H4 (adult word extension), H5 (age of acquisition, exploratory), and H6 (the CVCL encoder, exploratory). Appendix~\ref{app:rules} defines the decision rules and Appendix~\ref{app:posthoc} reports the analyses added after review.")
R(r"Pipeline control 1 tolerance & The rule text required",
  r"Pipeline control 1 tolerance & Pipeline control 1 checks that the agent code with DINOv2-small reproduces the earlier CIFAR-10 tier results exactly; pipeline control 2 checks that re-encoding CIFAR-100 with DINOv2-small on CPU reproduces the earlier distance matrix. The rule text required")
R(r"the optional H5 block, and the full T-sem block were not run; they remain pending. In adults, these variants exist only as sensitivity analyses of the primary test. \\",
  r"the optional H5 block, and the full T-sem block (the tier ordering across master seeds with re-drawn image pools) were not run; they remain pending. MAE with CLS pooling uses MAE's class token instead of the mean of its patch tokens; seeds 1 and 2 are two further random initializations of ResNet-50. In adults these variants entered only the sensitivity analyses of the primary test, and they are reported in the exploratory comparison on the same images (Table~\ref{tab:v2matched}). \\" + "\n" +
  r"Post hoc analyses after review & Specified on \VcPosthocDate, after two external reviews and after all results were known: by-image random slopes, exemplar distances, distance distributions, and H2 estimates with minimum detectable effects (Appendix~\ref{app:posthoc}). Outside every frozen plan; exploratory. \\")

# ---------- appendix B: preprocessing ----------
R(r"\par\smallskip{\raggedright\footnotesize\textit{Note.} Revisions are the first eight characters of the Hugging Face commit pinned in the embedding script.\par}",
  r"\par\smallskip{\raggedright\footnotesize\textit{Note.} Revisions are the first eight characters of the Hugging Face commit pinned in the embedding script.\par}" + "\n" + r"\end{table}" + "\n\n" +
  r"Each representation used its own preprocessing, recorded in the released specification file. On square images, \VcCropList{} resize the shorter side and keep the central 224 of 256 pixels per side (87.5\%), whereas \VcFullList{} resize the whole image to $224 \times 224$; OpenCLIP resizes the shorter side to 224 and crops 224, which keeps the full frame of a square image. Pixels are the $32 \times 32$ RGB values after the DINOv2-style resize and crop. MAE uses the mean of its patch tokens (CLS pooling was a sensitivity variant), and the random ResNet-50 uses seed 0 (seeds 1 and 2 were sensitivity variants)." + "\n" +
  r"\begin{table}[!h]\end{table}% placeholder removed below")
s = s.replace("\n" + r"\begin{table}[!h]\end{table}% placeholder removed below" + "\n" + r"\end{table}", "")

# ---------- appendix C notes ----------
R(r"$^{a}$Fitted with random intercepts only, because the model with by-person distance slopes did not pass the convergence and non-singularity check. The controls are not gates. Variants not run for these analyses: MAE with CLS pooling and random ResNet-50 seeds 1 and 2.",
  r"$^{a}$Fitted with random intercepts only, because the model with by-person distance slopes did not pass the convergence and non-singularity check; omitting the by-person slope tends to pull the slope toward zero relative to the other rows. Slopes are per standard deviation of each representation's distance (raw standard deviations in Appendix Table~\ref{tab:posthoc}). Regions are defined in Appendix~\ref{app:rules}. The controls are not gates. MAE with CLS pooling and random ResNet-50 seeds 1 and 2 entered only the sensitivity analyses of the primary test and are not shown.")
R(r"Regions at SESOI .10/.20/.30: S = support, I = inconclusive, N = no support. DINOv2-small and CVCL: descriptive. Not run: MAE with CLS pooling and random ResNet-50 seeds 1 and 2.",
  r"Regions at SESOI .10/.20/.30: S = support, I = inconclusive, N = no support (Appendix~\ref{app:rules}); support requires a negative effect with unadjusted $p < .05$ and does not depend on the SESOI, so S/S/S does not mean that the effect exceeds .30. D-all regions treat the \VtDallN{} pairs as independent and are descriptive. DINOv2-small and CVCL: descriptive. Not run in the CIFAR analyses: MAE with CLS pooling and random ResNet-50 seeds 1 and 2.")
R(r"with region at SESOI .10/.20/.30 (support requires $\rho > 0$ and Mantel $p < .05$). Mantel $p$ from 10{,}000 permutations. H2: region for human distance predicting learner difficulty.",
  r"with region at SESOI .10/.20/.30 (support requires $\rho > 0$ and Mantel $p < .05$ and does not depend on the SESOI; Appendix~\ref{app:rules}). Mantel $p$ from 10{,}000 permutations. H2: unadjusted region for human distance predicting learner difficulty (the Holm-adjusted reference analysis of Section~\ref{sec:link} is inconclusive for DINOv2-small on ImageNet-HSJ); estimates in Appendix Table~\ref{tab:h2post}.")

# ---------- new appendices D (rules) and E (post hoc) ----------
R(r"\end{document}", r"\input{appendix_rules_posthoc_v2c.tex}" + "\n\n" + r"\end{document}")

# ---------- guard: P. M. Vera's texts unchanged ----------
for frag in [r"The central example is a single photograph (\texttt{c3\_sup2}, a turtle).", r"The two agents thus departed from the adult pattern in opposite directions.",
             r"Visual distance predicted adult word extension across all eight trained encoders, extending the original finding beyond DINOv2.",
             r"Whether adults arrive at that inference through the same processes as any of these agents remains open.",
             r"A next step is to train agents to use relations among categories to infer how broadly a novel word applies."]:
    assert src.count(frag) == 1 and s.count(frag) == 1, frag
def para(t, start):
    i = t.index(start); return t[i:t.index("\n", i)]
for start in [r"The central example is a single photograph", r"Visual distance predicted adult word extension across all eight trained encoders",
              r"A next step is to train agents to use relations"]:
    a, b = para(src, start), para(s, start)
    if start.startswith("Visual distance"):   # only the editorial footnote inside this paragraph may change
        strip = lambda t: re.sub(r"\\footnote\{.*?\}\s(?=In an)", "", t)
        assert strip(a) == strip(b), "Conclusions paragraph changed"
    else:
        assert a == b, start
assert "AI assistant" not in s
open(sys.argv[2], "w").write(s); print("edits applied; Patricio's texts verified unchanged")
