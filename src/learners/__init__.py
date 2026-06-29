"""
src.learners
------------
Lexical learner zoo for exp_007 (concept carving / prototype-episodic
dissociation).

Every learner implements the SAME interface (see base.Learner) so the
experiment harness can swap them transparently inside an episode loop:

    fit(seed_emb, seed_concept_ids) -> None
    predict(emb)                    -> concept_id
    score(emb)                      -> per-concept similarity/logit vector
    retrieve(concept_id, cands)     -> index into cands  (bidirectional C2)

The zoo spans the prototype -> episodic continuum that exp_007 is built to
probe (plan §2, Axis B):

    CentroidLearner   — 1 centroid per concept                (pure prototype)
    KCentroidLearner  — K sub-centroids via intra-concept k-means (bridge)
    ExemplarKNN       — stores all seeds, k-NN classify       (pure episodic)
    LogRegLearner     — multinomial logistic regression       (reviewer baseline)
    LinSVMLearner     — linear SVM, one-vs-rest                (reviewer baseline)
    RandomLearner     — random-label floor                    (chance control)

H1 prediction: centroid accuracy collapses as native separability NS(c)
drops; exemplar/k-centroid hold. These classes are where that prediction
is operationalized.
"""

from __future__ import annotations

from .base import Learner

# Concrete learners are imported lazily by the harness via LEARNER_REGISTRY
# to keep optional sklearn imports out of the import path for code that only
# needs the interface.
LEARNER_REGISTRY: dict[str, str] = {
    "centroid":     "src.learners.centroid:CentroidLearner",
    "kcentroid":    "src.learners.kcentroid:KCentroidLearner",
    "exemplar_knn": "src.learners.exemplar_knn:ExemplarKNN",
    "logreg":       "src.learners.logreg:LogRegLearner",
    "linsvm":       "src.learners.linsvm:LinSVMLearner",
    "random":       "src.learners.random_baseline:RandomLearner",
}

__all__ = ["Learner", "LEARNER_REGISTRY"]
