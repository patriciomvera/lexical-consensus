# Shared helpers for the encoder comparison GLMMs (v2 of arXiv 2611.14116). Formulas mirror 04c/04e/04f of human_data_analyses.
suppressMessages(library(lme4))
CTL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
COND_MAP <- c(one="one", three_subordinate="sub", "3sub"="sub", three_basic="bas", "3bas"="bas", three_superordinate="sup", "3sup"="sup")
read_long <- function(path, key_path) {
  d <- read.csv(path); d$person <- factor(d$subid)
  d$condition <- factor(COND_MAP[as.character(d$condition)], levels = c("one", "sub", "bas", "sup"))
  stopifnot(!any(is.na(d$condition)))
  k <- read.csv(key_path)[1:12, ]; k$exp <- as.integer(k$exp); k$order <- gsub('"', '', k$order)
  d$order <- factor(k$order[match(d$exp, k$exp)], levels = c("1-3", "3-1")); stopifnot(!any(is.na(d$order)))
  d$level <- factor(d$level, levels = c("bas", "sub", "sup")); d$image <- factor(d$image); d
}
# zd standardised over the pooled item set (all experiments) -- plan §4.1; rank variant = S5 (percentile rank, then z)
add_zd <- function(d, rank = FALSE) {
  x <- if (rank) rank(d$dist, ties.method = "average") / nrow(d) else d$dist
  d$zd <- as.numeric(scale(x)); d
}
item_subset <- function(d, items) if (items == 8) droplevels(subset(d, in_target == 1)) else d
warn_msg <- function(m) paste(m@optinfo$conv$lme4$messages, collapse = ";")
# lme4 >= 2.0 records "boundary (singular) fit" among the conv messages; convergence is judged WITHOUT that message
conv_msgs <- function(m) { x <- m@optinfo$conv$lme4$messages; x[!grepl("singular", x)] }
fit_ok <- function(m) !is.null(m) && length(conv_msgs(m)) == 0 && !isSingular(m)          # primary: converged AND not singular
fit_converged <- function(m) !is.null(m) && length(conv_msgs(m)) == 0                       # fallback: converged (singular allowed, logged)
safe_glmer <- function(f, d, ctl = CTL) tryCatch(suppressWarnings(glmer(f, d, binomial, control = ctl)), error = function(e) NULL)
# ---- STAGE-1 LADDER (plan §5.2; Patricio 18-11 12:12 and 12:14; REVISION3 I1 + Ariadna's (a)/(b)) ----
# Fixed effects identical on every rung: selected ~ zd * condition + level (condition contr.sum). Rungs, richest -> simplest:
LADDER <- c(L1 = "(1 + zd | person) + (1 | image)",   # correlated random slope
            L2 = "(1 + zd || person) + (1 | image)",  # uncorrelated random slope
            L3 = "(1 | person) + (1 | image)",
            L4 = "(1 | person)")
# Convergence criterion (frozen; tolerances written explicitly = lme4 2.0.6 defaults): Laplace (nAGQ = 1), bobyqa, maxfun 2e5;
# converged <=> (i) no error, (ii) optimizer return code 0, (iii) no lme4 convergence message (gradient check tol 2e-3 with relTol NULL,
# Hessian check tol 1e-6; e.g. "failed to converge", "nearly unidentifiable", "unable to evaluate scaled gradient"), and
# (iv) NOT singular (isSingular, tol 1e-4): a singular fit counts as NOT converged. Other R warnings (e.g. glm.fit) are logged, not counted.
LADDER_CTL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                           check.conv.grad = .makeCC("warning", tol = 2e-3, relTol = NULL),
                           check.conv.singular = .makeCC(action = "message", tol = 1e-4),
                           check.conv.hess = .makeCC("warning", tol = 1e-6))
ladder_converged <- function(m) !is.null(m) && identical(as.integer(m@optinfo$conv$opt), 0L) && length(conv_msgs(m)) == 0 && !isSingular(m, tol = 1e-4)
fit_rung <- function(rung, dd) safe_glmer(as.formula(paste("selected ~ zd * condition + level +", LADDER[[rung]])), dd, LADDER_CTL)
