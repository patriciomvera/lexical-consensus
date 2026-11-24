# Post hoc v2c helpers (outside the frozen plans). Data reading, z-scoring and convergence criterion mirror
# visual_similarity_v2/encoder_equivalence_v2/encoder_equivalence/scripts/lib_glmm.R at a945ed2 (fit_ok: no lme4 convergence
# message other than the singular-fit note, and not singular).
suppressMessages(library(lme4))
CTL <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
COND_MAP <- c(one="one", three_subordinate="sub", "3sub"="sub", three_basic="bas", "3bas"="bas", three_superordinate="sup", "3sup"="sup")
read_ex <- function(path) {
  d <- read.csv(path); d$person <- factor(d$subid)
  d$condition <- factor(COND_MAP[as.character(d$condition)], levels = c("one", "sub", "bas", "sup")); stopifnot(!any(is.na(d$condition)))
  d$level <- factor(d$level, levels = c("bas", "sub", "sup")); d$image <- factor(d$image); d$cell <- interaction(d$image, d$condition, drop = TRUE); d }
conv_msgs <- function(m) { x <- m@optinfo$conv$lme4$messages; x[!grepl("singular", x)] }
fit_ok <- function(m) !is.null(m) && length(conv_msgs(m)) == 0 && !isSingular(m)
safe_glmer <- function(f, d) tryCatch(suppressWarnings(glmer(as.formula(f), d, binomial, control = CTL)), error = function(e) NULL)
coef_row <- function(tag, id, m, term, re, t0) {
  if (is.null(m)) return(data.frame(encoder = id, model = tag, re = re, term = term, estimate = NA, se = NA, lo = NA, hi = NA, p = NA,
                                    logLik = NA, AIC = NA, converged_ok = FALSE, singular = NA, warn = "fit error", nobs = NA, seconds = NA))
  co <- summary(m)$coefficients; e <- co[term, 1]; s <- co[term, 2]
  data.frame(encoder = id, model = tag, re = re, term = term, estimate = e, se = s, lo = e - qnorm(.975) * s, hi = e + qnorm(.975) * s, p = co[term, 4],
             logLik = as.numeric(logLik(m)), AIC = AIC(m), converged_ok = fit_ok(m), singular = isSingular(m),
             warn = paste(m@optinfo$conv$lme4$messages, collapse = ";"), nobs = nobs(m),
             seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
}
# first random-effects structure (in the declared order) whose fit passes fit_ok; returns list(model, re) or the last attempt
ladder <- function(fixed, res, d) {
  for (re in res) { m <- safe_glmer(paste(fixed, "+", re), d); if (fit_ok(m)) return(list(m = m, re = re, ok = TRUE)) }
  list(m = NULL, re = paste("none converged:", paste(res, collapse = " | ")), ok = FALSE)
}
