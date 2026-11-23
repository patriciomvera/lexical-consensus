# AGENTS addendum, H4 per encoder (people; REPLICA of the v1 confirmatory H4, proposal §2 and §3.1, P7).
# Model = human_data_analyses/scripts/04c_h4_person_re.R, fit "C1 todas 24, RE persona", UNCHANGED:
#   selected ~ zd + level + condition + (1 + zd | person) + (1 | image), all 24 test items, zd = scale(dist) over the 24-item rows,
#   bobyqa maxfun 2e5; if the slope fit errors / has any lme4 message / is singular -> (1 | person) + (1 | image) (same fallback as 04c).
# Input: the frozen per-trial tables of the human part, <long_dir>/h4_long_<ID>.csv (h4_prep_encoders.py; R0 = v1 distances).
# Output CSV: one row per encoder: logOR of zd, se, Wald 95% CI, p (two-sided Wald), RE used, warnings, singular, nobs, seconds, status.
# A fit that fails even with the fallback -> status "tecnico: ..." (region "inconcluso (técnico)" in 14). Nothing is interpreted here.
# Usage: Rscript 13_h4_by_encoder.R <long_dir> <out.csv> ID [ID ...]
suppressMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE); long_dir <- args[1]; out <- args[2]; ids <- args[-(1:2)]
set.seed(42)
ctl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
fs <- selected ~ zd + level + condition + (1 + zd | person) + (1 | image)
fn <- selected ~ zd + level + condition + (1 | person) + (1 | image)
res <- list()
for (id in ids) {
  t0 <- Sys.time()
  r <- tryCatch({
    d <- read.csv(file.path(long_dir, paste0("h4_long_", id, ".csv"))); d$person <- factor(d$subid)
    d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
    stopifnot(!any(is.na(d$condition)))
    d$zd <- as.numeric(scale(d$dist)); d$level <- factor(d$level); d$image <- factor(d$image)
    m <- tryCatch(glmer(fs, d, binomial, control = ctl), error = function(e) NULL)
    conv <- !is.null(m) && is.null(m@optinfo$conv$lme4$messages) && !isSingular(m); used <- "slope"
    if (!conv) { m <- glmer(fn, d, binomial, control = ctl); used <- "noslope" }
    z <- summary(m)$coefficients["zd", ]; ci <- z[1] + c(-1, 1) * qnorm(.975) * z[2]
    data.frame(encoder = id, RE = used, logOR = unname(z[1]), se = unname(z[2]), lo = ci[1], hi = ci[2], p = unname(z[4]),
               warn = paste(m@optinfo$conv$lme4$messages, collapse = ";"), singular = isSingular(m), nobs = nobs(m),
               persons = length(unique(d$person)), status = "ok", stringsAsFactors = FALSE)
  }, error = function(e) data.frame(encoder = id, RE = NA, logOR = NA, se = NA, lo = NA, hi = NA, p = NA, warn = "", singular = NA, nobs = NA,
                                    persons = NA, status = paste("tecnico:", conditionMessage(e)), stringsAsFactors = FALSE))
  if (r$status == "ok" && !all(is.finite(c(r$logOR, r$se, r$p)))) r$status <- "tecnico: estimate not finite"
  r$seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1); res[[length(res) + 1]] <- r
  cat(sprintf("%s H4 done in %.1f s (%s)\n", id, r$seconds, r$status)); flush.console()
}
write.csv(do.call(rbind, res), out, row.names = FALSE); cat("wrote", out, "\n")
