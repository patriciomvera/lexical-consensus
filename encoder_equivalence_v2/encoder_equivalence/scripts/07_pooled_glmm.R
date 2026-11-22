# Pooled GLMMs (plan §5.3 positive control; §5.4 S1, S2, S3; family A reference F0). Mirrors 04c/04e/04f formulas.
#  poscontrol : 8 items, selected ~ zd + level + condition + (1+zd|person) + (1|image)  [= C6]. Gate: beta(zd) < 0 and 95% Wald CI excludes 0.
#               (fallback as in 04c: (1|person)+(1|image) if singular/non-convergent; recorded)
#  S1         : Mbase / Mb / Mc of 04f (unchanged), on <items> (24 or 8). LR chi2(1) Mb vs Mbase, chi2(3) Mc vs Mb. Descriptive.
#  S2         : 8 items, Mb + (1|exp) + (0+zd|exp). Descriptive.
#  S3         : 8 items, Mb + level:order. Descriptive (I1).
#  F0         : no distance, selected ~ condition*order + level + (1|person) + (1|image), 24 items (family A reference). ID is only used to read rows.
# Usage: Rscript 07_pooled_glmm.R <long_dir> <key.csv> <out.csv> <model:poscontrol|S1|S2|S3|F0> <items:8|24> <suffix|-> ID [ID ...]
args <- commandArgs(trailingOnly = TRUE)
src <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "lib_glmm.R"); source(src)
long_dir <- args[1]; key <- args[2]; out <- args[3]; model <- args[4]; items <- as.integer(args[5]); suffix <- if (args[6] == "-") "" else args[6]; ids <- args[-(1:6)]
set.seed(42); RE <- "+ (1 + zd | person) + (1 | image)"; res <- list()
tab <- function(id, tag, m, extra = list()) {
  co <- summary(m)$coefficients
  data.frame(encoder = id, model = tag, items = items, term = rownames(co), estimate = co[, 1], se = co[, 2], z = co[, 3], p = co[, 4],
             warn = warn_msg(m), singular = isSingular(m), nobs = nobs(m), row.names = NULL, stringsAsFactors = FALSE)
}
lr <- function(id, tag, m0, m1) { l <- as.numeric(2 * (logLik(m1) - logLik(m0))); df <- attr(logLik(m1), "df") - attr(logLik(m0), "df")
  data.frame(encoder = id, model = tag, items = items, term = sprintf("LR_chi2(%d)", df), estimate = l, se = NA, z = NA, p = pchisq(l, df, lower.tail = FALSE),
             warn = "", singular = NA, nobs = nobs(m1), stringsAsFactors = FALSE) }
for (id in ids) {
  d <- add_zd(item_subset(read_long(file.path(long_dir, paste0("h4_long_", id, suffix, ".csv")), key), items)); t0 <- Sys.time()
  if (model == "poscontrol") {
    stopifnot(items == 8)
    m <- safe_glmer(as.formula(paste("selected ~ zd + level + condition", RE)), d); tag <- "poscontrol_slope"
    if (!fit_ok(m)) { m <- glmer(selected ~ zd + level + condition + (1 | person) + (1 | image), d, binomial, control = CTL); tag <- "poscontrol_noslope" }  # as 04c
    r <- tab(id, tag, m); zr <- r[r$term == "zd", ]; lo <- zr$estimate - qnorm(.975) * zr$se; hi <- zr$estimate + qnorm(.975) * zr$se
    r$gate_pass <- (zr$estimate < 0) && (hi < 0); r$zd_lo <- lo; r$zd_hi <- hi; res[[length(res) + 1]] <- r
  } else if (model == "S1") {
    # exactly 04f: Mbase, Mb = Mbase + zd:order, Mc = Mb + zd:condition:order (treatment contrasts, as in 04f); no fallback in 04f
    r <- tryCatch({ mb0 <- glmer(as.formula(paste("selected ~ zd*condition + condition*order + level", RE)), d, binomial, control = CTL)
      mb <- update(mb0, . ~ . + zd:order); mc <- update(mb, . ~ . + zd:condition:order)
      rbind(tab(id, "S1_Mb", mb), lr(id, "S1_Mb_vs_Mbase", mb0, mb), tab(id, "S1_Mc", mc), lr(id, "S1_Mc_vs_Mb", mb, mc)) },
      error = function(e) data.frame(encoder = id, model = "S1_failed", items = items, term = NA, estimate = NA, se = NA, z = NA, p = NA, warn = conditionMessage(e), singular = NA, nobs = nrow(d)))
    res[[length(res) + 1]] <- r
  } else if (model == "S2") {
    stopifnot(items == 8)
    m <- safe_glmer(as.formula(paste("selected ~ zd*condition + condition*order + level + zd:order", RE, "+ (1 | exp) + (0 + zd | exp)")), d)
    res[[length(res) + 1]] <- if (is.null(m)) data.frame(encoder = id, model = "S2_failed", items = items, term = NA, estimate = NA, se = NA, z = NA, p = NA, warn = "fit error", singular = NA, nobs = nrow(d)) else tab(id, "S2_Mb_expRE", m)
  } else if (model == "S3") {
    stopifnot(items == 8)
    m <- safe_glmer(as.formula(paste("selected ~ zd*condition + condition*order + level + zd:order + level:order", RE)), d)
    res[[length(res) + 1]] <- if (is.null(m)) data.frame(encoder = id, model = "S3_failed", items = items, term = NA, estimate = NA, se = NA, z = NA, p = NA, warn = "fit error", singular = NA, nobs = nrow(d)) else tab(id, "S3_Mb_levelxorder", m)
  } else if (model == "F0") {
    m <- glmer(selected ~ condition * order + level + (1 | person) + (1 | image), d, binomial, control = CTL); res[[length(res) + 1]] <- tab("F0_nodistance", "F0", m)
  } else stop("unknown model")
  cat(sprintf("%s %s items=%d done in %.1f s\n", id, model, items, as.numeric(difftime(Sys.time(), t0, units = "secs")))); flush.console()
}
all <- do.call(rbind, lapply(res, function(r) { for (c in c("gate_pass", "zd_lo", "zd_hi")) if (!c %in% names(r)) r[[c]] <- NA; r }))
write.csv(all, out, row.names = FALSE); cat("wrote", out, nrow(all), "rows\n")
