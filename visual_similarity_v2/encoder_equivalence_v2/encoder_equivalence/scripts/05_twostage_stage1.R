# STAGE 1 (plan §5.2, B1; ladder by decision of Patricio 18-11 12:12/12:14): per encoder x item set x experiment, the zd slope
# (contr.sum on condition, so coef "zd" = mean slope over the 4 conditions) for the rungs of LADDER (lib_glmm.R).
# zd standardised over the POOLED item set (all 12 experiments, or the pooled first-session subset for S4) before splitting.
# fit = "ladder": per experiment, rungs L1 -> L4 until the first converged one (every attempted rung is written: pass 1).
# fit = "L1".."L4": that rung only, in all 12 experiments (pass 2 of the selection, and the sensitivities at the common rung).
# The structure is SELECTED by 05b_select_structure.py from convergence only (never from Delta or p).
# Usage: Rscript 05_twostage_stage1.R <long_dir> <key.csv> <out.csv> <items:8|24> <variant:primary|first_session|rank> <suffix|-> <first_session.csv|-> <fit:ladder|L1|L2|L3|L4> ID [ID ...]
args <- commandArgs(trailingOnly = TRUE)
src <- file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "lib_glmm.R"); source(src)
long_dir <- args[1]; key <- args[2]; out <- args[3]; items <- as.integer(args[4]); variant <- args[5]
suffix <- if (args[6] == "-") "" else args[6]; fs_path <- args[7]; fit <- args[8]; ids <- args[-(1:8)]
stopifnot(items %in% c(8, 24), variant %in% c("primary", "first_session", "rank"), fit %in% c("ladder", names(LADDER)))
rungs <- if (fit == "ladder") names(LADDER) else fit
set.seed(42); res <- list()
for (id in ids) {
  d <- item_subset(read_long(file.path(long_dir, paste0("h4_long_", id, suffix, ".csv")), key), items)
  if (variant == "first_session") { f <- read.csv(fs_path); fe <- setNames(f$first_exp, f$subid); d <- droplevels(subset(d, exp == fe[as.character(subid)])) }
  d <- add_zd(d, rank = (variant == "rank"))
  for (x in sort(unique(d$exp))) {
    dd <- droplevels(subset(d, exp == x)); stopifnot(nlevels(dd$condition) == 4, length(unique(dd$order)) == 1)  # all 4 conditions present
    contrasts(dd$condition) <- contr.sum(4)
    for (r in rungs) {
      t0 <- Sys.time(); m <- fit_rung(r, dd); ok <- ladder_converged(m)
      b <- se <- NA; w <- if (is.null(m)) "fit error" else warn_msg(m); sg <- if (is.null(m)) NA else isSingular(m, tol = 1e-4)
      oc <- if (is.null(m)) NA else as.integer(m@optinfo$conv$opt)
      if (!is.null(m)) { co <- summary(m)$coefficients; b <- co["zd", 1]; se <- co["zd", 2] }
      res[[length(res) + 1]] <- data.frame(encoder = id, items = items, variant = variant, suffix = suffix, exp = x, order = as.character(dd$order[1]),
        n = nrow(dd), persons = nlevels(dd$person), rung = r, converged = ok, beta = b, se = se, warn = w, singular = sg, opt_code = oc,
        secs = round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1))
      cat(sprintf("%s items=%d %s exp=%d order=%s %s converged=%s beta=%.4f (%.1fs)\n", id, items, variant, x, dd$order[1], r, ok, b, res[[length(res)]]$secs)); flush.console()
      if (ok) break
    }
  }
}
r <- do.call(rbind, res); write.csv(r, out, row.names = FALSE); cat("wrote", out, nrow(r), "rows\n")
