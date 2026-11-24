# B1 (post hoc, v2c): by-image random slopes for distance as a robustness check of the adult positive-control model.
# Frozen model (plan v2 §5.3): 8 within-domain items, selected ~ zd + level + condition + (1 + zd | person) + (1 | image);
# fallback (1 | person) + (1 | image). Here: the same fixed effects with a by-image distance slope, ladder declared before fitting:
#   I1 (1 + zd | person) + (1 + zd | image); I2 (1 + zd | person) + (1 + zd || image); I3 (1 | person) + (1 + zd | image); I4 (1 | person) + (1 + zd || image)
# plus a stimulus-cell intercept variant: C1 (1 + zd | person) + (1 | image) + (1 | cell); C2 (1 | person) + (1 | image) + (1 | cell), cell = image x condition.
# Reference encoder only: the superordinate-only model (12 images) with a by-image slope or a cell intercept (LR tests as in 04g).
# Usage: Rscript p02_image_slopes.R <lib_ph.R> <data_dir> <out_dir> ID
args <- commandArgs(trailingOnly = TRUE); source(args[1]); dd <- args[2]; od <- args[3]; id <- args[4]
set.seed(42)
d <- read_ex(file.path(dd, paste0("ex_long_", id, ".csv")))
p8 <- droplevels(subset(d, in_target == 1)); p8$zd <- as.numeric(scale(p8$dist))
FX <- "selected ~ zd + level + condition"; out <- list()
t0 <- Sys.time(); f <- ladder(FX, c("(1 + zd | person) + (1 | image)", "(1 | person) + (1 | image)"), p8)
out[[1]] <- coef_row("pc_frozen_refit", id, f$m, "zd", f$re, t0)
t0 <- Sys.time(); f <- ladder(FX, c("(1 + zd | person) + (1 + zd | image)", "(1 + zd | person) + (1 + zd || image)",
                                     "(1 | person) + (1 + zd | image)", "(1 | person) + (1 + zd || image)"), p8)
out[[2]] <- coef_row("pc_image_slope", id, f$m, "zd", f$re, t0)
if (f$ok) { v <- as.data.frame(VarCorr(f$m)); out[[2]]$sd_image_slope <- v$sdcor[v$grp %in% c("image", "image.1") & v$var1 == "zd" & is.na(v$var2)][1] }
t0 <- Sys.time(); f <- ladder(FX, c("(1 + zd | person) + (1 | image) + (1 | cell)", "(1 | person) + (1 | image) + (1 | cell)"), p8)
out[[3]] <- coef_row("pc_cell_intercept", id, f$m, "zd", f$re, t0)
if (id == "R0_dinov2s") {
  p <- droplevels(subset(d, in_target == 1 & level == "sup")); p$zd <- as.numeric(scale(p$dist))
  for (re in c("(1 | person) + (1 | image)", "(1 | person) + (1 + zd | image)", "(1 | person) + (1 | image) + (1 | cell)")) {
    t0 <- Sys.time(); md <- safe_glmer(paste("selected ~ zd +", re), p); mc <- safe_glmer(paste("selected ~ condition +", re), p)
    mdc <- safe_glmer(paste("selected ~ zd + condition +", re), p)
    r <- coef_row("sup_only", id, mdc, "zd", re, t0)
    r$ok_all3 <- fit_ok(md) && fit_ok(mc) && fit_ok(mdc)
    r$LR_dist_given_cond <- if (!is.null(mdc) && !is.null(mc)) as.numeric(2 * (logLik(mdc) - logLik(mc))) else NA
    r$LR_cond_given_dist <- if (!is.null(mdc) && !is.null(md)) as.numeric(2 * (logLik(mdc) - logLik(md))) else NA
    r$p_dist_given_cond <- pchisq(r$LR_dist_given_cond, 1, lower.tail = FALSE); r$p_cond_given_dist <- pchisq(r$LR_cond_given_dist, 3, lower.tail = FALSE)
    out[[length(out) + 1]] <- r }
}
all <- do.call(rbind, lapply(out, function(x) { for (k in c("sd_image_slope", "ok_all3", "LR_dist_given_cond", "LR_cond_given_dist", "p_dist_given_cond", "p_cond_given_dist")) if (is.null(x[[k]])) x[[k]] <- NA; x }))
write.csv(all, file.path(od, paste0("p02_", id, ".csv")), row.names = FALSE); print(all[, c("model", "re", "estimate", "lo", "hi", "converged_ok", "seconds")])
