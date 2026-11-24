# B2 (post hoc, v2c): exemplar-based distances versus the centroid distance in the adult data.
# For each representation, three readouts of the same teaching images: dist (centroid, frozen), dmin (nearest exemplar), dmean (mean exemplar).
# (a) Positive-control model on the 8 within-domain items, selected ~ z + level + condition + RE, with the RE structure that converged
#     for the centroid model (frozen fallback order: (1 + z | person) + (1 | image), then (1 | person) + (1 | image)); the same structure is
#     used for the three readouts so that AIC is comparable; z standardized within the 8 items, per readout.
# (b) Superordinate-only model (12 images, as 04g): selected ~ z + condition + (1 | person) + (1 | image); LR chi2(3) for condition given
#     the readout, LR chi2(1) for the readout given condition, and AIC.
# Usage: Rscript p03_exemplar.R <lib_ph.R> <data_dir> <out_dir> ID
args <- commandArgs(trailingOnly = TRUE); source(args[1]); dd <- args[2]; od <- args[3]; id <- args[4]
set.seed(42)
d <- read_ex(file.path(dd, paste0("ex_long_", id, ".csv"))); out <- list()
p8 <- droplevels(subset(d, in_target == 1))
for (v in c("dist", "dmin", "dmean")) p8[[paste0("z_", v)]] <- as.numeric(scale(p8[[v]]))
t0 <- Sys.time(); base <- ladder("selected ~ z_dist + level + condition", c("(1 + z_dist | person) + (1 | image)", "(1 | person) + (1 | image)"), p8)
for (v in c("dist", "dmin", "dmean")) {
  zv <- paste0("z_", v); re <- gsub("z_dist", zv, base$re, fixed = TRUE); t0 <- Sys.time()
  m <- if (v == "dist") base$m else safe_glmer(paste("selected ~", zv, "+ level + condition +", re), p8)
  r <- coef_row(paste0("pc8_", v), id, m, zv, re, t0); out[[length(out) + 1]] <- r }
p <- droplevels(subset(d, in_target == 1 & level == "sup")); RE <- "(1 | person) + (1 | image)"
for (v in c("dist", "dmin", "dmean")) {
  p$z <- as.numeric(scale(p[[v]])); t0 <- Sys.time()
  md <- safe_glmer(paste("selected ~ z +", RE), p); mc <- safe_glmer(paste("selected ~ condition +", RE), p); mdc <- safe_glmer(paste("selected ~ z + condition +", RE), p)
  r <- coef_row(paste0("sup_", v), id, mdc, "z", RE, t0)
  r$ok_all3 <- fit_ok(md) && fit_ok(mc) && fit_ok(mdc)
  r$LR_cond_given_z <- as.numeric(2 * (logLik(mdc) - logLik(md))); r$LR_z_given_cond <- as.numeric(2 * (logLik(mdc) - logLik(mc)))
  r$p_cond_given_z <- pchisq(r$LR_cond_given_z, 3, lower.tail = FALSE); r$p_z_given_cond <- pchisq(r$LR_z_given_cond, 1, lower.tail = FALSE)
  r$AIC_z_only <- AIC(md); co <- summary(mdc)$coefficients
  r$b_sub <- co["conditionsub", 1]; r$b_bas <- co["conditionbas", 1]; r$b_sup <- co["conditionsup", 1]
  out[[length(out) + 1]] <- r }
keys <- unique(unlist(lapply(out, names)))
all <- do.call(rbind, lapply(out, function(x) { for (k in keys) if (is.null(x[[k]])) x[[k]] <- NA; x[keys] }))
write.csv(all, file.path(od, paste0("p03_", id, ".csv")), row.names = FALSE); print(all[, c("model", "re", "estimate", "lo", "hi", "AIC", "converged_ok", "seconds")])
