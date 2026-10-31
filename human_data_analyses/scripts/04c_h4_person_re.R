# H4 corrección (DESV §26): en all_raw_A.csv de XTMEM el subid se asigna tras unir los 12 experimentos
# (munge_anonymize_data.R, commit 220ed17), así que un subid repetido = misma persona (600 participaciones, 520 personas).
# Re-estima H4 con efectos aleatorios por persona real + sensibilidades.
suppressMessages(library(lme4)); set.seed(42)
d <- read.csv("data/h4_long.csv"); d$person <- factor(d$subid); d$partic <- factor(paste(d$exp, d$subid, sep="_"))
d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
nexp <- tapply(d$exp, d$subid, function(x) length(unique(x))); firstexp <- tapply(d$exp, d$subid, min)
res <- list()
fit <- function(dd, fs, fn, tag) {
  dd$zd <- as.numeric(scale(dd$dist)); dd$level <- factor(dd$level); dd$image <- factor(dd$image)
  m <- tryCatch(glmer(fs, dd, binomial, control=ctl), error=function(e) NULL)
  conv <- !is.null(m) && is.null(m@optinfo$conv$lme4$messages) && !isSingular(m); used <- "slope"
  if (!conv) { m <- glmer(fn, dd, binomial, control=ctl); used <- "noslope" }
  co <- summary(m)$coefficients; z <- co["zd",]; ci <- z[1] + c(-1,1)*qnorm(.975)*z[2]
  cat(sprintf("\n== %s | n=%d personas=%d particip=%d | RE=%s | warn=%s\n", tag, nrow(dd), length(unique(dd$person)), length(unique(dd$partic)), used, paste(m@optinfo$conv$lme4$messages, collapse=";")))
  print(round(co, 4)); cat(sprintf("zd logOR=%.4f CI[%.4f, %.4f] p=%.3g\n", z[1], ci[1], ci[2], z[4]))
  res[[tag]] <<- list(n=nrow(dd), persons=length(unique(dd$person)), RE=used, b=unname(z[1]), se=unname(z[2]), lo=ci[1], hi=ci[2], p=unname(z[4]))
  invisible(m) }
fs <- selected ~ zd + level + condition + (1 + zd | person) + (1 | image)
fn <- selected ~ zd + level + condition + (1 | person) + (1 | image)
fit(d, fs, fn, "C1 todas 24, RE persona")
fit(d, update(fs, . ~ . + (1 | partic)), update(fn, . ~ . + (1 | partic)), "C2 todas 24, RE persona + participacion")
fit(subset(d, exp == firstexp[as.character(subid)]), fs, fn, "C3 solo primera participacion (exp de menor numero; orden temporal no disponible)")
fit(subset(d, nexp[as.character(subid)] == 1), fs, fn, "C4 excluye personas que repiten")
fit(subset(d, !(exp %in% c(9, 10))), fs, fn, "C5 excl. no preregistrados, RE persona")
fit(subset(d, in_target == 1), fs, fn, "C6 8 items objetivo, RE persona")
p <- subset(d, in_target == 1); fit(subset(p, exp == firstexp[as.character(subid)]), fs, fn, "C7 8 items objetivo, primera participacion")
fit(subset(p, nexp[as.character(subid)] == 1), fs, fn, "C8 8 items objetivo, excluye repetidores")
saveRDS(res, "out/h4_person_re.rds")
cat("\nJSON\n"); for (k in names(res)) { r <- res[[k]]; cat(sprintf('{"model":"%s","n":%d,"persons":%d,"RE":"%s","logOR":%.4f,"lo":%.4f,"hi":%.4f,"p":%.3g}\n', k, r$n, r$persons, r$RE, r$b, r$lo, r$hi, r$p)) }
