# H4 GLMM (plan §H4). Primary (literal plan): all 24 test items x trial, level from file name.
# Sensitivities: (i) excl. non-preregistered exps; (ii) only the 8 target-category items; exploratory: zd x condition.
suppressMessages(library(lme4)); set.seed(42)
d <- read.csv("data/h4_long.csv"); d$subid <- paste(d$exp, d$subid, sep="_")  # subids repeat across experiments
d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
fit <- function(dd, f_slope, f_noslope, tag) {
  dd$zd <- as.numeric(scale(dd$dist)); dd$level <- factor(dd$level); dd$image <- factor(dd$image)
  m <- tryCatch(glmer(f_slope, dd, binomial, control=ctl), error=function(e) NULL)
  conv <- !is.null(m) && is.null(m@optinfo$conv$lme4$messages) && !isSingular(m); used <- "slope"
  if (!conv) { m <- glmer(f_noslope, dd, binomial, control=ctl); used <- "noslope" }
  co <- summary(m)$coefficients
  cat(sprintf("\n== %s | n=%d subj=%d | RE=%s | warnings=%s\n", tag, nrow(dd), length(unique(dd$subid)), used, paste(m@optinfo$conv$lme4$messages, collapse=";")))
  print(round(co, 4)); invisible(m)
}
fs <- selected ~ zd + level + condition + (1 + zd | subid) + (1 | image)
fn <- selected ~ zd + level + condition + (1 | subid) + (1 | image)
m1 <- fit(d, fs, fn, "PRIMARY all 24 items, level from file name")
fsi <- update(fs, . ~ . + zd:condition); fni <- update(fn, . ~ . + zd:condition)
m4 <- fit(d, fsi, fni, "EXPLORATORY zd x condition (all items)")
base <- update(m4, . ~ . - zd:condition); print(anova(base, m4))
