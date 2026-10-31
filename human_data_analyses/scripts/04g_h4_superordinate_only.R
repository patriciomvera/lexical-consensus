# EXPLORATORIO POST HOC (DESV §35; paperreview.ai Q1). Fuera del plan congelado y de la familia Holm.
suppressMessages(library(lme4)); set.seed(42)
d <- read.csv("data/h4_long.csv"); d$person <- factor(d$subid)
d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
p <- droplevels(subset(d, in_target == 1 & level == "sup")); p$image <- factor(p$image)
cat(sprintf("N obs=%d personas=%d imagenes=%d SD_dist=%.4f\n", nrow(p), nlevels(p$person), nlevels(p$image), sd(p$dist)))
print(round(tapply(p$selected, p$condition, mean), 4)); print(round(tapply(p$dist, p$condition, mean), 4))
p$zd <- as.numeric(scale(p$dist))
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
w <- function(m) paste(m@optinfo$conv$lme4$messages, collapse=";")
run <- function(re, tag){
  md  <- glmer(as.formula(paste("selected ~ zd", re)), p, binomial, control=ctl)
  mc  <- glmer(as.formula(paste("selected ~ condition", re)), p, binomial, control=ctl)
  mdc <- glmer(as.formula(paste("selected ~ zd + condition", re)), p, binomial, control=ctl)
  cat("==", tag, "| warn md/mc/mdc:", w(md), "|", w(mc), "|", w(mdc), "| singular:", isSingular(md), isSingular(mc), isSingular(mdc), "\n")
  l1 <- as.numeric(2*(logLik(mdc)-logLik(md))); l2 <- as.numeric(2*(logLik(mdc)-logLik(mc)))
  cat(sprintf("%s LR_cond_given_dist chi2(3)=%.3f p=%.4g\n", tag, l1, pchisq(l1, 3, lower.tail=FALSE)))
  cat(sprintf("%s LR_dist_given_cond chi2(1)=%.3f p=%.4g\n", tag, l2, pchisq(l2, 1, lower.tail=FALSE)))
  co <- summary(mdc)$coefficients; print(round(co, 4))
  for (r in rownames(co)[-1]) { ci <- co[r,1]+c(-1,1)*qnorm(.975)*co[r,2]
    cat(sprintf("%s COEF %s=%.4f CI[%.4f, %.4f] p=%.4g\n", tag, r, co[r,1], ci[1], ci[2], co[r,4])) }
  flush.console()
}
run("+ (1 | person) + (1 | image)", "PRIMARY")
run("+ (1 + zd | person) + (1 | image)", "SENS_SLOPE")
