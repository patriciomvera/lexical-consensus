# EXPLORATORIO POST HOC (DESV §33; no forma parte del plan congelado, no entra en la familia Holm).
# Pregunta (revision critica v2, M2): cuanto aporta la distancia por encima de nivel + condicion
# en los 8 items objetivo, y cual es la pendiente dentro de cada nivel. RE por persona (DESV §26).
suppressMessages(library(lme4)); set.seed(42)
d <- read.csv("data/h4_long.csv"); d$person <- factor(d$subid)
d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
sc <- function(dd){ cat(sprintf("SD_dist_raw=%.4f mean_dist=%.4f n=%d\n", sd(dd$dist), mean(dd$dist), nrow(dd))); dd$zd <- as.numeric(scale(dd$dist)); dd$level <- factor(dd$level, levels=c("bas","sub","sup")); dd$image <- factor(dd$image); dd }
a <- sc(d); cat("(arriba: escala de z(d) en el modelo de 24 items)\n")
p <- sc(subset(d, in_target == 1)); cat("(arriba: escala de z(d) en el modelo de 8 items objetivo)\n")
w <- function(m) paste(m@optinfo$conv$lme4$messages, collapse=";")
# M0: nivel + condicion, sin distancia; M1: + z(d) fijo; M2: + pendiente aleatoria de z(d) por persona (= C6)
m0 <- glmer(selected ~ level + condition + (1 | person) + (1 | image), p, binomial, control=ctl)
m1 <- glmer(selected ~ zd + level + condition + (1 | person) + (1 | image), p, binomial, control=ctl)
m2 <- glmer(selected ~ zd + level + condition + (1 + zd | person) + (1 | image), p, binomial, control=ctl)
cat("warn m0/m1/m2:", w(m0), "|", w(m1), "|", w(m2), "| singular:", isSingular(m0), isSingular(m1), isSingular(m2), "\n")
print(anova(m0, m1, m2))
l01 <- as.numeric(2*(logLik(m1)-logLik(m0))); l02 <- as.numeric(2*(logLik(m2)-logLik(m0)))
cat(sprintf("LR_M1vsM0 chi2(1)=%.2f dAIC=%.2f\n", l01, AIC(m0)-AIC(m1)))
cat(sprintf("LR_M2vsM0 chi2(3)=%.2f dAIC=%.2f\n", l02, AIC(m0)-AIC(m2)))
cat(sprintf("M1 zd=%.4f SE=%.4f\n", fixef(m1)["zd"], sqrt(vcov(m1)["zd","zd"])))
# Marginal R2 (Nakagawa, latente logit: var residual pi^2/3) para efectos fijos
r2m <- function(m){ vf <- var(as.vector(model.matrix(m) %*% fixef(m))); vr <- sum(sapply(VarCorr(m), function(v) v[1,1])); vf/(vf+vr+pi^2/3) }
cat(sprintf("R2m M0=%.4f M1=%.4f M2=%.4f\n", r2m(m0), r2m(m1), r2m(m2)))
# Pendiente de z(d) dentro de cada nivel (misma z del modelo objetivo)
m3 <- glmer(selected ~ 0 + level + level:zd + condition + (1 + zd | person) + (1 | image), p, binomial, control=ctl)
cat("warn m3:", w(m3), "singular:", isSingular(m3), "\n")
co <- summary(m3)$coefficients; print(round(co, 4))
for (lv in c("bas","sub","sup")) { k <- paste0("level", lv, ":zd"); z <- co[k,]; ci <- z[1]+c(-1,1)*qnorm(.975)*z[2]
  cat(sprintf("SLOPE level=%s zd=%.4f CI[%.4f, %.4f] p=%.3g\n", lv, z[1], ci[1], ci[2], z[4])) }
# SD de la distancia dentro de cada nivel (en unidades crudas)
print(tapply(p$dist, p$level, sd))
