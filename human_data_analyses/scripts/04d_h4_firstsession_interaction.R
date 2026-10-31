# H4 (DESV §29-30): (a) C3/C7 con la primera sesión real por persona (AcceptTime de all_raw_A.csv, XTMEM 220ed17;
# SubmitTime da la misma elección; difiere del criterio "exp de menor número" solo en subid 216: exp 7 vs 5).
# (b) EXPLORATORIO: interacción zd × condition con RE por persona; Wald + LR (arregla el fallo "object 'dd' not found" de 04b).
suppressMessages(library(lme4)); set.seed(42)
d <- read.csv("data/h4_long.csv"); d$person <- factor(d$subid)
d$condition <- factor(c(one="one",three_subordinate="sub","3sub"="sub",three_basic="bas","3bas"="bas",three_superordinate="sup","3sup"="sup")[d$condition], levels=c("one","sub","bas","sup"))
fsx <- read.csv("data/h4_first_session.csv"); first <- setNames(fsx$first_exp, fsx$subid)
ctl <- glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
prep <- function(dd) { dd$zd <- as.numeric(scale(dd$dist)); dd$level <- factor(dd$level); dd$image <- factor(dd$image); dd }
fs <- selected ~ zd + level + condition + (1 + zd | person) + (1 | image)
fn <- selected ~ zd + level + condition + (1 | person) + (1 | image)
fitm <- function(dd, f1, f0) { m <- tryCatch(glmer(f1, dd, binomial, control=ctl), error=function(e) NULL)
  ok <- !is.null(m) && is.null(m@optinfo$conv$lme4$messages) && !isSingular(m)
  if (!ok) m <- glmer(f0, dd, binomial, control=ctl); list(m=m, RE=if (ok) "slope" else "noslope") }
rep1 <- function(tag, dd) { r <- fitm(dd, fs, fn); z <- summary(r$m)$coefficients["zd",]; ci <- z[1] + c(-1,1)*qnorm(.975)*z[2]
  cat(sprintf("== %s | n=%d personas=%d | RE=%s | warn=%s\nzd logOR=%.4f CI[%.4f, %.4f] p=%.3g\n", tag, nrow(dd), length(unique(dd$person)), r$RE,
    paste(r$m@optinfo$conv$lme4$messages, collapse=";"), z[1], ci[1], ci[2], z[4])) }
d1 <- prep(subset(d, exp == first[as.character(subid)])); rep1("C3t primera sesion real (AcceptTime), 24 items", d1)
d7 <- prep(subset(d, in_target == 1 & exp == first[as.character(subid)])); rep1("C7t primera sesion real (AcceptTime), 8 items objetivo", d7)
# (b) interacción, todos los ítems
D <- prep(d)
fsi <- selected ~ zd * condition + level + (1 + zd | person) + (1 | image); fni <- selected ~ zd * condition + level + (1 | person) + (1 | image)
ri <- fitm(D, fsi, fni)
f0 <- if (ri$RE == "slope") fs else fn
m0 <- glmer(f0, D, binomial, control=ctl)
cat(sprintf("\n== EXPLORATORIO zd x condition | RE=%s | warn=%s | warn0=%s\n", ri$RE, paste(ri$m@optinfo$conv$lme4$messages, collapse=";"), paste(m0@optinfo$conv$lme4$messages, collapse=";")))
co <- summary(ri$m)$coefficients; print(round(co, 4))
for (k in grep("zd", rownames(co), value=TRUE)) { ci <- co[k,1] + c(-1,1)*qnorm(.975)*co[k,2]; cat(sprintf("%s: %.4f CI[%.4f, %.4f] z=%.2f p=%.3g\n", k, co[k,1], ci[1], ci[2], co[k,3], co[k,4])) }
# LR a mano: anova.merMod exige el mismo objeto de datos por nombre; ambos modelos se ajustaron sobre D (mismas filas)
stopifnot(nobs(m0) == nobs(ri$m)); l0 <- logLik(m0); l1 <- logLik(ri$m); df <- attr(l1, "df") - attr(l0, "df"); LR <- 2 * (as.numeric(l1) - as.numeric(l0))
cat(sprintf("LR (zd:condition, %d gl) = %.2f, p = %.3g | logLik0=%.2f logLik1=%.2f | AIC0=%.1f AIC1=%.1f\n", df, LR, pchisq(LR, df, lower.tail=FALSE), as.numeric(l0), as.numeric(l1), AIC(m0), AIC(ri$m)))
