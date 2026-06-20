# ERRATA — exp_008 Phase 4 Regression Table

## VIF Correction

**Affected file:** `regression_table.csv` (this directory)
**Run timestamp:** 20260608_173255
**Date of correction:** 2026-06-20
**Corrected by:** Patricio Vera

### What was wrong

The original `regression_table.csv` reports `PS_vif_perc = PS_vif_sem = 9.6930` for all learners.

This value is incorrect. It arose from a statsmodels artifact: the auxiliary regression used internally to compute the Variance Inflation Factor was fit **without an intercept** (`fit_intercept=False`). Omitting the intercept in the auxiliary regression inflates R² artificially, which inflates the VIF formula `VIF = 1 / (1 - R²_auxiliary)` even when the predictors are not actually collinear.

### Correct value

When the auxiliary regression is fit with an intercept (the standard procedure), the corrected VIF is:

```
PS_vif_perc = PS_vif_sem = 1.0105
```

This value is consistent with the Phase 1 correlation diagnostic:
- Pearson r(d_perc, d_sem) = 0.1555 (p = 3.71e-28)
- Spearman ρ(d_perc, d_sem) = 0.0886 (p = 4.21e-10)

Low correlation between predictors implies low multicollinearity, confirming VIF ≈ 1.01.

### What is NOT affected

The VIF correction changes **only** the reported VIF values. All of the following are unaffected:

- Regression coefficients (β_perc, β_sem) — unchanged
- Standard errors and confidence intervals — unchanged
- p-values — unchanged
- Partial R² values — unchanged
- R² of all three models (P, S, PS) — unchanged
- Likelihood ratio test statistics and p-values — unchanged
- Pre-registered decision Rule A/B/C outcome — Rule A holds cleanly

### Consequence for pre-registration

Under the original (buggy) VIF = 9.693, Rule B ("Ambiguous: high collinearity") would have appeared to trigger. Rule A criteria were all met independently of VIF, so the Rule B trigger was spurious. With the corrected VIF = 1.0105, Rule B is not triggered and Rule A holds without any ambiguity.

### Corrected artifacts

The corrected regression table is archived at:
```
results/exp_008_dissociation/exp_008_regression_table_corrected.csv
```

The corrected run report is at:
```
results/exp_008_dissociation/exp_008_report_corrected.md
```

This errata, the correction methodology, and its non-effect on the decision outcome are documented in Appendix I of the paper.
