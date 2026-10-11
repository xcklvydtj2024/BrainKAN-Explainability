# BrainKAN Estimands Definition

## Q3A: Operating-regime sensitivity
**Question:** Does the apparent curvature effect depend on the observed input regime?
**Estimand:** $\Delta C_e = C_e(X_{2BK}) - C_e(X_{0BK})$
where $C_e(X) = \frac{1}{16} \sum_o \mathbb{E}_{x \sim X} [|f''_{e,o}(x)|]$.

## Q3B: Common-reference structural geometry
**Question:** Is there detectable condition-related structural nonlinearity under a common reference domain?
**Estimand:** $G_e = \frac{1}{16} \sum_o (1 - R^2_{e,o,\text{ref}})$
evaluated exclusively on $X_{ref}^{(f)} = Q(X_{train}^{(f)})$, constructed securely per fold.

## Q3C: Effective gain
**Question:** Is there a detectable condition-related change in local sensitivity $|f'(x)|$?
**Estimand:** $G_{e,c} = \frac{1}{16} \sum_o \mathbb{E}_{x \sim X_{c}} [|f'_{e,o}(x)|]$.
Tested using the subject-level paired difference $dG_e = G_e^{2BK} - G_e^{0BK}$.
