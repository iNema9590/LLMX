# Expanded evaluation with confidence intervals

## What was done

Reviewers noted that 100 queries provide limited statistical evidence. We re-ran the
evaluation on **500 questions** and now report 95% bootstrap confidence intervals for every
result. All comparisons against ContextCite are **paired per question**: because both
methods see the same question, per-question difficulty is shared, and differencing removes
it. Each figure has two panels — absolute values with CI bands on top, and the paired
difference FACILE − ContextCite below. A star on the lower panel marks a Holm-adjusted
Wilcoxon p < 0.05.

The expanded evaluation **confirms the same overall trends** observed in the original experiments, providing stronger statistical support for our conclusions.

---

## Marginal attribution: claim holds

![nDCG@5 to Exact-Shapley](figures_ci/ndcg_at_5.png)

FACILE is above ContextCite at **every** budget.

![PR-AUC](figures_ci/pr_auc.png)

Same direction, significant at all budgets, but the advantage **decays with budget**
(≈ +0.035 at M = 32 down to ≈ +0.009 at M = 728).

---

## Surrogate fidelity: claim does not hold at low budget

![R²_util](figures_ci/r2_util.png)

![LDS](figures_ci/lds.png)

The two methods align
around M ≈ 80. Above it FACILE pulls clearly ahead and ContextCite flattens.

![R²_Δ](figures_ci/r2_delta.png)

Here FACILE wins from the start.

---

## Gold-document retrieval: no measurable difference

![Recall@k](figures_ci/recall_by_k_budget264.png)

![Top-k drop](figures_ci/topk_drop_by_k_budget264.png)
