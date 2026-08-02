# Final Method

[README](../README.md) | [Method](METHOD.md) |
[Reproducibility](REPRODUCIBILITY.md) |
[Results and artifacts](RESULTS_AND_ARTIFACTS.md)

This document is the public repository summary of the frozen experiment
method. The implementation executes preflight, inner validation, selection
freeze, final outer evaluation, aggregation, and QA as one deterministic
serial pipeline.

## Research design

The experiment evaluates whether a budget-conditioned, impact-aware
LambdaRank reranker changes Fraud-Amount-proxy coverage and classical Fraud
capture relative to a probability-based BCE baseline under limited Top-k
investigation budgets.

The ranker does not replace Fraud detection:

```text
converged BCE Fraud baseline
-> BCE Top-1000 candidate pool
-> budget-conditioned candidate reranking
-> unchanged BCE order for the remaining cases
-> complete-order Top-k evaluation
```

## Data and outer evaluation

The raw dataset contains 284,807 transactions, including 492 Fraud cases.
Stable full-row deduplication with `drop_duplicates(keep="first")` occurs
before any split or fit. It removes 1,081 rows and leaves 283,726 transactions:
283,253 legitimate and 473 Fraud.

Canonical outer seeds are `42, 7, 13, 123, 202`. Each seed defines a stratified
random 80:20 holdout with 226,980 training rows, including 378 Fraud, and
56,746 test rows, including 95 Fraud.

The evaluation is descriptive. Five splits do not establish statistical
significance or general robustness.

## BCE Fraud baseline

`CustomLogisticRegression` uses V1 through V28. `Time`, `Amount`, and `Class`
are not BCE features.

```text
loss                    binary cross-entropy
learning rate           0.1
tolerance               1e-6
iteration safety limit  10,000
initialization          zero
L2                      0
OOF folds               5
```

Each fold uses its own training-fitted `StandardScaler`. Five OOF folds produce
training-side ranker inputs. A separate full-outer-training BCE fit produces
test probabilities. Persisted final BCE fits must satisfy
`converged_by_tolerance=true`.

Only the BCE score has probabilistic Fraud semantics. The Brier score is
therefore reported only for BCE.

## Candidate pool and ranking group

The training candidate pool is the Top-1000 by `p_oof_train`. Validation and
test pools are the Top-1000 by the corresponding full-training BCE model.

Pool membership uses only the fit-separated BCE score and stable row position.
Labels, `Amount`, `Time`, relevance, ranker scores, and test metrics do not
influence membership. The pool is exactly one ranking group of 1,000
candidates; `Time` has no grouping role.

The pool is a method scope boundary: no reranker can promote a case that the
BCE baseline placed outside the Top-1000 candidates.

## Relevance, gain, and inner selection

Legitimate cases receive relevance 0. Fraud cases receive relevance 1 through
4 from q25/q50/q75 thresholds estimated only from the relevant
training-partition Fraud Amounts.

Candidate gain mappings are:

```text
linear       [0, 1, 2, 3, 4]
exponential  [0, 1, 3, 7, 15]
```

For every outer seed and target budget, three stratified inner folds reconstruct
fit-separated BCE scores and training-side Amount quantiles.

A gain is PLR-eligible when its mean inner PLR delta is positive and at least
two of three fold deltas are positive. Eligible gains are selected
lexicographically by:

1. mean Fraud retention;
2. minimum Fraud retention;
3. mean PLR delta;
4. mean Amount-nDCG;
5. mean cutoff-tie size, with smaller preferred;
6. deterministic gain identifier.

When no gain has an inner-validated positive PLR lift,
`NO_INNER_VALIDATED_POSITIVE_PLR_LIFT` remains visible. The 35 canonical
seed-budget configurations are frozen before outer-test labels or metrics are
used.

The final tree count is the half-up-rounded median of the three inner
`best_iteration` values, clamped to 1 through 500.

## Budget-conditioned LambdaRank

A separate model is fitted for every seed and budget.

- all budgets: 5, 10, 20, 50, 100, 200, 500;
- primary budgets: 20, 50, 100;
- objective: `lambdarank`;
- learning rate: 0.05;
- `num_leaves=7`;
- `min_child_samples=20`;
- `min_child_weight=0.001`;
- `reg_lambda=0`;
- `n_jobs=1`;
- maximum trees: 500;
- early stopping: 50 rounds;
- `truncation=k+3`;
- `eval_at=k`.

Truncation is derived deterministically from the target budget. It is not an
additional searched hyperparameter.

## Four comparison paths and complete order

1. **BCE** sorts the complete test set by `p_fraud`.
2. **p-only** learns an ordinal candidate order using `p_fraud`.
3. **Amount-Gain** learns an ordinal candidate order using `p_fraud`,
   `log1p(Amount)`, and their interaction.
4. **fixed reference** orders candidates by
   `p_fraud * log1p(Amount)` without training.

p-only uses the same selected configuration as Amount-Gain. It is not
Amount-free because its supervision remains connected to Amount-conditioned
relevance and gain selection.

For every reranked path, all candidates precede all non-candidates. Stable
BCE-based fallback resolves exact candidate-score ties. Non-candidates retain
their BCE-relative order. Ranker and fixed-reference scores are ordinal, not
Fraud probabilities.

## Evaluation

Matched-budget outputs include:

- PLR@k;
- Fraud@k, Precision@k, and Recall@k;
- Fraud Amount sum at k;
- Amount-nDCG@k;
- q90-Fraud coverage and BCE-miss recovery;
- high-Amount legitimate guardrails;
- replacement and boundary diagnostics;
- exact cutoff-tie bounds;
- candidate-pool coverage and ceiling utilization;
- complete-order ROC-AUC and Average Precision;
- Brier score for BCE only.

Aggregation uses the arithmetic mean and sample standard deviation across five
seeds (`ddof=1`).

Pool ceilings are post-hoc availability limits, not expected model
performance. Exact tie bounds are technical permutation bounds, not confidence
intervals.

## Presentation boundary

The presentation builder accepts only a completed, checksum-registered run. It
derives presentation data, figures, and tables without fitting models,
creating scores, selecting parameters, accessing raw data, or changing
rankings.

## Scientific boundaries

- `Amount` is only a proxy for potential financial impact.
- PLR and technical `prevented_loss_*` fields are proxy quantities, not real
  prevented losses.
- Only BCE scores have probabilistic Fraud semantics.
- The fixed reference is neither a trained model nor a loss function.
- The analysis does not claim statistical significance, causality, realized
  loss prevention, fairness, production readiness, scalability, external
  validity, or general model superiority.
- Budget points are outputs of separate budget-conditioned models, not prefixes
  of one global reranker.
