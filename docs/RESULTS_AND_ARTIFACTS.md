# Results and Artifacts

[README](../README.md) | [Method](METHOD.md) |
[Reproducibility](REPRODUCIBILITY.md) |
[Results and artifacts](RESULTS_AND_ARTIFACTS.md)

This document defines the roles of experiment outputs, presentation outputs,
and the three compact versioned reference files.

## Active-tree and generated-output policy

Generated experiment and presentation products are not versioned in the
supervisor-facing tree.

| Root | Role | Versioned? |
|---|---|---|
| `outputs/` | Canonical experiment products and registered intermediate artifacts | No |
| `generated/runs/` | Engineering experiment products | No |
| `generated/presentations/` | Presentation data, figures, tables, previews, and manifests | No |
| `thesis_build/` | Local thesis and LaTeX integration products | No |
| `reference_results/` | Three compact plausibility references | Yes |

Generated roots are ignored. Selective deletion inside a registered completed
run or presentation invalidates its integrity chain and is unsupported.

## Experiment completion authority

Directory existence and a process exit code do not establish a complete run.
The root `RUN_MANIFEST.json` is the semantic run authority.

A valid completed run establishes:

- `status=COMPLETE`;
- profile and evidence classification;
- all required semantic phases;
- one immutable effective configuration;
- data-source summary;
- QA pass;
- a safe relative registered-artifact inventory;
- concrete artifact checksums.

The experiment root retains selection evidence, final rankings, aggregate
tables, diagnostics, and QA artifacts registered by that manifest and its
checksum chain. Their internal directories are implementation details; the
manifest and semantic inspection are the public boundary.

A partial or failed root may contain individually valid files, but it is
`INCOMPLETE`, is not presentation compatible, and does not support fit-level
resume. Start a new experiment with a new output root. Do not repair a partial
run by deleting or replacing individual files.

## Ranking artifacts

Each complete ranking is identified by:

```text
(seed, target_budget, comparison_path)
```

The canonical profile therefore contains 5 x 7 x 4 = 140 complete ranking
groups. The mini-real profile contains 3 x 3 x 4 = 36. The smoke-synthetic
profile contains 1 x 3 x 4 = 12.

Every ranking is a complete, gap-free permutation of the corresponding test
set. For candidate-local raw scores:

- values are finite inside the candidate pool;
- values are structurally missing outside the pool;
- the BCE raw score remains finite over the complete ranking.

Those structural missing values are not failed scores and must not be imputed
or interpreted as numerical results.

## Presentation lifecycle

`fraud-detection build EXPERIMENT_PATH` validates the complete run and its
registered inputs before writing a presentation root.

The public presentation stages are:

1. presentation data;
2. figures;
3. tables and optional LaTeX preview.

Completion manifests are:

- `data/PRESENTATION_DATA_MANIFEST.json`;
- `figures/FIGURE_RENDER_MANIFEST.json`;
- `tables/TABLE_RENDER_MANIFEST.json`.

Build is fit-free. It does not access the raw dataset, fit models, create
scores, select parameters, change rankings, or modify the source experiment.

### Canonical presentation role

A canonical run selects the thesis-oriented catalog:

- 29 registered presentation-data files;
- 9 logical figures rendered as 27 PNG/PDF/SVG files;
- 9 logical tables represented by 18 CSV/LaTeX files.

### Engineering presentation role

`smoke-synthetic` and `mini-real` select an explicitly non-evidentiary
engineering catalog:

- 2 registered presentation-data files;
- 1 logical engineering figure;
- 1 logical engineering table.

Engineering artifacts must retain their profile and non-comparability
warnings. They are not thesis evidence.

LaTeX preview status is represented separately. Absence of a local LaTeX
engine may produce a supported skipped-preview state; it does not convert
unvalidated sources into complete output.

## Semantic inspection

`fraud-detection inspect <PATH>` is read-only.

For completed experiment and presentation roots it validates manifests,
registered files, and checksums. For a genuine partial experiment it reports
`INCOMPLETE`, identifies evidenced and missing phases, and states that
fit-level resume is unsupported.

Inspection never creates, modifies, repairs, deletes, fits, renders, or searches
parent directories.

## Reference results

`reference_results/` contains exactly three compact files:

- `central_topk_results.csv`: central five-seed summaries for the four
  comparison paths at k=20, 50, and 100;
- `selected_configuration_summary.csv`: the 35 canonical seed-budget
  selection records;
- `data_identity.json`: canonical data, seed, budget, and candidate-pool
  identities.

These files support plausibility review and compact comparison. They are:

- not experiment or presentation runtime inputs;
- not regenerated by ordinary CI;
- not substitutes for a complete validated run;
- protected from incidental overwrite;
- numerical sources only within their documented compact scope.

Exact numerical authority for a controlled final result remains the validated
experiment root and its registered presentation-data products. Rendered PDF,
SVG, PNG, or LaTeX output is never the primary numerical source.

## Evidence and claim boundaries

- `Amount`, PLR, and technical `prevented_loss_*` fields are proxies.
- Ranker and fixed-reference scores are ordinal, not Fraud probabilities.
- Engineering-profile outputs are not thesis evidence and are not comparable
  with canonical empirical results.
- Pool ceilings are availability limits, not expected performance.
- Tie bounds are exact technical permutation bounds, not confidence intervals.
- No artifact supports a claim of significance, causality, real prevented
  loss, fairness, production readiness, scalability, or general superiority.

## Controlled canonical handover

The canonical profile requires the validated local dataset and the frozen
method. A final accepted run must pass its manifest, QA, ranking, artifact, and
checksum contracts. Presentation outputs must be built from that same accepted
root.

Generated canonical outputs are not generally committed. Their acceptance,
external retention, thesis regeneration, and any refresh of the three compact
reference files require a separate explicit authorization. Ordinary
documentation or CI work must not trigger such a run.
