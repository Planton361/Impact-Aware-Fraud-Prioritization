# Impact-Aware Fraud Prioritization

This repository provides a reproducible Python workflow for evaluating
budget-conditioned fraud-ranking strategies and generating the presentation
data, figures, and tables used in the accompanying master's thesis.

[![CI](https://github.com/Planton361/Impact-Aware-Fraud-Prioritization/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Planton361/Impact-Aware-Fraud-Prioritization/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-150-2ea44f)](#documentation-tests-and-ci)
[![Thesis evidence](https://img.shields.io/badge/thesis%20evidence-canonical-6f42c1)](#experiment-profiles)
[![Wiki](https://img.shields.io/badge/Wiki-Guided%20Documentation-2f81f7?logo=github&logoColor=white)](https://github.com/Planton361/Impact-Aware-Fraud-Prioritization/wiki)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[Wiki](https://github.com/Planton361/Impact-Aware-Fraud-Prioritization/wiki) ·
[Quick Start](#reproduce-the-canonical-experiment) ·
[Method](docs/METHOD.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Results and artifacts](docs/RESULTS_AND_ARTIFACTS.md) ·
[Troubleshooting](#troubleshooting)

## Submission identity

This repository is the examiner-facing, history-free submission snapshot accompanying the master's thesis.

- Repository: `Planton361/Impact-Aware-Fraud-Prioritization`
- Submission branch: `main`
- Submission tag: `thesis-submission-2026`
- Wiki: [Guided documentation](https://github.com/Planton361/Impact-Aware-Fraud-Prioritization/wiki)
- Wiki tag: `thesis-submission-2026-wiki`
- Source freeze: `b0eeaef976e20e3ce8370c2e76960b11622ab838`
- The empirical method and reported results are frozen.
- The versioned files under `reference_results/` are compact plausibility references.
- The complete historical canonical run root is not included in this submission repository.
- Engineering profiles and CI outputs are not thesis evidence.

> [!IMPORTANT]
> - `Amount` is only a proxy for potential financial impact, not observed loss.
> - PLR and technical `prevented_loss_*` fields are proxy quantities, not
>   realized prevented losses.
> - Only BCE scores have fraud-probability semantics.
> - Ranker and transformation scores are ordinal, not probabilities.
> - p-only is not Amount-free because its supervision uses Amount-conditioned
>   relevance and gain selection.
> - The fixed reference is untrained and is neither a fitted model nor a loss
>   function.
> - Results are descriptive for the frozen data, splits, candidate pool, and
>   budgets.
> - The repository makes no claim of statistical significance, causality,
>   realized loss prevention, fairness, production readiness, scalability,
>   external validity, or general model superiority.

## Workflow overview

```text
Prepare once
→ validate the intended profile
→ execute one experiment
→ inspect completion
→ build fit-free presentation artifacts
→ inspect the presentation
```

| Path              | Purpose                                            | Evidence role                                                 |
| ----------------- | -------------------------------------------------- | ------------------------------------------------------------- |
| `canonical`       | Complete experiment on the registered real dataset | The only thesis-evidentiary profile                           |
| `smoke-synthetic` | Deterministic bounded engineering check            | Not thesis evidence and not comparable with canonical results |

`mini-real` is an additional real-data engineering profile. It is also not
thesis evidence and is not comparable with canonical empirical results.

---

## Reproduce the canonical experiment

**Primary reproduction path**

The `canonical` profile uses the registered real dataset and is the only
profile whose outputs may serve as thesis evidence.

### Canonical workflow in six steps

| Step | Action                                                                  | Operational role                       |
| ---: | ----------------------------------------------------------------------- | -------------------------------------- |
|    1 | [Prepare the environment](#1-prepare-the-environment)                   | Mutating setup; no experiment          |
|    2 | [Validate the canonical plan](#2-validate-the-canonical-plan)           | Safe dry run                           |
|    3 | [Execute the canonical experiment](#3-execute-the-canonical-experiment) | Complete computationally intensive run |
|    4 | [Inspect the completed run](#4-inspect-the-completed-run)               | Read-only completion validation        |
|    5 | [Build presentation artifacts](#5-build-presentation-artifacts)         | Fit-free artifact generation           |
|    6 | [Inspect the presentation](#6-inspect-the-presentation)                 | Read-only presentation validation      |

### 1. Prepare the environment

**Setup · modifies the local environment; does not run an experiment**

Prerequisites:

* Git and a local repository checkout
* Internet access for package installation
* Kaggle authentication when `data/creditcard.csv` is absent
* PowerShell, Bash, or the integrated PyCharm terminal
* A LaTeX engine only for a compiled table preview

Keep Kaggle credentials outside the repository.

For a fresh checkout, run the platform bootstrap. It:

1. obtains a SHA-256-verified workspace-local `uv` 0.12.1 binary;
2. installs managed CPython 3.12.13 below `.tools/python/`;
3. creates or reuses `.venv` from that managed interpreter;
4. installs the frozen dependency contract and CLI;
5. acquires, reuses, or validates the canonical dataset;
6. performs the installed bounded diagnostics.

`.tools/` holds only the local uv archive/binary, managed Python, and uv cache;
`.venv/` is the project environment. Nothing is installed system-wide, and
bootstrap never starts an experiment.

**Windows PowerShell**

```powershell
.\scripts\bootstrap.ps1
```

**macOS and Linux**

```bash
./scripts/bootstrap.sh
```

After bootstrap, use the executable inside `.venv`; activation is not
required. In PyCharm, optionally select `.venv` as the interpreter and prefer
the integrated terminal.

`fraud-detection setup` is the public setup command of an already installed
CLI. It is intended for later repair or repetition and is not a second
mandatory command immediately after bootstrap.

Continue only after bootstrap finishes successfully.

### 2. Validate the canonical plan

**Safe dry run · no data access, fit, scoring, or output**

Prerequisite: bootstrap completed successfully.

**Windows PowerShell**

```powershell
.\.venv\Scripts\fraud-detection.exe run --profile canonical --dry-run
```

**macOS and Linux**

```bash
.venv/bin/fraud-detection run --profile canonical --dry-run
```

This resolves the profile and pure execution plan without accessing the
dataset, creating an output directory, fitting models, or scoring data.

The dry run replaces a separate planning command. Review the profile, evidence
role, data path, and output path before proceeding.

### 3. Execute the canonical experiment

**Full experiment · fits models and writes a new run**

> [!WARNING]
> This starts the complete computationally intensive experiment. Run it only
> deliberately. The output root must be new, and no general runtime guarantee
> is provided.

**Windows PowerShell**

```powershell
.\.venv\Scripts\fraud-detection.exe run --profile canonical --yes
```

**macOS and Linux**

```bash
.venv/bin/fraud-detection run --profile canonical --yes
```

The default experiment root is:

```text
outputs/canonical-final
```

A different new root can be selected with `--output PATH`. Existing experiment
roots are never silently reused or overwritten.

The interactive display shows elapsed time and homogeneous progress counters.
It does not report a linear percentage or ETA across heterogeneous operations.

Continue to inspection only after the experiment command finishes.

### 4. Inspect the completed run

**Read-only inspection**

Prerequisite: the experiment command finished and the run root exists.

**Windows PowerShell**

```powershell
.\.venv\Scripts\fraud-detection.exe inspect outputs/canonical-final
```

**macOS and Linux**

```bash
.venv/bin/fraud-detection inspect outputs/canonical-final
```

Directory existence and process exit code alone do not establish completion.
Only semantic status `COMPLETE`, successful QA, all required phases, registered
artifacts, and verified checksums establish a completed run.

Build presentation artifacts only after inspection validates the experiment.

### 5. Build presentation artifacts

**Fit-free presentation build**

Prerequisite: `outputs/canonical-final` was inspected as complete.

**Windows PowerShell**

```powershell
.\.venv\Scripts\fraud-detection.exe build outputs/canonical-final
```

**macOS and Linux**

```bash
.venv/bin/fraud-detection build outputs/canonical-final
```

`build` is the public fit-free presentation command. Its default output is:

```text
generated/presentations/canonical-final
```

It creates:

* presentation data;
* figures;
* tables;
* an optional LaTeX preview;
* presentation completion manifests.

It does not access the raw dataset, fit models, create scores, select
parameters, modify rankings, or change the source experiment root.

Continue to presentation inspection only after the build finishes.

### 6. Inspect the presentation

**Read-only presentation inspection**

Prerequisite: the presentation build completed.

**Windows PowerShell**

```powershell
.\.venv\Scripts\fraud-detection.exe inspect generated/presentations/canonical-final
```

**macOS and Linux**

```bash
.venv/bin/fraud-detection inspect generated/presentations/canonical-final
```

Inspection validates the presentation manifests, registered outputs, and
checksums without creating, modifying, repairing, or deleting presentation
artifacts.

## Run the bounded engineering path

**Engineering-only path**

This deterministic synthetic path checks the shared pipeline. It is not thesis
evidence and is not comparable with canonical empirical results.

Run the commands sequentially. Let each command finish before starting the
next.

**Windows PowerShell**

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\fraud-detection.exe run --profile smoke-synthetic
.\.venv\Scripts\fraud-detection.exe build generated/runs/smoke-synthetic
```

**macOS and Linux**

```bash
./scripts/bootstrap.sh
.venv/bin/fraud-detection run --profile smoke-synthetic
.venv/bin/fraud-detection build generated/runs/smoke-synthetic
```

The engineering experiment is written to:

```text
generated/runs/smoke-synthetic
```

Its presentation is written to:

```text
generated/presentations/smoke-synthetic
```

---

## Validate an installed checkout

`fraud-detection check` is read-only. It does not install, repair, download,
fit, render, or modify the dataset.

```text
fraud-detection check
fraud-detection check --full
fraud-detection check --require-data
```

Use the executable inside `.venv` as shown in the workflow above.

* default mode performs quick environment and repository diagnostics;
* `--full` adds bounded syntax, dependency-integrity, import, and public-help
  checks;
* `--require-data` requires and validates the canonical local dataset;
* both options can be combined.

## Public commands

| Command   | Responsibility                                                      |
| --------- | ------------------------------------------------------------------- |
| `setup`   | Prepare or repair an installed environment; never run an experiment |
| `check`   | Perform read-only environment, repository, and data diagnostics     |
| `run`     | Validate or execute one complete experiment profile                 |
| `build`   | Generate fit-free presentation artifacts from a completed run       |
| `inspect` | Perform read-only semantic inspection of an exact path              |

For a fresh checkout, use the platform bootstrap. Bootstrap and the installed
`setup` command are not two normally consecutive workflow steps.

<details>
<summary><strong>Show command syntax and output modes</strong></summary>

### Command syntax

```text
fraud-detection setup
fraud-detection check
fraud-detection check --full
fraud-detection check --require-data
fraud-detection run --profile PROFILE --dry-run
fraud-detection run --profile PROFILE
fraud-detection run --profile PROFILE --yes
fraud-detection run --profile PROFILE --output PATH --yes
fraud-detection inspect PATH
fraud-detection build EXPERIMENT_PATH
fraud-detection build EXPERIMENT_PATH --output PATH
```

### Global output options

Global options precede the command:

```text
fraud-detection [--plain | --json] [--verbose] [--debug] [--no-color] COMMAND
```

* interactive TTY: restrained Rich progress for long-running commands;
* `--plain`: stable ANSI-free transition lines without heartbeat noise;
* `--json`: exactly one final ANSI-free semantic document;
* `--verbose`: include commands and additional diagnostic details;
* `--debug`: add Python tracebacks to failure output;
* `--no-color`: preserve interactive structure without ANSI color.

`--plain` and `--json` are mutually exclusive.

Run and build show elapsed time and homogeneous counters. They do not report a
linear percentage or ETA across heterogeneous operations.

</details>

## Experiment profiles

| Profile           | Data and scope                                                                      | Evidence role                       |
| ----------------- | ----------------------------------------------------------------------------------- | ----------------------------------- |
| `canonical`       | Real canonical dataset; five seeds; seven budgets; candidate pool 1,000             | The only thesis-evidentiary profile |
| `smoke-synthetic` | 5,000 deterministic synthetic rows; one seed; budgets 20/50/100; candidate pool 200 | Engineering only                    |
| `mini-real`       | Real canonical dataset; seeds 42/7/13; budgets 20/50/100; linear gain               | Engineering only                    |

All profiles use the shared experiment pipeline.

`smoke-synthetic` and `mini-real` verify engineering behavior. They do not
estimate canonical performance, are not thesis evidence, and must not be
compared with canonical empirical results.

## Reproducibility and completion

Directory existence and process exit code alone do not prove completion.

The root `RUN_MANIFEST.json` is the semantic authority for an experiment run.
A completed run requires:

* schema-supported semantic status `COMPLETE`;
* all required semantic phases;
* one immutable effective profile configuration;
* successful QA;
* a safe registered-artifact inventory;
* concrete artifact checksums.

`inspect` is read-only. It does not create, modify, repair, delete, fit,
render, or search parent directories.

Existing experiment roots are never silently reused or overwritten. Fit-level
resume is unsupported. An interrupted, partial, or failed attempt requires a
new output root.

Do not repair a partial run by deleting or replacing individual files.

A presentation build validates its completed experiment input and writes to a
separate presentation root. It does not modify the experiment root.

## Output locations

| Path                       | Role                                                        | Versioned |
| -------------------------- | ----------------------------------------------------------- | --------: |
| `data/creditcard.csv`      | Local canonical dataset                                     |        No |
| `outputs/`                 | Canonical experiment runs and registered artifacts          |        No |
| `generated/runs/`          | Non-evidentiary engineering runs                            |        No |
| `generated/presentations/` | Presentation data, figures, tables, previews, and manifests |        No |
| `reference_results/`       | Three compact plausibility references                       |       Yes |
| `docs/`                    | Method, reproducibility, and artifact documentation         |       Yes |

Generated roots are local and ignored by Git.

`reference_results/` is not a runtime input, is not regenerated by ordinary
CI, and does not replace a complete validated experiment run.

## Troubleshooting

<details>
<summary><strong>Managed Python bootstrap failed</strong></summary>

The bootstrap does not require a preinstalled Python interpreter. Check network,
TLS/proxy settings, available disk space, and write access to the checkout, then
rerun the appropriate bootstrap script.

</details>

<details>
<summary><strong>Kaggle authentication failed</strong></summary>

Configure a supported local Kaggle authentication method outside the
repository, then rerun bootstrap.

Do not commit or log Kaggle credentials.

</details>

<details>
<summary><strong>Dataset identity mismatch</strong></summary>

Do not use or overwrite the mismatching CSV.

Review the [local data contract](data/README.md), obtain the official dataset,
place it at `data/creditcard.csv`, and validate its registered identity before
starting a real-data profile.

</details>

<details>
<summary><strong>Output path already exists</strong></summary>

Experiment output roots must be new.

Inspect the existing path when its state is unknown, or choose a different
output path for a new attempt. Existing roots are not silently reused or
overwritten.

</details>

<details>
<summary><strong>Interrupted or incomplete run</strong></summary>

Inspect the partial root to identify evidenced and missing phases.

Do not repair the run by replacing individual files. Fit-level resume is
unsupported; start a new attempt with a new output root.

</details>

<details>
<summary><strong>No LaTeX engine</strong></summary>

The presentation build may use a supported skipped-preview state.

Presentation data, figures, tables, manifests, and integrity validation remain
available without a local LaTeX compiler.

</details>

## Scientific interpretation

The experiment evaluates four comparison paths:

| Path            | Interpretation                                                                                                |
| --------------- | ------------------------------------------------------------------------------------------------------------- |
| BCE             | Trained probabilistic fraud baseline over the complete test set                                               |
| p-only          | Trained budget-conditioned ordinal candidate reranker using `p_fraud`                                         |
| Amount-Gain     | Trained budget-conditioned ordinal candidate reranker using `p_fraud`, `log1p(Amount)`, and their interaction |
| fixed reference | Deterministic untrained candidate ordering using `p_fraud * log1p(Amount)`                                    |

p-only is not Amount-free because its supervision remains connected to
Amount-conditioned relevance and gain selection.

A separate model is fitted for every seed and target budget. Budget points are
not prefixes of one global reranker.

See the [final method](docs/METHOD.md) for the complete scientific contract.

## Documentation, tests, and CI

### Documentation

* [Final method](docs/METHOD.md)
* [Reproducibility and public commands](docs/REPRODUCIBILITY.md)
* [Results and artifact roles](docs/RESULTS_AND_ARTIFACTS.md)
* [Local data contract](data/README.md)

### Tests and CI

The fast test contract contains exactly:

* 150 tests;
* 90 unit tests;
* 60 contract tests.

Normal CI exposes four checks:

* `fast-checks`;
* `distributions`;
* `platform-smoke (ubuntu-24.04)`;
* `platform-smoke (windows-2025)`.

Normal CI does not access the canonical dataset, fit an experiment, or render
empirical presentation artifacts.

<details>
<summary><strong>Show local fast-suite and CI scope details</strong></summary>

The fast local suite is:

```bash
python -m pytest tests/unit tests/contract --durations=20 -q
```

The normal CI scope includes:

* dependency-integrity checks;
* static quality checks;
* compile and import checks;
* the exact unit and contract test counts;
* public command help;
* source-distribution and wheel validation;
* isolated package installation;
* bounded platform checks;
* dry-run purity.

The normal development and CI path does not perform the canonical experiment.

</details>

---

## Citation and license

Software citation metadata is provided in [CITATION.cff](CITATION.cff).

The repository software and documentation are licensed under the
[MIT License](LICENSE).

The external dataset has separate licensing and attribution requirements. See
the [local data contract](data/README.md#dataset-license-and-attribution).
