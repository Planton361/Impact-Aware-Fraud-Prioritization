# Reproducibility

[README](../README.md) | [Method](METHOD.md) |
[Reproducibility](REPRODUCIBILITY.md) |
[Results and artifacts](RESULTS_AND_ARTIFACTS.md)

This document defines the environment, setup, data identity, public commands,
runtime lifecycle, and platform limits for reproducing the repository.

## Environment

The package Python contract is `>=3.12,<3.13`. Fresh-clone bootstrap pins uv
0.12.1 and managed CPython 3.12.13. uv supplies this CPython distribution using
its documented Astral python-build-standalone mechanism. The accepted reference
environment was:

| Package | Version |
|---|---:|
| Python | 3.12.10 |
| NumPy | 2.5.1 |
| pandas | 3.0.5 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| LightGBM | 4.7.0 |
| PyArrow | 25.0.0 |
| Matplotlib | 3.11.1 |
| Pillow | 12.3.0 |
| pytest | 9.1.1 |
| Typer | 0.27.0 |
| Rich | 15.0.0 |
| Ruff | 0.16.1 |

During A6 acceptance, Ubuntu CI resolved CPython 3.12.13 and Windows CI
resolved CPython 3.12.10. The supported Python contract remains
`>=3.12,<3.13`. This cross-platform evidence does not claim low-level
floating-point equality between platforms.

## Preferred bootstrap

Windows PowerShell:

```powershell
.\scripts\bootstrap.ps1
```

macOS/Linux:

```bash
./scripts/bootstrap.sh
```

The thin platform wrapper parses the tracked toolchain lock, downloads an
immutable uv archive, verifies its SHA-256 before extraction, and delegates to
the authoritative Python bootstrap with managed CPython 3.12.13. Bootstrap:

1. stores uv, managed Python, and its cache under `.tools/`;
2. creates or reuses `.venv` from the managed interpreter;
3. installs the pinned experiment and bootstrap requirements;
4. installs the repository package;
5. reuses or acquires the canonical dataset according to the current setup
   contract;
6. validates data identity when applicable;
7. executes the installed `fraud-detection check --full`.

Bootstrap may use the network for package installation and, when the canonical
CSV is absent, Kaggle dataset acquisition. Keep Kaggle credentials outside the
repository. Bootstrap never starts an experiment.

After bootstrap, execute the environment-specific console script directly; no
activation is required.

Supported bootstrap platforms are Windows x86_64 and ARM64, macOS x86_64 and
ARM64, and GNU Linux x86_64 and ARM64. Linux musl and translated Rosetta shells
are rejected rather than selecting a non-native archive. A rerun rehashes the
retained uv archive, reuses valid local tooling and a compatible `.venv`, and
replaces only an incompatible repository-root `.venv`. No PATH, shell profile,
registry, package-manager, system-Python, or user-level uv configuration is
modified.

## Environment repair

For a fresh clone, always use the platform bootstrap rather than manually
creating an environment. After a successful bootstrap, `fraud-detection setup`
remains available for installed-environment repair or repetition.

For manual data setup, obtain the official CSV described in the
[local data contract](../data/README.md), place it at
`data/creditcard.csv`, and verify its raw SHA-256.

## Public commands

### `setup`

`fraud-detection setup` is the mutating environment-preparation command. It may
create or reuse `.venv`, install packages, and acquire, reuse, or validate the
canonical dataset. It never starts an experiment.

### `check`

`fraud-detection check` is read-only.

- default: quick environment and repository diagnostics;
- `--full`: bounded syntax, dependency-integrity, import, and public-help
  diagnostics;
- `--require-data`: require and validate the canonical local dataset;
- `--full --require-data`: combine both scopes.

Check does not install, repair, fit, render, download, or modify the dataset.
Exit codes are 0 for required checks passing, 1 for diagnostic failure, 2 for
usage error, and 130 for interruption.

### `run`

`fraud-detection run` executes one complete profile through the shared serial
pipeline.

```text
fraud-detection run --profile smoke-synthetic --dry-run
fraud-detection run --profile smoke-synthetic
fraud-detection run --profile mini-real --yes
fraud-detection run --profile canonical --yes
```

`--dry-run` resolves the profile and pure execution plan without dataset
access, output creation, fitting, or scoring. Non-interactive real-data runs
require `--yes`.

Default outputs are:

- `generated/runs/smoke-synthetic`;
- `generated/runs/mini-real`;
- `outputs/canonical-final`.

Every output path must be new. Existing experiment roots are never silently
reused or overwritten.

### `build`

`fraud-detection build EXPERIMENT_PATH` validates a completed run manifest and
registered artifacts before building presentation data, figures, tables, and
an optional LaTeX preview.

The default output is:

```text
generated/presentations/<experiment-root-name>
```

Build is fit-free. It does not access the raw dataset, create scores, change
rankings, or modify the source experiment. `--force` is limited to conscious
replacement of the selected safe presentation directory.

### `inspect`

`fraud-detection inspect PATH` is read-only. It identifies an exact repository,
completed experiment, partial experiment, or presentation root. Completed
outputs receive semantic manifest, artifact, and checksum validation.
Inspection does not search parent directories or repair anything.

## Profiles and evidence roles

| Profile | Data and scope | Evidence classification |
|---|---|---|
| `smoke-synthetic` | Deterministic synthetic data; one seed; budgets 20/50/100; two OOF and two inner folds; candidate pool 200 | Non-evidentiary engineering profile; not comparable with canonical empirical results |
| `mini-real` | Canonical data; seeds 42/7/13; budgets 20/50/100; five OOF and three inner folds; candidate pool 1,000; linear gain only | Engineering mini profile; not thesis evidence and not comparable with canonical empirical results |
| `canonical` | Canonical data; five seeds; seven budgets; five OOF and three inner folds; both gain schemes; candidate pool 1,000 | Thesis-evidentiary profile |

All profiles use the same experiment pipeline. Engineering profiles verify the
workflow; they do not estimate canonical performance.

## Output modes

Global options precede the command:

```text
fraud-detection [--plain | --json] [--verbose] [--debug] [--no-color] COMMAND
```

- interactive TTY: restrained Rich progress for long-running commands;
- `--plain`: stable, ANSI-free transition lines without heartbeat noise;
- `--json`: exactly one final ANSI-free semantic document;
- `--no-color`: preserve the interactive structure without ANSI color;
- `--debug`: add a traceback on standard error.

Run and build show elapsed time and homogeneous counters. They do not report a
linear percentage or ETA across heterogeneous operations.

## Completion, failure, and interruption

Directory existence and process exit code alone do not prove completion.

A complete experiment requires a valid root `RUN_MANIFEST.json` with:

- schema-supported semantic status `COMPLETE`;
- all required phases;
- one immutable effective profile configuration;
- QA pass;
- safe registered artifact inventory;
- concrete checksums.

Partial or interrupted roots are `INCOMPLETE` and are not presentation
compatible. Fit-level resume is unsupported. The CLI does not delete partial
outputs automatically; start a new run with a new output path.

A complete presentation requires successful data, figure, and table stages and
their manifests. Interruption leaves the source experiment unchanged and may
leave a partial presentation root.

## Data contract

The raw file must have 284,807 rows, including 492 Fraud cases, and the columns
`Time`, `V1` through `V28`, `Amount`, and `Class`.

```text
raw SHA-256
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89

deduplicated DataFrame SHA-256
525bfe7a3155e7a5b01cf52ffdacec38a09725667b307cb6c553047c28120875
```

Stable full-row deduplication removes 1,081 rows and yields 283,726 rows:
283,253 legitimate and 473 Fraud. Any mismatch is a hard stop.

## Fit-free presentation sequence

A canonical example is:

```text
fraud-detection build outputs/canonical-final
fraud-detection inspect generated/presentations/canonical-final
```

The builder validates the completed experiment root before writing. Presentation
data and rendering are deterministic projections of registered experiment
artifacts; they are not another empirical path.

LaTeX compilation is optional. Source, schema, inventory, and hash validation
remain available when no LaTeX engine is installed.

## Tests and CI

The fast local suite is:

```text
python -m pytest tests/unit tests/contract --durations=20 -q
```

The contract is exactly 150 tests: 90 unit and 60 contract. The total runtime
target is at most 15 seconds. Ruff is pinned to `0.16.1`.

Normal CI exposes four checks:

- `fast-checks`: dependency integrity, static quality, compile/import checks,
  the exact 150-test contract, and public command help;
- `distributions`: one sdist, one wheel built from the extracted sdist,
  distribution inspection, isolated wheel installation, and public API and
  command checks;
- `platform-smoke (ubuntu-24.04)`: Linux packaging and public-path checks plus
  dry-run purity;
- `platform-smoke (windows-2025)`: Windows packaging and public-path checks
  plus dry-run purity.

Normal CI does not access the canonical dataset, fit the experiment, or render
empirical presentation outputs.

## Distribution boundary

CI creates one source distribution, extracts it, and builds one wheel from that
extracted sdist. It verifies PEP 639 MIT metadata, required archive exclusions,
and the package and console-script inventory before installing the wheel in an
isolated environment. The isolated installation verifies dependencies, public
APIs, and public help.

This is an installability and structural contract. It is not a PyPI,
production, performance, or standalone scientific-execution claim.

## Frozen A7 acceptance boundary

The frozen code, test, and package acceptance included static gates, the test
suite, sdist and wheel inspection, isolated installation, one deterministic
`smoke-synthetic` run, one engineering presentation build, semantic
inspection, and cleanup. The smoke acceptance is engineering evidence, not
thesis evidence.

## Reference and generated-output policy

`outputs/`, `generated/`, and `thesis_build/` are ignored local roots.
`reference_results/` contains three compact versioned plausibility references.
They are not runtime inputs, are not regenerated by ordinary CI, and do not
replace a complete checksum-registered experiment root.

A new canonical acceptance, presentation regeneration, or reference refresh
requires a separately authorized controlled step. See
[Results and artifacts](RESULTS_AND_ARTIFACTS.md).
