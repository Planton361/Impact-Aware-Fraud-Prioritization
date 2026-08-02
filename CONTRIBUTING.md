# Contributing

## Scope

Focused contributions are welcome for documentation, reproducibility,
packaging, platform compatibility, tests that protect documented behavior, and
clearly demonstrated implementation defects.

The empirical experiment is frozen. Changes to data identity, deduplication,
seeds, folds, budgets, gains, candidate-pool scope, model parameters, selection
rules, rankings, metrics, or scientific claim boundaries are out of scope
unless a new research protocol is explicitly authorized.

## Development setup

Use either GitHub Codespaces / the Dev Container or the platform bootstrap in
the [README](README.md). Bootstrap may acquire data; the Dev Container does not
download the canonical dataset automatically.

## Required checks

```text
python -m ruff check src scripts tests
python -m compileall -q src scripts
python -m pytest tests/unit tests/contract --durations=20 -q
```

The exact fast contract is 150 tests: 90 unit and 60 contract tests. A
canonical experiment is not a normal contribution prerequisite. Smoke or
canonical execution requires explicit scope and justification.

## Data and generated files

Never commit `data/creditcard.csv`, Kaggle credentials, `.env` files,
`outputs/`, `generated/`, `thesis_build/`, local virtual environments, or
generated presentation files. Do not modify `reference_results/` unless an
explicitly authorized controlled refresh is part of the task.

## Pull requests

Pull requests need one focused purpose, an explanation of user-visible
behavior, and tests or documentation appropriate to the change. Avoid unrelated
formatting and do not silently change commands, profiles, paths, evidence
roles, or claim boundaries.

## Scientific language

Preserve these boundaries:

- `Amount` is a proxy for potential financial impact;
- PLR and technical `prevented_loss_*` fields are proxy quantities;
- BCE is the only probabilistic score;
- ranker and fixed-reference scores are ordinal;
- p-only is not Amount-free;
- the fixed reference is untrained and not a loss function.
