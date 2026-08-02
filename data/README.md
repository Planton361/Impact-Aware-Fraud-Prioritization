# Local dataset

The canonical external input is the official Kaggle
[`mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset stored as:

```text
data/creditcard.csv
```

The CSV is ignored by Git and must never be committed or included in a source
or release archive.

## Expected identity

The raw file contains 284,807 rows and 31 columns (`Time`, `V1` through `V28`,
`Amount`, `Class`), including 492 Fraud cases.

```text
raw SHA-256
76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

Stable full-row deduplication uses `drop_duplicates(keep="first")` before any
split or fit. It removes 1,081 duplicate rows and produces:

- 283,726 total rows;
- 283,253 legitimate cases;
- 473 Fraud cases.

```text
deduplicated DataFrame SHA-256
525bfe7a3155e7a5b01cf52ffdacec38a09725667b307cb6c553047c28120875
```

Any identity mismatch is a hard stop.

## Preferred setup

Windows PowerShell:

```powershell
.\scripts\bootstrap.ps1
```

macOS/Linux:

```bash
./scripts/bootstrap.sh
```

Bootstrap reuses a valid existing CSV. When the file is absent, setup may use
the network and Kaggle authentication to acquire it. Keep credentials outside
the repository.

The installed mutating command is:

```text
fraud-detection setup
```

The read-only data validation command is:

```text
fraud-detection check --require-data
```

## Manual setup

Alternatively, download `creditcard.csv` from the official Kaggle dataset
page, place it at `data/creditcard.csv`, and verify the raw SHA-256 before use.

## Dataset license and attribution

The repository MIT License applies only to the original repository software
and documentation. It does not license the external dataset.

The Kaggle data card currently identifies the dataset license as
`Database: Open Database, Contents: Database Contents`. The corresponding Open
Data Commons terms are the
[Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/)
and the
[Database Contents License 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Users remain responsible for following the current license and attribution
notices on the original dataset page, especially when publishing or
redistributing a database or an adapted database.

The dataset page attributes collection and analysis to a research
collaboration between Worldline and the Machine Learning Group of the
Université Libre de Bruxelles. It requests citation of:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
> “Calibrating Probability with Undersampling for Unbalanced Classification.”
> IEEE Symposium on Computational Intelligence and Data Mining, 2015.

Software citation metadata is provided separately in
[`CITATION.cff`](../CITATION.cff).

`Amount` is only a proxy for potential financial impact, not observed loss.
See [Reproducibility](../docs/REPRODUCIBILITY.md#data-contract) for the full
environment and validation contract.
