# Security Policy

## Supported versions

Security fixes apply only to the current public default branch and, when one
exists, the latest public tagged release. Historical branches, commits,
development snapshots, and superseded releases are unsupported.

## Reporting a vulnerability

Do not disclose sensitive vulnerability details in a public issue. Use GitHub
private vulnerability reporting when the repository exposes **Report a
vulnerability**. If that option is unavailable, open only a minimal issue
asking the maintainer for a private contact channel.

Do not include exploit details, credentials, raw data, transaction records, or
personal information in that issue.

## Security scope

In scope:

- unsafe path handling;
- arbitrary file overwrite or deletion;
- command execution;
- credential exposure;
- dependency or packaging vulnerabilities;
- manifest or checksum bypasses that permit invalid artifacts to appear valid.

The following are not security vulnerabilities:

- model accuracy, false positives, or false negatives;
- methodological disagreement or empirical-result interpretation;
- dataset-license questions;
- missing production hardening;
- claims that the research model is not production-ready.

This is research software, not a deployed fraud-decision system.
