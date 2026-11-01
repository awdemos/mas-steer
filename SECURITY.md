# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Email hello@neuronengineering.dev with a clear description, reproduction steps,
and the commit or version affected. Do not open public issues for undisclosed
security bugs.

## Adversarial reviews

Each significant release is passed through the `adversarial-security-review`
workflow. The latest review is stored in `reports/adversarial-mas-steer-*.md`.

## Scope and residual risk

- The CLI is intended for local, user-driven model experimentation.
- File-path arguments are validated against parent-directory traversal.
- Prompt and config inputs are size-bounded.
- The default backend is a deterministic mock model; real model backends are
  supplied by the user, who remains responsible for their own secret handling,
  supply-chain pinning, and runtime isolation.
