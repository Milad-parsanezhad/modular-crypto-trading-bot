# Security Policy

## Scope

This repository contains research code, public-data connectors and PAPER/backtest infrastructure. Real-money LIVE execution is not authorized by the current research contract.

## Reporting a vulnerability

Please do not publish exploitable credentials, private endpoints, wallet secrets, database passwords or exchange tokens in a public issue.

When reporting a security problem, include only the minimum technical detail needed to reproduce the issue and redact secrets from logs/screenshots.

## Credential rules

Never commit:

- exchange API keys or secrets;
- GitHub, Railway, database or cloud tokens;
- private keys / seed phrases;
- passwords or connection strings containing credentials;
- `.env` files containing secrets.

Use environment variables or the hosting platform's secret store. Public research artifacts must contain provenance and hashes, not private credentials.

## Execution safety

Any code path that could place a real order must remain fail-closed unless a separately reviewed production-readiness process explicitly authorizes it. Research, backtest and PAPER evidence cannot silently enable LIVE execution.

## Dependency / model artifact safety

Serialized Python/ML artifacts (for example `joblib` or PyTorch files) should only be loaded from trusted experiment artifacts with a recorded digest and compatible dependency environment. Treat untrusted pickle/joblib files as executable code.
