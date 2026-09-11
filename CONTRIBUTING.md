# Contributing

Thank you for contributing to this research repository.

## Research-first contribution standard

Changes that affect data, features, labels, strategies, model selection, risk, execution or reported performance must preserve the project's **Evidence Before Opinion** contract.

A research change should state:

1. the hypothesis being tested;
2. what data are available at decision time;
3. which period/data are development, validation, test, shadow or forward;
4. whether any test/terminal evidence has already been observed;
5. transaction-cost and execution assumptions;
6. the baseline being challenged;
7. the promotion gate and failure conditions;
8. the expected artifacts/provenance outputs.

## Prohibited research shortcuts

Do not:

- use future or revised information as if it were point-in-time;
- tune a model on a terminal test and then call that same test untouched;
- remove losing assets/periods after inspecting outcomes without labeling the analysis exploratory;
- report synthetic smoke-test performance as thesis evidence;
- omit modeled fees/slippage when comparing trading rules;
- fabricate unavailable order-flow, on-chain, OI, funding or liquidation history;
- enable LIVE execution from a research result.

## Pull requests

Prefer a focused branch and pull request. Include:

- purpose and hypothesis;
- changed scientific contract;
- exact tests/workflows executed;
- source commit and artifact identifiers for material experiments;
- result status using project terminology (`TESTED`, `REJECTED`, `BLOCKED`, `CHALLENGER`, `DATA_UNAVAILABLE`);
- explicit PAPER/LIVE authorization state.

Negative results are welcome and should be preserved.

## Code quality

Before submitting:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

For optional stacks, install only the required extra (`ml`, `deep`, or `rl`). Keep dependencies explicit and record the environment for scientific runs.

## Security

Never commit API keys, exchange credentials, passwords, tokens or private wallet material. See `SECURITY.md`.
