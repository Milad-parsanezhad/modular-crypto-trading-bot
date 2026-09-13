# Versioned Thesis Evidence Ledger

This directory records each research version against the objective and
capability that existed at that stage. Early prototypes are not retroactively
required to satisfy v0.51 standards, but their limitations under today's
standard remain explicit.

Each `ledger/v*.json` record separates:

- engineering result;
- scientific-method result;
- economic result;
- provenance strength;
- historical limitations and contribution to the next version.

Evidence classes:

- `VERIFIED_CANONICAL`: commit/run/artifact lineage is directly recorded;
- `VERIFIED_REPOSITORY_RECORD`: result is supported by a versioned repository document;
- `RETROSPECTIVE_RECONSTRUCTION`: reconstructed later and never represented as prospective evidence.

The ledger never authorizes PAPER, LIVE or Kraken access. Those remain separate
fail-closed decisions.
