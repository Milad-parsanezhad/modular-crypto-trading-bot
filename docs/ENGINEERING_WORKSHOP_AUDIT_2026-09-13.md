# Engineering Workshop Audit — 2026-09-13

## Scope

Forensic engineering/research audit of the MSc crypto-trading-bot repository, GitHub workflow surface, repository routing, public deployment shell and Railway routing. The objective is to identify root causes, distinguish scientific failures from infrastructure failures, and repair low-risk defects without changing frozen scientific outcomes.

Current safety contract remains:

`RESEARCH_ONLY / PAPER_OFF / LIVE_OFF / KRAKEN_SEALED`

No finding in this audit is profitability evidence or trading authorization.

## Executive result

The project is not suffering from a single broken repository. The dominant risks are **lineage divergence, stale navigation, missing default-branch hardening, CI infrastructure blockage, historical CI fan-out, and deployment drift**.

A selective reconciliation branch/PR was created instead of performing a destructive mega-merge.

## Findings and disposition

| ID | Severity | Finding | Evidence / consequence | Disposition |
|---|---|---|---|---|
| W-01 | HIGH | `main` and `deploy/research-v50` are materially diverged | Research lineage is hundreds of commits ahead of the old merge base while `main` also has later default-branch commits. Blind merge risks flattening preregistration and provenance. | **MITIGATED** — canonical navigation and anti-mega-merge policy added; use selective reconciliation. |
| W-02 | HIGH | Current status on `main` was stale | README/project-status navigation still centered on v0.25 while service/provenance identifies v0.50 and active v0.51. Agents could make decisions from a dated snapshot. | **FIXED IN AUDIT PR** — canonical v0.50/v0.51 status document + README/AGENTS routing. |
| W-03 | HIGH | Proven fail-closed execution validation was missing from `main` | NaN/Inf quantities/prices/fill inputs could pass normal comparisons and create invalid simulated state. | **FIXED IN AUDIT PR** — finite/policy validation + regression tests. |
| W-04 | HIGH | Risk snapshot validation was not fail-closed on `main` | NaN/Inf/negative state could avoid threshold comparisons. | **FIXED IN AUDIT PR** — invalid snapshots return `INVALID_RISK_SNAPSHOT` with kill switch; sizing validates finite inputs. |
| W-05 | HIGH | Multi-asset PIT join ordering bug remained on `main` | Group-first `merge_asof` sorting can violate pandas global monotonicity and normal interleaved panels can fail; decision-row order could change. | **FIXED IN AUDIT PR** — time-first global sort + restore input order + regression test. |
| W-06 | HIGH | `ForwardPaperRunner` defaulted to paper execution enabled | Direct construction could silently move from observation to simulated execution despite project fail-closed policy. | **FIXED IN AUDIT PR** — observation-only default + regression test. |
| W-07 | HIGH | Forward-paper zero-depth/exit governance defects remained on `main` | Forced minimum fill could create phantom liquidity; drawdown gates could block risk-reducing exits; oversized accumulated positions could be impossible to exit in one request. | **FIXED IN AUDIT PR** — true zero fill, risk-reducing exits, cap-compliant exit chunks + regression tests. |
| W-08 | MEDIUM | README advertised optional dependency groups that `main` did not define | `pip install -e '.[ml]'`, `.[deep]`, `.[rl]` did not represent a valid authoritative dependency contract. | **FIXED IN AUDIT PR** — optional groups restored from the audited research lineage. |
| W-09 | MEDIUM | Security policy forbade `.env` commits but `.gitignore` did not ignore them | Easy accidental credential leakage. | **FIXED IN AUDIT PR** — `.env`/local residue ignore rules added. |
| W-10 | MEDIUM | Canonical runtime Python was not pinned in the main repo | Railway/local resolver can choose a different interpreter than the audited deployment/CI environment. | **FIXED IN AUDIT PR** — `.python-version = 3.11`. |
| W-11 | HIGH | No general always-on integrity workflow covered every proposed main change | Recent main commits could exist without a default general status gate because historical workflows are version/path specific. | **FIXED, CURRENTLY INFRA-BLOCKED** — `main-integrity.yml` added; first run failed before any step and produced no log blob, consistent with the separately reproduced hosted-runner incident. |
| W-12 | MEDIUM | Historical CI fan-out is excessive | An unrelated audit PR triggered v17/v18/v19/v20 research workflows, including workflows that run public-data experiments or evidence harvests. This consumes runner capacity and can confuse historical evidence with ordinary regression CI. | **OPEN / GOVERNANCE FOLLOW-UP** — do not rewrite historical evidence workflows casually. Path-scope or retire PR triggers in a dedicated workflow-governance change after confirming which historical checks remain required. |
| W-13 | HIGH | GitHub-hosted runner capacity is currently unreliable | New `main-integrity` and an independent existing workflow failed before step execution with no logs; active v0.51 work records the same `runner_id=0` class. | **EXTERNAL BLOCKER** — classify as infrastructure, not code/scientific failure. Re-run once GitHub runner allocation recovers. |
| W-14 | MEDIUM | Branch protection/ruleset enforcement could not be established | Repository ruleset API reports that the private repository requires GitHub Pro or public visibility for the queried feature. | **ACCOUNT/PLAN LIMITATION** — CI exists, but required-check enforcement must be configured when account capability permits. |
| W-15 | MEDIUM | Public deployment shell and Railway runtime are drifted | Railway `thesis-trading-bot-v08` is deployed from `miladchicomobot` commit `dbd48ce...`, while shell `main` is now `09af215...`. | **OPEN — DEPLOY APPROVAL REQUIRED** — do not force redeploy as part of scientific audit; explicitly redeploy after the shell/runtime change is approved. |
| W-16 | MEDIUM | Thesis Railway service is hosted inside the `crypto-intelligence-studio-v6` Railway project | Operational placement can cause agents to conflate the MSc trading bot with the separate fundamental intelligence platform. The dedicated `crypto-bot-research` project currently has no services after resource-provision failure. | **DOCUMENTED / OPEN** — repository contracts now separate the projects. Move the service only when Railway resource capacity and migration approval exist. |
| W-17 | HIGH | v0.51 prospective collection is infrastructure-blocked | Active PR records GitHub no-runner failures and Railway free-plan service-provision limit. Strict first-seen rules prohibit retroactively upgrading late bars to prospective evidence. | **SCIENTIFIC DECISION DEFERRED** — do not backfill missed bars as prospective; resume only with timestamp-valid collection capacity. |
| W-18 | MEDIUM | Forward-paper marks are not restart-persistent across all assets | v0.50 engineering audit notes in-process last marks can fall back to average entry price after restart. | **BLOCKER BEFORE PAPER REACTIVATION** — add persistent synchronized mark snapshots before any future paper authorization. No immediate runtime risk while PAPER is off. |
| W-19 | MEDIUM | Deployment shell is a second code surface | Shell/main can drift in dependencies/endpoints from canonical scientific repo. | **MITIGATED** — AGENTS/source-of-truth contract; longer-term goal should be a generated/versioned deployment contract rather than duplicated scientific logic. |
| W-20 | LOW | Dependency update monitoring was absent | Python/action dependency drift could go unnoticed. | **FIXED IN AUDIT PR** — Dependabot config for pip and GitHub Actions added. |

## Scientific reconciliation rule

Only engineering changes that are **scientific-result neutral** should be selectively ported into `main` without creating a new experiment. Examples: input validation, fail-closed guards, deterministic data-join correctness, dependency metadata and documentation routing.

The following are not engineering-only changes and require experiment-specific governance: model architecture, labels, features with predictive meaning, thresholds, costs, sample windows, venue/asset selection, holdout access, overlap/arbitration policy, and PAPER/LIVE authorization.

## CI interpretation rule

A GitHub Actions `failure` is not enough to classify a code failure. Use this hierarchy:

1. Did a runner start?
2. Did checkout/setup execute?
3. Did dependency installation execute?
4. Did tests execute?
5. Which assertion or command failed?

If the job has no executed steps/logs because a runner was never allocated, classify it as `INFRASTRUCTURE_BLOCKED`, not `TEST_FAILED`.

## Deployment interpretation rule

GitHub source state, Railway source configuration, and active Railway deployment SHA are three separate facts. A repository commit does not prove the runtime was redeployed.

Before claiming a deployment contains a fix, verify the Railway deployment commit SHA and `/health` response after the deploy.

## Next admissible actions

1. Let PR #65 receive real runner execution when GitHub-hosted runners recover; require full `main-integrity` pass before merge.
2. After PR #65 passes, merge it through review rather than force-moving `main`.
3. Create a dedicated CI-governance change to path-scope historical v17-v20 PR workflows and reduce unnecessary runner fan-out without altering frozen artifacts.
4. If the owner approves deployment, redeploy the public shell and verify Railway commit SHA + health endpoint.
5. Do not claim any v0.51 economic result until timestamp-valid prospective support exists under the frozen first-seen contract.
6. Before any future PAPER reactivation, add persistent synchronized marking, dedicated database routing, restart/recovery tests, and explicit paper-promotion governance.

## Current audit disposition

`ENGINEERING_RECONCILIATION_PREPARED / CI_RUNNER_BLOCKED / SCIENTIFIC_STATE_UNCHANGED / PAPER_OFF / LIVE_OFF`
