# v0.20 Phase-Q Source Notes

Date: 2026-09-10

External infrastructure research used for this stage:

- GitHub Actions artifacts are persistent run outputs and can be listed/downloaded through the GitHub Actions artifacts API.
- Artifact retention can be set per upload within repository/organization limits; the project retains Phase-Q artifacts for 90 days.
- Scheduled workflow execution can be delayed relative to nominal cron timing, so actual stored timestamps are treated as authoritative.

These infrastructure facts motivate the separate artifact-harvest/quality-monitor design. They do not constitute market evidence or alpha evidence.
