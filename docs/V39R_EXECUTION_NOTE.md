# v0.39R Execution Note

The preferred execution stack for this research stage is:

- **GitHub Actions** for deterministic CI, unit tests, and reproducible research jobs;
- **Railway** for a long-running hosted research service or scheduled runtime when continuous execution/log inspection is needed;
- **Neon** only when persistent PostgreSQL storage becomes necessary for experiment registries, evidence ledgers, model/run metadata, or result querying.

Neon and Railway are complementary rather than substitutes: Railway runs the service; Neon stores relational state.

For v0.39R reconstruction, GitHub Actions is sufficient and avoids introducing infrastructure state before the architecture is validated.
