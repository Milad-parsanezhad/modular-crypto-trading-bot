from pathlib import Path


ACTIVE_RESEARCH_WORKFLOWS = (
    ".github/workflows/v17-strategy-lab.yml",
    ".github/workflows/v22c-ict-localization.yml",
    ".github/workflows/v22d-multimodal-fusion.yml",
    ".github/workflows/v23r-ml-rebuild-audit.yml",
    ".github/workflows/v23r-repo-recovery-guard.yml",
)


def _text(path: str) -> str:
    p = Path(path)
    assert p.exists(), f"active research workflow missing: {path}"
    return p.read_text(encoding="utf-8")


def test_active_research_workflows_use_exact_sha_and_current_action_generation():
    for path in ACTIVE_RESEARCH_WORKFLOWS:
        text = _text(path)
        assert "EXPECTED_SHA" in text, f"immutable source SHA missing: {path}"
        assert "git rev-parse HEAD" in text, f"HEAD provenance verification missing: {path}"
        assert "actions/checkout@v4" not in text, f"legacy checkout action remains: {path}"
        assert "actions/setup-python@v5" not in text, f"legacy setup-python action remains: {path}"
        assert "actions/checkout@v7" in text, f"current checkout action missing: {path}"
        assert "actions/setup-python@v7" in text, f"current setup-python action missing: {path}"


def test_active_research_workflows_do_not_restore_shared_pip_cache():
    for path in ACTIVE_RESEARCH_WORKFLOWS:
        text = _text(path)
        assert "cache: pip" not in text, f"shared pip cache reintroduced: {path}"
        assert "PIP_NO_CACHE_DIR" in text or "--no-cache-dir" in text, f"no-cache policy missing: {path}"


def test_current_ml_dag_contains_all_integrity_and_replication_gates():
    text = _text(".github/workflows/v23r-ml-rebuild-audit.yml")
    required = (
        "dependency-integrity:",
        "upstream-evidence-integrity:",
        "deterministic-audit-tests:",
        "real-data-panel-audit:",
        "external-venue-replication:",
        "final-evidence-gate:",
        "tests/test_ict_localization_v22c.py",
        "tests/test_multimodal_fusion_v22d.py",
        "tests/test_deep_temporal_v22.py",
        "tests/test_external_ml_replication_v23r.py",
    )
    for token in required:
        assert token in text, f"required research-CI gate missing: {token}"


def test_recovery_guard_contains_one_minute_breaker_and_forced_fallback():
    text = _text(".github/workflows/v23r-repo-recovery-guard.yml")
    assert "timeout-minutes: 1" in text
    assert "Force direct-clone failure and prove the fallback path end-to-end" in text
    assert "forced direct-clone failure" in text
    assert "fallback_fetch" in text
