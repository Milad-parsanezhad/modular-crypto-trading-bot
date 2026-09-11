from pathlib import Path


ACTIVE_RESEARCH_WORKFLOWS = (
    ".github/workflows/v17-strategy-lab.yml",
    ".github/workflows/v22c-ict-localization.yml",
    ".github/workflows/v22d-multimodal-fusion.yml",
    ".github/workflows/v23r-ml-rebuild-audit.yml",
    ".github/workflows/v23r-repo-recovery-guard.yml",
)

CHECKOUT_V7_SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_V7_SHA = "5fda3b95a4ea91299a34e894583c3862153e4b97"
UPLOAD_ARTIFACT_V7_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"


def _text(path: str) -> str:
    p = Path(path)
    assert p.exists(), f"active research workflow missing: {path}"
    return p.read_text(encoding="utf-8")


def test_active_research_workflows_use_exact_event_sha_and_node24_generation_actions():
    for path in ACTIVE_RESEARCH_WORKFLOWS:
        text = _text(path)
        assert "EXPECTED_SHA" in text, f"immutable source SHA missing: {path}"
        assert "git rev-parse HEAD" in text, f"HEAD provenance verification missing: {path}"
        assert "actions/checkout@v4" not in text, f"legacy checkout action remains: {path}"
        assert "actions/setup-python@v5" not in text, f"legacy setup-python action remains: {path}"
        checkout_current = "actions/checkout@v7" in text or f"actions/checkout@{CHECKOUT_V7_SHA}" in text
        setup_current = "actions/setup-python@v7" in text or f"actions/setup-python@{SETUP_PYTHON_V7_SHA}" in text
        assert checkout_current, f"current checkout generation missing: {path}"
        assert setup_current, f"current setup-python generation missing: {path}"


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
        "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,ADA/USDT",
    )
    for token in required:
        assert token in text, f"required research-CI gate missing: {token}"


def test_recovery_guard_is_strict_timeboxed_and_exercises_independent_fallback():
    text = _text(".github/workflows/v23r-repo-recovery-guard.yml")
    assert "timeout-minutes: 1" in text
    assert "Force direct-clone failure and prove the fallback path end-to-end" in text
    assert "forced direct-clone failure" in text
    assert "fallback_fetch" in text
    assert "STALE_PARTIAL_FILE" in text
    assert "RECOVERY_MODE=fallback_exact_ref" in text
    assert "test \"${branch}\" = \"${EXPECTED_REF}\"" in text
    assert "|| true" not in text, "recovery identity checks must never be softened"


def test_recovery_guard_pins_supply_chain_actions_to_immutable_shas():
    text = _text(".github/workflows/v23r-repo-recovery-guard.yml")
    assert f"actions/checkout@{CHECKOUT_V7_SHA}" in text
    assert f"actions/setup-python@{SETUP_PYTHON_V7_SHA}" in text
    assert f"actions/upload-artifact@{UPLOAD_ARTIFACT_V7_SHA}" in text
    assert "actions/checkout@v7" not in text
    assert "actions/setup-python@v7" not in text
    assert "actions/upload-artifact@v7" not in text


def test_research_workflows_are_fail_closed_for_live_execution():
    audit = _text(".github/workflows/v23r-ml-rebuild-audit.yml")
    multimodal = _text(".github/workflows/v22d-multimodal-fusion.yml")
    assert "live_execution_authorized" in audit
    assert "assert not d.get(\"live_execution_authorized\", False)" in audit
    assert "live_execution_authorized" in multimodal
