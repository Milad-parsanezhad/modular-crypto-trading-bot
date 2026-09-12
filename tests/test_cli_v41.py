from __future__ import annotations

from research_bot.cli import build_parser, cmd_doctor


def test_cli_has_research_commands_but_no_live_command() -> None:
    parser = build_parser()
    action = next(a for a in parser._actions if getattr(a, "choices", None))
    choices = set(action.choices)
    assert {"doctor", "manifest", "status", "features", "characterize"}.issubset(choices)
    assert "live" not in choices
    assert "order" not in choices
    assert "paper" not in choices


def test_doctor_passes_fail_closed_governance() -> None:
    assert cmd_doctor(None) == 0
