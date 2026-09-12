from __future__ import annotations

"""Compatibility launcher for the frozen v0.46 empirical runner.

The scientific runner and frozen v0.39 governor semantics are unchanged. This
launcher only replaces the v0.39 allocator function reference with the v0.46
numerical-compatibility copy that requests a writable NumPy array under pandas 3.
"""

import importlib.util
from pathlib import Path

from research_bot.economic_state_v46 import allocate_portfolio_risk_writable_v46

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v46_economic_state_representation.py"

spec = importlib.util.spec_from_file_location("v46_frozen_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load frozen v0.46 runner: {RUNNER}")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

# Both dynamic v0.39 module instances resolve allocate_portfolio_risk from their
# own module globals. Patch only that function reference; all other financial
# system functions, policy values and simulation code remain frozen.
runner.base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46
runner.corrected.base.allocate_portfolio_risk = allocate_portfolio_risk_writable_v46

if __name__ == "__main__":
    runner.main()
