from __future__ import annotations

"""Performance-equivalent wrapper for the frozen v0.41 characterization.

Scientific protocol is unchanged.  Only the implementation of competing-risk
inference is replaced by a batched/vectorized function that is mathematically
identical to the scalar implementation.
"""

import importlib.util
from pathlib import Path

from research_bot.event_competing_risk_vectorized_v41 import (
    predict_competing_risks_vectorized_v41,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "run_v41_competing_risk_characterization.py"
spec = importlib.util.spec_from_file_location("v41_base_characterization", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.41 characterization runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
runner.predict_competing_risks_v41 = predict_competing_risks_vectorized_v41

if __name__ == "__main__":
    runner.main()
