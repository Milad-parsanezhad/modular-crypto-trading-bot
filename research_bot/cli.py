from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .event_competing_risk_v41 import preregistration_manifest_v41
from .mother_strategy_v39 import build_mother_features_v39, mother_strategy_manifest_v39
from .two_stage_hurdle_v40 import preregistration_manifest_v40


ROOT = Path(__file__).resolve().parents[1]


def _json(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def cmd_doctor(_: argparse.Namespace) -> int:
    manifests = {
        "v39": mother_strategy_manifest_v39(),
        "v40": preregistration_manifest_v40(),
        "v41": preregistration_manifest_v41(),
    }
    assert manifests["v39"]["kraken_holdout"] == "SEALED"
    assert manifests["v39"]["paper_execution"] is False
    assert manifests["v39"]["live_execution"] is False
    assert manifests["v40"]["kraken_touched"] is False
    assert manifests["v41"]["kraken_touched"] is False
    assert manifests["v40"]["paper_execution"] is False
    assert manifests["v41"]["paper_execution"] is False
    assert manifests["v40"]["live_execution"] is False
    assert manifests["v41"]["live_execution"] is False
    _json({"status": "PASS", "execution": "RESEARCH_ONLY", "kraken": "SEALED"})
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    manifests = {
        "v39": mother_strategy_manifest_v39,
        "v40": preregistration_manifest_v40,
        "v41": preregistration_manifest_v41,
    }
    _json(manifests[args.version]())
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    result_files = {
        "v39": ROOT / "docs" / "V39_RESULTS_2026-09-12.md",
        "v40": ROOT / "docs" / "V40_RESULTS_2026-09-12.md",
        "v41": ROOT / "docs" / "V41_RESULTS_2026-09-12.md",
    }
    _json(
        {
            "package": "modular-crypto-research-bot",
            "mode": "RESEARCH_ONLY",
            "mother_strategy": "v0.39",
            "latest_experiment": "v0.41",
            "result_documents": {k: v.exists() for k, v in result_files.items()},
            "paper_execution": False,
            "live_execution": False,
            "kraken_holdout": "SEALED",
        }
    )
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.input)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    out = build_mother_features_v39(frame)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output}")
    return 0


def cmd_characterize(args: argparse.Namespace) -> int:
    scripts = {
        "v39": ROOT / "scripts" / "run_v39_development_characterization_corrected.py",
        "v40": ROOT / "scripts" / "run_v40_two_stage_characterization.py",
        "v41": ROOT / "scripts" / "run_v41_competing_risk_characterization_fast.py",
    }
    script = scripts[args.version]
    if not script.exists():
        raise SystemExit(f"missing characterization runner: {script}")
    cmd = [sys.executable, str(script), "--output-dir", args.output_dir]
    return int(subprocess.call(cmd, cwd=str(ROOT)))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modular-crypto-bot",
        description="Fail-closed academic crypto strategy research CLI. No live-order command exists.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="verify safety/governance invariants")
    doctor.set_defaults(func=cmd_doctor)

    manifest = sub.add_parser("manifest", help="print a frozen research manifest")
    manifest.add_argument("--version", choices=("v39", "v40", "v41"), default="v41")
    manifest.set_defaults(func=cmd_manifest)

    status = sub.add_parser("status", help="print current research/execution state")
    status.set_defaults(func=cmd_status)

    features = sub.add_parser("features", help="build causal v0.39 mother features from OHLCV CSV")
    features.add_argument("--input", required=True)
    features.add_argument("--output", required=True)
    features.set_defaults(func=cmd_features)

    characterize = sub.add_parser("characterize", help="run a frozen development characterization")
    characterize.add_argument("--version", choices=("v39", "v40", "v41"), default="v41")
    characterize.add_argument("--output-dir", required=True)
    characterize.set_defaults(func=cmd_characterize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
