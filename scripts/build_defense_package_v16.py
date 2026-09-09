from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from research_bot.formal_forward_v16 import build_formal_forward_evaluation, chapter4_markdown


def _load_snapshots(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*_v15_forward_evidence.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        raise ValueError(f"no v0.15 snapshots found in {input_dir}")
    return rows


def _svg_line(values: list[float], labels: list[str], title: str, y_label: str) -> str:
    width, height = 960, 420
    left, right, top, bottom = 80, 30, 50, 65
    plot_w = width - left - right
    plot_h = height - top - bottom
    if not values:
        values = [0.0]
        labels = [""]
    lo, hi = min(values), max(values)
    if hi == lo:
        pad = abs(hi) * 0.05 or 1.0
        lo, hi = lo - pad, hi + pad
    else:
        pad = (hi - lo) * 0.08
        lo, hi = lo - pad, hi + pad

    def xy(i: int, v: float) -> tuple[float, float]:
        x = left if len(values) == 1 else left + plot_w * i / (len(values) - 1)
        y = top + plot_h * (hi - v) / (hi - lo)
        return x, y

    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (xy(i, v) for i, v in enumerate(values)))
    first_label = labels[0] if labels else ""
    last_label = labels[-1] if labels else ""
    mid = (hi + lo) / 2.0
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700">{title}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black" stroke-width="1"/>
<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black" stroke-width="1"/>
<polyline fill="none" stroke="black" stroke-width="2.5" points="{points}"/>
<text x="{left-10}" y="{top+5}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{hi:.6f}</text>
<text x="{left-10}" y="{top+plot_h/2+5}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{mid:.6f}</text>
<text x="{left-10}" y="{top+plot_h+5}" text-anchor="end" font-family="Arial, sans-serif" font-size="12">{lo:.6f}</text>
<text x="{left}" y="{height-28}" font-family="Arial, sans-serif" font-size="11">{first_label}</text>
<text x="{left+plot_w}" y="{height-28}" text-anchor="end" font-family="Arial, sans-serif" font-size="11">{last_label}</text>
<text x="20" y="{top+plot_h/2}" transform="rotate(-90 20 {top+plot_h/2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">{y_label}</text>
</svg>'''


def _write_timeline_csv(evaluation: dict, path: Path) -> None:
    rows = list(evaluation.get("timeline") or [])
    fields = ["captured_at", "equity", "cumulative_return", "current_drawdown", "observations_total", "fills_total", "equity_points_total"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _defense_summary(evaluation: dict) -> str:
    s = evaluation["sample_progress"]
    p = evaluation["paper_performance"]
    policy = evaluation["metric_policy"]
    return "\n".join([
        "# Thesis Defense Evidence Summary — v0.16",
        "",
        f"Formal state: `{evaluation['formal_state']}`",
        f"LIVE promotion: `{evaluation['live_promotion']}`",
        "",
        "## What can be demonstrated now",
        "",
        "- production Railway service is operational in PAPER mode;",
        "- PostgreSQL-backed observations/fills/equity are being persisted;",
        "- scheduled GitHub Actions export reproducible forward evidence;",
        "- fail-closed LIVE safety semantics are continuously checked;",
        "- the defense package is regenerated from immutable v0.15 artifacts.",
        "",
        "## Current forward evidence",
        "",
        f"- observations: `{s['observations_total']}`;",
        f"- simulated fills: `{s['fills_total']}`;",
        f"- equity points: `{s['equity_points_total']}`;",
        f"- latest cumulative paper return: `{p['latest_cumulative_return']}`;",
        f"- latest drawdown: `{p['latest_current_drawdown']}`;",
        f"- pre-registered sample gate passed: `{s['preregistered_gate_passed']}`;",
        f"- formal Sharpe/Sortino authorized: `{policy['risk_ratios_authorized']}`.",
        "",
        "## Defense wording",
        "",
        "The system is engineering-complete for a forward-paper thesis demonstration. The forward sample is evaluated prospectively. Until the pre-registered evidence gate is satisfied and a separate formal statistical review is completed, the work does not claim validated alpha, profitability, or real-money live readiness.",
        "",
    ])


def _manifest(output_dir: Path, files: Iterable[Path]) -> dict:
    entries = []
    for path in sorted(files, key=lambda x: x.name):
        data = path.read_bytes()
        entries.append({"file": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"package": "v0.16-formal-forward-defense", "files": entries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/v16_input")
    parser.add_argument("--output-dir", default="artifacts/v16_defense")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = build_formal_forward_evaluation(_load_snapshots(input_dir))
    timeline = list(evaluation.get("timeline") or [])

    json_path = output_dir / "v16_formal_forward_evaluation.json"
    chapter_path = output_dir / "chapter4_forward_results.md"
    summary_path = output_dir / "defense_summary.md"
    csv_path = output_dir / "forward_timeline.csv"
    equity_svg = output_dir / "equity_curve.svg"
    dd_svg = output_dir / "drawdown_curve.svg"
    sample_svg = output_dir / "sample_growth.svg"

    json_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8")
    chapter_path.write_text(chapter4_markdown(evaluation), encoding="utf-8")
    summary_path.write_text(_defense_summary(evaluation), encoding="utf-8")
    _write_timeline_csv(evaluation, csv_path)

    labels = [str(x.get("captured_at") or "") for x in timeline]
    equity_svg.write_text(_svg_line([float(x.get("equity") or 0.0) for x in timeline], labels, "Forward Paper Equity", "Equity (USDT)"), encoding="utf-8")
    dd_svg.write_text(_svg_line([100.0 * float(x.get("current_drawdown") or 0.0) for x in timeline], labels, "Forward Paper Drawdown", "Drawdown (%)"), encoding="utf-8")
    sample_svg.write_text(_svg_line([float(x.get("observations_total") or 0.0) for x in timeline], labels, "Forward Evidence Growth", "Observations"), encoding="utf-8")

    package_files = [json_path, chapter_path, summary_path, csv_path, equity_svg, dd_svg, sample_svg]
    manifest = _manifest(output_dir, package_files)
    manifest_path = output_dir / "defense_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "formal_state": evaluation["formal_state"],
        "snapshots": evaluation["source_contract"]["snapshot_count"],
        "observations": evaluation["sample_progress"]["observations_total"],
        "fills": evaluation["sample_progress"]["fills_total"],
        "sample_gate": evaluation["sample_progress"]["preregistered_gate_passed"],
        "live_promotion": evaluation["live_promotion"],
        "output_dir": str(output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
