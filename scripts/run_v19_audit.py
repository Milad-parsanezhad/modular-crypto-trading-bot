from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import ccxt
import pandas as pd

from research_bot.audit_v19 import (
    V19AuditConfig,
    audit_cost_aware_conversion,
    audit_manifest,
    audit_v11_regime_replication_construction,
)
from research_bot.coinex_public import PERIOD_MS, fetch_coinex_klines, utc_now_ms
from research_bot.integrity_v19 import finalize_payload_hash


CORE_EXTERNAL_COHORT = [
    "BTC/USDT",
    "ETH/USDT",
    "DOGE/USDT",
    "DOT/USDT",
    "NEAR/USDT",
    "SOL/USDT",
    "SUI/USDT",
    "UNI/USDT",
    "XRP/USDT",
    "ZEC/USDT",
]


def _drop_incomplete(df: pd.DataFrame, step_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    x = df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="raise")
    return x[x["timestamp"] + pd.Timedelta(milliseconds=step_ms) <= now].reset_index(drop=True)


def fetch_coinex(symbols: list[str], bars: int) -> dict[str, pd.DataFrame]:
    end_ms = utc_now_ms()
    out: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = fetch_coinex_klines(symbol=symbol, period="4hour", market_type="spot", end_ms=end_ms, bars=bars)
        frame = _drop_incomplete(frame, PERIOD_MS["4hour"])
        if not frame.empty:
            out[symbol] = frame
    return out


def fetch_ccxt_cohort(exchange_id: str, symbols: list[str], bars: int, timeframe: str = "4h") -> tuple[dict[str, pd.DataFrame], list[dict]]:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    exchange.load_markets()
    step_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    now_ms = exchange.milliseconds()
    since = now_ms - int((bars + 80) * step_ms)
    out: dict[str, pd.DataFrame] = {}
    failures: list[dict] = []
    try:
        for symbol in symbols:
            if symbol not in exchange.markets:
                failures.append({"symbol": symbol, "reason": "SYMBOL_NOT_LISTED"})
                continue
            rows = []
            cursor = since
            loops = 0
            try:
                while cursor < now_ms and len(rows) < bars + 80 and loops < 25:
                    loops += 1
                    limit = min(300, bars + 80 - len(rows))
                    batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
                    if not batch:
                        break
                    rows.extend(batch)
                    nxt = int(batch[-1][0]) + step_ms
                    if nxt <= cursor:
                        break
                    cursor = nxt
                    if exchange.rateLimit:
                        time.sleep(exchange.rateLimit / 1000.0)
                if not rows:
                    failures.append({"symbol": symbol, "reason": "EMPTY_OHLCV"})
                    continue
                df = pd.DataFrame(rows, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df = df.drop_duplicates("timestamp").sort_values("timestamp")
                df = _drop_incomplete(df, step_ms).tail(bars).reset_index(drop=True)
                if not df.empty:
                    out[symbol] = df
            except Exception as exc:
                failures.append({"symbol": symbol, "reason": f"FETCH_ERROR:{type(exc).__name__}:{exc}"})
    finally:
        exchange.close()
    return out, failures


def render_markdown(payload: dict) -> str:
    a = payload["cost_audit"]
    b = payload["regime_audit"]
    am = a["cost_aware_portfolio"]
    an = a["naive_portfolio"]
    bp = b["plain_cross_sectional_ichimoku"]
    bc = b["market_regime_conditioned_ichimoku"]
    return f"""# v0.19 Audit-Corrected Research Results\n\nGenerated: {payload['run_generated_at']}\n\nStatus: **AUDIT / NO PROMOTION**\n\n## Bugs corrected\n\n- v0.18-A calibrated uncertainty on in-sample residuals. v0.19 uses chronological expanding out-of-fold residuals inside the training segment.\n- v0.18 backtests could leave a final open position without paying its terminal liquidation cost. v0.19 charges terminal close turnover/cost.\n- v0.18 did not persist coverage timestamps and dataset fingerprints in Experiment A. v0.19 records both.\n- v0.18-B did not reproduce the v0.11 portfolio geometry: it replaced the cross-sectional top-quartile Ichimoku baseline with per-asset `score > 0` and used per-asset regime gates. v0.19 restores cross-sectional ranking and market-level regime classification.\n\n## Important reproducibility limitation\n\nThe original v0.18 artifact did not archive the raw OHLCV input or its deterministic fingerprint. Therefore v0.19 cannot reconstruct the exact original v0.18 sample byte-for-byte. The audit uses a refreshed public-data reconstruction with the same requested bar count and frozen methodology. Its numerical results must **not** be interpreted as a direct corrected replacement of the original v0.18 numbers.\n\n## A — Cost-aware reconstruction audit\n\n| Metric | Naive | Cost-aware |\n|---|---:|---:|\n| Net return | {an.get('net_total_return', float('nan')):.4%} | {am.get('net_total_return', float('nan')):.4%} |\n| Sharpe | {an.get('sharpe', float('nan')):.3f} | {am.get('sharpe', float('nan')):.3f} |\n| Max DD | {an.get('max_drawdown', float('nan')):.4%} | {am.get('max_drawdown', float('nan')):.4%} |\n| Turnover | {an.get('turnover_sum', float('nan')):.2f} | {am.get('turnover_sum', float('nan')):.2f} |\n\nBootstrap 95% CI for cost-aware minus naive mean return: `{a['paired_moving_block_bootstrap'].get('ci_low')}` to `{a['paired_moving_block_bootstrap'].get('ci_high')}`.\n\nEvidence status: `{a['evidence_status']}`. This is a refreshed reconstruction audit, not fresh alpha evidence and not a replacement of the immutable v0.18 artifact.\n\n## B — Corrected v0.11 regime-construction audit\n\nSymbols used: {', '.join(b['symbols'])}\n\n| Metric | Plain cross-sectional Ichimoku | Market-regime conditioned |\n|---|---:|---:|\n| Net return | {bp.get('net_total_return', float('nan')):.4%} | {bc.get('net_total_return', float('nan')):.4%} |\n| Sharpe | {bp.get('sharpe', float('nan')):.3f} | {bc.get('sharpe', float('nan')):.3f} |\n| Max DD | {bp.get('max_drawdown', float('nan')):.4%} | {bc.get('max_drawdown', float('nan')):.4%} |\n| Turnover | {bp.get('turnover_sum', float('nan')):.2f} | {bc.get('turnover_sum', float('nan')):.2f} |\n\nBootstrap 95% CI for conditioned minus plain mean return: `{b['paired_moving_block_bootstrap'].get('ci_low')}` to `{b['paired_moving_block_bootstrap'].get('ci_high')}`.\n\nEvidence status: `{b['evidence_status']}`. This is a construction audit, not a fresh independent replication.\n\n## Integrity\n\nFinal payload SHA-256 (self-excluding): `{payload['manifest_sha256']}`.\n\n## Research decision\n\n`LIVE_EXECUTION = false`\n\n`PAPER_STRATEGY_REPLACEMENT = false`\n\nThe next fresh evidence stream is prospective multi-venue microstructure collection under v0.19.\n"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bars", type=int, default=1800)
    p.add_argument("--external-exchange", default="okx")
    p.add_argument("--output", default="artifacts/v19/v19_audit.json")
    p.add_argument("--markdown", default="artifacts/v19/V19_AUDIT_RESULTS.md")
    args = p.parse_args()

    cfg = V19AuditConfig()
    coinex = fetch_coinex(["BTC/USDT", "ETH/USDT"], args.bars)
    if len(coinex) < 2:
        raise RuntimeError(f"CoinEx audit dataset incomplete: {sorted(coinex)}")
    cost = audit_cost_aware_conversion(coinex, cfg)
    cost["evidence_status"] = "REFRESHED_RECONSTRUCTION_AUDIT_NOT_ORIGINAL_V18_SAMPLE"
    cost["original_v18_raw_dataset_recoverable"] = False
    cost["reconstruction_note"] = (
        "v0.18 did not persist raw OHLCV or its fingerprint; v0.19 uses the same requested bar count "
        "on the later public-data endpoint and therefore cannot isolate code-fix effects from sample drift"
    )

    ext, failures = fetch_ccxt_cohort(args.external_exchange, CORE_EXTERNAL_COHORT, args.bars)
    external_used = args.external_exchange
    if len(ext) < cfg.min_cross_section_assets:
        fallback = "kucoin" if args.external_exchange != "kucoin" else "okx"
        ext, failures2 = fetch_ccxt_cohort(fallback, CORE_EXTERNAL_COHORT, args.bars)
        failures.extend([{"fallback": fallback, **x} for x in failures2])
        external_used = fallback
    if len(ext) < cfg.min_cross_section_assets:
        raise RuntimeError(f"insufficient external cohort after fallback: {sorted(ext)}")

    regime = audit_v11_regime_replication_construction(ext, cfg)
    payload = audit_manifest(cost, regime)
    payload["external_exchange"] = external_used
    payload["external_provider_failures"] = failures
    payload["run_generated_at"] = datetime.now(timezone.utc).isoformat()
    finalize_payload_hash(payload, "manifest_sha256")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md = Path(args.markdown); md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "markdown": str(md),
        "external_exchange": external_used,
        "external_symbols": regime["symbols"],
        "manifest_sha256": payload["manifest_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
