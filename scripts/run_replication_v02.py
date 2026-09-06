from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from research_bot.data import fetch_with_fallback
from research_bot.features import FEATURE_COLUMNS, add_features
from research_bot.regime import REGIME_COLUMNS, add_regime_features
from research_bot.replication import symmetric_cusum_events, triple_barrier_events


def event_equity_metrics(returns: pd.Series) -> dict:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {"n_events": 0}
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {"n_events": int(len(r)), "total_return": float(equity.iloc[-1] - 1.0), "mean_event_return": float(r.mean()), "median_event_return": float(r.median()), "event_hit_rate": float((r > 0).mean()), "max_drawdown_event_equity": float(dd.min())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", default="coinex")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="4h")
    p.add_argument("--limit", type=int, default=3000)
    p.add_argument("--fee-bps", type=float, default=10.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--cusum-mult", type=float, default=0.8)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--pt-mult", type=float, default=1.5)
    p.add_argument("--sl-mult", type=float, default=1.0)
    p.add_argument("--prob-threshold", type=float, default=0.60)
    p.add_argument("--output", default="artifacts/replication_v02_report.json")
    args = p.parse_args()

    fallback = [args.exchange] + [x for x in ["coinex", "kraken", "okx"] if x != args.exchange]
    raw, used_exchange = fetch_with_fallback(fallback, args.symbol, args.timeframe, args.limit)
    feat = add_regime_features(add_features(raw)).set_index("timestamp", drop=False)
    vol = np.log(feat["close"]).diff().ewm(span=50, min_periods=20, adjust=False).std().shift(1)
    events = symmetric_cusum_events(feat["close"], vol * args.cusum_mult)
    labels = triple_barrier_events(feat, events, volatility=vol, horizon_bars=args.horizon, profit_take_mult=args.pt_mult, stop_loss_mult=args.sl_mult, allow_overlap=False)
    labels = labels[(~labels["ambiguous"]) & (labels["label"] != 0)].copy()
    if len(labels) < 80:
        raise RuntimeError(f"Too few non-ambiguous labeled events: {len(labels)}")

    feature_cols = [c for c in FEATURE_COLUMNS + REGIME_COLUMNS if c in feat.columns]
    event_features = feat.loc[labels["event_index"], feature_cols].copy()
    event_features.index = range(len(event_features))
    labels = labels.reset_index(drop=True)
    y = (labels["label"] == 1).astype(int)
    split = int(len(labels) * 0.70)
    train_end = max(1, split - args.horizon)
    X_train, y_train = event_features.iloc[:train_end], y.iloc[:train_end]
    X_test = event_features.iloc[split:]
    test_labels = labels.iloc[split:].copy().reset_index(drop=True)

    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_leaf_nodes=15, l2_regularization=1.0, random_state=42))])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    trade = prob >= args.prob_threshold
    gross = np.where(trade, test_labels["event_return"].to_numpy(), 0.0)
    friction = np.where(trade, (args.fee_bps + args.slippage_bps) / 10000.0 * 2.0, 0.0)
    net = pd.Series(gross - friction)
    oracle_direction = (test_labels["label"] == 1).to_numpy()

    report = {
        "research_status": "replication_lab_v0.2_component_test_not_production",
        "exchange": used_exchange, "symbol": args.symbol, "timeframe": args.timeframe,
        "bars_raw": int(len(raw)), "events_cusum": int(len(events)), "events_nonoverlap_labeled": int(len(labels)),
        "train_events": int(len(X_train)), "test_events": int(len(X_test)), "purge_events": int(args.horizon),
        "trades": int(trade.sum()), "trade_rate": float(trade.mean()),
        "classification_accuracy_at_0_5": float(((prob >= 0.5) == oracle_direction).mean()),
        "metrics": event_equity_metrics(net),
        "frictions": {"fee_bps_per_side": args.fee_bps, "slippage_bps_per_side": args.slippage_bps, "round_trip_assumption": True},
        "method": {"sampling": "symmetric CUSUM on log returns with lagged EWMA volatility threshold", "labeling": "long-side triple barrier, no overlapping events, ambiguous same-bar hits excluded", "model": "HistGradientBoostingClassifier", "abstention": f"trade only when P(profit barrier first) >= {args.prob_threshold}"},
        "limitations": ["OHLCV component test, not tick-data information-driven-bar replication", "CPCV/PBO/Deflated Sharpe not yet implemented", "Funding/order-book impact/capacity/on-chain not included", "Event metrics are not annualized"]
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
