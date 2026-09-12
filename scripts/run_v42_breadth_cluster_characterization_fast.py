from __future__ import annotations

"""Compute-optimized wrapper for the frozen v0.42 characterization.

Scientific protocol is unchanged. This wrapper performs only two engineering
optimizations:

1) memoize deterministic, seed-independent competing-risk design expansions;
2) fetch the three independent development venues concurrently, while keeping
   every symbol fetch sequential and rate-limited *within each venue*.

No feature, label, threshold, seed, fold, venue, symbol, cost, gate, eligibility
rule, model architecture, or hyperparameter is changed. Availability rows are
reassembled in the original frozen venue/symbol order.
"""

import importlib.util
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd

import research_bot.event_competing_risk_v41 as cr


_ORIGINAL_HAZARD = cr.build_hazard_training_table_v41
_ORIGINAL_EVENT = cr.event_design_matrix_v41
_HAZARD_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, tuple]] = OrderedDict()
_EVENT_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, object]] = OrderedDict()
_MAX_HAZARD_CACHE = 2
_MAX_EVENT_CACHE = 24


def _policy_key(policy) -> tuple:
    p = policy or cr.CompetingRiskPolicyV41()
    return (
        int(p.max_hold_bars),
        float(p.conformal_alpha),
        int(p.minimum_expert_events),
        int(p.minimum_expert_cause_events),
        int(p.minimum_family_conformal_events),
        float(p.hazard_cap),
        bool(p.require_target_probability_gt_stop),
    )


def _cached_hazard(events: pd.DataFrame, feature_columns: Sequence[str], policy=None):
    features = tuple(feature_columns)
    key = (id(events), len(events), features, _policy_key(policy))
    hit = _HAZARD_CACHE.get(key)
    if hit is not None and hit[0] is events:
        _HAZARD_CACHE.move_to_end(key)
        return hit[1]
    value = _ORIGINAL_HAZARD(events, feature_columns, policy)
    _HAZARD_CACHE[key] = (events, value)
    _HAZARD_CACHE.move_to_end(key)
    while len(_HAZARD_CACHE) > _MAX_HAZARD_CACHE:
        _HAZARD_CACHE.popitem(last=False)
    return value


def _cached_event(events: pd.DataFrame, feature_columns: Sequence[str]):
    features = tuple(feature_columns)
    key = (id(events), len(events), features)
    hit = _EVENT_CACHE.get(key)
    if hit is not None and hit[0] is events:
        _EVENT_CACHE.move_to_end(key)
        return hit[1]
    value = _ORIGINAL_EVENT(events, feature_columns)
    _EVENT_CACHE[key] = (events, value)
    _EVENT_CACHE.move_to_end(key)
    while len(_EVENT_CACHE) > _MAX_EVENT_CACHE:
        _EVENT_CACHE.popitem(last=False)
    return value


# Patch deterministic representation builders before loading the frozen runner.
cr.build_hazard_training_table_v41 = _cached_hazard
cr.event_design_matrix_v41 = _cached_event

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
spec = importlib.util.spec_from_file_location("v42_frozen_characterization", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 characterization runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _fetch_one_venue(venue: str):
    if venue == runner.V42_RESERVED_HOLDOUT:
        raise RuntimeError("Kraken is sealed in v0.42")
    ex = runner.base._exchange(venue)
    markets = ex.load_markets()
    local_frames: dict[tuple[str, str], pd.DataFrame] = {}
    local_bars: dict[str, int] = {}
    local_rows: list[dict] = []
    for symbol in runner.V42_SYMBOL_CANDIDATES:
        if symbol not in markets:
            local_bars[symbol] = 0
            local_rows.append({
                "venue": venue,
                "symbol": symbol,
                "available": False,
                "bars": 0,
                "first_bar": None,
                "last_bar": None,
                "eligibility_reason": "MISSING_MARKET",
            })
            continue
        try:
            frame = runner.base.fetch_fixed_ohlcv(ex, symbol)
        except Exception as exc:
            local_bars[symbol] = 0
            local_rows.append({
                "venue": venue,
                "symbol": symbol,
                "available": False,
                "bars": 0,
                "first_bar": None,
                "last_bar": None,
                "eligibility_reason": f"FETCH_OR_HISTORY_ERROR:{type(exc).__name__}",
            })
            continue
        local_frames[(venue, symbol)] = frame
        local_bars[symbol] = int(len(frame))
        local_rows.append({
            "venue": venue,
            "symbol": symbol,
            "available": True,
            "bars": int(len(frame)),
            "first_bar": frame["timestamp"].iloc[0],
            "last_bar": frame["timestamp"].iloc[-1],
            "eligibility_reason": "AVAILABLE",
        })
    return venue, local_frames, local_bars, local_rows


def _parallel_fetch_availability():
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    bars: dict[str, dict[str, int]] = {v: {} for v in runner.V42_DEVELOPMENT_VENUES}
    rows_by_venue: dict[str, list[dict]] = {}
    payload_by_venue = {}
    with ThreadPoolExecutor(max_workers=len(runner.V42_DEVELOPMENT_VENUES)) as pool:
        futures = {
            venue: pool.submit(_fetch_one_venue, venue)
            for venue in runner.V42_DEVELOPMENT_VENUES
        }
        # Resolve futures in frozen venue order so raised errors and assembled
        # metadata remain deterministic even though network work is concurrent.
        for venue in runner.V42_DEVELOPMENT_VENUES:
            payload_by_venue[venue] = futures[venue].result()

    rows: list[dict] = []
    for venue in runner.V42_DEVELOPMENT_VENUES:
        _, local_frames, local_bars, local_rows = payload_by_venue[venue]
        frames.update(local_frames)
        bars[venue] = local_bars
        rows_by_venue[venue] = local_rows
        rows.extend(local_rows)
    return frames, bars, pd.DataFrame(rows)


runner._fetch_availability = _parallel_fetch_availability


if __name__ == "__main__":
    runner.main()
